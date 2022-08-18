# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import json
import logging
import os
import shutil
import sys

import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from syne_tune.backend.trial_backend import TrialBackend
from syne_tune.num_gpu import get_num_gpus
from syne_tune.report import retrieve
from syne_tune.backend.trial_status import TrialResult, Status
from syne_tune.constants import ST_CHECKPOINT_DIR
from syne_tune.util import experiment_path, random_string

logger = logging.getLogger(__name__)


if "OMP_NUM_THREADS" not in os.environ:
    logger.debug(
        "OMP_NUM_THREADS is not set, it is going to be set to 1 to avoid performance issues in case of many "
        "workers are used locally. Overrides this behavior by setting a custom value."
    )
    os.environ["OMP_NUM_THREADS"] = "1"


class LocalBackend(TrialBackend):
    def __init__(
        self,
        entry_point: str,
        rotate_gpus: bool = True,
        delete_checkpoints: bool = False,
    ):
        """
        A backend running locally by spawning sub-process concurrently.
        Note that no resource management is done so the concurrent number of
        trials should be adjusted to the machine capacity.

        :param entry_point: python main file to be tuned
        :param rotate_gpus: in case several GPUs are present, each trial is
            scheduled on a different GPU. A new trial is preferentially
            scheduled on a free GPU, and otherwise the GPU with least prior
            assignments is chosen. If False, then all GPUs are used at the same
            time for all trials.
        :param delete_checkpoints: If True, checkpoints of stopped or completed
            trials are deleted

        """
        super(LocalBackend, self).__init__(delete_checkpoints)

        assert Path(
            entry_point
        ).exists(), f"the script provided to tune does not exist ({entry_point})"
        self.entry_point = entry_point

        self.trial_subprocess = {}

        # GPU rotation
        # Note that the initialization is delayed until first used, so we can
        # be sure it happens on the instance running the training evaluations
        self.rotate_gpus = rotate_gpus
        self.num_gpus = None
        self.trial_gpu = None
        self.gpu_times_assigned = None

        # sets the path where to write files, can be overidden later by Tuner.
        self.set_path(Path(experiment_path(tuner_name=random_string(length=10))))

    def trial_path(self, trial_id: int) -> Path:
        return self.local_path / str(trial_id)

    def _checkpoint_trial_path(self, trial_id: int):
        return self.trial_path(trial_id) / "checkpoints"

    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        src_checkpoint_path = self._checkpoint_trial_path(src_trial_id)
        tgt_checkpoint_path = self._checkpoint_trial_path(tgt_trial_id)
        shutil.copytree(src_checkpoint_path, tgt_checkpoint_path)

    def delete_checkpoint(self, trial_id: int):
        checkpoint_path = self._checkpoint_trial_path(trial_id)
        shutil.rmtree(checkpoint_path, ignore_errors=True)

    def _prepare_for_schedule(self, num_gpus=None):
        """
        Called at the start of each `_schedule`.
        In particular, we initialize variables related to GPU scheduling, if
        `rotate_gpus' is set. This is done before the first call of `_schedule`,
        so we can be sure it runs on the target instance.

        """
        if self.rotate_gpus and self.num_gpus is None:
            if num_gpus is None:
                self.num_gpus = get_num_gpus()
            else:
                self.num_gpus = num_gpus
            logging.info(f"Detected {self.num_gpus} GPUs")
            if self.num_gpus > 1:
                self.trial_gpu = dict()  # Maps running trials to GPUs
                # To break ties among GPUs (free ones have precedence)
                self.gpu_times_assigned = [0] * self.num_gpus
            else:
                # Nothing to rotate over
                self.rotate_gpus = False

    def _gpu_for_new_trial(self) -> int:
        """
        Selects GPU for trial to be scheduled on. GPUs not assigned to other
        running trials have precedence. Ties are resolved by selecting a GPU
        with the least number of previous assignments.
        The number of assignments is incremented for the GPU returned.

        """
        assert self.rotate_gpus
        free_gpus = set(range(self.num_gpus)).difference(self.trial_gpu.values())
        if free_gpus:
            eligible_gpus = free_gpus
            logging.debug(f"Free GPUs: {free_gpus}")
        else:
            eligible_gpus = range(self.num_gpus)
        # We select the GPU which has the least prior assignments. Selection
        # over all GPUs currently free. If all GPUs are currently assigned,
        # selection is over all GPUs. In this case, the assignment will go to
        # a GPU currently occupied (this happens if the number of GPUs is
        # smaller than the number of workers).
        res_gpu, _ = min(
            ((gpu, self.gpu_times_assigned[gpu]) for gpu in eligible_gpus),
            key=lambda x: x[1],
        )
        self.gpu_times_assigned[res_gpu] += 1
        return res_gpu

    def _schedule(self, trial_id: int, config: Dict):
        self._prepare_for_schedule()
        trial_path = self.trial_path(trial_id)
        os.makedirs(trial_path, exist_ok=True)
        with open(trial_path / "std.out", "a") as stdout:
            with open(trial_path / "std.err", "a") as stderr:
                logging.debug(
                    f"scheduling {trial_id}, {self.entry_point}, {config}, logging into {trial_path}"
                )
                config_copy = config.copy()
                config_copy[ST_CHECKPOINT_DIR] = str(trial_path / "checkpoints")
                config_str = " ".join(
                    [f"--{key} {value}" for key, value in config_copy.items()]
                )

                def np_encoder(object):
                    if isinstance(object, np.generic):
                        return object.item()

                with open(trial_path / "config.json", "w") as f:
                    # the encoder fixes json error "TypeError: Object of type 'int64' is not JSON serializable"
                    json.dump(config, f, default=np_encoder)

                cmd = f"{sys.executable} {self.entry_point} {config_str}"

                env = dict(os.environ)
                self._allocate_gpu(trial_id, env)

                logging.info(f"running subprocess with command: {cmd}")

                self.trial_subprocess[trial_id] = subprocess.Popen(
                    cmd.split(" "), stdout=stdout, stderr=stderr, env=env
                )

    def _allocate_gpu(self, trial_id: int, env: dict):
        if self.rotate_gpus:
            gpu = self._gpu_for_new_trial()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.trial_gpu[trial_id] = gpu
            logging.debug(f"Assigned GPU {gpu} to trial_id {trial_id}")

    def _deallocate_gpu(self, trial_id: int):
        if self.rotate_gpus and trial_id in self.trial_gpu:
            del self.trial_gpu[trial_id]

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        """
        :param trial_ids: list of trial-ids whose status must be retrieved
        :return:
        """
        res = []
        for trial_id in trial_ids:
            trial_path = self.trial_path(trial_id)
            status = self._read_status(trial_id)

            if status != Status.in_progress:
                # Trial completed or failed: Deallocate GPU
                self._deallocate_gpu(trial_id)

            # If the job has finished, we read its end-time in a time-stamp.
            # If the time-stamp does not exist and the job finished, we create it. As a consequence the end-time is
            # an (over)-approximation. It is not clear how to avoid this without running the command in shell mode
            # (which allows to write a time-stamp when the process finishes) but it is probably OK if all_results
            # is called every few seconds.
            if os.path.exists(trial_path / "end"):
                training_end_time = self._read_time_stamp(trial_id=trial_id, name="end")
            else:
                training_end_time = datetime.now()

                # if the time-stamp is not present, we check whether the job has finished,
                # if this is the case we create a time-stamp to mark now as the end-time.
                if self._is_process_done(trial_id=trial_id):
                    self._write_time_stamp(trial_id=trial_id, name="end")

            metrics = retrieve(log_lines=self.stdout(trial_id=trial_id))
            trial_results = self._trial_dict[trial_id].add_results(
                metrics=metrics,
                status=status,
                training_end_time=training_end_time,
            )
            res.append(trial_results)
        return res

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        self._file_path(trial_id=trial_id, filename="pause").touch()
        self._kill_process(trial_id)
        self._deallocate_gpu(trial_id)

    def _resume_trial(self, trial_id: int):
        pause_path = self._file_path(trial_id=trial_id, filename="pause")
        try:
            pause_path.unlink()
        except FileNotFoundError:
            logger.info(f"Pause lock file {str(pause_path)} not found")

    def _stop_trial(self, trial_id: int, result: Optional[dict]):
        self._file_path(trial_id=trial_id, filename="stop").touch()
        self._kill_process(trial_id)
        self._deallocate_gpu(trial_id)

    def _kill_process(self, trial_id: int):
        # send a kill process to the process
        process = self.trial_subprocess[trial_id]
        try:
            process.kill()
        except ProcessLookupError as e:
            pass

    def _file_path(self, trial_id: int, filename: str):
        return Path(self.trial_path(trial_id=trial_id) / filename)

    def _write_time_stamp(self, trial_id: int, name: str):
        time_stamp_path = self._file_path(trial_id=trial_id, filename=name)
        with open(time_stamp_path, "w") as f:
            f.write(str(datetime.now().timestamp()))

    def _read_time_stamp(self, trial_id: int, name: str):
        time_stamp_path = self._file_path(trial_id=trial_id, filename=name)
        with open(time_stamp_path, "r") as f:
            return datetime.fromtimestamp(float(f.readline()))

    def _is_process_done(self, trial_id: int) -> bool:
        return self.trial_subprocess[trial_id].poll() is not None

    def _read_status(self, trial_id: int):
        if self._file_path(trial_id=trial_id, filename="stop").exists():
            return Status.stopped
        elif self._file_path(trial_id=trial_id, filename="pause").exists():
            return Status.paused
        else:
            code = self.trial_subprocess[trial_id].poll()
            if code is None:
                return Status.in_progress
            else:
                if code == 0:
                    return Status.completed
                else:
                    return Status.failed

    def stdout(self, trial_id: int) -> List[str]:
        with open(self.trial_path(trial_id=trial_id) / "std.out", "r") as f:
            return f.readlines()

    def stderr(self, trial_id: int) -> List[str]:
        with open(self.trial_path(trial_id=trial_id) / "std.err", "r") as f:
            return f.readlines()

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        self.local_path = Path(results_root)

    def entrypoint_path(self) -> Path:
        return Path(self.entry_point)

    def set_entrypoint(self, entry_point: str):
        self.entry_point = entry_point

    def __str__(self):
        return f"local entry_point {Path(self.entry_point).name}"
