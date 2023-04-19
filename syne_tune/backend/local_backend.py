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
import logging
import os
import shutil
import sys
from operator import itemgetter
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from syne_tune.backend.trial_backend import TrialBackend, BUSY_STATUS
from syne_tune.num_gpu import get_num_gpus
from syne_tune.report import retrieve
from syne_tune.backend.trial_status import TrialResult, Status
from syne_tune.constants import ST_CHECKPOINT_DIR, ST_CONFIG_JSON_FNAME_ARG
from syne_tune.util import experiment_path, random_string, dump_json_with_numpy

logger = logging.getLogger(__name__)


if "OMP_NUM_THREADS" not in os.environ:
    logger.debug(
        "OMP_NUM_THREADS is not set, it is going to be set to 1 to avoid "
        "performance issues in case of many workers being used locally. "
        "Overrides this behavior by setting a custom value."
    )
    os.environ["OMP_NUM_THREADS"] = "1"


class LocalBackend(TrialBackend):
    """
    A backend running locally by spawning sub-process concurrently. Note that
    no resource management is done so the concurrent number of trials should
    be adjusted to the machine capacity.

    Additional arguments on top of parent class
    :class:`~syne_tune.backend.trial_backend.TrialBackend`:

    :param entry_point: Path to Python main file to be tuned
    :param rotate_gpus: In case several GPUs are present, each trial is
        scheduled on a different GPU. A new trial is preferentially
        scheduled on a free GPU, and otherwise the GPU with least prior
        assignments is chosen. If ``False``, then all GPUs are used at the same
        time for all trials. Defaults to ``True``.
    :param num_gpus_per_trial: Number of GPUs to be allocated to each trial.
        Must be not larger than the total number of GPUs available.
        Defaults to 1
    """

    def __init__(
        self,
        entry_point: str,
        delete_checkpoints: bool = False,
        pass_args_as_json: bool = False,
        rotate_gpus: bool = True,
        num_gpus_per_trial: int = 1,
    ):
        super(LocalBackend, self).__init__(
            delete_checkpoints=delete_checkpoints, pass_args_as_json=pass_args_as_json
        )

        assert Path(
            entry_point
        ).exists(), f"the script provided to tune does not exist ({entry_point})"
        self.entry_point = entry_point
        self.local_path = None
        self.trial_subprocess = dict()

        # GPU rotation
        # Note that the initialization is delayed until first used, so we can
        # be sure it happens on the instance running the training evaluations
        self.rotate_gpus = rotate_gpus
        self.num_gpus = None
        # Maps ``trial_id`` to list of GPUs currently assigned to this trial
        self.trial_gpu = None
        self.gpu_times_assigned = None
        self.num_gpus_per_trial = num_gpus_per_trial
        # sets the path where to write files, can be overridden later by Tuner.
        self.set_path(str(Path(experiment_path(tuner_name=random_string(length=10)))))
        # Trials which may currently be busy (status in ``BUSY_STATUS``). The
        # corresponding jobs are polled for status in ``busy_trial_ids``.
        self._busy_trial_id_candidates = set()

    def trial_path(self, trial_id: int) -> Path:
        """
        :param trial_id: ID of trial
        :return: Directory where files related to trial are written to
        """
        return self.local_path / str(trial_id)

    def checkpoint_trial_path(self, trial_id: int) -> Path:
        """
        :param trial_id: ID of trial
        :return: Directory where checkpoints for trial are written to and
            read from
        """
        return self.trial_path(trial_id) / "checkpoints"

    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        src_checkpoint_path = self.checkpoint_trial_path(src_trial_id)
        tgt_checkpoint_path = self.checkpoint_trial_path(tgt_trial_id)
        shutil.copytree(src_checkpoint_path, tgt_checkpoint_path)

    def delete_checkpoint(self, trial_id: int):
        checkpoint_path = self.checkpoint_trial_path(trial_id)
        shutil.rmtree(checkpoint_path, ignore_errors=True)

    def _prepare_for_schedule(self, num_gpus=None):
        """
        Called at the start of each :meth:`_schedule`.
        In particular, we initialize variables related to GPU scheduling, if
        ``rotate_gpus`` is set. This is done before the first call of
        :meth:`_schedule`, so we can be sure it runs on the target instance.
        """
        if self.rotate_gpus and self.num_gpus is None:
            if num_gpus is None:
                self.num_gpus = get_num_gpus()
            else:
                self.num_gpus = num_gpus
            logger.info(f"Detected {self.num_gpus} GPUs")
            if self.num_gpus_per_trial > self.num_gpus:
                logger.warning(
                    f"num_gpus_per_trial = {self.num_gpus_per_trial} is too "
                    f"large, reducing to {self.num_gpus}"
                )
                self.num_gpus_per_trial = self.num_gpus
            if self.num_gpus > 1:
                self.trial_gpu = dict()  # Maps running trials to GPUs
                # To break ties among GPUs (free ones have precedence)
                self.gpu_times_assigned = [0] * self.num_gpus
            else:
                # Nothing to rotate over
                self.rotate_gpus = False

    def _gpus_for_new_trial(self) -> List[int]:
        """
        Selects ``num_gpus_per_trial`` GPUs for trial to be scheduled on. GPUs
        not assigned to other running trials have precedence. Ties are resolved
        by selecting GPUs with the least number of previous assignments.
        The number of assignments is incremented for the GPUs returned.
        """
        assert self.rotate_gpus
        assigned_gpus = list(gpu for gpus in self.trial_gpu.values() for gpu in gpus)
        free_gpus = [x for x in range(self.num_gpus) if x not in assigned_gpus]
        num_extra = self.num_gpus_per_trial - len(free_gpus)
        if num_extra > 0:
            candidate_gpus = assigned_gpus[:num_extra]
            res_gpu = free_gpus  # Pick all free ones
        else:
            res_gpu = []
            candidate_gpus = free_gpus
        # We select the GPU which has the least prior assignments. Selection
        # over all GPUs currently free. If all GPUs are currently assigned,
        # selection is over all GPUs. In this case, the assignment will go to
        # a GPU currently occupied (this happens if the number of GPUs is
        # smaller than the number of workers).
        num_extra = self.num_gpus_per_trial - len(res_gpu)
        top_list = sorted(
            ((gpu, self.gpu_times_assigned[gpu]) for gpu in candidate_gpus),
            key=itemgetter(1),
        )[:num_extra]
        res_gpu = res_gpu + [gpu for gpu, _ in top_list]
        for gpu in res_gpu:
            self.gpu_times_assigned[gpu] += 1
        return res_gpu

    def _schedule(self, trial_id: int, config: Dict[str, Any]):
        self._prepare_for_schedule()
        trial_path = self.trial_path(trial_id)
        os.makedirs(trial_path, exist_ok=True)
        with open(trial_path / "std.out", "a") as stdout:
            with open(trial_path / "std.err", "a") as stderr:
                logger.debug(
                    f"scheduling {trial_id}, {self.entry_point}, {config}, logging into {trial_path}"
                )
                config_json_fname = str(trial_path / "config.json")
                if self.pass_args_as_json:
                    config_for_args = {ST_CONFIG_JSON_FNAME_ARG: config_json_fname}
                else:
                    config_for_args = config.copy()
                config_for_args[ST_CHECKPOINT_DIR] = str(
                    self.checkpoint_trial_path(trial_id)
                )
                config_str = " ".join(
                    [f"--{key} {value}" for key, value in config_for_args.items()]
                )

                dump_json_with_numpy(config, config_json_fname)
                cmd = f"{sys.executable} {self.entry_point} {config_str}"
                env = dict(os.environ)
                self._allocate_gpu(trial_id, env)
                logger.info(f"running subprocess with command: {cmd}")

                self.trial_subprocess[trial_id] = subprocess.Popen(
                    cmd.split(" "), stdout=stdout, stderr=stderr, env=env
                )
        self._busy_trial_id_candidates.add(trial_id)  # Mark trial as busy

    def _allocate_gpu(self, trial_id: int, env: Dict[str, Any]):
        if self.rotate_gpus:
            gpus = self._gpus_for_new_trial()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
            self.trial_gpu[trial_id] = gpus
            part = f"GPU {gpus[0]}" if len(gpus) == 1 else f"GPUs {gpus}"
            # logger.debug(f"Assigned {part} to trial_id {trial_id}")
            logger.info(f"*** Assigned {part} to trial_id {trial_id}")  # DEBUG

    def _deallocate_gpu(self, trial_id: int):
        if self.rotate_gpus and trial_id in self.trial_gpu:
            del self.trial_gpu[trial_id]

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
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

    def _release_from_worker(self, trial_id: int):
        if trial_id in self._busy_trial_id_candidates:
            self._busy_trial_id_candidates.remove(trial_id)

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        self._file_path(trial_id=trial_id, filename="pause").touch()
        self._kill_process(trial_id)
        self._deallocate_gpu(trial_id)
        self._release_from_worker(trial_id)

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
        self._release_from_worker(trial_id)

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

    def _get_busy_trial_ids(self) -> List[Tuple[int, str]]:
        busy_list = []
        for trial_id in self._busy_trial_id_candidates:
            status = self._read_status(trial_id)
            if status in BUSY_STATUS:
                busy_list.append((trial_id, status))
        return busy_list

    def busy_trial_ids(self) -> List[Tuple[int, str]]:
        # Note that at this point, ``self._busy_trial_id_candidates`` contains
        # trials whose jobs have been busy in the past, but they may have
        # stopped or terminated since. We query the current status for all
        # these jobs and update ``self._busy_trial_id_candidates`` accordingly.
        if self._busy_trial_id_candidates:
            busy_list = self._get_busy_trial_ids()
            # Update internal candidate list
            self._busy_trial_id_candidates = set(trial_id for trial_id, _ in busy_list)
            return busy_list
        else:
            return []

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
