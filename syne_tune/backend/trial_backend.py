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
from collections import defaultdict

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from syne_tune.backend.trial_status import TrialResult, Trial, Status
from syne_tune.constants import ST_WORKER_TIMESTAMP

logger = logging.getLogger(__name__)


class TrialBackend:
    def __init__(self, delete_checkpoints: bool = False):
        """
        If `delete_checkpoints` is True, the checkpoints written by a trial are
        deleted once the trial is stopped or is registered as completed. Also,
        as part of `stop_all` called at the end of the tuning loop, all remaining
        checkpoints are deleted.

        :param delete_checkpoints: See above

        """
        self.delete_checkpoints = delete_checkpoints
        self.trial_ids = []
        self._trial_dict = {}

        # index of the last metric that was seen for each trial-id
        self._last_metric_seen_index = defaultdict(lambda: 0)

    def start_trial(
        self, config: dict, checkpoint_trial_id: Optional[int] = None
    ) -> TrialResult:
        """
        :param config: program arguments of `script`
        :param checkpoint_trial_id: id of a trial to be resumed, if given the checkpoint of this trial-id is be copied
        to the checkpoint of the new trial-id.
        """
        trial_id = self.new_trial_id()
        if checkpoint_trial_id is not None:
            self.copy_checkpoint(
                src_trial_id=checkpoint_trial_id, tgt_trial_id=trial_id
            )
        self.trial_ids.append(trial_id)
        self._schedule(trial_id=trial_id, config=config)
        now = datetime.now()
        trial = TrialResult(
            trial_id=trial_id,
            config=config,
            creation_time=now,
            status=Status.in_progress,
            metrics=[],
        )
        self._trial_dict[trial_id] = trial

        return trial

    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        """
        Copy the checkpoint folder from one trial to the other.

        :param src_trial_id:
        :param tgt_trial_id:
        """
        raise NotImplementedError()

    def delete_checkpoint(self, trial_id: int):
        """
        Removes checkpoint folder for a trial. It is OK for the folder not to
        exist.

        :param trial_id:
        """
        raise NotImplementedError()

    def resume_trial(self, trial_id: int, new_config: Optional[dict] = None):
        """
        :param trial_id: id of the trial to be resumed
        :param new_config: If given, the config maintained in trial.config is
            replaced by new_config
        :return:
        """
        assert trial_id < len(
            self.trial_ids
        ), "cannot resume a trial id that is not present"
        # todo assert that status is not running
        trial = self._trial_dict[trial_id]
        self._resume_trial(trial_id)
        if new_config is not None:
            trial.config = new_config
        self._schedule(
            trial_id=trial_id,
            config=trial.config,
        )

    def _resume_trial(self, trial_id: int):
        """
        update internal backend information when a trial gets resumed
        """
        raise NotImplementedError()

    def pause_trial(self, trial_id: int, result: Optional[dict] = None):
        """
        Checks that the operation is valid and call backend internal implementation to actually pause the trial.
        If the status is queried after this function, it should be `paused`.

        :param trial_id: ID of trial to pause
        :param result: Result dict based on which scheduler decided to pause the
            trial
        """
        # todo assert trial_id is valid
        # todo assert trial_id has not been stopped or paused before
        self._pause_trial(trial_id=trial_id, result=result)

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        """
        Backend specific operation that pauses the trial.
        """
        raise NotImplementedError()

    def stop_trial(self, trial_id: int, result: Optional[dict] = None):
        """
        Checks that the operation is valid and call backend internal implementation to actually stop the trial.
        If the status is queried after this function, it should be `stopped`.
        :param trial_id: ID of trial to stop
        :param result: Result dict based on which scheduler decided to stop the
            trial
        """
        # todo assert trial_id is valid
        # todo assert trial_id has not been stopped or paused before
        self._stop_trial(trial_id=trial_id, result=result)
        if self.delete_checkpoints:
            logger.info(f"Removing checkpoints for trial_id = {trial_id}")
            self.delete_checkpoint(trial_id=trial_id)  # checkpoint not needed anymore

    def _stop_trial(self, trial_id: int, result: Optional[dict]):
        """
        Backend specific operation that stops the trial.
        """
        raise NotImplementedError()

    def new_trial_id(self) -> int:
        return len(self.trial_ids)

    def _schedule(self, trial_id: int, config: dict):
        raise NotImplementedError()

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        """
        :param trial_ids:
        :return: list of results corresponding to the trial-id passed, contains all the results obtained since the start
        of the trial.
        """
        pass

    def fetch_status_results(
        self, trial_ids: List[int]
    ) -> Tuple[Dict[int, Tuple[Trial, str]], List[Tuple[int, dict]]]:
        """
        :param trial_ids: trials whose information should be fetch.
        :return: A tuple containing 1) a dictionary from trial-id to Trial and status information 2) list of
        trial-id/results pair for each new result that was emitted since the last call. The list of results is sorted
         by the worker time-stamp (last time-stamp appears last).
        """
        all_trial_results = self._all_trial_results(trial_ids)
        results = []
        for trial_result in all_trial_results:
            trial_id = trial_result.trial_id
            self._trial_dict[trial_id] = trial_result
            if len(trial_result.metrics) > 0:
                if trial_result.status in [
                    Status.paused,
                    Status.stopping,
                    Status.stopped,
                ]:
                    # metrics obtained after a stopping decision from a scheduler are hidden.
                    new_metrics = []
                else:
                    # we return the list of all new metrics, which may be empty if no new metrics were generated.
                    position_last_seen = self._last_metric_seen_index[trial_id]
                    new_metrics = trial_result.metrics[position_last_seen:]
                    self._last_metric_seen_index[trial_id] += len(new_metrics)
                    if (
                        self.delete_checkpoints
                        and trial_result.status == Status.completed
                    ):
                        logger.info(f"Removing checkpoints for trial_id = {trial_id}")
                        self.delete_checkpoint(trial_id=trial_id)
                for new_metric in new_metrics:
                    results.append((trial_id, new_metric))

        trial_status_dict = {}
        for trial_id in trial_ids:
            trial_result = self._trial_dict[trial_id]
            # we cast TrialResult to Trial to avoid downstream code depending on TrialResult which we should ultimately
            # remove (since it duplicates several information such as status or list of results)
            trial = Trial(
                trial_id=trial_result.trial_id,
                config=trial_result.config,
                creation_time=trial_result.creation_time,
            )
            trial_status_dict[trial_id] = (trial, trial_result.status)
        results = sorted(results, key=lambda result: result[1][ST_WORKER_TIMESTAMP])
        return trial_status_dict, results

    def stdout(self, trial_id: int) -> List[str]:
        """
        :param trial_id:
        :return: lines of the log of the trial (stdout)
        """
        raise NotImplementedError()

    def stderr(self, trial_id: int) -> List[str]:
        """
        :param trial_id:
        :return: lines of the log of the trial (stderr)
        """
        raise NotImplementedError()

    def stop_all(self):
        trial_results = self._all_trial_results(self.trial_ids)
        for trial in trial_results:
            if trial.status == Status.in_progress:
                self.stop_trial(trial_id=trial.trial_id)
        if self.delete_checkpoints:
            # Delete all remaining checkpoints (e.g., of paused trials)
            logger.info("Removing all remaining checkpoints of trials")
            for trial_id in self.trial_ids:
                self.delete_checkpoint(trial_id=trial_id)

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        """
        :param results_root: the local folder that should contains the results of the tuning experiment.
        Used by Tuner to indicate a desired path where the results should be written to. This is used
         to unify the location of backend files and Tuner results when possible (in the local backend).
         By default, the backend does not do anything since not all backends may be able to unify their files
         locations.
        :param tuner_name: name of the tuner can be used for instance to save checkpoints on remote storage.
        """
        pass

    def entrypoint_path(self) -> Path:
        """
        :return: the path of the entrypoint to be executed
        """
        pass

    def set_entrypoint(self, entry_point: str):
        """
        Update the entrypoint to point path.
        :param entry_point: new path of the entrypoint.
        :return:
        """
        pass

    def on_tuner_save(self):
        """
        Called by :class:`Tuner` at the end of `save`
        """
        pass
