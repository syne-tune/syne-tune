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
from typing import Dict, List, Tuple, Optional, Any
import logging

from syne_tune.backend.trial_status import TrialResult, Trial, Status
from syne_tune.constants import ST_WORKER_TIMESTAMP

logger = logging.getLogger(__name__)


TrialAndStatusInformation = Dict[int, Tuple[Trial, str]]


TrialIdAndResultList = List[Tuple[int, dict]]


BUSY_STATUS = {Status.in_progress, Status.stopping}


class TrialBackend:
    """
    Interface for backend to execute evaluations of trials.

    :param delete_checkpoints: If ``True``, the checkpoints written by a trial
        are deleted once the trial is stopped or is registered as
        completed. Checkpoints of paused trials may also be removed, if the
        scheduler supports early checkpoint removal. Also, as part of
        :meth:`stop_all` called at the end of the tuning loop, all remaining
        checkpoints are deleted. Defaults to ``False`` (no checkpoints are
        removed).
    :param pass_args_as_json: Normally, the hyperparameter configuration is
        passed as command line arguments to the trial evaluation script. This
        works if all hyperparameters have elementary types. If
        ``pass_args_as_json == True``, the configuration is instead written into
        a JSON file, whose name is passed as command line argument
        :const:`~syne_tune.constants.ST_CONFIG_JSON_FNAME_ARG`. The trial
        evaluation script then loads the configuration from this file. This allows
        the configuration to contain entries with complex types (e.g., lists or
        dictionaries), as long as they are JSON-serializable.
        Defaults to ``False``.
    """

    def __init__(
        self,
        delete_checkpoints: bool = False,
        pass_args_as_json: bool = False,
    ):
        self.delete_checkpoints = delete_checkpoints
        self.pass_args_as_json = pass_args_as_json
        self.trial_ids = []
        self._trial_dict = dict()

        # index of the last metric that was seen for each trial-id
        self._last_metric_seen_index = defaultdict(lambda: 0)

    def start_trial(
        self, config: Dict[str, Any], checkpoint_trial_id: Optional[int] = None
    ) -> TrialResult:
        """Start new trial with new trial ID

        :param config: Configuration for new trial
        :param checkpoint_trial_id: If given, the new trial starts from the
            checkpoint written by this previous trial
        :return: New trial, which includes new trial ID
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

        :param src_trial_id: Source trial ID (copy from)
        :param tgt_trial_id: Target trial ID (copy to)
        """
        raise NotImplementedError

    def delete_checkpoint(self, trial_id: int):
        """
        Removes checkpoint folder for a trial. It is OK for the folder not to
        exist.

        :param trial_id: ID of trial for which checkpoint files are deleted
        """
        raise NotImplementedError

    def resume_trial(
        self, trial_id: int, new_config: Optional[dict] = None
    ) -> TrialResult:
        """Resume paused trial

        :param trial_id: ID of (paused) trial to be resumed
        :param new_config: If given, the config maintained in ``trial.config`` is
            replaced by ``new_config``
        :return: Information for resumed trial
        """
        assert trial_id < len(
            self.trial_ids
        ), "cannot resume a trial id that is not present"
        trial = self._trial_dict[trial_id]
        assert (
            trial.status == Status.paused
        ), f"Cannot resume trial_id {trial_id} from status '{trial.status}', must be '{Status.paused}'"
        self._resume_trial(trial_id)
        if new_config is not None:
            trial.config = new_config
        self._schedule(
            trial_id=trial_id,
            config=trial.config,
        )
        trial.status = Status.in_progress
        return trial

    def _resume_trial(self, trial_id: int):
        """Called in :meth:`resume_trial`, before job is scheduled.

        :param trial_id: See ``resume_trial``
        """
        raise NotImplementedError

    def pause_trial(self, trial_id: int, result: Optional[dict] = None):
        """Pauses a running trial

        Checks that the operation is valid and calls backend internal
        implementation to actually pause the trial.
        If the status is queried after this function, it should be ``"paused"``.

        :param trial_id: ID of trial to pause
        :param result: Result dict based on which scheduler decided to pause the
            trial
        """
        assert trial_id < len(self.trial_ids), f"Invalid trial_id = {trial_id}"
        self._trial_dict[trial_id].status = Status.paused
        self._pause_trial(trial_id=trial_id, result=result)
        self._cleanup_after_trial(trial_id)

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        """Implements :meth:`pause_trial`.

        :param trial_id: ID of trial to pause
        :param result: Result dict based on which scheduler decided to pause the
            trial
        """
        raise NotImplementedError

    def stop_trial(self, trial_id: int, result: Optional[dict] = None):
        """Stops (and terminates) a running trial

        Checks that the operation is valid and calls backend internal
        implementation to actually stop the trial. f the status is queried after
        this function, it should be ``"stopped"``.

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
        self._cleanup_after_trial(trial_id)

    def _cleanup_after_trial(self, trial_id: int):
        """
        This is called whenever a trial is stopped or paused.
        Note that ``delete_checkpoints`` should not be dealt with here, since
        checkpoints must not be deleted when a trial is paused.

        :param trial_id: ID of trial to clean up after
        """
        pass

    def _stop_trial(self, trial_id: int, result: Optional[dict]):
        """Backend specific operation that stops the trial.

        :param trial_id: ID of trial to stop
        :param result: Result dict based on which scheduler decided to stop the
            trial
        """
        raise NotImplementedError

    def new_trial_id(self) -> int:
        return len(self.trial_ids)

    def _schedule(self, trial_id: int, config: Dict[str, Any]):
        """Schedules job for trial evaluation.

        Called by :meth:`start_trial`, :meth:`resume_trial`.

        :param trial_id: ID of trial to schedule
        :param config: Configuration for this trial
        """
        raise NotImplementedError

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        """Returns results for selected trials

        :param trial_ids: IDs of trials for which results are to be queried
        :return: list of results corresponding to ``trial_ids``, contains all the
            results obtained since the start of each trial.
        """
        raise NotImplementedError

    def fetch_status_results(
        self, trial_ids: List[int]
    ) -> (TrialAndStatusInformation, TrialIdAndResultList):
        """
        :param trial_ids: Trials whose information should be fetched.
        :return: A tuple containing 1) a dictionary from trial-id to Trial and status
            information; 2) a list of (trial-id, results) pairs for each new result
            emitted since the last call. The list of results is sorted by the worker
            time-stamp.
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
                for new_metric in new_metrics:
                    results.append((trial_id, new_metric))

        trial_status_dict = dict()
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

    def busy_trial_ids(self) -> List[Tuple[int, str]]:
        """Returns list of ids for currently busy trials

        A trial is busy if its status is
        :const:`~syne_tune.backend.trial_status.Status.in_progress` or
        :const:`~syne_tune.backend.trial_status.Status.stopping`.
        If the execution setup is able to run ``n_workers`` jobs in parallel,
        then if this method returns a list of size ``n``, the tuner may start
        ``n_workers - n`` new jobs.

        :return: List of ``(trial_id, status)``
        """
        raise NotImplementedError

    def stdout(self, trial_id: int) -> List[str]:
        """Fetch ``stdout`` log for trial

        :param trial_id: ID of trial
        :return: Lines of the log of the trial (stdout)
        """
        raise NotImplementedError

    def stderr(self, trial_id: int) -> List[str]:
        """Fetch ``stderr`` log for trial

        :param trial_id: ID of trial
        :return: Lines of the log of the trial (stderr)
        """
        raise NotImplementedError

    def stop_all(self):
        """Stop all trials which are in progress."""
        trial_results = self._all_trial_results(self.trial_ids)
        for trial in trial_results:
            if trial.status == Status.in_progress:
                self.stop_trial(trial_id=trial.trial_id)
        if self.delete_checkpoints:
            # Delete all remaining checkpoints (e.g., of paused trials)
            # We loop over all trials here, but ``delete_checkpoints`` does nothing
            # if the checkpoint has already been deleted before
            logger.info("Removing all remaining checkpoints of trials")
            for trial_id in self.trial_ids:
                self.delete_checkpoint(trial_id=trial_id)
                self._cleanup_after_trial(trial_id)

    def set_path(
        self, results_root: Optional[str] = None, tuner_name: Optional[str] = None
    ):
        """
        :param results_root: The local folder that should contain the results of
            the tuning experiment. Used by :class:`~syne_tune.Tuner` to indicate
            a desired path where the results should be written to. This is used
            to unify the location of backend files and :class:`~syne_tune.Tuner`
            results when possible (in the local backend). By default, the backend
            does not do anything since not all backends may be able to unify their
            file locations.
        :param tuner_name: Name of the tuner, can be used for instance to save
            checkpoints on remote storage.
        """
        pass

    def entrypoint_path(self) -> Path:
        """
        :return: Entrypoint path of script to be executed
        """
        raise NotImplementedError

    def set_entrypoint(self, entry_point: str):
        """Update the entrypoint.

        :param entry_point: New path of the entrypoint.
        """
        raise NotImplementedError

    def on_tuner_save(self):
        """
        Called at the end of :meth:`~syne_tune.Tuner.save`.
        """
        pass
