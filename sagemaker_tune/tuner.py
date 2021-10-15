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
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Callable, Tuple, Optional, Dict

import dill as dill

from sagemaker_tune.backend.backend import Backend
from sagemaker_tune.backend.sagemaker_backend.sagemaker_backend import SagemakerBackend
from sagemaker_tune.backend.trial_status import Status, Trial
from sagemaker_tune.constants import SMT_TUNER_CREATION_TIMESTAMP
from sagemaker_tune.optimizer.scheduler import SchedulerDecision, TrialScheduler
from sagemaker_tune.tuner_callback import TunerCallback, StoreResultsCallback
from sagemaker_tune.tuning_status import TuningStatus, print_best_metric_found
from sagemaker_tune.util import RegularCallback, experiment_path, name_from_base

logger = logging.getLogger(__name__)


DEFAULT_SLEEP_TIME = 5.0

class Tuner:

    def __init__(
            self,
            backend: Backend,
            scheduler: TrialScheduler,
            stop_criterion: Callable[[TuningStatus], bool],
            n_workers: int,
            sleep_time: float = DEFAULT_SLEEP_TIME,
            results_update_interval: float = 10.0,
            print_update_interval: float = 30.0,
            max_failures: int = 1,
            tuner_name: Optional[str] = None,
            asynchronous_scheduling: bool = True,
            metadata: Optional[Dict] = None,
    ):
        """
        Allows to run an tuning job, call `run` after initializing.
        :param backend:
        :param scheduler:
        :param stop_criterion: the tuning stops when this predicates returns True, called each iteration with the
        current tuning status, for instance pass `stop_criterion=lambda status: status.num_trials_completed > 200`
        to stop after 200 completed jobs.
        :param n_workers: Number of workers used here. Note that the backend
            needs to support (at least) this number of workers to be run
            in parallel
        :param sleep_time: time to sleep when all workers are busy
        :param results_update_interval: frequency at which results are updated and stored in seconds
        :param max_failures: max failures allowed,
        :param tuner_name: name associated with the tuning experiment, default to a random date string. must be unique.
        :param asynchronous_scheduling: whether to use asynchronous scheduling when scheduling new trials. If `True`,
        trials are scheduled as soon as a worker is available, if `False`, the tuner waits that all trials are finished
         before scheduling a new batch.
        :param metadata: dictionary of user-metadata that will be persistend in {tuner_path}/metadata.json, in addition
        to the metadata provided by the user, `SMT_TUNER_CREATION_TIMESTAMP` is always included which measures
        the time-stamp when the tuner started to run.
        """
        self.backend = backend
        self.scheduler = scheduler
        self.n_workers = n_workers
        self.sleep_time = sleep_time
        self.results_update_interval = results_update_interval
        self.stop_criterion = stop_criterion
        self.asynchronous_scheduling = asynchronous_scheduling
        self.metadata = metadata
        self.max_failures = max_failures
        self.print_update_interval = print_update_interval

        # we keep track of the last result seen to send it to schedulers when trials complete.
        self.last_seen_result_per_trial = {}

        if tuner_name is not None:
            tuner_name = tuner_name.replace("_", "-")
        else:
            tuner_name = Path(self.backend.entrypoint_path()).stem
        self.name = name_from_base(tuner_name, default="smt-tuner")
        self.tuner_path = Path(experiment_path(tuner_name=self.name))

        logger.info(f"results of trials will be saved on {self.tuner_path}")

        # inform the backend to the name of the tuner. This allows the local backend
        # to store the logs and tuner results in the same folder.
        self.backend.set_path(results_root=self.tuner_path, tuner_name=self.name)

    def run(
            self,
            callbacks: Optional[List[TunerCallback]] = None,
    ) -> TuningStatus:
        """
        Launches the tuning.
        :param callbacks: called when events happens in the tuning loop such as when a result is seen, by default
        a callback that stores results every `results_update_interval` is used.
        :return: the tuning status when finished
        """
        if callbacks is None:
            callbacks = [self._default_callback()]

        for callback in callbacks:
            callback.on_tuning_start(self)

        self.tuner_path.mkdir(exist_ok=True, parents=True)

        with open(self.tuner_path / "metadata.json", "w") as f:
            metadata = {
                SMT_TUNER_CREATION_TIMESTAMP: time.time(),
            }
            if self.metadata is not None:
                metadata.update(self.metadata)
            json.dump(metadata, f)

        all_done_trials = OrderedDict()
        # `running_trial_ids` contains the ids of all trials currently running,
        # whether they were started from scratch or were resumed from a pausing
        # state
        running_trials_ids = set()
        tuning_status = TuningStatus(
            metric_names=self.scheduler.metric_names(),
            metric_mode=self.scheduler.metric_mode())

        # allows to print the status only every 30s
        status_printer = RegularCallback(
            call_seconds_frequency=self.print_update_interval,
            callback=lambda tuning_status: logger.info("tuning status\n" + str(tuning_status)),
        )

        tuner_saver = RegularCallback(
            callback=lambda tuner: tuner.save(),
            call_seconds_frequency=self.results_update_interval
        )

        while not self.stop_criterion(tuning_status) and tuning_status.num_trials_failed < self.max_failures:
            for callback in callbacks:
                callback.on_loop_start()

            # fetch new results
            trial_status_dict, new_results = self.backend.fetch_status_results(trial_ids=list(running_trials_ids))

            for callback in callbacks:
                callback.on_fetch_status_results(trial_status_dict=trial_status_dict, new_results=new_results)

            # update status with new results and all done trials
            tuning_status.update(
                trial_status_dict=trial_status_dict,
                new_results=new_results
            )

            assert len(running_trials_ids) <= self.n_workers

            # gets list of trials that are done with the new results (could be because they completed or because the
            # scheduler decided to interrupt them
            # Note: `done_trials` includes trials which are paused
            done_trials = self._update_running_trials(trial_status_dict, new_results, callbacks=callbacks)

            # update the list of done trials and remove those from `running_trials_ids`
            all_done_trials.update(done_trials)
            running_trials_ids.difference_update(done_trials.keys())

            running_trials_threshold = self.n_workers \
                if self.asynchronous_scheduling else 1
            num_running_trials = len(running_trials_ids)
            if len(new_results) == 0 and \
                    num_running_trials >= running_trials_threshold:
                # Note: For synchronous scheduling, we need to sleep here if at
                # least one worker is busy
                logger.debug(
                    f"{num_running_trials} of {self.n_workers} workers are "
                    f"busy, wait for {self.sleep_time} seconds")
                time.sleep(self.sleep_time)

                for callback in callbacks:
                    callback.on_tuning_sleep(self.sleep_time)

            else:
                # Schedule as many trials as we have free workers
                for i in range(self.n_workers - num_running_trials):
                    trial_id = self._schedule_new_task()
                    if trial_id is None:
                        logger.info("Searcher ran out of candidates, tuning job is stopping.")
                        # todo should also stop parent loop
                        break
                    running_trials_ids.add(trial_id)

                # Save tuner state only if there have been new results
                tuner_saver(tuner=self)

            status_printer(tuning_status)

            for callback in callbacks:
                callback.on_loop_end()

        print_best_metric_found(
            tuning_status=tuning_status,
            metric_names=self.scheduler.metric_names(),
            mode=self.scheduler.metric_mode(),
        )

        # save intermediate checkpoint of tuner
        self.save()

        # stop every job still running
        self.backend.stop_all()

        # in case too many errors were triggered, exit the loop and show log of last failed job.
        if tuning_status.num_trials_failed >= self.max_failures:
            self._handle_failure(all_done_trials=all_done_trials)

        for callback in callbacks:
            callback.on_tuning_end()

        return tuning_status

    def _schedule_new_task(self) -> Optional[int]:
        """
        Schedules a new task according to scheduler suggestion.
        :return: the trial-id of the task suggested, None if the scheduler was done.
        """
        trial_id = self.backend.new_trial_id()
        suggestion = self.scheduler.suggest(trial_id=trial_id)
        if suggestion is None:
            logger.info("Searcher ran out of candidates, tuning job is stopping.")
            raise StopIteration
        elif suggestion.spawn_new_trial_id:
            # we schedule a new trial, possibly using the checkpoint of `checkpoint_trial_id`
            # if given.
            trial = self.backend.start_trial(
                config=suggestion.config.copy(),
                checkpoint_trial_id=suggestion.checkpoint_trial_id
            )
            self.scheduler.on_trial_add(trial=trial)
            logger.info(f"(trial {trial_id}) - scheduled {suggestion}")
            return trial_id
        else:
            # suggestion is a trial_id to resume, with possibly a new configuration
            log_msg = f"Resuming trial {suggestion.checkpoint_trial_id}"
            if suggestion.config is not None:
                log_msg += f" with new_config = {suggestion.config}"
            logger.info(log_msg)
            self.backend.resume_trial(
                trial_id=suggestion.checkpoint_trial_id, new_config=suggestion.config)
            return suggestion.checkpoint_trial_id

    def _handle_failure(self, all_done_trials: Dict[int, Tuple[Trial, str]]):
        logger.error(f"Stopped as {self.max_failures} failures were reached")
        for trial_id, (trial, status) in all_done_trials.items():
            if status == Status.failed:
                logger.error(f"showing log of first failure")
                logger.error(f"{trial}")
                stdout = "".join(self.backend.stdout(trial_id))
                stderr = "".join(self.backend.stderr(trial_id))
                logger.error(stdout)
                logger.error(stderr)
                raise ValueError(f"Trial - {trial_id} failed")

    def save(self, folder: Optional[str] = None):
        if folder is None:
            tuner_serialized_path = self.tuner_path / "tuner.dill"
        else:
            tuner_serialized_path = Path(folder) / "tuner.dill"
        with open(tuner_serialized_path, "wb") as f:
            logger.debug(f"saving tuner in {tuner_serialized_path}")
            dill.dump(self, f)
            # ugly hack to reinitialize the session, we could remove it by having kwargs/args of SagemakerFramework
            # plus the class (for instance PyTorch)
            if isinstance(self.backend, SagemakerBackend):
                self.backend.initialize_sagemaker_session()

    @staticmethod
    def load(tuner_path: Optional[str]):
        with open(Path(tuner_path) / "tuner.dill", "rb") as f:
            tuner = dill.load(f)
            tuner.tuner_path = Path(experiment_path(tuner_name=tuner.name))
            return tuner

    def _update_running_trials(
            self,
            trial_status_dict: Dict[int, Tuple[Trial, str]],
            new_results: List[Tuple[int, Dict]],
            callbacks: List[TunerCallback],
    ) -> Dict[int, Tuple[Trial, str]]:
        """
        Updates schedulers with new results.
        :return: dictionary mapping trial-ids trials that are finished to Trial and status.
        Trials can be finished because:
         1) the scheduler decided to stop or pause.
         2) the trial failed.
         3) the trial completed.
        """
        # gets the list of jobs from running_jobs that are done
        done_trials = {}

        for trial_id, result in new_results:
            if trial_id not in done_trials:
                trial, status = trial_status_dict[trial_id]

                # communicate new result to the searcher and the scheduler
                self.last_seen_result_per_trial[trial_id] = result
                decision = self.scheduler.on_trial_result(trial=trial, result=result)

                for callback in callbacks:
                    callback.on_trial_result(
                        trial=trial,
                        status=status,
                        result=result,
                        decision=decision,
                    )

                is_done = False
                if decision == SchedulerDecision.STOP:
                    if status != Status.completed:
                        # we override the status immediately, this avoid calling the backend status another time to
                        # update after the change which may be expensive
                        status = Status.stopped
                        self.backend.stop_trial(trial_id)
                    is_done = True
                elif decision == SchedulerDecision.PAUSE:
                    status = Status.paused
                    self.backend.pause_trial(trial_id)
                    is_done = True
                if is_done:
                    self.scheduler.on_trial_remove(trial=trial)
                    done_trials[trial_id] = (trial, status)

        for trial_id, (trial, status) in trial_status_dict.items():
            # Status "completed" and "failed" are signaled to scheduler.
            if status == Status.completed:
                # since the code above updates `trial_status_dict[trial_id]` after a pause/stop scheduling decision
                # this callback is never called after a pause/stop scheduler decision.
                logger.info(f"Trial trial_id {trial_id} completed.")
                assert trial_id in self.last_seen_result_per_trial, \
                    f"trial {trial_id} completed and no metrics got observed"
                last_result = self.last_seen_result_per_trial[trial_id]
                if not trial_id in done_trials:
                    self.scheduler.on_trial_complete(trial, last_result)
                for callback in callbacks:
                    callback.on_trial_complete(trial, last_result)
                done_trials[trial_id] = (trial, status)

            # Status "in_progress", "stopped" and "stopping" are not signaled, although the first one could be added
            # to notify the scheduler of pending runtimes (even in the absence of new results).
            # The stopped/stopping are not needed a priori since they come after a stop decision.
            if status == Status.failed:
                logger.info(f"Trial trial_id {trial_id} failed.")
                self.scheduler.on_trial_error(trial)
                done_trials[trial_id] = (trial, status)

        return done_trials

    def _default_callback(self):
        """
        :return: default callback to store results
        """
        return StoreResultsCallback(
            csv_file=str(self.tuner_path / "results.csv.zip"), results_update_interval=self.results_update_interval
        )
