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
from typing import List, Callable, Tuple, Optional, Dict, Set
import dill as dill

from syne_tune.backend.trial_backend import TrialBackend
from syne_tune.backend.trial_status import Status, Trial
from syne_tune.config_space import to_dict, Domain
from syne_tune.constants import ST_TUNER_CREATION_TIMESTAMP, ST_TUNER_START_TIMESTAMP
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialScheduler
from syne_tune.tuner_callback import TunerCallback, StoreResultsCallback
from syne_tune.tuning_status import TuningStatus, print_best_metric_found
from syne_tune.util import (
    RegularCallback,
    experiment_path,
    name_from_base,
    check_valid_sagemaker_name,
)

logger = logging.getLogger(__name__)

DEFAULT_SLEEP_TIME = 5.0


class Tuner:
    def __init__(
        self,
        trial_backend: TrialBackend,
        scheduler: TrialScheduler,
        stop_criterion: Callable[[TuningStatus], bool],
        n_workers: int,
        sleep_time: float = DEFAULT_SLEEP_TIME,
        results_update_interval: float = 10.0,
        print_update_interval: float = 30.0,
        max_failures: int = 1,
        tuner_name: Optional[str] = None,
        asynchronous_scheduling: bool = True,
        wait_trial_completion_when_stopping: bool = False,
        callbacks: Optional[List[TunerCallback]] = None,
        metadata: Optional[dict] = None,
        suffix_tuner_name: bool = True,
        save_tuner: bool = True,
    ):
        """
        Allows to run an tuning job, call `run` after initializing.
        :param trial_backend:
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
        :param tuner_name: name associated with the tuning experiment, default to the name of the entrypoint.
        It can only consists in alpha-digits characters, possibly separated by '-'. A postfix with a date time-stamp
        is added to ensure unicity.
        :param asynchronous_scheduling: whether to use asynchronous scheduling when scheduling new trials. If `True`,
        trials are scheduled as soon as a worker is available, if `False`, the tuner waits that all trials are finished
         before scheduling a new batch.
        :param wait_trial_completion_when_stopping: how to deal with running trials when stopping criterion is
        met. If `True`, the tuner waits that all trials are finished, if `False`, all trials are terminated.
        :param callbacks: called when events happens in the tuning loop such as when a result is seen, by default
        a callback that stores results every `results_update_interval` is used.
        :param metadata: dictionary of user-metadata that will be persistend in {tuner_path}/metadata.json, in addition
        to the metadata provided by the user, `SMT_TUNER_CREATION_TIMESTAMP` is always included which measures
        the time-stamp when the tuner started to run.
        :param suffix_tuner_name: If True, a timestamp is appended to the provided `tuner_name` that ensures uniqueness
        otherwise the name is left unchanged and is expected to be unique.
        :param save_tuner: If True, the `Tuner` object is serialized at the end
            of tuning, including its dependencies (e.g., scheduler). This allows
            all details of the experiment to be recovered
        """
        self.trial_backend = trial_backend
        self.scheduler = scheduler
        self.n_workers = n_workers
        self.sleep_time = sleep_time
        self.results_update_interval = results_update_interval
        self.stop_criterion = stop_criterion
        self.asynchronous_scheduling = asynchronous_scheduling
        self.wait_trial_completion_when_stopping = wait_trial_completion_when_stopping
        self.metadata = self._enrich_metadata(metadata)
        self.save_tuner = save_tuner

        self.max_failures = max_failures
        self.print_update_interval = print_update_interval

        if tuner_name is not None:
            check_valid_sagemaker_name(tuner_name)
        else:
            tuner_name = Path(self.trial_backend.entrypoint_path()).stem.replace(
                "_", "-"
            )
        if suffix_tuner_name or tuner_name is None:
            self.name = name_from_base(tuner_name, default="st-tuner")
        else:
            self.name = tuner_name

        # we keep track of the last result seen to send it to schedulers when trials complete.
        self.last_seen_result_per_trial = {}
        self.trials_scheduler_stopped = set()
        self.tuner_path = Path(experiment_path(tuner_name=self.name))

        # inform the backend to the folder of the Tuner. This allows the local backend
        # to store the logs and tuner results in the same folder.
        self.trial_backend.set_path(results_root=self.tuner_path, tuner_name=self.name)
        self.callbacks = (
            callbacks if callbacks is not None else [self._default_callback()]
        )

        self.tuning_status = None
        self.tuner_saver = None

    def run(self):
        """
        Launches the tuning.
        :return: the tuning status when finished
        """
        try:
            logger.info(f"results of trials will be saved on {self.tuner_path}")

            if self.tuning_status is None:
                self.tuning_status = TuningStatus(
                    metric_names=self.scheduler.metric_names()
                )
            # prints the status every print_update_interval seconds
            self.status_printer = RegularCallback(
                call_seconds_frequency=self.print_update_interval,
                callback=lambda tuning_status: logger.info(
                    "tuning status (last metric is reported)\n" + str(tuning_status)
                ),
            )
            # saves the tuner every results_update_interval seconds
            if self.save_tuner:
                self.tuner_saver = RegularCallback(
                    callback=lambda tuner: tuner.save(),
                    call_seconds_frequency=self.results_update_interval,
                )

            self.metadata[ST_TUNER_START_TIMESTAMP] = time.time()

            for callback in self.callbacks:
                callback.on_tuning_start(self)

            self.tuner_path.mkdir(exist_ok=True, parents=True)

            self._save_metadata()

            done_trials_statuses = OrderedDict()
            # `running_trial_ids` contains the ids of all trials currently running,
            # whether they were started from scratch or were resumed from a pausing
            # state
            running_trials_ids = set()

            config_space_exhausted = False
            stop_condition_reached = self._stop_condition()

            while (
                # we stop when either the stop condition is reached
                not stop_condition_reached
                # or when all trials are done if the wait_trial_completion is activated
                or self.wait_trial_completion_when_stopping
                and len(running_trials_ids) > 0
            ):
                for callback in self.callbacks:
                    callback.on_loop_start()

                new_done_trial_statuses, new_results = self._process_new_results(
                    running_trials_ids=running_trials_ids,
                )

                if new_results and self.save_tuner:
                    # Save tuner state only if there have been new results
                    self.tuner_saver(tuner=self)

                # update the list of done trials and remove those from `running_trials_ids`
                # Note: It is important to update `running_trials_ids` before
                # calling `_schedule_new_tasks`.
                # Otherwise, a trial can be registered as paused in
                # `_process_new_results`, and immediately be resumed in
                # `_schedule_new_tasks`. If `new_done_trial_statuses` is subtracted from
                # `running_trials_ids` afterwards only, this trial is removed from
                # `running_trials_ids` even though it is running. Also, its status remains
                # paused, because the next call of `_process_new_results` only considers
                # trials in `running_trials_ids`.
                done_trials_statuses.update(new_done_trial_statuses)
                running_trials_ids.difference_update(new_done_trial_statuses.keys())

                if (
                    config_space_exhausted
                    or self.wait_trial_completion_when_stopping
                    and stop_condition_reached
                ):
                    # if the search space is exhausted, we loop until the running trials are done or until the
                    # stop condition is reached
                    if len(running_trials_ids) > 0:
                        if config_space_exhausted:
                            logger.debug(
                                f"Configuration space exhausted, waiting for completion of running trials "
                                f"{running_trials_ids}"
                            )
                        else:
                            logger.debug(
                                f"Stopping criterion reached, waiting for completion of running trials "
                                f"{running_trials_ids}"
                            )
                        self._sleep()
                    else:
                        break
                else:
                    try:
                        self._schedule_new_tasks(running_trials_ids=running_trials_ids)
                    except StopIteration:
                        logger.info(
                            "Tuning is finishing as the whole configuration space got exhausted."
                        )
                        config_space_exhausted = True
                        print(
                            "Tuning is finishing as the whole configuration space got exhausted."
                        )

                self.status_printer(self.tuning_status)

                for callback in self.callbacks:
                    callback.on_loop_end()

                stop_condition_reached = self._stop_condition()
        except Exception as e:
            logger.error(
                "An error happened during the tuning, cleaning up resources and logging final resources "
                "before throwing the exception."
            )
            raise e
        finally:
            # graceful termination block called when the tuner reached its stop condition, when an error happened or
            # when the job got interrupted (can happen in spot-instances or when sending a SIGINT signal with ctrl+C).
            # the block displays the best configuration found and stops trials that may still be running.
            print_best_metric_found(
                tuning_status=self.tuning_status,
                metric_names=self.scheduler.metric_names(),
                mode=self.scheduler.metric_mode(),
            )

            # Callbacks (typically includes writing final results)
            for callback in self.callbacks:
                callback.on_tuning_end()

            # Serialize Tuner object
            if self.save_tuner:
                self.save()

            logger.info("Stopping trials that may still be running.")
            self.trial_backend.stop_all()

            # notify tuning status that jobs were stopped without having to query their status in the backend since
            # we know that all trials were stopped
            self.tuning_status.mark_running_job_as_stopped()

            # in case too many errors were triggered, show log of last failed job and terminates with an error
            if self.tuning_status.num_trials_failed > self.max_failures:
                self._handle_failure(done_trials_statuses=done_trials_statuses)

            logger.info(
                f"Tuning finished, results of trials can be found on {self.tuner_path}"
            )

    def _sleep(self):
        time.sleep(self.sleep_time)
        for callback in self.callbacks:
            callback.on_tuning_sleep(self.sleep_time)

    @staticmethod
    def _set_metadata(metadata: dict, name: str, value):
        if name in metadata:
            logger.warning(
                f"Entry {name} in metadata is used, but will be overwritten:\n"
                f"Old value: {metadata[name]}\n"
                f"Overwrite: {value}\n"
            )
        metadata[name] = value

    def _enrich_metadata(self, metadata: dict):
        """
        :return: adds creation time stamp, metric names and mode, entrypoint and backend to the metadata.
        """
        res = metadata if metadata is not None else dict()
        self._set_metadata(res, ST_TUNER_CREATION_TIMESTAMP, time.time())
        self._set_metadata(res, "metric_names", self.scheduler.metric_names())
        self._set_metadata(res, "metric_mode", self.scheduler.metric_mode())
        self._set_metadata(res, "entrypoint", self.trial_backend.entrypoint_path().stem)
        self._set_metadata(res, "backend", str(type(self.trial_backend).__name__))
        self._set_metadata(
            res, "scheduler_name", str(self.scheduler.__class__.__name__)
        )
        config_space_json = json.dumps(
            {
                k: to_dict(v) if isinstance(v, Domain) else v
                for k, v in self.scheduler.config_space.items()
            }
        )
        self._set_metadata(res, "config_space", config_space_json)
        return res

    def _save_metadata(self):
        with open(self.tuner_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def _stop_condition(self) -> bool:
        return (
            self.stop_criterion(self.tuning_status)
            or self.tuning_status.num_trials_failed > self.max_failures
        )

    def _process_new_results(self, running_trials_ids: Set[int]):
        """
        Communicates new results from the backend to the scheduler
        :param running_trials_ids: list of trials currently running
        :return: dictionary from trial-id to status of trials that are not running and new results observed
        """

        # fetch new results
        trial_status_dict, new_results = self.trial_backend.fetch_status_results(
            trial_ids=list(running_trials_ids)
        )

        for callback in self.callbacks:
            callback.on_fetch_status_results(
                trial_status_dict=trial_status_dict, new_results=new_results
            )

        assert len(running_trials_ids) <= self.n_workers

        # Gets list of trials that are done with the new results.
        # The trials can be finished for different reasons:
        # - they completed,
        # - they were stopped independently of the scheduler, e.g. due to a timeout argument or a manual interruption
        # - scheduler decided to interrupt them.
        # Note: `done_trials` includes trials which are paused.
        done_trials_statuses = self._update_running_trials(
            trial_status_dict, new_results, callbacks=self.callbacks
        )
        trial_status_dict.update(done_trials_statuses)

        # update status with new results and all done trials
        self.tuning_status.update(
            trial_status_dict=trial_status_dict, new_results=new_results
        )

        return done_trials_statuses, new_results

    def _schedule_new_tasks(self, running_trials_ids: Set[int]):
        """
        Schedules new tasks if resources are available or sleep.
        :param running_trials_ids: set if trial-ids currently running, gets updated if new trials are scheduled.
        """
        running_trials_threshold = self.n_workers if self.asynchronous_scheduling else 1
        num_running_trials = len(running_trials_ids)
        if num_running_trials >= running_trials_threshold:
            # Note: For synchronous scheduling, we need to sleep here if at
            # least one worker is busy
            logger.debug(
                f"{num_running_trials} of {self.n_workers} workers are "
                f"busy, wait for {self.sleep_time} seconds"
            )
            self._sleep()

        else:
            # Schedule as many trials as we have free workers
            for i in range(self.n_workers - num_running_trials):
                trial_id = self._schedule_new_task()
                running_trials_ids.add(trial_id)

    def _schedule_new_task(self) -> Optional[int]:
        """
        Schedules a new task according to scheduler suggestion.
        :return: the trial-id of the task suggested, None if the scheduler was done.
        """
        trial_id = self.trial_backend.new_trial_id()
        suggestion = self.scheduler.suggest(trial_id=trial_id)
        if suggestion is None:
            logger.info("Searcher ran out of candidates, tuning job is stopping.")
            raise StopIteration
        elif suggestion.spawn_new_trial_id:
            # we schedule a new trial, possibly using the checkpoint of `checkpoint_trial_id`
            # if given.
            trial = self.trial_backend.start_trial(
                config=suggestion.config.copy(),
                checkpoint_trial_id=suggestion.checkpoint_trial_id,
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
            self.trial_backend.resume_trial(
                trial_id=suggestion.checkpoint_trial_id, new_config=suggestion.config
            )
            return suggestion.checkpoint_trial_id

    def _handle_failure(self, done_trials_statuses: Dict[int, Tuple[Trial, str]]):
        logger.error(f"Stopped as {self.max_failures} failures were reached")
        for trial_id, (_, status) in done_trials_statuses.items():
            if status == Status.failed:
                logger.error(f"showing log of first failure")
                stdout = "".join(self.trial_backend.stdout(trial_id))
                stderr = "".join(self.trial_backend.stderr(trial_id))
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
            self.trial_backend.on_tuner_save()  # callback

    @staticmethod
    def load(tuner_path: Optional[str]):
        with open(Path(tuner_path) / "tuner.dill", "rb") as f:
            tuner = dill.load(f)
            tuner.tuner_path = Path(experiment_path(tuner_name=tuner.name))
            return tuner

    def _update_running_trials(
        self,
        trial_status_dict: Dict[int, Tuple[Trial, str]],
        new_results: List[Tuple[int, dict]],
        callbacks: List[TunerCallback],
    ) -> Dict[int, Tuple[Trial, str]]:
        """
        Updates schedulers with new results and sends decision to stop/pause trials to the backend.
        :return: dictionary mapping trial-ids that are finished to status.
        Trials can be finished because:
         1) the scheduler decided to stop or pause.
         2) the trial failed.
         3) the trial was stopped independently of the scheduler, e.g. due to a timeout argument or a manual interruption.
         4) the trial completed.
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

                if decision == SchedulerDecision.STOP:
                    if status != Status.completed:
                        # we override the status immediately, this avoids calling the backend status another time to
                        # update after the change which may be expensive
                        status = Status.stopped
                        self.trial_backend.stop_trial(trial_id=trial_id, result=result)
                    self.scheduler.on_trial_remove(trial=trial)
                    done_trials[trial_id] = (trial, status)
                    self.trials_scheduler_stopped.add(trial_id)

                elif decision == SchedulerDecision.PAUSE:
                    status = Status.paused
                    self.trial_backend.pause_trial(trial_id=trial_id, result=result)
                    self.scheduler.on_trial_remove(trial=trial)
                    done_trials[trial_id] = (trial, status)

        for trial_id, (trial, status) in trial_status_dict.items():
            # Status "completed", "stopped" and "failed" are signaled to scheduler.
            # Status "in_progress" and "stopping" are not signaled, although the first one could be added
            # to notify the scheduler of pending runtimes (even in the absence of new results).

            if status == Status.completed:
                # since the code above updates `trial_status_dict[trial_id]` after a pause/stop scheduling decision
                # this callback is never called after a pause/stop scheduler decision.
                if (
                    trial_id not in done_trials
                    or done_trials[trial_id][1] != Status.paused
                ):
                    logger.info(f"Trial trial_id {trial_id} completed.")
                assert (
                    trial_id in self.last_seen_result_per_trial
                ), f"trial {trial_id} completed and no metrics got observed"
                last_result = self.last_seen_result_per_trial[trial_id]
                if not trial_id in done_trials:
                    self.scheduler.on_trial_complete(trial, last_result)
                for callback in callbacks:
                    callback.on_trial_complete(trial, last_result)
                done_trials[trial_id] = (trial, status)

            if status == Status.failed:
                logger.info(f"Trial trial_id {trial_id} failed.")
                self.scheduler.on_trial_error(trial)
                done_trials[trial_id] = (trial, status)

            # For the case when the trial is stopped independently of the scheduler, we choose to use
            # scheduler.on_trial_error(...) since it was not the scheduler's decision to stop the trial.
            if (
                status == Status.stopped
                and trial_id not in self.trials_scheduler_stopped
            ):
                logger.info(
                    f"Trial trial_id {trial_id} was stopped independently of the scheduler."
                )
                self.scheduler.on_trial_error(trial)
                done_trials[trial_id] = (trial, status)

        return done_trials

    def _default_callback(self):
        """
        :return: default callback to store results
        """
        return StoreResultsCallback()
