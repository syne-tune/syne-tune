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
from datetime import timedelta
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import subprocess

from syne_tune.report import retrieve
from syne_tune.backend import LocalBackend
from syne_tune.backend.trial_status import TrialResult, Status, Trial
from syne_tune.backend.simulator_backend.time_keeper import SimulatedTimeKeeper
from syne_tune.backend.simulator_backend.events import (
    SimulatorState,
    StartEvent,
    CompleteEvent,
    StopEvent,
    OnTrialResultEvent,
)
from syne_tune.constants import ST_CHECKPOINT_DIR, ST_WORKER_TIMESTAMP, ST_TUNER_TIME
from syne_tune.tuner import DEFAULT_SLEEP_TIME

logger = logging.getLogger(__name__)


DEFAULT_DELAY = 0.05


@dataclass
class SimulatorConfig:
    """
    Configures the simulator:

    delay_on_trial_result:
        Time from `report` called on worker to result registered at back-end
    delay_complete_after_final_report:
        Time from final `report` called on worker to job completion being
        registered at back-end
    delay_complete_after_stop:
        Time from stop signal received at worker to job completion being
        registered at back-end
    delay_start:
        Time from start command being sent at back-end and job starting on
        the worker (which is free)
    delay_stop:
        Time from stop signal being sent at back-end to signal received at
        worker (which is running)
    """

    delay_on_trial_result: float = DEFAULT_DELAY
    delay_complete_after_final_report: float = DEFAULT_DELAY
    delay_complete_after_stop: float = DEFAULT_DELAY
    delay_start: float = DEFAULT_DELAY
    delay_stop: float = DEFAULT_DELAY

    def __post_init__(self):
        assert self.delay_on_trial_result >= 0
        assert self.delay_complete_after_final_report >= 0
        assert self.delay_complete_after_stop >= 0
        assert self.delay_start >= 0
        assert self.delay_stop >= 0
        # Otherwise, the final result may arrive after the job completion
        assert self.delay_on_trial_result <= self.delay_complete_after_final_report


class SimulatorBackend(LocalBackend):
    def __init__(
        self,
        entry_point: str,
        elapsed_time_attr: str,
        simulator_config: Optional[SimulatorConfig] = None,
        tuner_sleep_time: float = DEFAULT_SLEEP_TIME,
        debug_resource_attr: Optional[str] = None,
    ):
        """
        This simulator back-end drives experiments with tabulated training
        evaluation functions, which return their computation time rather than
        spend it. To this end, time (on the tuning instance) is simulated using
        a `time_keeper` and an event priority queue in `_simulator_state`.

        Time is advanced both by `Tuner.run` waiting, and by non-negligible
        computations during the tuning loop (in particular, we take care of
        `scheduler.suggest` and `scheduler.on_trial_result` there).

        When the `entry_point` script is executed, we wait for all results to
        be returned. In each result, the value for key `elapsed_time_attr`
        contains the time since start of the script. These values are used
        to place worker events on the simulated time line (represented by
        `simulator_state`).
        NOTE: If a trial is resumed, the elapsed_time value contains the time
        since start of the last recent resume, NOT the cumulative time used by
        the trial.

        Each method call starts by advancing time by what was spent outside,
        since the last recent call to the back-end. Then, all events in
        `simulator_state` are processed whose time is before the current time
        in `time_keeper`. The method ends by `time_keeper.mark_exit()`.

        Note: In this basic version of the simulator back-end, we still call a
        Python main function as a subprocess, which returns the requested
        metrics by looking them up or running a surrogate. This is flexible,
        but has the overhead of loading a table at every call. For faster
        simulations, use :class:`BlackboxRepositoryBackend` after bringing your
        tabulated data or surrogate benchmark into the blackbox repository.

        :param entry_point: Python main file to be tuned
        :param elapsed_time_attr: See above
        :param simulator_config: Parameters for simulator
        :param tuner_sleep_time: Effective sleep time in `Tuner.run`. This
            information is needed in `SimulatorCallback`

        """
        super().__init__(entry_point=entry_point, rotate_gpus=False)
        self.elapsed_time_attr = elapsed_time_attr
        if simulator_config is None:
            self.simulator_config = SimulatorConfig()
        else:
            self.simulator_config = simulator_config
        self.tuner_sleep_time = tuner_sleep_time
        self._debug_resource_attr = debug_resource_attr
        # Start with empty event queue
        self._simulator_state = SimulatorState()
        self._time_keeper = SimulatedTimeKeeper()
        self._next_results_to_fetch = dict()
        logger.setLevel(logging.INFO)  # Suppress DEBUG for this class

    @property
    def time_keeper(self) -> SimulatedTimeKeeper:
        return self._time_keeper

    @staticmethod
    def _debug_message(
        event_name: str, time: float, trial_id: int, pushed: bool = False, **kwargs
    ):
        msg_part = "push " if pushed else ""
        msg = f"[{msg_part}{event_name}:"
        parts = [f"time = {time:.2f}", f"trial_id = {trial_id}"] + [
            f"{k} = {v}" for k, v in kwargs.items()
        ]
        msg += ", ".join(parts) + "]"
        logger.debug(msg)

    def start_trial(
        self, config: Dict, checkpoint_trial_id: Optional[int] = None
    ) -> Trial:
        # Overwritten to record the correct `creation_time`
        trial_id = self.new_trial_id()
        if checkpoint_trial_id is not None:
            self.copy_checkpoint(
                src_trial_id=checkpoint_trial_id, tgt_trial_id=trial_id
            )
        self.trial_ids.append(trial_id)
        self._schedule(trial_id=trial_id, config=config)
        now = self._time_keeper.time_stamp()
        trial = Trial(
            trial_id=trial_id,
            config=config,
            creation_time=now,
        )
        self._trial_dict[trial_id] = trial

        return trial

    def _process_events_until_now(self):
        """
        We process all events in the queue with times before
        `time_keeper.time()`.
        """
        time_now = self._time_keeper.time()
        next_event = self._simulator_state.next_until(time_now)
        while next_event is not None:
            time_event, event = next_event
            trial_id = event.trial_id
            if isinstance(event, StartEvent):
                self._debug_message("StartEvent", time=time_event, trial_id=trial_id)
                # Run training script and push event for each result
                self._process_start_event(trial_id=trial_id, time_event=time_event)
            elif isinstance(event, CompleteEvent):
                trial_result = self._trial_dict[trial_id]
                status = event.status
                self._debug_message(
                    "CompleteEvent", time=time_event, trial_id=trial_id, status=status
                )
                training_end_time = self._time_keeper.start_time_stamp + timedelta(
                    seconds=time_event
                )
                if isinstance(trial_result, TrialResult):
                    trial_result.status = status
                    trial_result.training_end_time = training_end_time
                else:
                    # No results reported for the trial. This can happen if
                    # the trial failed
                    self._trial_dict[trial_id] = trial_result.add_results(
                        metrics=[], status=status, training_end_time=training_end_time
                    )
            elif isinstance(event, StopEvent):
                self._debug_message("StopEvent", time=time_event, trial_id=trial_id)
                # Remove all remaining events for `trial_id`. This includes
                # the `CompleteEvent` pushed with `StartEvent`, so there can
                # be no confusion with the 2nd `CompleteEvent` pushed by
                # `_stop_trial`.
                self._simulator_state.remove_events(trial_id)
            elif isinstance(event, OnTrialResultEvent):
                result = copy.copy(event.result)
                if self._debug_resource_attr:
                    k = self._debug_resource_attr
                    debug_kwargs = {k: result.get(k)}
                else:
                    debug_kwargs = dict()
                self._debug_message(
                    "OnTrialResultEvent",
                    time=time_event,
                    trial_id=trial_id,
                    **debug_kwargs,
                )
                # Append timestamps to `result`. This is done here, but not in
                # the other back-ends, for which timestamps are only added when
                # results are written out.
                result[ST_TUNER_TIME] = time_event
                if trial_id in self._next_results_to_fetch:
                    self._next_results_to_fetch[trial_id].append(result)
                else:
                    self._next_results_to_fetch[trial_id] = [result]
                trial_result = self._trial_dict[trial_id]
                if isinstance(trial_result, TrialResult):
                    trial_result.metrics.append(result)
                else:
                    self._trial_dict[trial_id] = trial_result.add_results(
                        metrics=[result],
                        status=Status.in_progress,
                        training_end_time=None,
                    )
            else:
                raise TypeError(f"Event at time {time_event} of unknown type: {event}")
            next_event = self._simulator_state.next_until(time_now)

    def _process_start_event(
        self, trial_id: int, time_event: float, config: Optional[dict] = None
    ):
        # Run training script and record results
        status, results = self._run_job_and_collect_results(trial_id, config=config)
        time_final_result = time_event
        deb_it = 0  # DEBUG
        for i, result in enumerate(results):
            elapsed_time = result.get(self.elapsed_time_attr)
            assert elapsed_time is not None, (
                f"Result for trial_id = {trial_id} does not contain "
                + f"{self.elapsed_time_attr} entry. Your code needs "
                + "to report elapsed time, and the attribute name "
                + "must be set as elapsed_time_attr here."
            )
            _time_result = time_event + float(elapsed_time)
            time_result = _time_result + self.simulator_config.delay_on_trial_result
            self._simulator_state.push(
                OnTrialResultEvent(trial_id=trial_id, result=result),
                event_time=time_result,
            )
            time_final_result = max(time_final_result, _time_result)
            # DEBUG:
            if deb_it < 10:
                if self._debug_resource_attr:
                    k = self._debug_resource_attr
                    debug_kwargs = {k: result.get(k)}
                else:
                    debug_kwargs = dict()
                self._debug_message(
                    "OnTrialResultEvent",
                    time=time_result,
                    trial_id=trial_id,
                    pushed=True,
                    **debug_kwargs,
                )
                deb_it += 1
        time_complete = (
            time_final_result + self.simulator_config.delay_complete_after_final_report
        )
        self._simulator_state.push(
            CompleteEvent(trial_id=trial_id, status=status), event_time=time_complete
        )
        self._debug_message(
            "CompleteEvent", time=time_complete, trial_id=trial_id, pushed=True
        )

    def _advance_by_outside_time(self):
        self._time_keeper.advance(self._time_keeper.real_time_since_last_recent_exit())

    def fetch_status_results(
        self, trial_ids: List[int]
    ) -> Tuple[Dict[int, Tuple[Trial, str]], List[Tuple[int, Dict]]]:
        self._advance_by_outside_time()
        # Process all events in the past
        self._process_events_until_now()
        # Results are collected in `_next_results_to_fetch`
        results = []
        for trial_id in trial_ids:
            result_list = self._next_results_to_fetch.get(trial_id)
            if result_list is not None:
                results.extend((trial_id, result) for result in result_list)
                self._last_metric_seen_index[trial_id] += len(result_list)
                del self._next_results_to_fetch[trial_id]
        if self._next_results_to_fetch:
            # Note: This tends to happen regularly with fast-running
            # benchmarks
            warn_msg = [
                "The following trials reported results, but are not covered "
                "by trial_ids. These results will be ignored:"
            ]
            for trial_id, result_list in self._next_results_to_fetch.items():
                status = self._trial_dict[trial_id].status
                msg_line = f"  trial_id {trial_id}: status = {status}, "
                if self._debug_resource_attr is None:
                    msg_line += f"num_results = {len(result_list)}"
                else:
                    resources = [
                        result[self._debug_resource_attr] for result in result_list
                    ]
                    msg_line += f"resources = {resources}"
                warn_msg.append(msg_line)
                self._last_metric_seen_index[trial_id] += len(result_list)
            logger.debug("\n".join(warn_msg))
            self._next_results_to_fetch = dict()

        if len(results) > 0 and ST_WORKER_TIMESTAMP in results[0]:
            results = sorted(results, key=lambda result: result[1][ST_WORKER_TIMESTAMP])

        trial_status_dict = dict()
        for trial_id in trial_ids:
            trial_result = self._trial_dict[trial_id]
            status = (
                trial_result.status
                if isinstance(trial_result, TrialResult)
                else Status.in_progress
            )
            trial = Trial(
                trial_id=trial_result.trial_id,
                config=trial_result.config,
                creation_time=trial_result.creation_time,
            )
            trial_status_dict[trial_id] = (trial, status)

        self._time_keeper.mark_exit()
        return trial_status_dict, results

    def _schedule(self, trial_id: int, config: Dict):
        """
        This is called by `start_trial` or `resume_trial`. We register a start
        event here. `config` can be ignored, it will be in
        `trial(trial_id).config` once the start event is executed.

        Note: This call is "non-blocking": The start event is registered
        here (in the future), but is not yet processed.
        """
        self._advance_by_outside_time()
        # Process all events in the past
        self._process_events_until_now()
        _time_start = self._time_keeper.time()
        time_start = _time_start + self.simulator_config.delay_start
        self._simulator_state.push(StartEvent(trial_id=trial_id), event_time=time_start)
        self._debug_message(
            "StartEvent", time=time_start, trial_id=trial_id, pushed=True
        )
        logger.debug(f"Simulated time since start: {_time_start:.2f} secs")
        self._time_keeper.mark_exit()

    def _all_trial_results(self, trial_ids: List[int]) -> List[TrialResult]:
        """
        Note: Since this is not used anymore in `fetch_results`, it can
        simply just return all registered trials which already have some
        results.
        This will not return trials which have just been started, but did not
        report any results yet.
        """
        results = []
        for trial_id in trial_ids:
            trial_result = self._trial_dict[trial_id]
            # Filter out entries which have not obtained any results
            if isinstance(trial_result, TrialResult):
                results.append(trial_result)
        return results

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        self._stop_or_pause_trial(trial_id, status=Status.paused)

    def _resume_trial(self, trial_id: int):
        pass

    def _stop_trial(self, trial_id: int, result: Optional[dict]):
        self._stop_or_pause_trial(trial_id, status=Status.stopped)

    def _stop_or_pause_trial(self, trial_id: int, status: str):
        """
        This is called by `stop_trial` or `pause_trial`.

        Note: This call is "blocking": Stop and complete events
        are not just registered here, but also processed.
        """
        self._advance_by_outside_time()
        time_stop = self._time_keeper.time() + self.simulator_config.delay_stop
        self._simulator_state.push(StopEvent(trial_id=trial_id), event_time=time_stop)
        self._debug_message("StopEvent", time=time_stop, trial_id=trial_id, pushed=True)
        # Note: We need to call `_process_events_until_now` twice. If we first
        # pushed the final `CompleteEvent`, it would be removed by the
        # `StopEvent`.
        self._time_keeper.advance_to(time_stop + 1e-3)
        # Process events up to and including `StopEvent`
        self._process_events_until_now()
        time_complete = (
            self._time_keeper.time() + self.simulator_config.delay_complete_after_stop
        )
        self._simulator_state.push(
            CompleteEvent(trial_id=trial_id, status=status), event_time=time_complete
        )
        self._debug_message(
            "CompleteEvent", time=time_complete, trial_id=trial_id, pushed=True
        )
        # Process final `CompleteEvent`
        self._time_keeper.advance_to(time_complete + 1e-3)
        self._process_events_until_now()
        self._time_keeper.mark_exit()

    def _run_job_and_collect_results(
        self, trial_id: int, config: Optional[dict] = None
    ) -> (str, List[dict]):
        """
        Runs training evaluation script for trial `trial_id`, using the config
        `trial(trial_id).config`. This is a blocking call, we wait for the
        script to finish, then parse all its results and return them.

        :param trial_id:
        :return: (final status, list of all results reported)
        """
        assert (
            trial_id in self._trial_dict
        ), f"Trial with trial_id = {trial_id} not registered with back-end"
        if config is None:
            config = self._trial_dict[trial_id].config

        # Run training script and fetch all results
        trial_path = self.trial_path(trial_id)
        os.makedirs(trial_path, exist_ok=True)
        config_copy = config.copy()
        config_copy[ST_CHECKPOINT_DIR] = str(trial_path / "checkpoints")
        config_str = " ".join(
            [f"--{key} {value}" for key, value in config_copy.items()]
        )

        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()

        with open(trial_path / "config.json", "w") as f:
            # the encoder fixes json error "TypeError: Object of type 'int64' is not JSON serializable"
            json.dump(config, f, default=np_encoder)
        cmd = f"python {self.entry_point} {config_str}"
        env = dict(os.environ)
        logging.info(f"running script with command: {cmd}")
        with open(trial_path / "std.out", "a") as stdout:
            with open(trial_path / "std.err", "a") as stderr:
                return_status = subprocess.run(
                    cmd.split(" "), stdout=stdout, stderr=stderr, env=env
                )
        if return_status.returncode == 0:
            status = Status.completed
        else:
            status = Status.failed
        # Read all reported results
        # Results are also read if the process failed
        # Note that `retrieve` returns all results, even those already
        # received before (in case the trial is resumed at least once).
        all_results = retrieve(log_lines=self.stdout(trial_id=trial_id))
        num_already_before = self._last_metric_seen_index[trial_id]
        assert num_already_before <= len(all_results), (
            f"Found {len(all_results)} total results, but have already "
            + f"processed {num_already_before} before!"
        )
        results = all_results[num_already_before:]

        return status, results
