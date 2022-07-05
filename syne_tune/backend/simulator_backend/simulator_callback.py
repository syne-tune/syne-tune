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

from syne_tune.tuner_callback import StoreResultsCallback
from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune import Tuner
from syne_tune.constants import ST_TUNER_TIME
from syne_tune import StoppingCriterion
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

logger = logging.getLogger(__name__)


class SimulatorCallback(StoreResultsCallback):
    """
    Callback to be used in `Tuner.run` in order to support the
    :class:`SimulatorBackend`.

    This is doing two things. First, `on_tuning_sleep` is advancing the
    `time_keeper` of the simulator back-end by `tuner_sleep_time` (also
    defined in the back-end). The real sleep time in `Tuner` must be 0.

    Second, we need to make sure that results written out are annotated by
    simulated time, not real time. This is already catered for by
    `SimulatorBackend` adding ST_TUNER_TIME entries to each result it
    receives.

    Third (and most subtle), we need to make sure the stop criterion in
    `Tuner.run` is using simulated time instead of real time when making
    a decision based on `max_wallclock_time`. By default, `StoppingCriterion`
    takes `TuningStatus` as an input, which counts real time and knows nothing
    about simulated time. To this end, we modify `stop_criterion` of the tuner
    to instead depend on the ST_TUNER_TIME fields in the results received.
    This allows us to keep both `Tuner` and `TuningStatus` independent of the
    time keeper.

    """

    def __init__(self):
        # Note: `results_update_interval` is w.r.t. real time, not
        # simulated time. Storing results intermediately is not important for
        # the simulator back-end, so the default is larger
        super().__init__(add_wallclock_time=True)
        self._tuner_sleep_time = None
        self._time_keeper = None
        self._tuner = None
        self._backup_stop_criterion = None

    def _modify_stop_criterion(self, tuner: "Tuner"):
        stop_criterion = tuner.stop_criterion
        if not isinstance(stop_criterion, StoppingCriterion):
            # Note: We could raise an exception here ...
            logger.warning(
                "The stop_criterion argument to Tuner is not of type "
                + "StoppingCriterion. This can be problematic when using "
                + "the SimulatorBackend. If your stop_criterion depends on "
                + "wallclock time, you'll get wrong behaviour. It is highly "
                + "recommended to use StoppingCriterion!"
            )
        elif stop_criterion.max_wallclock_time is not None:
            # Since `TuningStatus` is measuring real time, not simulated time,
            # we need to replace the `max_wallclock_time` part of this criterion
            # by `max_metric_value` w.r.t. ST_TUNER_TIME. Note that
            # `SimulatorBackend` is adding ST_TUNER_TIME to any result it
            # receives
            self._backup_stop_criterion = stop_criterion
            max_wallclock_time = stop_criterion.max_wallclock_time
            new_stop_criterion = StoppingCriterion(
                max_num_trials_started=stop_criterion.max_num_trials_started,
                max_num_trials_completed=stop_criterion.max_num_trials_completed,
                max_cost=stop_criterion.max_cost,
                max_num_trials_finished=stop_criterion.max_num_trials_finished,
                max_metric_value={ST_TUNER_TIME: max_wallclock_time},
                max_num_evaluations=stop_criterion.max_num_evaluations,
            )
            tuner.stop_criterion = new_stop_criterion

    def on_tuning_start(self, tuner: "Tuner"):
        super(SimulatorCallback, self).on_tuning_start(tuner=tuner)
        if tuner.sleep_time != 0:
            logger.warning(
                "Setting sleep time of tuner to 0 as it is required for simulations."
            )
            tuner.sleep_time = 0
        backend = tuner.trial_backend
        assert isinstance(
            backend, SimulatorBackend
        ), "Use SimulatorCallback only together with SimulatorBackend"
        assert (
            tuner.sleep_time == 0
        ), "Initialize Tuner with sleep_time = 0 if you use the SimulatorBackend"
        self._time_keeper = backend.time_keeper
        scheduler = tuner.scheduler
        if isinstance(scheduler, FIFOScheduler):
            # Assign backend.time_keeper. It is important to do this here,
            # just at the start of an experiment, and not already at
            # construction of backend and scheduler. Otherwise, the way in
            # which backend and scheduler are serialized and deserialized for
            # remote tuning, leads to issues (in particular, the backend and its
            # time_keeper are recreated, so the scheduler refers to the wrong
            # time_keeper object then).
            scheduler.set_time_keeper(self._time_keeper)
        self._time_keeper.start_of_time()
        self._tuner_sleep_time = backend.tuner_sleep_time
        # Modify `tuner.stop_criterion` in case it depends on wallclock time
        self._modify_stop_criterion(tuner)
        self._tuner = tuner

    def on_tuning_sleep(self, sleep_time: float):
        self._time_keeper.advance(self._tuner_sleep_time)

    def on_tuning_end(self):
        super().on_tuning_end()
        # Restore `stop_criterion`
        self._tuner.stop_criterion = self._backup_stop_criterion
        self._tuner = None
