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
import copy
import pytest

from benchmarking.definitions.definition_nasbench201 import (
    nasbench201_default_params,
    nasbench201_benchmark,
)
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune import Tuner, StoppingCriterion
from syne_tune.optimizer.schedulers.hb_replay.hyperband_wrapper import (
    HyperbandSchedulerWrapper
)
from syne_tune.optimizer.schedulers.hb_replay.replay_scheduler import (
    replay_scheduling_events
)
from syne_tune.optimizer.schedulers.hb_replay.test_hyperband import (
    state_representation_hyperband
)


class RecordedEventsSimulatorCallback(SimulatorCallback):
    """
    On top of :class:`SimulatorCallback`, this is also storing recorded
    events and scheduler state several times during the course of the
    experiment, which runs for `max_wallclock_time` seconds.
    """

    def __init__(
            self,
            max_wallclock_time: float,
            num_writeout: int = 10
    ):
        super().__init__()
        self._max_wallclock_time = max_wallclock_time
        self._num_writeout = num_writeout
        self._period_between_writeout = max_wallclock_time / num_writeout
        self.events_and_states = []
        self._next_threshold = self._period_between_writeout

    def on_tuning_start(self, tuner: "Tuner"):
        super().on_tuning_start(tuner=tuner)
        assert isinstance(tuner.scheduler, HyperbandSchedulerWrapper)

    def _writeout_event_and_state(self):
        scheduler = self._tuner.scheduler
        scheduler_state = state_representation_hyperband(scheduler)
        self.events_and_states.append(
            (copy.copy(scheduler.events_to_replay), scheduler_state)
        )

    def on_tuning_sleep(self, sleep_time: float):
        super().on_tuning_sleep(sleep_time)
        current_time = self._time_keeper.time()
        if current_time >= self._next_threshold:
            self._next_threshold += self._period_between_writeout
            self._writeout_event_and_state()

    def on_tuning_end(self):
        if len(self.events_and_states) < self._num_writeout:
            self._writeout_event_and_state()
        # Resets `self._tuner`, so call here only
        super().on_tuning_end()


# This test runs very fast, but needs the nasbench201 files to be downloaded
# for the blackbox repository, which can take a long time
@pytest.mark.skip("Needs blackbox repository")
def test_hyperband_replay_nasbench201():
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 4
    default_params = nasbench201_default_params({"backend": "simulated"})
    benchmark = nasbench201_benchmark(default_params)
    # Benchmark must be tabulated to support simulation:
    assert benchmark.get("supports_simulated", False)
    config_space = benchmark["config_space"]
    mode = benchmark["mode"]
    metric = benchmark["metric"]
    blackbox_name = benchmark.get("blackbox_name")
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None
    dataset_name = "cifar100"

    # Simulator back-end specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=benchmark["elapsed_time_attr"],
        time_this_resource_attr=benchmark.get("time_this_resource_attr"),
        dataset=dataset_name,
    )

    # Run experiment to record events and build up scheduler state
    print("\n*** Run experiment and record events ***\n")
    scheduler_kwargs = dict(
        searcher="random",
        max_t=default_params["max_resource_level"],
        grace_period=default_params["grace_period"],
        reduction_factor=default_params["reduction_factor"],
        resource_attr=benchmark["resource_attr"],
        mode=mode,
        metric=metric,
        random_seed=random_seed,
        search_options={"debug_log": False},
    )
    scheduler = HyperbandSchedulerWrapper(config_space, **scheduler_kwargs)
    max_wallclock_time = 600
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    print_update_interval = 700
    results_update_interval = 300
    # It is important to set `sleep_time` to 0 here (mandatory for simulator
    # backend)
    events_callback = RecordedEventsSimulatorCallback(
        max_wallclock_time=max_wallclock_time
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
        callbacks=[events_callback],
    )
    tuner.run()

    # For each writeout: Replay events and compare states
    for ind, (events_to_replay, scheduler_state) in enumerate(
            events_callback.events_and_states):
        print(f"\n*** Writeout {ind}: Replay events ***\n")
        replayed_scheduler = replay_scheduling_events(
            events_to_replay=events_to_replay,
            config_space=config_space,
            **scheduler_kwargs
        )["trial_scheduler"]
        replayed_state = state_representation_hyperband(replayed_scheduler)
        print(f"\n*** Writeout {ind}: Compare scheduler states ***\n")
        assert scheduler_state == replayed_state, (
                "States are different! scheduler_state:\n" + repr(scheduler_state)
                + "\nreplayed_state:\n" + repr(replayed_state)
        )
        print("OK, original and replayed state are the same")
