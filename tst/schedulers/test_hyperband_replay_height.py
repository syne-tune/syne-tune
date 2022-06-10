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

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.hb_replay.hyperband_wrapper import (
    HyperbandSchedulerWrapper
)
from syne_tune.optimizer.schedulers.hb_replay.replay_scheduler import (
    replay_scheduling_events
)
from syne_tune.optimizer.schedulers.hb_replay.test_hyperband import (
    state_representation_hyperband
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint
from syne_tune.util import repository_root_path


def test_hyperband_replay_height():
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        "sleep_time": 0.01,
    }
    entry_point = (
        repository_root_path()
        / "examples"
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )
    mode = "min"
    metric = "mean_loss"

    # Run experiment to record events and build up scheduler state
    print("\n*** Run experiment and record events ***\n")
    trial_backend = LocalBackend(entry_point=str(entry_point))
    stop_criterion = StoppingCriterion(max_wallclock_time=3)
    scheduler_kwargs = dict(
        metric= metric,
        mode=mode,
        searcher="random",
        resource_attr="epoch",
        max_t=max_steps,
        search_options={"debug_log": False},
    )
    scheduler = HyperbandSchedulerWrapper(config_space, **scheduler_kwargs)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )
    tuner.run()

    # Replay recorded events
    print("\n*** Replay events ***\n")
    original_state = state_representation_hyperband(scheduler)
    replayed_scheduler = replay_scheduling_events(
        events_to_replay=scheduler.events_to_replay,
        config_space=config_space,
        **scheduler_kwargs
    )["trial_scheduler"]
    replayed_state = state_representation_hyperband(replayed_scheduler)

    print("\n*** Compare scheduler states ***\n")
    assert original_state == replayed_state, (
        "States are different! original_state:\n" + repr(original_state)
        + "\nreplayed_state:\n" + repr(replayed_state)
    )
    print("OK, original and replayed state are the same")
