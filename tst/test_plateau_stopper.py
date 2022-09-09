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
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner
from syne_tune.stopping_criterion import PlateauStopper
from syne_tune.config_space import randint
from syne_tune.util import script_height_example_path
from syne_tune.backend.trial_status import Trial, Status
from syne_tune.tuning_status import TuningStatus

from tst.util_test import temporary_local_backend


def test_plateau_scheduler_integration():
    max_steps = 5
    num_workers = 1
    random_seed = 382378624

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        "sleep_time": 0.001,
    }

    entry_point = str(script_height_example_path())
    metric = "mean_loss"
    mode = "min"

    trial_backend = temporary_local_backend(entry_point=entry_point)

    search_options = {"debug_log": False, "num_init_random": num_workers}

    myscheduler = RandomSearch(
        config_space,
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
        points_to_evaluate=[
            {"width": 10, "height": 0},
            {"width": 7, "height": 0},
            {"width": 6, "height": 0},
        ],
    )

    stop_criterion = PlateauStopper(
        metric=metric, mode=mode, std=0.1, num_trials=3, patience=1
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()

    assert tuner.tuning_status.num_trials_finished == 3


def test_plateau_stopper():
    metric = "loss"

    stop_criterion = PlateauStopper(
        metric=metric, mode="min", std=0.1, num_trials=2, patience=1
    )

    status = TuningStatus(metric_names=[metric])

    trial0 = Trial(trial_id=0, config={"x": 1.0}, creation_time=None)
    trial1 = Trial(trial_id=1, config={"x": 5.0}, creation_time=None)
    status.update(
        trial_status_dict={
            0: (trial0, Status.completed),
            1: (trial1, Status.completed),
        },
        new_results=[
            (0, {metric: 4.0}),
            (1, {metric: 3.0}),
        ],
    )
    assert not stop_criterion(status)

    trial2 = Trial(trial_id=2, config={"x": 4.0}, creation_time=None)
    trial3 = Trial(trial_id=3, config={"x": 6.0}, creation_time=None)
    trial4 = Trial(trial_id=4, config={"x": 7.0}, creation_time=None)

    status.update(
        trial_status_dict={
            2: (trial2, Status.completed),
            3: (trial3, Status.completed),
            4: (trial4, Status.completed),
        },
        new_results=[
            (2, {metric: 3.00000}),
            (3, {metric: 3.000}),
            (4, {metric: 3.0000}),
        ],
    )

    assert stop_criterion(status)

    trial5 = Trial(trial_id=5, config={"x": 10.0}, creation_time=None)

    status.update(
        trial_status_dict={
            5: (trial5, Status.completed),
        },
        new_results=[
            (5, {metric: 1.00000}),
        ],
    )

    assert not stop_criterion(status)
