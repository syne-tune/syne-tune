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

import pytest

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend.trial_status import Status
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.util import script_height_example_path
from tst.util_test import temporary_local_backend

_parameterizations = [
    (False, Status.stopped),  # Worker should be stopped after 0.5 second
    (
        True,
        Status.completed,
    ),  # Worker should complete (NOT be stopped) after 0.5 second
]


@pytest.mark.timeout(5)
@pytest.mark.parametrize("wait_for_completion, desired_status", _parameterizations)
def test_tuner_wait_trial_completion_when_stopping(wait_for_completion, desired_status):
    """
    This test check the behavior of the wait_trial_completion_when_stopping parametr of the tuner.

    In this scenario, starting the tuner will start n_workes (1) workers,
    each aiming to evaluate the objective for 0.2s (100 steps of 0.02s) given a set of params.

    In case the tuner does not wait until worker completion (wait_trial_completion_when_stopping=False)
    the jobs will be stopped after the stopping criterion is fulfilled (walltime reachig 0.1s).
    This means all worker jobs will be aborted and and with Status.stopped

    In case the tuner does wait until worker completion (wait_trial_completion_when_stopping=True)
    the jobs will not be stopped after the stopping criterion is fulfilled (walltime reachig 0.1s).
    They will run until completion 0.1s later (full 0.2s passed) and then conclude naturally with Status.compled
    """
    max_steps = 10
    sleep_time_bench = 0.02
    sleep_time_tuner = 0.01
    max_wallclock_time = 0.1
    mode = "min"
    metric = "mean_loss"
    num_workers = 1

    config_space = {
        "steps": max_steps,
        "sleep_time": sleep_time_bench,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = script_height_example_path()
    scheduler = RandomSearch(config_space, metric=metric, mode=mode)
    trial_backend = temporary_local_backend(entry_point=entry_point)
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    tuner = Tuner(
        trial_backend=trial_backend,
        sleep_time=sleep_time_tuner,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
        wait_trial_completion_when_stopping=wait_for_completion,
    )
    tuner.run()
    for trial, status in tuner.tuning_status.last_trial_status_seen.items():
        assert status == desired_status
