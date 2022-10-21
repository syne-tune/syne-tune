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
from pathlib import Path

import numpy as np
import pytest

from syne_tune import StoppingCriterion
from syne_tune import Tuner
from syne_tune.backend.trial_status import Status
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.util import script_height_example_path
from tst.util_test import temporary_local_backend


@pytest.fixture
def tunertools():
    max_steps = 100
    sleep_time = 0.01
    max_wallclock_time = 0.5
    mode = "min"
    metric = "mean_loss"

    config_space = {
        "steps": max_steps,
        "sleep_time": sleep_time,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = script_height_example_path()
    scheduler = RandomSearch(config_space, metric=metric, mode=mode)
    trial_backend = temporary_local_backend(entry_point=entry_point)
    stop_criterion = StoppingCriterion(
        max_wallclock_time=max_wallclock_time, min_metric_value={"mean_loss": -np.inf}
    )
    return scheduler, stop_criterion, trial_backend


@pytest.mark.parametrize(
    "wait_trial_completion_when_stopping,desired_status",
    [
        (False, Status.stopped),  # Worker should be stopped after 0.5 second
        (True, Status.completed),  # Worker should complete (NOT be stopped) after 0.5 second
    ])
def test_tuner_not_wait_trial_completion_when_stopping(
        tunertools,
        wait_trial_completion_when_stopping: bool,
        desired_status: str
):
    # Worker should be stopped after 1 second given the max_wallclock_time is 1s
    scheduler, stop_criterion, trial_backend = tunertools
    tuner = Tuner(
        trial_backend=trial_backend,
        sleep_time=0.01,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        wait_trial_completion_when_stopping=wait_trial_completion_when_stopping
    )
    tuner.run()
    for trial, status in tuner.tuning_status.last_trial_status_seen.items():
        assert status == desired_status


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # Run the tests without capturing stdout if this file is executed
    pytest.main(args=["-s", Path(__file__)])
