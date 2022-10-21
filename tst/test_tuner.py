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

import pytest

from syne_tune import Tuner
from syne_tune.backend.trial_status import Status
from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.util import script_height_example_path
from tst.util_test import temporary_local_backend

_parameterizations = [
    ("dummy", "dummy", False, Status.stopped),  # Worker should be stopped after 0.5 second
    ("dummy", "dummy", True, Status.completed),  # Worker should complete (NOT be stopped) after 0.5 second
    ("dummy", "dummy", True, Status.completed),  # Worker should complete (NOT be stopped) after 0.5 second
]


@pytest.mark.parametrize("dummy, dummy2, wait_for_completion, desired_status", _parameterizations)
def test_tuner_wait_trial_completion_when_stopping(dummy, dummy2, wait_for_completion, desired_status):
    max_steps = 10
    sleep_time_bench = 0.2
    sleep_time_tuner = 0.1
    max_wallclock_time = 1
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
    stop_criterion = lambda status: status.wallclock_time > max_wallclock_time
    tuner = Tuner(
        trial_backend=trial_backend,
        sleep_time=sleep_time_tuner,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
        wait_trial_completion_when_stopping=wait_for_completion
    )
    tuner.run()
    for trial, status in tuner.tuning_status.last_trial_status_seen.items():
        assert status == desired_status


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # Run the tests without capturing stdout if this file is executed
    pytest.main(args=["-s", Path(__file__)])
