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
from pathlib import Path

import pytest
import itertools

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband_impl import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune import Tuner
from syne_tune import StoppingCriterion
from syne_tune.config_space import randint, choice, uniform
from syne_tune.util import script_checkpoint_example_path
from tst.util_test import temporary_local_backend


_grid_parameterizations = list(
    itertools.product(
        ["fifo"],
        ["grid"],
        ["min"],
    )
)



@pytest.mark.parametrize("scheduler, searcher, mode", _grid_parameterizations)
def test_grid_scheduler(scheduler, searcher, mode):
    max_steps = 100
    num_workers = 2
    random_seed = 382378624

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": uniform(-100, 100),
        "sleep_time": 0.001,
    }

    num_samples = {"width":5, "height":10}
    metric = "mean_loss"

    entry_point = (
            Path(__file__).parent.parent
            / "examples"
            / "training_scripts"
            / "height_example"
            / "train_height.py"
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    #entry_point = str(script_checkpoint_example_path())
    #trial_backend = temporary_local_backend(entry_point=entry_point)

    search_options = {"debug_log": True, "num_init_random": num_workers, "num_samples":num_samples}

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )
    stop_criterion = StoppingCriterion(max_wallclock_time=60)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()