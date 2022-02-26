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

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import uniform


@pytest.mark.skip("this unit test takes about a minute and is skipped for now")
@pytest.mark.parametrize("scheduler, searcher, cost", [
    ('fifo', 'bayesopt', 1),  # ignored cost
    ('fifo', 'bayesopt_cost', 1),  # linear cost
    ('fifo', 'bayesopt_cost', 2),  # quadratic cost
])
def test_cost_aware_bayesopt(scheduler, searcher, cost):
    num_workers = 2

    config_space = {
        "x1": uniform(-5, 10),
        "x2": uniform(0, 15),
        "cost": cost  # cost_value = x2 ** cost
    }

    trial_backend = LocalBackend(
        entry_point=Path(__file__).parent.parent / "examples"
                    / "training_scripts" / "cost_aware_hpo" / "train_cost_aware_example.py")

    search_options = {
        'num_init_random': num_workers,
        'cost_attr': 'elapsed_time',  # Name of the cost metric captured by the reporter (mandatory)
    }
    stop_criterion = StoppingCriterion(max_wallclock_time=18)

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode='max',
        metric='objective',
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        stop_criterion=stop_criterion,
        n_workers=num_workers,
    )

    tuner.run()
