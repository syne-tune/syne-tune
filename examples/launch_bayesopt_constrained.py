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
"""
Example for running constrained Bayesian optimization on a toy example
"""
import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune import Tuner
from syne_tune.config_space import uniform
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 2

    config_space = {
        "x1": uniform(-5, 10),
        "x2": uniform(0, 15),
        "constraint_offset": 1.0,  # the lower, the stricter
    }

    entry_point = str(
        Path(__file__).parent / "training_scripts" / "constrained_hpo"
        / "train_constrained_example.py")
    mode = "max"
    metric = "objective"
    constraint_attr = 'my_constraint_metric'

    # Local back-end
    trial_backend = LocalBackend(entry_point=entry_point)

    # Bayesian constrained optimization:
    #   max_x f(x)   s.t. c(x) <= 0
    # Here, `metric` represents f(x), `constraint_attr` represents c(x).
    search_options = {
        'num_init_random': n_workers,
        'constraint_attr': constraint_attr,
    }
    scheduler = FIFOScheduler(
        config_space,
        searcher='bayesopt_constrained',
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed)

    stop_criterion = StoppingCriterion(max_wallclock_time=30)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
