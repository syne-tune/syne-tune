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

# If you like to run the code linked in this tutorial, please make sure to use
# the current `PyPI` release. If you cloned the source repository, this is
# obtained as follows:
#
# ```bash
# git checkout -b basic_tutorial v0.11
# ```
#
# This gives you a local branch `basic_tutorial`, in which you can play around
# with the code.
import logging
from pathlib import Path

from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune import Tuner, StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 4
    max_wallclock_time = 3 * 3600  # Run for 3 hours
    max_resource_level = 81  # Maximum number of training epochs

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training script
    entry_point = str(Path(__file__).parent / "traincode_report_withcheckpointing.py")
    mode = 'max'
    metric = 'accuracy'
    resource_attr = 'epoch'
    max_resource_attr = 'epochs'

    # Search space (or configuration space)
    # For each tunable parameter, need to define type, range, and encoding
    # (linear, logarithmic)
    config_space = {
        'n_units_1': randint(4, 1024),
        'n_units_2': randint(4, 1024),
        'batch_size': randint(8, 128),
        'dropout_1': uniform(0, 0.99),
        'dropout_2': uniform(0, 0.99),
        'learning_rate': loguniform(1e-6, 1),
        'weight_decay': loguniform(1e-8, 1),
    }

    # Additional fixed parameters
    config_space.update({
        max_resource_attr: max_resource_level,
        'dataset_path': './',
    })

    # Local back-end: Responsible for scheduling trials
    # The local back-end runs trials as sub-processes on a single instance
    trial_backend = LocalBackend(entry_point=entry_point)

    # Scheduler:
    # 'HyperbandScheduler' runs asynchronous successive halving, or Hyperband.
    # It starts a trial whenever a worker is free.
    # The 'promotion' variant pauses each trial at certain resource levels
    # (called rungs). Trials which outperform others at the same rung, are
    # promoted later on, to run to the next higher rung.
    # We configure this scheduler with Bayesian optimization: configurations
    # for new trials are selected by optimizing an acquisition function based
    # on a Gaussian process surrogate model. The latter models learning curves
    # f(x, r), x the configuration, r the number of epochs done, not just final
    # values f(x).
    searcher = 'bayesopt'
    search_options = {
        'num_init_random': n_workers + 2,
        'gp_resource_kernel': 'exp-decay-sum',  # GP surrogate model
    }
    scheduler = HyperbandScheduler(
        config_space,
        type='promotion',
        searcher=searcher,
        search_options=search_options,
        grace_period=1,
        reduction_factor=3,
        resource_attr=resource_attr,
        max_resource_attr=max_resource_attr,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )

    # The experiment is stopped after `max_wallclock_time` seconds
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

    # Everything comes together in the tuner
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
