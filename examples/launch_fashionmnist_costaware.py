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
Example for cost-aware promotion-based Hyperband
"""
import logging

from benchmarking.definitions.definition_mlp_on_fashion_mnist import mlp_fashionmnist_default_params, \
    mlp_fashionmnist_benchmark
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune import Tuner, StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    #logging.getLogger().setLevel(logging.INFO)

    # We pick the MLP on FashionMNIST benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    random_seed = 31415927
    n_workers = 4
    default_params = mlp_fashionmnist_default_params()
    benchmark = mlp_fashionmnist_benchmark(default_params)
    mode = benchmark['mode']
    metric = benchmark['metric']

    # If you don't like the default config_space, change it here. But let
    # us use the default
    config_space = benchmark['config_space']

    # Local back-end
    trial_backend = LocalBackend(entry_point=benchmark['script'])

    # Cost-aware variant of ASHA, using a random searcher
    scheduler = HyperbandScheduler(
        config_space,
        searcher='random',
        max_t=default_params['max_resource_level'],
        grace_period=default_params['grace_period'],
        reduction_factor=default_params['reduction_factor'],
        resource_attr=benchmark['resource_attr'],
        mode=mode,
        metric=metric,
        type='cost_promotion',
        rung_system_kwargs={'cost_attr': benchmark['elapsed_time_attr']},
        random_seed=random_seed)

    stop_criterion = StoppingCriterion(max_wallclock_time=120)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
