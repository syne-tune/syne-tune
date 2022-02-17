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
Example for running the simulator back-end on a tabulated benchmark
"""
import logging

from benchmarking.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend

from benchmarking.definitions.definition_nasbench201 import nasbench201_default_params, nasbench201_benchmark
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune import Tuner
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 4
    default_params = nasbench201_default_params({'backend': 'simulated'})
    benchmark = nasbench201_benchmark(default_params)
    # Benchmark must be tabulated to support simulation:
    assert benchmark.get('supports_simulated', False)
    mode = benchmark['mode']
    metric = benchmark['metric']
    blackbox_name = benchmark.get('blackbox_name')
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None
    dataset_name = 'cifar100'

    # If you don't like the default config_space, change it here. But let
    # us use the default
    config_space = benchmark['config_space']

    # Simulator back-end specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=benchmark['elapsed_time_attr'],
        time_this_resource_attr=benchmark.get('time_this_resource_attr'),
        dataset=dataset_name)

    searcher = 'random'
    # Hyperband (or successive halving) scheduler of the stopping type.
    scheduler = HyperbandScheduler(
        config_space,
        searcher=searcher,
        max_t=default_params['max_resource_level'],
        grace_period=default_params['grace_period'],
        reduction_factor=default_params['reduction_factor'],
        resource_attr=benchmark['resource_attr'],
        mode=mode,
        metric=metric,
        random_seed=random_seed)
    # Make scheduler aware of time_keeper
    scheduler.set_time_keeper(trial_backend.time_keeper)

    max_wallclock_time = 600
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    # Printing the status during tuning takes a lot of time, and so does
    # storing results.
    print_update_interval = 700
    results_update_interval = 300
    # It is important to set `sleep_time` to 0 here (mandatory for simulator
    # backend)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )
    tuner.run()
