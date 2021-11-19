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
Example for running synchronous Hyperband together with the simulator
back-end on a tabulated benchmark
"""
import logging
import argparse

from syne_tune.backend.simulator_backend.simulator_backend import \
    SimulatorBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.synchronous.hyperband_impl import \
    SynchronousGeometricHyperbandScheduler
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion

from examples.training_scripts.nasbench201.nasbench201 import \
    nasbench201_benchmark, nasbench201_default_params


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description='Synchronous Hyperband on NASBench201')
    parser.add_argument(
        '--dataset_s3_bucket', type=str, required=True,
        help='Name of S3 bucket where NASBench201 files can be downloaded')
    params = vars(parser.parse_args())
    dataset_s3_bucket = params['dataset_s3_bucket']

    random_seed = 31415927
    n_workers = 4
    default_params = nasbench201_default_params({'backend': 'simulated'})
    benchmark = nasbench201_benchmark(default_params)
    # Benchmark must be tabulated to support simulation:
    assert benchmark.get('supports_simulated', False)
    mode = benchmark['mode']
    metric = benchmark['metric']

    # If you don't like the default config_space, change it here. But let
    # us use the default
    config_space = benchmark['config_space']
    # TODO: Needs better solution!
    config_space['dataset_s3_bucket'] = dataset_s3_bucket

    # Simulator back-end
    # If the benchmark provides a table object, use that. Otherwise, call the
    # training script
    backend = SimulatorBackend(
        entry_point=benchmark['script'],
        elapsed_time_attr=benchmark['elapsed_time_attr'],
        table_class_name=benchmark.get('benchmark_table_class'))

    # Synchronous Hyperband, using the default setup of rung levels using
    # geometric sequences. We use 3 brackets here.
    # Note: batch_size must be equal to the number of workers the Tuner
    # is running with.
    scheduler = SynchronousGeometricHyperbandScheduler(
        config_space,
        max_resource_level=default_params['max_resource_level'],
        grace_period=default_params['grace_period'],
        reduction_factor=default_params['reduction_factor'],
        brackets=3,
        max_resource_attr=default_params['max_resource_attr'],
        resource_attr=benchmark['resource_attr'],
        batch_size=n_workers,
        mode=mode,
        metric=metric,
        random_seed=random_seed)

    max_wallclock_time = 600
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    # Printing the status during tuning takes a lot of time, and so does
    # storing results.
    print_update_interval = 700
    results_update_interval = 300
    # It is important to set `sleep_time` to 0 here (mandatory for simulator
    # backend)
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        asynchronous_scheduling=False,
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
