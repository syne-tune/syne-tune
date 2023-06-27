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
Example for running the simulator backend on a tabulated benchmark
"""
import logging

from syne_tune.experiments.benchmark_definitions.nas201 import nas201_benchmark
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    n_workers = 4
    dataset_name = "cifar100"
    benchmark = nas201_benchmark(dataset_name)

    # Simulator backend specialized to tabulated blackboxes
    max_resource_attr = benchmark.max_resource_attr
    trial_backend = BlackboxRepositoryBackend(
        elapsed_time_attr=benchmark.elapsed_time_attr,
        max_resource_attr=max_resource_attr,
        blackbox_name=benchmark.blackbox_name,
        dataset=dataset_name,
    )

    # Asynchronous successive halving (ASHA)
    blackbox = trial_backend.blackbox
    scheduler = ASHA(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        ),
        max_resource_attr=max_resource_attr,
        resource_attr=blackbox.fidelity_name(),
        mode=benchmark.mode,
        metric=benchmark.metric,
        search_options={"debug_log": False},
        random_seed=random_seed,
    )

    max_wallclock_time = 3600
    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    # Printing the status during tuning takes a lot of time, and so does
    # storing results.
    print_update_interval = 700
    results_update_interval = 300
    # It is important to set ``sleep_time`` to 0 here (mandatory for simulator
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
