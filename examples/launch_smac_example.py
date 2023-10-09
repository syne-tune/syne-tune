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
An example showing how to launch the tuning of a python function ``Rosenbrock2D`` with SMAC optimizer wrapper
or a simulation with nasbench201 from blackbox repository.

Requires that SMAC is installed, see SMAC documentation to install it on your machine:
https://github.com/automl/SMAC3/tree/main
"""

import logging

from syne_tune.backend import PythonBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.config_space import uniform
from syne_tune.experiments.benchmark_definitions import nas201_benchmark
from syne_tune.optimizer.baselines import SMAC
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


def rosenbrock2D(x0: float, x1: float) -> float:
    # Use the same function as Ask & Tell SMAC example
    # https://github.com/automl/SMAC3/blob/main/examples/1_basics/3_ask_and_tell.py

    from syne_tune import Reporter

    reporter = Reporter()

    cost = 100.0 * (x1 - x1**2.0) ** 2.0 + (1 - x0) ** 2.0
    print(f"Cost for {x0}/{x1}: {cost}")
    reporter(cost=cost)


def run_smac_python_function():
    config_space = {
        "x0": uniform(lower=-5, upper=10),
        "x1": uniform(lower=-5, upper=10),
    }

    trial_backend = PythonBackend(
        tune_function=rosenbrock2D,
        config_space=config_space,
    )
    scheduler = SMAC(
        config_space=config_space,
        metric="cost",
        points_to_evaluate=[{"x0": -3, "x1": 4}],
    )
    tuner = Tuner(
        scheduler=scheduler,
        trial_backend=trial_backend,
        n_workers=1,
        results_update_interval=10,
        stop_criterion=StoppingCriterion(max_wallclock_time=20),
    )

    tuner.run()


def run_smac_simulated_backend():
    # Simulate tuning on NASBench201 with simulated backend

    dataset_name = "cifar100"
    benchmark = nas201_benchmark(dataset_name)

    max_resource_attr = benchmark.max_resource_attr
    trial_backend = BlackboxRepositoryBackend(
        elapsed_time_attr=benchmark.elapsed_time_attr,
        max_resource_attr=max_resource_attr,
        blackbox_name=benchmark.blackbox_name,
        dataset=dataset_name,
    )

    blackbox = trial_backend.blackbox

    scheduler = SMAC(
        config_space=blackbox.configuration_space_with_max_resource_attr(
            benchmark.max_resource_attr
        ),
        metric=benchmark.metric,
        mode=benchmark.mode,
    )

    tuner = Tuner(
        scheduler=scheduler,
        trial_backend=trial_backend,
        n_workers=4,
        results_update_interval=10,
        stop_criterion=StoppingCriterion(max_num_trials_finished=1000),
        sleep_time=0,
        callbacks=[SimulatorCallback()],
    )

    tuner.run()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    run_smac_simulated_backend()

    run_smac_python_function()
