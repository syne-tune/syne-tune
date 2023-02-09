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
This example show how to launch a tuning job with a custom benchmark
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

from benchmarking.commons.benchmark_definitions import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_sagemaker import main as main_sagemaker
from benchmarking.commons.hpo_main_local import main as main_local
from benchmarking.nursery.launch_sagemaker.baselines import (
    methods as all_methods,
    Methods,
)
from syne_tune.config_space import randint
from syne_tune.remote.estimators import (
    DEFAULT_CPU_INSTANCE_SMALL,
)


def height_benchmark(
    sagemaker_backend: bool = False, **kwargs
) -> Dict[str, RealBenchmarkDefinition]:
    """
    This function defines the benchmark definition based on our training script
    In order to use this benchmark, you need to specify --benchmark height_benchmark in input params
    """
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    script = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )
    mode = "min"
    metric = "mean_loss"

    benchargs = dict(
        script=script,
        config_space=config_space,
        max_wallclock_time=5 * 60,
        n_workers=n_workers,
        instance_type=DEFAULT_CPU_INSTANCE_SMALL,
        metric=metric,
        mode=mode,
        max_resource_attr="epochs",
        framework="PyTorch",
    )
    benchargs.update(kwargs)
    return {"height_benchmark": RealBenchmarkDefinition(**benchargs)}


if __name__ == "__main__":
    """
    Use this example to run the custom-defined benchmark with remote workers for selected SF methods.
    The benchmark name needs to be passed using --benchmark <bench_name> command line argument.
    Methods can be specified using --method <method>, for example <--method BO>
    """
    parser = argparse.ArgumentParser(
        description="Launch a custom <train_height.py> benchmark using selected backend"
    )
    parser.add_argument(
        "--backend",
        choices=["local", "sagemaker"],
        required=True,
        type=str,
        help="Backed to use for experiment",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark to run, should be <height_benchmark> for this example",
    )
    args, unknown = parser.parse_known_args()

    logging.getLogger().setLevel(logging.INFO)
    single_fidelity_methods = {
        Methods.RS: all_methods[Methods.RS],
        Methods.BO: all_methods[Methods.BO],
    }

    print(f"Got the backend argument of {args.backend}, starting tuning")
    if args.backend == "local":
        main_local(
            methods=single_fidelity_methods,
            benchmark_definitions=height_benchmark,
        )
    elif args.backend == "sagemaker":
        main_sagemaker(
            methods=single_fidelity_methods,
            benchmark_definitions=height_benchmark,
        )
