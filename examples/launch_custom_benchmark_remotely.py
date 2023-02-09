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
This example show how to launch a tuning job that will orchestrated on Sagemaker rather than on your local machine.
"""

import logging
from pathlib import Path

from benchmarking.commons.launch_remote_sagemaker import launch_remote
from benchmarking.nursery.launch_sagemaker.baselines import (
    methods as all_methods,
    Methods,
)
from examples.launch_custom_benchmark import height_benchmark

if __name__ == "__main__":
    """
    Use this example to run the custom-defined benchmark remotely using specified methods.
    The benchmark name needs to be passed using --benchmark <bench_name> command line argument.
    For the custom height example, one needs to specify `--benchmark height_benchmark`
    """
    logging.getLogger().setLevel(logging.INFO)
    entry_point = Path(__file__).parent / "launch_custom_benchmark.py"
    single_fidelity_methods = {
        Methods.RS: all_methods[Methods.RS],
        Methods.BO: all_methods[Methods.BO],
    }
    launch_remote(
        entry_point=entry_point,
        methods=single_fidelity_methods,
        benchmark_definitions=height_benchmark,
        extra_args=[
            dict(
                name="backend",
                choices=["local", "sagemaker"],
                required=True,
                type=str,
                help="Backed to use for experiment",
            )
        ],
    )
