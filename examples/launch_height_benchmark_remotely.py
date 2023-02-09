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
This example show how to launch a tuning job that will be executed on Sagemaker rather than on your local machine.
"""
import logging
from pathlib import Path

from benchmarking.commons.launch_remote_sagemaker import launch_remote
from benchmarking.nursery.launch_sagemaker.baselines import (
    methods as all_methods,
    Methods,
)
from examples.training_scripts.height_example.launch_height_benchmark_sagemaker import (
    height_benchmark,
)

if __name__ == "__main__":
    """
    Use this example to run the custom-defined benchmark on sagemaker remotely using all methods.
    The benchmark name needs to be passed using --benchmark <bench_name> command line argument.
    """
    logging.getLogger().setLevel(logging.INFO)
    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "launch_height_benchmark_sagemaker.py"
    )
    single_fidelity_methods = {
        Methods.RS: all_methods[Methods.RS],
        Methods.BO: all_methods[Methods.BO],
    }
    launch_remote(
        entry_point=entry_point,
        methods=single_fidelity_methods,
        benchmark_definitions=height_benchmark,
    )
