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
from typing import Dict

from benchmarking.commons.benchmark_definitions import RealBenchmarkDefinition
from benchmarking.nursery.odsc_tutorial.resnet_cifar10.code.resnet_cifar10_definition import (
    resnet_cifar10_benchmark,
)


def benchmark_definitions(
    sagemaker_backend: bool = False, **kwargs
) -> Dict[str, RealBenchmarkDefinition]:
    return {
        "resnet_cifar10": resnet_cifar10_benchmark(
            sagemaker_backend=sagemaker_backend, **kwargs
        ),
    }