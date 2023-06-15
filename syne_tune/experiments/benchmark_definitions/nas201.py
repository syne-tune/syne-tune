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
from syne_tune.experiments.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)


NAS201_MAX_WALLCLOCK_TIME = {
    "cifar10": 5 * 3600,
    "cifar100": 6 * 3600,
    "ImageNet16-120": 8 * 3600,
}


NAS201_N_WORKERS = {
    "cifar10": 4,
    "cifar100": 4,
    "ImageNet16-120": 8,
}


def nas201_benchmark(dataset_name):
    return SurrogateBenchmarkDefinition(
        max_wallclock_time=NAS201_MAX_WALLCLOCK_TIME[dataset_name],
        n_workers=NAS201_N_WORKERS[dataset_name],
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


nas201_benchmark_definitions = {
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}
