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
from benchmarking.nursery.benchmark_automl.benchmark_definitions import (
    BenchmarkDefinition,
    fcnet_benchmark,
    lcbench_benchmark,
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
    return BenchmarkDefinition(
        max_wallclock_time=NAS201_MAX_WALLCLOCK_TIME[dataset_name],
        n_workers=NAS201_N_WORKERS[dataset_name],
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}

# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions[
        "lcbench-" + task.replace("_", "-").replace(".", "")
    ] = lcbench_benchmark(task, datasets=lc_bench_datasets)
