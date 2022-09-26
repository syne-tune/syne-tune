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
    fcnet_benchmark,
    lcbench_benchmark,
)
from benchmarking.nursery.benchmark_dehb.benchmark_definitions import (
    nas201_benchmark,
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
