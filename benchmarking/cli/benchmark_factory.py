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
import logging

from benchmarking.definitions.definition_nasbench201 import (
    nasbench201_benchmark,
    nasbench201_default_params,
    DATASET_NAMES as NB201_DATASET_NAMES,
)
from benchmarking.definitions.definition_nashpobench import (
    nashpobench_benchmark,
    nashpobench_default_params,
    DATASET_NAMES as FCNET_DATASET_NAMES,
)
from benchmarking.definitions.definition_lcbench import (
    lcbench_benchmark,
    lcbench_default_params,
    DATASET_NAMES as LCBENCH_DATASET_NAMES,
)
from benchmarking.definitions.definition_mlp_on_fashion_mnist import (
    mlp_fashionmnist_benchmark,
    mlp_fashionmnist_default_params,
)
from benchmarking.definitions.definition_resnet_cifar10 import (
    resnet_cifar10_benchmark,
    resnet_cifar10_default_params,
)
from benchmarking.nursery.lstm_wikitext2.definition_lstm_wikitext2 import (
    lstm_wikitext2_benchmark,
    lstm_wikitext2_default_params,
)

logger = logging.getLogger(__name__)

__all__ = ["supported_benchmarks", "benchmark_factory"]


MULTI_DATASET_BENCHMARKS = [
    "nasbench201_",
    "nashpobench_",
    "lcbench_",
]


_NB201_BENCHMARKS = {
    f"nasbench201_{dataset}": (nasbench201_benchmark, nasbench201_default_params)
    for dataset in NB201_DATASET_NAMES
}

_FCNET_BENCHMARKS = {
    f"nashpobench_{dataset}": (nashpobench_benchmark, nashpobench_default_params)
    for dataset in FCNET_DATASET_NAMES
}

_LCBENCH_BENCHMARKS = {
    f"lcbench_{dataset}": (lcbench_benchmark, lcbench_default_params)
    for dataset in LCBENCH_DATASET_NAMES
}

BENCHMARKS = {
    **_NB201_BENCHMARKS,
    **_FCNET_BENCHMARKS,
    **_LCBENCH_BENCHMARKS,
    "mlp_fashionmnist": (mlp_fashionmnist_benchmark, mlp_fashionmnist_default_params),
    "resnet_cifar10": (resnet_cifar10_benchmark, resnet_cifar10_default_params),
    "lstm_wikitext2": (lstm_wikitext2_benchmark, lstm_wikitext2_default_params),
}


def supported_benchmarks():
    return BENCHMARKS.keys()


def benchmark_factory(params):
    name = params["benchmark_name"]
    assert (
        name in supported_benchmarks()
    ), f"benchmark_name = {name} not supported, choose from:\n{supported_benchmarks()}"

    for prefix in MULTI_DATASET_BENCHMARKS:
        if name.startswith(prefix):
            dataset_name = name[len(prefix) :]
            params["dataset_name"] = dataset_name

    benchmark, default_params = BENCHMARKS[name]
    # We want to use `default_params` of the benchmark as input if not in
    # `params`
    default_params = default_params(params)
    _params = default_params.copy()
    # Note: `_params.update(params)` does not work, because `params` contains
    # None values
    for k, v in params.items():
        if v is not None:
            _params[k] = v
    return benchmark(_params), default_params
