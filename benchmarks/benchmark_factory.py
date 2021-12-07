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

from benchmarks.definitions.nasbench201 import nasbench201_benchmark, \
    nasbench201_default_params
from benchmarks.definitions.nas_hpo_bench import nashpobench_benchmark, nashpobench_default_params

from examples.training_scripts.mlp_on_fashion_mnist.mlp_on_fashion_mnist \
    import mlp_fashionmnist_benchmark, mlp_fashionmnist_default_params
from examples.training_scripts.resnet_cifar10.resnet_cifar10 import \
    resnet_cifar10_benchmark, resnet_cifar10_default_params
from examples.training_scripts.lstm_wikitext2.lstm_wikitext2 import \
    lstm_wikitext2_benchmark, lstm_wikitext2_default_params

logger = logging.getLogger(__name__)

__all__ = ['supported_benchmarks',
           'benchmark_factory']


BENCHMARKS = {
    'mlp_fashionmnist': (
        mlp_fashionmnist_benchmark, mlp_fashionmnist_default_params),
    'nasbench201': (
        nasbench201_benchmark, nasbench201_default_params),
    'nasbench201_cifar10': (
        nasbench201_benchmark, nasbench201_default_params),
    'nasbench201_cifar100': (
        nasbench201_benchmark, nasbench201_default_params),
    'nasbench201_ImageNet16-120': (
        nasbench201_benchmark, nasbench201_default_params),
    'resnet_cifar10': (
        resnet_cifar10_benchmark, resnet_cifar10_default_params),
    'lstm_wikitext2': (
        lstm_wikitext2_benchmark, lstm_wikitext2_default_params),
    'nashpobench_protein_structure': (
        nashpobench_benchmark, nashpobench_default_params),
    'nashpobench_naval_propulsion': (
        nashpobench_benchmark, nashpobench_default_params),
    'nashpobench_parkinsons_telemonitoring': (
        nashpobench_benchmark, nashpobench_default_params),
    'nashpobench_slice_localization': (
        nashpobench_benchmark, nashpobench_default_params),
}


def supported_benchmarks():
    return BENCHMARKS.keys()


def benchmark_factory(params):
    name = params['benchmark_name']
    assert name in supported_benchmarks(), \
        f"benchmark_name = {name} not supported, choose from:\n{supported_benchmarks()}"

    if name.startswith('nasbench201_'):
        dataset_name = name[len('nasbench201_'):]
        params['dataset_name'] = dataset_name

    if name.startswith('nashpobench_'):
        dataset_name = name[len('nashpobench_'):]
        params['dataset_name'] = dataset_name

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
