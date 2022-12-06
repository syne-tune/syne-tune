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

from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.benchmark_definitions.lstm_wikitext2 import (
    lstm_wikitext2_benchmark,
)
from benchmarking.commons.benchmark_definitions.resnet_cifar10 import (
    resnet_cifar10_benchmark,
)
from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)
from benchmarking.commons.benchmark_definitions.distilbert_on_imdb import (
    distilbert_imdb_benchmark,
)
from benchmarking.commons.benchmark_definitions.transformer_wikitext2 import (
    transformer_wikitext2_benchmark,
)


def real_benchmark_definitions(
    sagemaker_backend: bool = False, **kwargs
) -> Dict[str, RealBenchmarkDefinition]:
    return {
        "resnet_cifar10": resnet_cifar10_benchmark(sagemaker_backend, **kwargs),
        "lstm_wikitext2": lstm_wikitext2_benchmark(sagemaker_backend, **kwargs),
        "mlp_fashionmnist": mlp_fashionmnist_benchmark(sagemaker_backend, **kwargs),
        "distilbert_imdb": distilbert_imdb_benchmark(sagemaker_backend, **kwargs),
        "transformer_wikitext2": transformer_wikitext2_benchmark(
            sagemaker_backend, **kwargs
        ),
    }
