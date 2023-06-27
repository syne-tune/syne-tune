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
from benchmarking.benchmark_definitions.distilbert_on_imdb import (  # noqa: F401
    distilbert_imdb_benchmark,
)
from benchmarking.benchmark_definitions.finetune_transformer_glue import (  # noqa: F401
    finetune_transformer_glue_benchmark,
    finetune_transformer_glue_all_benchmarks,
)
from benchmarking.benchmark_definitions.finetune_transformer_swag import (  # noqa: F401
    finetune_transformer_swag_benchmark,
)
from benchmarking.benchmark_definitions.lstm_wikitext2 import (  # noqa: F401
    lstm_wikitext2_benchmark,
)
from benchmarking.benchmark_definitions.mlp_on_fashionmnist import (  # noqa: F401
    mlp_fashionmnist_benchmark,
)
from benchmarking.benchmark_definitions.real_benchmark_definitions import (  # noqa: F401
    real_benchmark_definitions,
)
from benchmarking.benchmark_definitions.resnet_cifar10 import (  # noqa: F401
    resnet_cifar10_benchmark,
)
from benchmarking.benchmark_definitions.transformer_wikitext2 import (  # noqa: F401
    transformer_wikitext2_benchmark,
)

__all__ = [
    "distilbert_imdb_benchmark",
    "finetune_transformer_glue_benchmark",
    "finetune_transformer_glue_all_benchmarks",
    "finetune_transformer_swag_benchmark",
    "lstm_wikitext2_benchmark",
    "mlp_fashionmnist_benchmark",
    "real_benchmark_definitions",
    "resnet_cifar10_benchmark",
    "transformer_wikitext2_benchmark",
]
