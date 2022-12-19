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
from benchmarking.commons.benchmark_definitions.common import (  # noqa: F401
    SurrogateBenchmarkDefinition,
    RealBenchmarkDefinition,
)
from benchmarking.commons.benchmark_definitions.fcnet import (  # noqa: F401
    fcnet_benchmark_definitions,
    fcnet_benchmark,
)
from benchmarking.commons.benchmark_definitions.nas201 import (  # noqa: F401
    nas201_benchmark_definitions,
    nas201_benchmark,
)
from benchmarking.commons.benchmark_definitions.lcbench import (  # noqa: F401
    lcbench_benchmark_definitions,
    lcbench_selected_benchmark_definitions,
    lcbench_benchmark,
)
from benchmarking.commons.benchmark_definitions.yahpo import (  # noqa: F401
    yahpo_nb301_benchmark_definitions,
    yahpo_lcbench_benchmark_definitions,
    yahpo_lcbench_selected_benchmark_definitions,
    yahpo_iaml_benchmark_definitions,
    yahpo_iaml_selected_benchmark_definitions,
    yahpo_rbv2_benchmark_definitions,
    yahpo_rbv2_selected_benchmark_definitions,
    yahpo_nb301_benchmark,
    yahpo_lcbench_benchmark,
    yahpo_iaml_benchmark,
    yahpo_rbv2_benchmark,
)
from benchmarking.commons.benchmark_definitions.resnet_cifar10 import (  # noqa: F401
    resnet_cifar10_benchmark,
)
from benchmarking.commons.benchmark_definitions.lstm_wikitext2 import (  # noqa: F401
    lstm_wikitext2_benchmark,
)
from benchmarking.commons.benchmark_definitions.real_benchmark_definitions import (  # noqa: F401
    real_benchmark_definitions,
)
from benchmarking.commons.benchmark_definitions.distilbert_on_imdb import (  # noqa: F401
    distilbert_imdb_benchmark,
)
from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import (  # noqa: F401
    mlp_fashionmnist_benchmark,
)
from benchmarking.commons.benchmark_definitions.transformer_wikitext2 import (  # noqa: F401
    transformer_wikitext2_benchmark,
)

__all__ = [
    "SurrogateBenchmarkDefinition",
    "RealBenchmarkDefinition",
    "fcnet_benchmark_definitions",
    "fcnet_benchmark",
    "nas201_benchmark_definitions",
    "nas201_benchmark",
    "lcbench_benchmark_definitions",
    "lcbench_selected_benchmark_definitions",
    "lcbench_benchmark",
    "yahpo_nb301_benchmark_definitions",
    "yahpo_lcbench_benchmark_definitions",
    "yahpo_lcbench_selected_benchmark_definitions",
    "yahpo_iaml_benchmark_definitions",
    "yahpo_iaml_selected_benchmark_definitions",
    "yahpo_rbv2_benchmark_definitions",
    "yahpo_rbv2_selected_benchmark_definitions",
    "yahpo_nb301_benchmark",
    "yahpo_lcbench_benchmark",
    "yahpo_iaml_benchmark",
    "yahpo_rbv2_benchmark",
    "resnet_cifar10_benchmark",
    "lstm_wikitext2_benchmark",
    "real_benchmark_definitions",
    "distilbert_imdb_benchmark",
    "mlp_fashionmnist_benchmark",
    "transformer_wikitext2_benchmark",
]
