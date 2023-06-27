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
from syne_tune.experiments.benchmark_definitions.common import (  # noqa: F401
    SurrogateBenchmarkDefinition,
    RealBenchmarkDefinition,
)
from syne_tune.experiments.benchmark_definitions.fcnet import (  # noqa: F401
    fcnet_benchmark_definitions,
    fcnet_benchmark,
)
from syne_tune.experiments.benchmark_definitions.nas201 import (  # noqa: F401
    nas201_benchmark_definitions,
    nas201_benchmark,
)
from syne_tune.experiments.benchmark_definitions.lcbench import (  # noqa: F401
    lcbench_benchmark_definitions,
    lcbench_selected_benchmark_definitions,
    lcbench_benchmark,
)
from syne_tune.experiments.benchmark_definitions.yahpo import (  # noqa: F401
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
]
