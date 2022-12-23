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
from benchmarking.commons.benchmark_definitions import (
    yahpo_iaml_benchmark_definitions,
    yahpo_rbv2_benchmark_definitions,
    yahpo_nb301_benchmark_definitions,
)
from benchmarking.commons.benchmark_definitions.yahpo import (
    yahpo_rbv2_metrics,
    yahpo_iaml_methods,
    yahpo_rbv2_methods,
)


# RESTRICT_FIDELITIES = False
RESTRICT_FIDELITIES = True


benchmark_definitions_iaml = {
    k: v
    for method in yahpo_iaml_methods
    for k, v in yahpo_iaml_benchmark_definitions(
        method, restrict_fidelities=RESTRICT_FIDELITIES
    ).items()
}


benchmark_definitions_rbv2 = dict()
for method in yahpo_rbv2_methods:
    definition = yahpo_rbv2_benchmark_definitions(
        method, restrict_fidelities=RESTRICT_FIDELITIES
    )
    for metric in yahpo_rbv2_metrics:
        key = f"rbv2-{method}-{metric[0]}"
        prefix = f"yahpo-rbv2_{method}_{metric[0]}"
        benchmark_definitions_rbv2[key] = {
            k: v for k, v in definition.items() if k.startswith(prefix)
        }


# benchmark_definitions = benchmark_definitions_iaml
# benchmark_definitions = benchmark_definitions_rbv2
benchmark_definitions = yahpo_nb301_benchmark_definitions
