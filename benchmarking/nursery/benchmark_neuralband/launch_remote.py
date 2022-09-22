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
from benchmarking.commons.launch_remote import launch_remote
from benchmarking.commons.benchmark_definitions import benchmark_definitions


if __name__ == "__main__":
    from benchmarking.nursery.benchmark_neuralband.baselines import (
        methods,
        Methods,
    )
    from benchmarking.nursery.benchmark_neuralband.benchmark_main import (
        extra_args,
        map_extra_args,
    )

    def _is_expensive_method(method: str) -> bool:
        return method in {
            Methods.MOBSTER,
            Methods.NeuralBandSH,
            Methods.NeuralBandHB,
            Methods.NeuralBand_UCB,
            Methods.NeuralBand_TS,
            Methods.NeuralBandEpsilon,
        }

    launch_remote(
        methods, benchmark_definitions, extra_args, map_extra_args, _is_expensive_method
    )
