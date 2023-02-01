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
from pathlib import Path

from benchmarking.commons.launch_remote_simulator import launch_remote
from benchmarking.nursery.benchmark_dyhpo.benchmark_definitions import (
    benchmark_definitions,
)
from benchmarking.nursery.benchmark_dyhpo.baselines import methods
from benchmarking.nursery.benchmark_dyhpo.hpo_main import extra_args


if __name__ == "__main__":

    def _is_expensive_method(method: str) -> bool:
        return method not in ["RS", "BO", "ASHA"]

    entry_point = Path(__file__).parent / "hpo_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        extra_args=extra_args,
        is_expensive_method=_is_expensive_method,
    )
