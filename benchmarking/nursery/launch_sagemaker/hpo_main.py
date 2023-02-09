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
from typing import Dict, Any

from benchmarking.commons.hpo_main_sagemaker import main
from benchmarking.nursery.launch_sagemaker.baselines import methods, Methods
from benchmarking.commons.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from syne_tune.util import recursive_merge


extra_args = [
    dict(
        name="grace_period",
        type=int,
        default=1,
        help="Minimum resource level in Hyperband",
    ),
    dict(
        name="reduction_factor",
        type=int,
        default=3,
        help="Reduction factor in Hyperband",
    ),
]


def map_extra_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if method in (Methods.ASHA, Methods.MOBSTER):
        new_dict = {
            "scheduler_kwargs": {
                "grace_period": args.grace_period,
                "reduction_factor": args.reduction_factor,
            },
        }
        method_kwargs = recursive_merge(method_kwargs, new_dict)
    return method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions)
