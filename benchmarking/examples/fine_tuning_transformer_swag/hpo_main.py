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

from benchmarking.examples.fine_tuning_transformer_swag.baselines import methods
from benchmarking.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from benchmarking.benchmark_definitions.finetune_transformer_swag import (
    MAX_RESOURCE_ATTR,
    BATCH_SIZE_ATTR,
)
from syne_tune.experiments.launchers.hpo_main_local import main


extra_args = [
    dict(
        name="num_train_epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs",
    ),
    dict(
        name="batch_size",
        type=int,
        default=8,
        help="Training batch size (per device)",
    ),
]


def map_method_args(args, method: str, method_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # We need to change ``method_kwargs.config_space``, based on ``extra_args``
    new_method_kwargs = method_kwargs.copy()
    new_config_space = new_method_kwargs["config_space"].copy()
    new_config_space[MAX_RESOURCE_ATTR] = args.num_train_epochs
    new_config_space[BATCH_SIZE_ATTR] = args.batch_size
    new_method_kwargs["config_space"] = new_config_space
    return new_method_kwargs


if __name__ == "__main__":
    main(methods, benchmark_definitions, extra_args, map_method_args)
