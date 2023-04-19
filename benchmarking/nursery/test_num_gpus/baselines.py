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
import copy

from benchmarking.commons.default_baselines import (
    MOBSTER,
)
from benchmarking.commons.baselines import MethodArguments


class Methods:
    MOBSTER = "MOBSTER"


def _modify_config_space(method_arguments: MethodArguments) -> MethodArguments:
    result = copy.copy(method_arguments)
    result.config_space = dict(
        method_arguments.config_space, num_gpus=method_arguments.num_gpus_per_trial
    )
    return result


methods = {
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        _modify_config_space(method_arguments), type="promotion"
    ),
}
