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
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)
from benchmarking.commons.baselines import MethodArguments


class Methods:
    RS = "RS"
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"
    ASHA_TANH = "ASHA-TANH"
    MOBSTER_TANH = "MOBSTER-TANH"
    ASHA_RELU = "ASHA-RELU"
    MOBSTER_RELU = "MOBSTER-RELU"


def _modify_config_space(
    method_arguments: MethodArguments, value: str
) -> MethodArguments:
    result = copy.copy(method_arguments)
    result.config_space = dict(
        method_arguments.config_space,
        hp_activation_fn_1=value,
        hp_activation_fn_2=value,
    )
    return result


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
    ),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
    ),
    # Fix activations to "tanh"
    Methods.ASHA_TANH: lambda method_arguments: ASHA(
        _modify_config_space(method_arguments, value="tanh"),
        type="promotion",
    ),
    Methods.MOBSTER_TANH: lambda method_arguments: MOBSTER(
        _modify_config_space(method_arguments, value="tanh"),
        type="promotion",
    ),
    # Fix activations to "relu"
    Methods.ASHA_RELU: lambda method_arguments: ASHA(
        _modify_config_space(method_arguments, value="relu"),
        type="promotion",
    ),
    Methods.MOBSTER_RELU: lambda method_arguments: MOBSTER(
        _modify_config_space(method_arguments, value="relu"),
        type="promotion",
    ),
}
