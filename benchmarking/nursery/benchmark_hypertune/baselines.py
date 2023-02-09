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
from benchmarking.commons.baselines import (
    convert_categorical_to_ordinal_numeric,
)
from benchmarking.commons.default_baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncBOHB,
)


class Methods:
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    HYPERTUNE_INDEP = "HYPERTUNE-INDEP"
    HYPERTUNE_JOINT = "HYPERTUNE-JOINT"
    SYNCHB = "SYNCHB"
    BOHB = "BOHB"


def conv_numeric_only(margs) -> Dict[str, Any]:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
    )


methods = {
    Methods.ASHA: lambda method_arguments: ASHA(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
        type="promotion",
    ),
    Methods.MOBSTER_JOINT: lambda method_arguments: MOBSTER(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
        type="promotion",
    ),
    Methods.MOBSTER_INDEP: lambda method_arguments: MOBSTER(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperTune(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperTune(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
        type="promotion",
        search_options=dict(model="gp_multitask"),
    ),
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
    ),
    Methods.BOHB: lambda method_arguments: SyncBOHB(
        method_arguments,
        config_space=conv_numeric_only(method_arguments),
    ),
}
