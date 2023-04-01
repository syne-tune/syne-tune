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
from benchmarking.commons.default_baselines import (
    ASHA,
)


class Methods:
    ASHA = "ASHA"
    ASHA_RCP_10 = "ASHA-RCP-10"
    ASHA_RCP_5 = "ASHA-RCP-5"
    ASHA_RCP_30 = "ASHA-RCP-30"
    ASHA_RCP_BASE_RANDOM = "ASHA-RCP-BASE-RANDOM"
    ASHA_RCP_BASE_BYLEVEL = "ASHA-RCP-BASE-BYLEVEL"


methods = {
    # Methods.ASHA: lambda method_arguments: ASHA(method_arguments, type="promotion"),
    Methods.ASHA_RCP_10: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            approx_steps=10,
        ),
    ),
    Methods.ASHA_RCP_5: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            approx_steps=5,
        ),
    ),
    Methods.ASHA_RCP_30: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            approx_steps=30,
        ),
    ),
    Methods.ASHA_RCP_BASE_RANDOM: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            baseline="random",
        ),
    ),
    Methods.ASHA_RCP_BASE_BYLEVEL: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            baseline="by_level",
        ),
    ),
}
