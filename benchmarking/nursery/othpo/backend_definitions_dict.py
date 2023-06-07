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
# File to store specifications for backends
# metric, mode, active_task_str, uses_fidelity

BACKEND_DEFS = {
    "SimOpt": (
        "profit",
        "max",
        "time_idx",
        False,
    ),
    "YAHPO": (
        "auc",
        "max",
        "trainsize",
        True,
    ),
    "XGBoost": (
        "metric_error",
        "min",
        "data_size",
        False,
    ),
}
