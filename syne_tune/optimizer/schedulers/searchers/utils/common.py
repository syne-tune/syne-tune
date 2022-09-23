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
from typing import Union, Dict, Callable


Hyperparameter = Union[str, int, float]

Configuration = Dict[str, Hyperparameter]

# Type of `filter_observed_data`, which is (optionally) used to filter the
# observed data in `TuningJobState.trials_evaluations` when determining
# the best config (incumbent) or the exclusion list. One use case is
# warm-starting, where the observed data can come from a number of tasks, only
# one of which is active.

ConfigurationFilter = Callable[[Configuration], bool]
