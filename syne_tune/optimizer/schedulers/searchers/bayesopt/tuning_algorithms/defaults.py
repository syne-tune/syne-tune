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
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    LBFGSOptimizeAcquisition,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
)


DEFAULT_ACQUISITION_FUNCTION = EIAcquisitionFunction
DEFAULT_LOCAL_OPTIMIZER_CLASS = LBFGSOptimizeAcquisition
DEFAULT_NUM_INITIAL_CANDIDATES = 250
DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS = 3
