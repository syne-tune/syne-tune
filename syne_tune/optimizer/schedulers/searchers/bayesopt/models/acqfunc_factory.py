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
from functools import partial

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    AcquisitionFunctionConstructor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
    LCBAcquisitionFunction,
)


SUPPORTED_ACQUISITION_FUNCTIONS = (
    "ei",
    "lcb",
)


def acquisition_function_factory(name: str, **kwargs) -> AcquisitionFunctionConstructor:
    assert (
        name in SUPPORTED_ACQUISITION_FUNCTIONS
    ), f"name = {name} not supported. Choose from:\n{SUPPORTED_ACQUISITION_FUNCTIONS}"
    if name == "ei":
        return EIAcquisitionFunction
    else:  # name == "lcb"
        kappa = kwargs.get("kappa", 1.0)
        return partial(LCBAcquisitionFunction, kappa=kappa)
