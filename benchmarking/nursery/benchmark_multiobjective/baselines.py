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
from typing import Dict, Any, Optional, List, Union

from benchmarking.commons.baselines import MethodArguments
from benchmarking.commons.default_baselines import (
    RandomSearch,
    BayesianOptimization, _baseline_kwargs,
)
from syne_tune.config_space import Float, Integer, Categorical
from syne_tune.optimizer.baselines import MOREA, _assert_searcher_must_be
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import \
    LinearScalarizedScheduler
from syne_tune.optimizer.schedulers.searchers import StochasticSearcher, StochasticAndFilterDuplicatesSearcher

NUM_TO_SAMPLE = 1000


def MOREABench(method_arguments: MethodArguments, **kwargs):
    kwargs = _baseline_kwargs(method_arguments, kwargs)
    for name, space in kwargs["config_space"].items():
        if isinstance(space, Float) or isinstance(space, Integer):
            space = Categorical(categories=space.sample(size=NUM_TO_SAMPLE))
        kwargs["config_space"][name] = space
    return MOREA(**kwargs)


def LSOBOBench(method_arguments: MethodArguments, **kwargs):
    return LinearScalarizedScheduler(**_baseline_kwargs(method_arguments, kwargs))


class Methods:
    RS = "RS"
    MOREA = "MOREA"
    LSOBO = "LSOBO"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.MOREA: lambda method_arguments: MOREABench(method_arguments),
    Methods.LSOBO: lambda method_arguments: LSOBOBench(method_arguments),
}
