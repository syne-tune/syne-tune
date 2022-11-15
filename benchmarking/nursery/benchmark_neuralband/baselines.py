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
from benchmarking.commons.baselines import (
    search_options,
    convert_categorical_to_ordinal_numeric,
)
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.neuralbands.neuralband import NeuralbandScheduler
from syne_tune.optimizer.schedulers.neuralbands.neuralband_supplement import (
    NeuralbandUCBScheduler,
    NeuralbandTSScheduler,
    NeuralbandEGreedyScheduler,
)


class Methods:
    RS = "RS"
    ASHA = "ASHA"
    HP = "HP"
    GP = "GP"
    BOHB = "BOHB"
    MOBSTER = "MOB"
    TPE = "TPE"
    NeuralBandSH = "NeuralBandSH"
    NeuralBandHB = "NeuralBandHB"
    NeuralBand_UCB = "NeuralBandUCB"
    NeuralBand_TS = "NeuralBandTS"
    NeuralBandEpsilon = "NeuralBandEpsilon"


def conv_numeric_only(margs) -> dict:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
    )


methods = {
    Methods.RS: lambda method_arguments: FIFOScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.HP: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        brackets=method_arguments.num_brackets,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.BOHB: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        brackets=method_arguments.num_brackets,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.TPE: lambda method_arguments: FIFOScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.GP: lambda method_arguments: FIFOScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.MOBSTER: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        brackets=method_arguments.num_brackets,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandSH: lambda method_arguments: NeuralbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        gamma=0.05,
        nu=0.02,
        max_while_loop=50,
        step_size=5,
        brackets=method_arguments.num_brackets,
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandHB: lambda method_arguments: NeuralbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        gamma=0.04,
        nu=0.02,
        max_while_loop=50,
        step_size=5,
        brackets=method_arguments.num_brackets,
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBand_UCB: lambda method_arguments: NeuralbandUCBScheduler(
        config_space=conv_numeric_only(method_arguments),
        lamdba=0.1,
        nu=0.001,
        max_while_loop=50,
        step_size=5,
        searcher="random",
        brackets=method_arguments.num_brackets,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBand_TS: lambda method_arguments: NeuralbandTSScheduler(
        config_space=conv_numeric_only(method_arguments),
        lamdba=0.1,
        nu=0.001,
        max_while_loop=50,
        step_size=5,
        searcher="random",
        brackets=method_arguments.num_brackets,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandEpsilon: lambda method_arguments: NeuralbandEGreedyScheduler(
        config_space=conv_numeric_only(method_arguments),
        epsilon=0.1,
        max_while_loop=1000,
        step_size=5,
        searcher="random",
        brackets=method_arguments.num_brackets,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
}
