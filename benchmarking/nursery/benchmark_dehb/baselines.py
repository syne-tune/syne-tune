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
    convert_categorical_to_ordinal,
    convert_categorical_to_ordinal_numeric,
    search_options,
)
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)


class Methods:
    ASHA = "ASHA"
    SYNCHB = "SYNCHB"
    DEHB = "DEHB"
    BOHB = "BOHB"
    ASHA_ORD = "ASHA-ORD"
    SYNCHB_ORD = "SYNCHB-ORD"
    DEHB_ORD = "DEHB-ORD"
    BOHB_ORD = "BOHB-ORD"
    ASHA_STOP = "ASHA-STOP"
    SYNCMOBSTER = "SYNCMOBSTER"


def conv_numeric_only(margs) -> dict:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
    )


def conv_numeric_then_rest(margs) -> dict:
    return convert_categorical_to_ordinal(
        convert_categorical_to_ordinal_numeric(
            margs.config_space, kind=margs.fcnet_ordinal
        )
    )


methods = {
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        type="promotion",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.SYNCHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random_encoded",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.ASHA_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random",
        type="promotion",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNCHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB_ORD: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random_encoded",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.ASHA_STOP: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        type="stopping",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.SYNCMOBSTER: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
}
