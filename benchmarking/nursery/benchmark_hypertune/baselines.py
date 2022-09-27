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
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
)


class Methods:
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    HYPERTUNE_INDEP = "HYPERTUNE-INDEP"
    HYPERTUNE_JOINT = "HYPERTUNE-JOINT"
    SYNCHB = "SYNCHB"
    BOHB = "BOHB"


def conv_numeric_only(margs) -> dict:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
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
    Methods.MOBSTER_JOINT: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        type="promotion",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER_INDEP: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        type="promotion",
        search_options=dict(
            search_options(method_arguments),
            model="gp_independent",
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="hypertune",
        type="promotion",
        search_options=dict(
            search_options(method_arguments),
            model="gp_independent",
            hypertune_distribution_num_samples=method_arguments.num_samples,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperbandScheduler(
        config_space=conv_numeric_only(method_arguments),
        searcher="hypertune",
        type="promotion",
        search_options=dict(
            search_options(method_arguments),
            model="gp_multitask",
            hypertune_distribution_num_samples=method_arguments.num_samples,
        ),
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
}
