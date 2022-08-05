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
from syne_tune.optimizer.schedulers import HyperbandScheduler, FIFOScheduler

methods = {
    "RS": lambda config_space, metric, mode, random_seed, max_t, resource_attr: FIFOScheduler(
        config_space=config_space,
        searcher="random",
        metric=metric,
        mode=mode,
        random_seed=random_seed,
    ),
    "HB": lambda config_space, metric, mode, random_seed, max_t, resource_attr: HyperbandScheduler(
        config_space=config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=mode,
        metric=metric,
        max_t=max_t,
        resource_attr=resource_attr,
        random_seed=random_seed,
    ),
    "GP": lambda config_space, metric, mode, random_seed, max_t, resource_attr: FIFOScheduler(
        config_space,
        searcher="bayesopt",
        search_options={"debug_log": False},
        metric=metric,
        mode=mode,
        random_seed=random_seed,
    ),
    "MOBSTER": lambda config_space, metric, mode, random_seed, max_t, resource_attr: HyperbandScheduler(
        config_space,
        searcher="bayesopt",
        search_options={"debug_log": False},
        mode=mode,
        metric=metric,
        max_t=max_t,
        resource_attr=resource_attr,
        random_seed=random_seed,
    ),
}
