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
from dataclasses import dataclass
from typing import Dict, Optional

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune.config_space import (
    Categorical,
    ordinal,
)


@dataclass
class MethodArguments:
    config_space: Dict
    metric: str
    mode: str
    random_seed: int
    max_resource_attr: str
    resource_attr: str
    num_brackets: Optional[int] = None
    verbose: Optional[bool] = False


class Methods:
    ASHA = "ASHA"
    SYNCHB = "SYNCHB"
    DEHB = "DEHB"
    BOHB = "BOHB"
    ASHA_ORD = "ASHA-ORD"
    SYNCHB_ORD = "SYNCHB-ORD"
    DEHB_ORD = "DEHB-ORD"
    BOHB_ORD = "BOHB-ORD"


def _convert_categorical_to_ordinal(args: MethodArguments) -> dict:
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in args.config_space.items()
    }


def _search_options(args: MethodArguments) -> dict:
    if args.verbose:
        return dict()
    else:
        return {"debug_log": False}


methods = {
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        type="promotion",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNCHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random_encoded",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.ASHA_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random",
        type="promotion",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNCHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DEHB_ORD: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="random_encoded",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="kde",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.nursery.benchmark_dehb.benchmark_definitions import (
        benchmark_definitions,
    )

    benchmarks = ["fcnet-protein", "nas201-cifar10", "lcbench-Fashion-MNIST"]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            time_this_resource_attr=benchmark.time_this_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        for method_name, method_fun in methods.items():
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_resource_attr=benchmark.max_resource_attr,
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                )
            )
            scheduler.suggest(0)
            scheduler.suggest(1)
