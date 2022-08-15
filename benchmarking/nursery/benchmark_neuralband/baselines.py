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
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.neuralbands.neuralband import NeuralbandScheduler
from syne_tune.optimizer.schedulers.neuralbands.neuralband_supplement import (
    NeuralbandUCBScheduler,
    NeuralbandTSScheduler,
    NeuralbandEGreedyScheduler,
)


@dataclass
class MethodArguments:
    config_space: Dict
    metric: str
    mode: str
    random_seed: int
    max_t: int
    resource_attr: str
    transfer_learning_evaluations: Optional[Dict] = None
    use_surrogates: bool = False


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


methods = {
    Methods.RS: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.HP: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        brackets=3,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.BOHB: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        mode=method_arguments.mode,
        brackets=3,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.TPE: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.GP: lambda method_arguments: FIFOScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options={"debug_log": False},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.MOBSTER: lambda method_arguments: HyperbandScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandSH: lambda method_arguments: NeuralbandScheduler(
        gamma=0.05,
        nu=0.02,
        max_while_loop=50,
        step_size=5,
        brackets=1,
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandHB: lambda method_arguments: NeuralbandScheduler(
        gamma=0.04,
        nu=0.02,
        max_while_loop=50,
        step_size=5,
        brackets=3,
        config_space=method_arguments.config_space,
        searcher="random",
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBand_UCB: lambda method_arguments: NeuralbandUCBScheduler(
        lamdba=0.1,
        nu=0.001,
        max_while_loop=50,
        step_size=5,
        config_space=method_arguments.config_space,
        searcher="random",
        brackets=3,
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBand_TS: lambda method_arguments: NeuralbandTSScheduler(
        lamdba=0.1,
        nu=0.001,
        max_while_loop=50,
        step_size=5,
        config_space=method_arguments.config_space,
        searcher="random",
        brackets=3,
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.NeuralBandEpsilon: lambda method_arguments: NeuralbandEGreedyScheduler(
        epsilon=0.1,
        max_while_loop=1000,
        step_size=5,
        config_space=method_arguments.config_space,
        searcher="random",
        brackets=3,
        search_options={"debug_log": False},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_t=method_arguments.max_t,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.nursery.benchmark_automl.benchmark_main import (
        get_transfer_learning_evaluations,
    )
    from benchmarking.nursery.benchmark_automl.benchmark_definitions import (
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
                    max_t=max(backend.blackbox.fidelity_values),
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    transfer_learning_evaluations=get_transfer_learning_evaluations(
                        blackbox_name=benchmark.blackbox_name,
                        test_task=benchmark.dataset_name,
                        datasets=benchmark.datasets,
                    ),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                )
            )
            scheduler.suggest(0)
            scheduler.suggest(1)
