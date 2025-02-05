from dataclasses import dataclass
from typing import Dict, Optional, Any

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.baselines import ZeroShotTransfer
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.legacy_median_stopping_rule import (
    LegacyMedianStoppingRule,
)
from syne_tune.optimizer.schedulers.transfer_learning import RUSHScheduler
from syne_tune.optimizer.schedulers.transfer_learning.legacy_bounding_box import (
    LegacyBoundingBox,
)
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    QuantileBasedSurrogateSearcher,
)


@dataclass
class MethodArguments:
    config_space: Dict[str, Any]
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    max_t: Optional[int] = None
    max_resource_attr: Optional[str] = None
    transfer_learning_evaluations: Optional[Dict] = None
    use_surrogates: bool = False


class Methods:
    RS = "RS"
    ASHA = "ASHA"
    MSR = "RS-MSR"
    ASHA_BB = "ASHA-BB"
    ASHA_CTS = "ASHA-CTS"
    GP = "GP"
    BOHB = "BOHB"
    REA = "REA"
    MOBSTER = "MOB"
    TPE = "TPE"
    BORE = "BORE"
    ZERO_SHOT = "ZS"
    RUSH = "RUSH"


def _max_resource_attr_or_max_t(
    args: MethodArguments, max_t_name: str = "max_t"
) -> Dict[str, Any]:
    if args.max_resource_attr is not None:
        return {"max_resource_attr": args.max_resource_attr}
    else:
        assert args.max_t is not None
        return {max_t_name: args.max_t}


def search_options(args: MethodArguments) -> Dict[str, Any]:
    return {"debug_log": False}


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
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.MSR: lambda method_arguments: LegacyMedianStoppingRule(
        scheduler=FIFOScheduler(
            config_space=method_arguments.config_space,
            searcher="random",
            metric=method_arguments.metric,
            mode=method_arguments.mode,
            random_seed=method_arguments.random_seed,
        ),
        resource_attr=method_arguments.resource_attr,
        running_average=False,
    ),
    Methods.ASHA_BB: lambda method_arguments: LegacyBoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: HyperbandScheduler(
            new_config_space,
            searcher="random",
            metric=metric,
            mode=mode,
            search_options=search_options(method_arguments),
            resource_attr=method_arguments.resource_attr,
            random_seed=method_arguments.random_seed,
            **_max_resource_attr_or_max_t(method_arguments),
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        config_space=method_arguments.config_space,
        transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
        num_hyperparameters_per_task=10,
    ),
    Methods.ASHA_CTS: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher=QuantileBasedSurrogateSearcher(
            mode=method_arguments.mode,
            config_space=method_arguments.config_space,
            metric=method_arguments.metric,
            transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
            random_seed=method_arguments.random_seed,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.GP: lambda method_arguments: FIFOScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.REA: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher=RegularizedEvolution(
            config_space=method_arguments.config_space,
            metric=method_arguments.metric,
            mode=method_arguments.mode,
            random_seed=method_arguments.random_seed,
            population_size=10,
            sample_size=5,
        ),
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.BOHB: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.TPE: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.BORE: lambda method_arguments: FIFOScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        search_options={"classifier": "mlp"},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
    ),
    Methods.MOBSTER: lambda method_arguments: HyperbandScheduler(
        method_arguments.config_space,
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.ZERO_SHOT: lambda method_arguments: ZeroShotTransfer(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
        use_surrogates=method_arguments.use_surrogates,
        random_seed=method_arguments.random_seed,
    ),
    Methods.RUSH: lambda method_arguments: RUSHScheduler(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        transfer_learning_evaluations=method_arguments.transfer_learning_evaluations,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **_max_resource_attr_or_max_t(method_arguments),
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
