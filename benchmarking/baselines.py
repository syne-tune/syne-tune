from dataclasses import dataclass
from typing import Optional, List

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer import baselines
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving

# from syne_tune.optimizer.schedulers.searchers.conformal.surrogate_searcher import (
#    SurrogateSearcher,
# )
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
import syne_tune.optimizer.baselines as legacy_baselines
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)


# import syne_tune.optimizer.baselines as baselines


@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    points_to_evaluate: List[dict]
    max_t: Optional[int] = None
    max_resource_attr: Optional[str] = None
    use_surrogates: bool = False
    num_brackets: Optional[int] = 1
    verbose: Optional[bool] = False


class Methods:
    # single fidelity
    LegacyBORE = "LegacyBORE"
    LegacyRS = "LegacyRS"
    LegacyTPE = "LegacyTPE"
    LegacyREA = "LegacyREA"
    LegacyBOTorch = "LegacyBOTorch"
    LegacyCQR = "LegacyCQR"
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    GP = "GP"
    CQR = "CQR"
    #   HEBO = "HEBO"

    # multifidelity
    ASHA = "ASHA"
    ASHABORE = "ASHABORE"
    ASHACQR = "ASHACQR"
    BOHB = "BOHB"
    LegacyASHA = "LegacyASHA"
    LegacyASHABORE = "LegacyASHABORE"
    LegacyASHACQR = "LegacyASHACQR"
    LegacyBOHB = "LegacyBOHB"
    #    BOREHB = "BOREHB"
    MOBSTER = "MOB"
    HYPERTUNE = "HT"

    #    QR = "QR"
    #    CQR = "CQR"

    @staticmethod
    def multifidelity(method):
        return f"{method}-last"


def _max_resource_attr_or_max_t(
    args: MethodArguments, max_t_name: str = "max_t"
) -> dict:
    if args.max_resource_attr is not None:
        return {"max_resource_attr": args.max_resource_attr}
    else:
        assert args.max_t is not None
        return {max_t_name: args.max_t}


methods = {
    # Legacy Methods
    Methods.LegacyRS: lambda method_arguments: legacy_baselines.RandomSearch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.LegacyREA: lambda method_arguments: legacy_baselines.REA(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.LegacyTPE: lambda method_arguments: legacy_baselines.KDE(
        config_space=method_arguments.config_space,
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.LegacyBORE: lambda method_arguments: legacy_baselines.BORE(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.LegacyCQR: lambda method_arguments: legacy_baselines.CQR(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    Methods.LegacyBOTorch: lambda method_arguments: legacy_baselines.BoTorch(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        mode=method_arguments.mode,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
    ),
    # Methods.GP: lambda method_arguments: FIFOScheduler(
    #     method_arguments.config_space,
    #     searcher="bayesopt",
    #     search_options={"debug_log": False},
    #     metric=method_arguments.metric,
    #     mode=method_arguments.mode,
    #     random_seed=method_arguments.random_seed,
    #     points_to_evaluate=method_arguments.points_to_evaluate,
    # ),
    # Methods.LegacyASHA: lambda method_arguments: HyperbandScheduler(
    #     config_space=method_arguments.config_space,
    #     searcher="random",
    #     search_options={"debug_log": False},
    #     mode=method_arguments.mode,
    #     metric=method_arguments.metric,
    #     resource_attr=method_arguments.resource_attr,
    #     random_seed=method_arguments.random_seed,
    #     points_to_evaluate=method_arguments.points_to_evaluate,
    #     **_max_resource_attr_or_max_t(method_arguments),
    # ),
    Methods.LegacyASHABORE: lambda method_arguments: legacy_baselines.ASHABORE(
        config_space=method_arguments.config_space,
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.LegacyASHACQR: lambda method_arguments: legacy_baselines.ASHACQR(
        config_space=method_arguments.config_space,
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    Methods.LegacyBOHB: lambda method_arguments: legacy_baselines.BOHB(
        config_space=method_arguments.config_space,
        search_options={"debug_log": False, "min_bandwidth": 0.1},
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        points_to_evaluate=method_arguments.points_to_evaluate,
        **_max_resource_attr_or_max_t(method_arguments),
    ),
    # New methods
    Methods.BORE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.TPE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.CQR: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="cqr",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.ASHACQR: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="cqr",
        time_attr=method_arguments.resource_attr,
    ),
    Methods.ASHABORE: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="bore",
        time_attr=method_arguments.resource_attr,
    ),
    Methods.BOHB: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="kde",
        time_attr=method_arguments.resource_attr,
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.benchmarks import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")
    # sys.exit(0)
    benchmarks = ["fcnet-protein", "nas201-cifar10", "lcbench-Fashion-MNIST"]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {k: v.sample() for k, v in backend.blackbox.configuration_space.items()}
            for _ in range(4)
        ]
        print(f"Checking initialization of {list(methods.keys())[::-1]}")
        for method_name, method_fun in list(methods.items())[::-1]:
            print(f"checking initialization of: {method_name}, {benchmark_name}")
            # if method_name != Methods.QHB_XGB:
            #     continue

            scheduler = method_fun(
                MethodArguments(
                    config_space=backend.blackbox.configuration_space,
                    metric=benchmark.metric,
                    mode=benchmark.mode,
                    random_seed=0,
                    max_t=max(backend.blackbox.fidelity_values),
                    resource_attr=next(iter(backend.blackbox.fidelity_space.keys())),
                    use_surrogates=benchmark_name == "lcbench-Fashion-MNIST",
                    points_to_evaluate=points_to_evaluate,
                )
            )
            if isinstance(scheduler, TrialScheduler):
                print(scheduler.suggest())
                print(scheduler.suggest())
            else:
                print(scheduler.suggest(0))
                print(scheduler.suggest(1))
