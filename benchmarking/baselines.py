from dataclasses import dataclass
from typing import Optional, List

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)


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
    BORE = "BORE"
    RS = "RS"
    TPE = "TPE"
    REA = "REA"
    BOTorch = "BOTorch"
    CQR = "CQR"
    BOHB = "BOHB"

    # multifidelity
    ASHA = "ASHA"
    ASHABORE = "ASHABORE"
    ASHACQR = "ASHACQR"


methods = {
    Methods.RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="random_search",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BORE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.TPE: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.CQR: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="cqr",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BOTorch: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="botorch",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.REA: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher="regularized_evolution",
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.BOHB: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="kde",
        time_attr=method_arguments.resource_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHA: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="random_search",
        time_attr=method_arguments.resource_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHACQR: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="cqr",
        time_attr=method_arguments.resource_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
    Methods.ASHABORE: lambda method_arguments: AsynchronousSuccessiveHalving(
        config_space=method_arguments.config_space,
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
        searcher="bore",
        time_attr=method_arguments.resource_attr,
        searcher_kwargs={"points_to_evaluate": method_arguments.points_to_evaluate},
    ),
}


if __name__ == "__main__":
    # Run a loop that initializes all schedulers on all benchmark to see if they all work
    from benchmarking.benchmarks import (
        benchmark_definitions,
    )

    print(f"Checking initialization of {list(methods.keys())[::-1]}")

    benchmarks = [
        "fcnet-protein",
        "nas201-cifar10",
        "lcbench-Fashion-MNIST",
        "tabrepo-RandomForest-2dplanes",
        "hpob_5636_3492",
    ]
    for benchmark_name in benchmarks:
        benchmark = benchmark_definitions[benchmark_name]
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )
        points_to_evaluate = [
            {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in backend.blackbox.configuration_space.items()
            }
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
