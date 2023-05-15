from typing import Callable

import numpy as np

from benchmarking.commons.baselines import MethodArguments
from benchmarking.commons.benchmark_definitions import SurrogateBenchmarkDefinition
from benchmarking.commons.hpo_main_simulator import get_transfer_learning_evaluations
from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.experiments import ExperimentResult, load_experiment
from syne_tune.optimizer.scheduler import TrialScheduler

from benchmarking.nursery.benchmark_multiobjective.baselines import (
    methods,
    Methods
)
from benchmarking.nursery.benchmark_multiobjective.benchmark_definitions import (
    benchmark_definitions,
)

def run_experiment(
        method: str,
        seed: int,
        benchmark: SurrogateBenchmarkDefinition,
        experiment_tag: str
) -> ExperimentResult:

    np.random.seed(seed)

    print(
        f"Starting experiment ({method}/{benchmark.blackbox_name}/{benchmark.dataset_name}/{seed}) of {experiment_tag}"
    )

    backend = BlackboxRepositoryBackend(
        elapsed_time_attr=benchmark.elapsed_time_attr,
        max_resource_attr=benchmark.max_resource_attr,
        blackbox_name=benchmark.blackbox_name,
        dataset=benchmark.dataset_name,
        surrogate=benchmark.surrogate,
        surrogate_kwargs=benchmark.surrogate_kwargs,
        add_surrogate_kwargs=benchmark.add_surrogate_kwargs,
    )

    resource_attr = next(iter(backend.blackbox.fidelity_space.keys()))
    max_resource_level = int(max(backend.blackbox.fidelity_values))
    if benchmark.max_resource_attr is not None:
        config_space = dict(
            backend.blackbox.configuration_space,
            **{benchmark.max_resource_attr: max_resource_level},
        )
        method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
    else:
        config_space = backend.blackbox.configuration_space
        method_kwargs = {"max_t": max_resource_level}

    scheduler = methods[method](
        MethodArguments(
            config_space=config_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=seed,
            resource_attr=resource_attr,
            **method_kwargs,
        )
    )

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    metadata = {
        "seed": seed,
        "algorithm": method,
        "tag": experiment_tag,
        "benchmark": f"{benchmark.blackbox_name}/{benchmark.dataset_name}",
    }
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        results_update_interval=600,
        print_update_interval=600,
        tuner_name=experiment_tag,
        metadata=metadata,
    )
    tuner.run()
    return load_experiment(tuner_name=tuner.name)


if __name__ == "__main__":


    method = Methods.MOREA

    res = run_experiment(
        method=method,
        seed=123,
        benchmark=benchmark_definitions['nas201-cifar10'],
        experiment_tag="TEST"
    )

    res.plot(
        metric_to_plot="metric_valid_error",
        figure_path=f"/tmp/moo-benchmark-{method}-metric_valid_error.jpeg",
    )
    res.plot(
        metric_to_plot="metric_params",
        figure_path=f"/tmp/moo-benchmark-{method}-metric_params.jpeg",
    )
    # res.plot_hypervolume( # TODO FIX, this is currently hanging
    #     metrics_to_plot=["metric_valid_error", "metric_params"],
    #     figure_path=f"/tmp/moo-benchmark-{name}-hypervolume.jpeg",
    #     reference_point=np.array([1.01, 1.01])
    # )
