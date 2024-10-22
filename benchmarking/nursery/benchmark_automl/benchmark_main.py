from typing import Dict, Any
import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.experiments.launchers.hpo_main_simulator import (
    get_transfer_learning_evaluations,
)
from benchmarking.nursery.benchmark_automl.baselines import MethodArguments

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


def parse_args(methods: Dict[str, Any], benchmark_definitions: Dict[str, Any]):
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        required=False,
        default=3,
        help="number of seeds to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        default=1,
        help="if 1 run all the seeds [0, ``num_seeds``-1], otherwise run seed ``num_seeds`` only",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        required=False,
        default=0,
        help="first seed to run (if ``run_all_seed`` == 1)",
    )
    parser.add_argument(
        "--method", type=str, required=False, help="a method to run from baselines.py"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help="a benchmark to run from benchmark_definitions.py",
    )
    args, _ = parser.parse_known_args()
    args.run_all_seeds = bool(args.run_all_seeds)
    if args.run_all_seeds:
        seeds = list(range(args.start_seed, args.num_seeds))
    else:
        seeds = [args.num_seeds]
    method_names = [args.method] if args.method is not None else list(methods.keys())
    benchmark_names = (
        [args.benchmark]
        if args.benchmark is not None
        else list(benchmark_definitions.keys())
    )
    return args, method_names, benchmark_names, seeds


def main(methods: Dict[str, Any], benchmark_definitions: Dict[str, Any]):
    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions
    )
    experiment_tag = args.experiment_tag

    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))
    print(combinations)
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )

        max_resource_attr = benchmark.max_resource_attr
        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
            surrogate_kwargs=benchmark.surrogate_kwargs,
            add_surrogate_kwargs=benchmark.add_surrogate_kwargs,
        )

        resource_attr = next(iter(backend.blackbox.fidelity_space.keys()))
        max_resource_level = int(max(backend.blackbox.fidelity_values))
        if max_resource_attr is not None:
            config_space = dict(
                backend.blackbox.configuration_space,
                **{max_resource_attr: max_resource_level},
            )
            method_kwargs = {"max_resource_attr": max_resource_attr}
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
                transfer_learning_evaluations=get_transfer_learning_evaluations(
                    blackbox_name=benchmark.blackbox_name,
                    test_task=benchmark.dataset_name,
                    datasets=benchmark.datasets,
                ),
                use_surrogates="lcbench" in benchmark_name,
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
            "benchmark": benchmark_name,
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


if __name__ == "__main__":
    from benchmarking.nursery.benchmark_automl.baselines import methods
    from benchmarking.nursery.benchmark_automl.benchmark_definitions import (
        benchmark_definitions,
    )

    main(methods, benchmark_definitions)
