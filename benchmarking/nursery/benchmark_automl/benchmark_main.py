from typing import Dict, Optional, List

import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

from syne_tune.blackbox_repository import load
from syne_tune.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend
from benchmarking.nursery.benchmark_automl.baselines import MethodArguments, methods
from benchmarking.nursery.benchmark_automl.benchmark_definitions import benchmark_definitions

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from coolname import generate_slug


def get_transfer_learning_evaluations(
        blackbox_name: str,
        test_task: str,
        datasets: Optional[List[str]],
        n_evals: Optional[int] = None
) -> Dict:
    """
    :param blackbox_name:
    :param test_task: task where the performance would be tested, it is excluded from transfer-learning evaluations
    :param datasets: subset of datasets to consider, only evaluations from those datasets are provided to
    transfer-learning methods. If none, all datasets are used.
    :param n_evals: maximum number of evaluations to be returned
    :return:
    """
    task_to_evaluations = load(blackbox_name)

    # todo retrieve right metric
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            configuration_space=bb.configuration_space,
            hyperparameters=bb.hyperparameters,
            objectives_evaluations=bb.objectives_evaluations[..., metric_index:metric_index + 1],
            objectives_names=[bb.objectives_names[metric_index]],
        )
        for task, bb in task_to_evaluations.items()
        if task != test_task and (datasets is None or task in datasets)
    }

    if n_evals is not None:
        # subsample n_evals / n_tasks of observations on each tasks
        def subsample(transfer_evaluations: TransferLearningTaskEvaluations, n: int) -> TransferLearningTaskEvaluations:
            random_indices = np.random.permutation(len(transfer_evaluations.hyperparameters))[:n]
            return TransferLearningTaskEvaluations(
                configuration_space=transfer_evaluations.configuration_space,
                hyperparameters=transfer_evaluations.hyperparameters.loc[random_indices].reset_index(drop=True),
                objectives_evaluations=transfer_evaluations.objectives_evaluations[random_indices],
                objectives_names=transfer_evaluations.objectives_names,
            )
        n = n_evals // len(transfer_learning_evaluations)
        transfer_learning_evaluations = {
            task: subsample(transfer_evaluations, n)
            for task, transfer_evaluations in transfer_learning_evaluations.items()
        }

    return transfer_learning_evaluations

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default=generate_slug(2))
    parser.add_argument("--num_seeds", type=int, required=False, default=3, help='number of seeds to run')
    parser.add_argument("--run_all_seed", type=int, default=1, help='if 1 run only `seed=num_seeds`, otherwise runs all the seeds [0, `num_seeds`-1]')
    parser.add_argument("--method", type=str, required=False, help='a method to run from baselines.py')
    parser.add_argument("--benchmark", type=str, required=False, help='a benchmark to run from benchmark_definitions.py')
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag

    if args.run_all_seed == 1:
        seeds = list(range(args.num_seeds))
    else:
        seeds = [args.num_seeds]

    method_names = [args.method] if args.method is not None else list(methods.keys())
    benchmark_names = [args.benchmark] if args.benchmark is not None else list(benchmark_definitions.keys())

    # logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(logging.WARNING)

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(combinations)
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}")

        backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            time_this_resource_attr=benchmark.time_this_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
        )

        # todo move into benchmark definition
        max_t = max(backend.blackbox.fidelity_values)
        resource_attr = next(iter(backend.blackbox.fidelity_space.keys()))

        scheduler = methods[method](MethodArguments(
            config_space=backend.blackbox.configuration_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=seed,
            max_t=max_t,
            resource_attr=resource_attr,
            transfer_learning_evaluations=get_transfer_learning_evaluations(
                blackbox_name=benchmark.blackbox_name,
                test_task=benchmark.dataset_name,
                datasets=benchmark.datasets,
            ),
            use_surrogates='lcbench' in benchmark_name,
        ))

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_evaluations=benchmark.max_num_evaluations,
        )
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=600,
            tuner_name=f"{experiment_tag}-{method}-{seed}-{benchmark_name}".replace("_", "-"),
            metadata={
                "seed": seed,
                "algorithm": method,
                "tag": experiment_tag,
                "benchmark": benchmark_name
            }
        )
        tuner.run()
