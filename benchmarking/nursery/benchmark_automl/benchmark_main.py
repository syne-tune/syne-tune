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
from typing import Optional, List

import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from benchmarking.nursery.benchmark_automl.baselines import MethodArguments

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


def get_transfer_learning_evaluations(
    blackbox_name: str,
    test_task: str,
    datasets: Optional[List[str]],
    n_evals: Optional[int] = None,
) -> dict:
    """
    :param blackbox_name:
    :param test_task: task where the performance would be tested, it is excluded from transfer-learning evaluations
    :param datasets: subset of datasets to consider, only evaluations from those datasets are provided to
    transfer-learning methods. If none, all datasets are used.
    :param n_evals: maximum number of evaluations to be returned
    :return:
    """
    task_to_evaluations = load_blackbox(blackbox_name)

    # todo retrieve right metric
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            configuration_space=bb.configuration_space,
            hyperparameters=bb.hyperparameters,
            objectives_evaluations=bb.objectives_evaluations[
                ..., metric_index : metric_index + 1
            ],
            objectives_names=[bb.objectives_names[metric_index]],
        )
        for task, bb in task_to_evaluations.items()
        if task != test_task and (datasets is None or task in datasets)
    }

    if n_evals is not None:
        # subsample n_evals / n_tasks of observations on each tasks
        def subsample(
            transfer_evaluations: TransferLearningTaskEvaluations, n: int
        ) -> TransferLearningTaskEvaluations:
            random_indices = np.random.permutation(
                len(transfer_evaluations.hyperparameters)
            )[:n]
            return TransferLearningTaskEvaluations(
                configuration_space=transfer_evaluations.configuration_space,
                hyperparameters=transfer_evaluations.hyperparameters.loc[
                    random_indices
                ].reset_index(drop=True),
                objectives_evaluations=transfer_evaluations.objectives_evaluations[
                    random_indices
                ],
                objectives_names=transfer_evaluations.objectives_names,
            )

        n = n_evals // len(transfer_learning_evaluations)
        transfer_learning_evaluations = {
            task: subsample(transfer_evaluations, n)
            for task, transfer_evaluations in transfer_learning_evaluations.items()
        }

    return transfer_learning_evaluations


def parse_args(methods: dict, benchmark_definitions: dict):
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
        "--run_all_seed",
        type=int,
        default=1,
        help="if 1 run all the seeds [0, `num_seeds`-1], otherwise run seed `num_seeds` only",
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
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="verbose log output?",
    )
    parser.add_argument(
        "--num_brackets",
        type=int,
        required=False,
        help="number of brackets",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="first seed to run (if run_all_seed)",
    )
    args, _ = parser.parse_known_args()
    args.verbose = bool(args.verbose)
    if args.run_all_seed == 1:
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


def main(methods: dict, benchmark_definitions: dict):
    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions
    )
    experiment_tag = args.experiment_tag

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
        logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
        logging.getLogger(
            "syne_tune.backend.simulator_backend.simulator_backend"
        ).setLevel(logging.WARNING)

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
                verbose=args.verbose,
                transfer_learning_evaluations=get_transfer_learning_evaluations(
                    blackbox_name=benchmark.blackbox_name,
                    test_task=benchmark.dataset_name,
                    datasets=benchmark.datasets,
                ),
                num_brackets=args.num_brackets,
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
        if args.num_brackets is not None:
            metadata["num_brackets"] = args.num_brackets
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
