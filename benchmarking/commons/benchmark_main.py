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
from typing import Optional, List, Callable, Union, Dict

import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from benchmarking.commons.baselines import MethodArguments
from benchmarking.commons.benchmark_definitions.common import BenchmarkDefinition
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


BenchmarkDefinitions = Union[
    Dict[str, BenchmarkDefinition], Dict[str, Dict[str, BenchmarkDefinition]]
]


def is_dict_of_dict(benchmark_definitions: BenchmarkDefinitions) -> bool:
    assert isinstance(benchmark_definitions, dict) and len(benchmark_definitions) > 0
    val = next(iter(benchmark_definitions.values()))
    return isinstance(val, dict)


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


def parse_args(
    methods: dict,
    benchmark_definitions: BenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
):
    try:
        default_experiment_tag = generate_slug(2)
    except Exception:
        default_experiment_tag = "syne_tune_experiment"
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=default_experiment_tag,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=30,
        help="number of seeds to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        default=1,
        help="if 1 run all the seeds [0, `num_seeds`-1], otherwise run seed `num_seeds` only",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="first seed to run (if `run_all_seed` == 1)",
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
        "--support_checkpointing",
        type=int,
        default=1,
        help="if 0, trials are started from scratch when resumed",
    )
    parser.add_argument(
        "--save_tuner",
        type=int,
        default=0,
        help="Serialize Tuner object at the end of tuning?",
    )
    # Internal parameter, to support nested dict for `benchmark_definitions`
    nested_dict = is_dict_of_dict(benchmark_definitions)
    if nested_dict:
        parser.add_argument(
            "--benchmark_key",
            type=str,
        )
    if extra_args is not None:
        for kwargs in extra_args:
            name = kwargs.pop("name")
            parser.add_argument(name, **kwargs)
    args, _ = parser.parse_known_args()
    args.verbose = bool(args.verbose)
    args.support_checkpointing = bool(args.support_checkpointing)
    args.save_tuner = bool(args.save_tuner)
    args.run_all_seeds = bool(args.run_all_seeds)
    if args.run_all_seeds:
        seeds = list(range(args.start_seed, args.num_seeds))
    else:
        seeds = [args.num_seeds]
    method_names = [args.method] if args.method is not None else list(methods.keys())
    if args.benchmark is not None:
        benchmark_names = [args.benchmark]
    else:
        if nested_dict:
            # If `parse_args` is called from `launch_remote`, `benchmark_key` is
            # not set. In this case, `benchmark_names` is not needed
            k = args.benchmark_key
            if k is None:
                bm_dict = dict()
            else:
                bm_dict = benchmark_definitions.get(k)
                assert (
                    bm_dict is not None
                ), f"{k} (value of --benchmark_key) is not among keys of benchmark_definition [{list(benchmark_definitions.keys())}]"
        else:
            bm_dict = benchmark_definitions
        benchmark_names = list(bm_dict.keys())

    return args, method_names, benchmark_names, seeds


def main(
    methods: dict,
    benchmark_definitions: BenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[Callable] = None,
    use_transfer_learning: bool = False,
):
    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions, extra_args
    )
    experiment_tag = args.experiment_tag
    if is_dict_of_dict(benchmark_definitions):
        assert (
            args.benchmark_key is not None
        ), "Use --benchmark_key if benchmark_definitions is a nested dictionary"
        benchmark_definitions = benchmark_definitions[args.benchmark_key]

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
            support_checkpointing=args.support_checkpointing,
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

        if extra_args is not None:
            assert map_extra_args is not None
            extra_args = map_extra_args(args)
            method_kwargs.update(extra_args)
        if use_transfer_learning:
            method_kwargs["transfer_learning_evaluations"] = (
                get_transfer_learning_evaluations(
                    blackbox_name=benchmark.blackbox_name,
                    test_task=benchmark.dataset_name,
                    datasets=benchmark.datasets,
                ),
            )
        scheduler = methods[method](
            MethodArguments(
                config_space=config_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=seed,
                resource_attr=resource_attr,
                verbose=args.verbose,
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
        if extra_args is not None:
            metadata.update(extra_args)
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
            save_tuner=args.save_tuner,
        )
        tuner.run()
