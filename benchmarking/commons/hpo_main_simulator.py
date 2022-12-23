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
from typing import Optional, List, Union, Dict, Any
import numpy as np
import itertools
from tqdm import tqdm

from benchmarking.commons.baselines import MethodArguments, MethodDefinitions
from benchmarking.commons.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)
from benchmarking.commons.hpo_main_common import (
    parse_args as _parse_args,
    set_logging_level,
    get_metadata,
)
from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


SurrogateBenchmarkDefinitions = Union[
    Dict[str, SurrogateBenchmarkDefinition],
    Dict[str, Dict[str, SurrogateBenchmarkDefinition]],
]


def is_dict_of_dict(benchmark_definitions: SurrogateBenchmarkDefinitions) -> bool:
    assert isinstance(benchmark_definitions, dict) and len(benchmark_definitions) > 0
    val = next(iter(benchmark_definitions.values()))
    return isinstance(val, dict)


def get_transfer_learning_evaluations(
    blackbox_name: str,
    test_task: str,
    datasets: Optional[List[str]],
    n_evals: Optional[int] = None,
    ignore_hash: bool = False,
) -> dict:
    """
    :param blackbox_name: name of blackbox
    :param test_task: task where the performance would be tested, it is excluded from transfer-learning evaluations
    :param datasets: subset of datasets to consider, only evaluations from those datasets are provided to
    transfer-learning methods. If none, all datasets are used.
    :param n_evals: maximum number of evaluations to be returned
    :return:
    """
    task_to_evaluations = load_blackbox(blackbox_name, ignore_hash=ignore_hash)

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
    methods: Dict[str, Any],
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
) -> (Any, List[str], List[str], List[int]):
    """Parse command line arguments for simulator backend experiments.

    :param methods: If ``--method`` is not given, then ``method_names`` are the
        keys of this dictionary
    :param benchmark_definitions: Dictionary with tabulated or surrogate
        benchmarks. If ``--benchmark`` is not given, then ``benchmark_names`` are
        keys of this dictionary.
        Can be nested (only for internal use).
    :param extra_args: List of dictionaries, containing additional arguments
        to be passed. Must contain ``name`` for argument name (without leading
        ``"--"``), and other kwargs to ``parser.add_argument``. Optional
    :return: ``(args, method_names, benchmark_names, seeds)``, where ``args`` is
        result of ``parser.parse_known_args()``, ``method_names`` see ``methods``,
        'benchmark_names`` see ``benchmark_definitions``, and ``seeds`` are list of
        seeds specified by ``--num_seeds`` and ``--start_seed``
    """
    if extra_args is None:
        extra_args = []
    else:
        extra_args = extra_args.copy()
    nested_dict = is_dict_of_dict(benchmark_definitions)
    extra_args.extend(
        [
            dict(
                name="benchmark",
                type=str,
                help="Benchmark to run from benchmark_definitions",
            ),
            dict(
                name="verbose",
                type=int,
                default=0,
                help="Verbose log output?",
            ),
            dict(
                name="support_checkpointing",
                type=int,
                default=1,
                help="If 0, trials are started from scratch when resumed",
            ),
            dict(
                name="fcnet_ordinal",
                type=str,
                choices=("none", "equal", "nn", "nn-log"),
                default="none",
                help="Ordinal encoding for fcnet categorical HPs",
            ),
            dict(
                name="ignore_blackbox_hash",
                type=int,
                default=0,
                help="Ignore mechanism to check whether blackbox files are up to date?",
            ),
        ]
    )
    if nested_dict:
        extra_args.append(
            dict(
                name="benchmark_key",
                type=str,
                required=True,
            )
        )
    args, method_names, seeds = _parse_args(methods, extra_args)
    args.verbose = bool(args.verbose)
    args.support_checkpointing = bool(args.support_checkpointing)
    args.ignore_blackbox_hash = bool(args.ignore_blackbox_hash)
    if args.benchmark is not None:
        benchmark_names = [args.benchmark]
    else:
        if nested_dict:
            # If ``parse_args`` is called from ``launch_remote``, ``benchmark_key`` is
            # not set. In this case, ``benchmark_names`` is not needed
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
    methods: MethodDefinitions,
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
    use_transfer_learning: bool = False,
):
    """
    Runs sequence of experiments with simulator backend sequentially. The loop
    runs over methods selected from ``methods``, repetitions and benchmarks
    selected from ``benchmark_definitions``, with the range being controlled by
    command line arguments.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_extra_args: Maps ``args`` returned by :func:`parse_args` to dictionary
        for extra argument values. Needed if ``extra_args`` given
    :param use_transfer_learning: If True, we use transfer tuning. Defaults to
        False
    """
    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions, extra_args
    )
    experiment_tag = args.experiment_tag
    if is_dict_of_dict(benchmark_definitions):
        assert (
            args.benchmark_key is not None
        ), "Use --benchmark_key if benchmark_definitions is a nested dictionary"
        benchmark_definitions = benchmark_definitions[args.benchmark_key]
    set_logging_level(args)

    combinations = list(itertools.product(method_names, seeds, benchmark_names))
    print(combinations)
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]
        if args.n_workers is not None:
            benchmark.n_workers = args.n_workers
        if args.max_wallclock_time is not None:
            benchmark.max_wallclock_time = args.max_wallclock_time
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )

        max_resource_attr = benchmark.max_resource_attr
        backend = BlackboxRepositoryBackend(
            blackbox_name=benchmark.blackbox_name,
            elapsed_time_attr=benchmark.elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            support_checkpointing=args.support_checkpointing,
            dataset=benchmark.dataset_name,
            surrogate=benchmark.surrogate,
            surrogate_kwargs=benchmark.surrogate_kwargs,
            add_surrogate_kwargs=benchmark.add_surrogate_kwargs,
            ignore_hash=args.ignore_blackbox_hash,
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
            method_kwargs["scheduler_kwargs"] = extra_args
        if use_transfer_learning:
            method_kwargs["transfer_learning_evaluations"] = (
                get_transfer_learning_evaluations(
                    blackbox_name=benchmark.blackbox_name,
                    test_task=benchmark.dataset_name,
                    datasets=benchmark.datasets,
                    ignore_hash=args.ignore_blackbox_hash,
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
                fcnet_ordinal=args.fcnet_ordinal,
                use_surrogates="lcbench" in benchmark_name,
                **method_kwargs,
            )
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_evaluations=benchmark.max_num_evaluations,
        )
        metadata = get_metadata(
            seed, method, experiment_tag, benchmark_name, extra_args=extra_args
        )
        metadata["fcnet_ordinal"] = args.fcnet_ordinal
        if benchmark.add_surrogate_kwargs is not None:
            metadata["predict_curves"] = int(
                benchmark.add_surrogate_kwargs["predict_curves"]
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
            tuner_name=experiment_tag,
            metadata=metadata,
            save_tuner=args.save_tuner,
        )
        tuner.run()
