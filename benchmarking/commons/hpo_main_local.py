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
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import itertools
from tqdm import tqdm

from syne_tune.backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.commons.baselines import MethodArguments, MethodDefinitions
from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_common import (
    parse_args as _parse_args,
    set_logging_level,
    get_metadata,
)
from benchmarking.commons.utils import get_master_random_seed, effective_random_seed


RealBenchmarkDefinitions = Callable[..., Dict[str, RealBenchmarkDefinition]]


def get_benchmark(
    args, benchmark_definitions: RealBenchmarkDefinitions, **benchmark_kwargs
):
    if args.n_workers is not None:
        benchmark_kwargs["n_workers"] = args.n_workers
    if args.max_wallclock_time is not None:
        benchmark_kwargs["max_wallclock_time"] = args.max_wallclock_time
    if args.instance_type is not None:
        benchmark_kwargs["instance_type"] = args.instance_type
    return benchmark_definitions(**benchmark_kwargs)[args.benchmark]


def parse_args(
    methods: Dict[str, Any], extra_args: Optional[List[dict]] = None
) -> (Any, List[str], List[int]):
    """Parse command line arguments for local backend experiments.

    :param methods: If ``--method`` is not given, then ``method_names`` are the
        keys of this dictionary
    :param extra_args: List of dictionaries, containing additional arguments
        to be passed. Must contain ``name`` for argument name (without leading
        ``"--"``), and other kwargs to ``parser.add_argument``. Optional
    :return: ``(args, method_names, seeds)``, where ``args`` is result of
        ``parser.parse_known_args()``, ``method_names`` see ``methods``, and
        ``seeds`` are list of seeds specified by ``--num_seeds`` and ``--start_seed``
    """
    if extra_args is None:
        extra_args = []
    else:
        extra_args = extra_args.copy()
    extra_args.extend(
        [
            dict(
                name="benchmark",
                type=str,
                default="resnet_cifar10",
                help="Benchmark to run",
            ),
            dict(
                name="verbose",
                type=int,
                default=0,
                help="Verbose log output?",
            ),
            dict(
                name="instance_type",
                type=str,
                help="AWS SageMaker instance type",
            ),
        ]
    )
    args, method_names, seeds = _parse_args(methods, extra_args)
    args.verbose = bool(args.verbose)
    return args, method_names, seeds


def main(
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    """
    Runs sequence of experiments with local backend sequentially. The loop runs
    over methods selected from ``methods`` and repetitions, both controlled by
    command line arguments.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_extra_args: Maps ``args`` returned by :func:`parse_args` to dictionary
        for extra argument values. Needed if ``extra_args`` given
    """
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark
    master_random_seed = get_master_random_seed(args.random_seed)

    set_logging_level(args)
    benchmark = get_benchmark(args, benchmark_definitions)

    combinations = list(itertools.product(method_names, seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        random_seed = effective_random_seed(master_random_seed, seed)
        np.random.seed(random_seed)
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )
        trial_backend = LocalBackend(entry_point=str(benchmark.script))

        method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
        if extra_args is not None:
            assert map_extra_args is not None
            extra_args = map_extra_args(args)
            method_kwargs.update(extra_args)
        scheduler = methods[method](
            MethodArguments(
                config_space=benchmark.config_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=random_seed,
                resource_attr=benchmark.resource_attr,
                verbose=args.verbose,
                max_size_data_for_model=args.max_size_data_for_model,
                **method_kwargs,
            )
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_evaluations=benchmark.max_num_evaluations,
        )
        metadata = get_metadata(
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            benchmark_name=benchmark_name,
            random_seed=master_random_seed,
            max_size_data_for_model=args.max_size_data_for_model,
            benchmark=benchmark,
            extra_args=extra_args,
        )
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            tuner_name=experiment_tag,
            metadata=metadata,
            save_tuner=args.save_tuner,
        )
        tuner.run()
