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
from typing import Optional, List, Callable, Dict

import numpy as np
import itertools
from tqdm import tqdm

from syne_tune.backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.commons.baselines import MethodArguments
from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_common import (
    parse_args as _parse_args,
    set_logging_level,
    get_metadata,
)


RealBenchmarkDefinitions = Callable[..., Dict[str, RealBenchmarkDefinition]]


def get_benchmark(
    args, benchmark_definitions: RealBenchmarkDefinitions, **benchmark_kwargs
):
    if args.n_workers is not None:
        benchmark_kwargs["n_workers"] = args.n_workers
    if args.max_wallclock_time is not None:
        benchmark_kwargs["max_wallclock_time"] = args.max_wallclock_time
    return benchmark_definitions(**benchmark_kwargs)[args.benchmark]


def parse_args(methods: dict, extra_args: Optional[List[dict]] = None):
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
        ]
    )
    args, method_names, seeds = _parse_args(methods, extra_args)
    args.verbose = bool(args.verbose)
    return args, method_names, seeds


def main(
    methods: dict,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark

    set_logging_level(args)
    benchmark = get_benchmark(args, benchmark_definitions)

    combinations = list(itertools.product(method_names, seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        np.random.seed(seed)
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
                random_seed=seed,
                resource_attr=benchmark.resource_attr,
                verbose=args.verbose,
                **method_kwargs,
            )
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_evaluations=benchmark.max_num_evaluations,
        )
        metadata = get_metadata(
            seed, method, experiment_tag, benchmark_name, benchmark, extra_args
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
