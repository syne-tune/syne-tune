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
from typing import Optional, List, Callable
from pathlib import Path
from tqdm import tqdm
import itertools

from benchmarking.commons.hpo_main_simulator import (
    parse_args,
    SurrogateBenchmarkDefinitions,
    is_dict_of_dict,
)
from benchmarking.commons.utils import (
    filter_none,
    message_sync_from_s3,
    basic_cpu_instance_sagemaker_estimator,
    find_or_create_requirements_txt,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from syne_tune.util import random_string


def launch_remote(
    entry_point: Path,
    methods: dict,
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[Callable] = None,
    is_expensive_method: Optional[Callable[[str], bool]] = None,
):
    if is_expensive_method is None:
        # Nothing is expensive
        is_expensive_method = lambda method: False
    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions, extra_args
    )
    nested_dict = is_dict_of_dict(benchmark_definitions)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    find_or_create_requirements_txt(entry_point)

    combinations = []
    for method in method_names:
        seed_range = seeds if is_expensive_method(method) else [None]
        combinations.extend([(method, seed) for seed in seed_range])
    if nested_dict:
        benchmark_keys = list(benchmark_definitions.keys())
        combinations = list(itertools.product(combinations, benchmark_keys))
    else:
        combinations = [(x, None) for x in combinations]

    for (method, seed), benchmark_key in tqdm(combinations):
        tuner_name = method
        if seed is not None:
            tuner_name += f"-{seed}"
        if benchmark_key is not None:
            tuner_name += f"-{benchmark_key}"
        sm_args = sagemaker_estimator_args(
            entry_point=entry_point,
            experiment_tag=args.experiment_tag,
            tuner_name=tuner_name,
        )
        hyperparameters = {
            "experiment_tag": experiment_tag,
            "method": method,
            "support_checkpointing": int(args.support_checkpointing),
            "save_tuner": int(args.save_tuner),
            "verbose": int(args.verbose),
        }
        if extra_args is not None:
            assert map_extra_args is not None
            hyperparameters.update(filter_none(map_extra_args(args)))
        if seed is not None:
            hyperparameters["num_seeds"] = seed + 1
            hyperparameters["start_seed"] = seed
        else:
            hyperparameters["num_seeds"] = args.num_seeds
            hyperparameters["start_seed"] = args.start_seed
        if args.benchmark is not None:
            hyperparameters["benchmark"] = args.benchmark
        if benchmark_key is not None:
            hyperparameters["benchmark_key"] = benchmark_key
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {sm_args['checkpoint_s3_uri']}"
        )
        est = basic_cpu_instance_sagemaker_estimator(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
