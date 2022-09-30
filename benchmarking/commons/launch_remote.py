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

from sagemaker.pytorch import PyTorch

from benchmarking.commons.benchmark_main import (
    parse_args,
    BenchmarkDefinitions,
    is_dict_of_dict,
)
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


def _filter_none(a: dict) -> dict:
    return {k: v for k, v in a.items() if v is not None}


def message_sync_from_s3(experiment_tag: str) -> str:
    return (
        "Launched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_tag)} "
        f'~/syne-tune/{experiment_tag}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )


def launch_remote(
    entry_point: Path,
    methods: dict,
    benchmark_definitions: BenchmarkDefinitions,
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
    benchmark_name = args.benchmark
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
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
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=tuner_name, experiment_name=experiment_tag
        )
        sm_args = dict(
            entry_point=entry_point.name,
            source_dir=str(entry_point.parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
            debugger_hook_config=False,
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
            hyperparameters.update(_filter_none(map_extra_args(args)))
        if seed is not None:
            hyperparameters["num_seeds"] = seed
            hyperparameters["run_all_seed"] = 0
        else:
            hyperparameters["num_seeds"] = args.num_seeds
            hyperparameters["start_seed"] = args.start_seed
        if benchmark_name is not None:
            hyperparameters["benchmark"] = benchmark_name
        if benchmark_key is not None:
            hyperparameters["benchmark_key"] = benchmark_key
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
