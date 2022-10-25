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
from typing import Optional, List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import itertools
import logging

from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
    parse_args,
)
from benchmarking.commons.utils import (
    filter_none,
    message_sync_from_s3,
    sagemaker_estimator,
    find_or_create_requirements_txt,
    combine_requirements_txt,
)
from syne_tune.util import random_string

logger = logging.getLogger(__name__)


def get_hyperparameters(
    seed: int,
    method: str,
    experiment_tag: str,
    args,
    benchmark: RealBenchmarkDefinition,
    map_extra_args: Optional[callable],
) -> Dict[str, Any]:
    hyperparameters = {
        "experiment_tag": experiment_tag,
        "benchmark": args.benchmark,
        "method": method,
        "save_tuner": int(args.save_tuner),
        "num_seeds": seed + 1,
        "start_seed": seed,
        "n_workers": benchmark.n_workers,
        "max_wallclock_time": benchmark.max_wallclock_time,
    }
    if map_extra_args is not None:
        hyperparameters.update(filter_none(map_extra_args(args)))
    return hyperparameters


def launch_remote(
    entry_point: Path,
    methods: dict,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    benchmark = get_benchmark(args, benchmark_definitions)

    synetune_requirements_file = find_or_create_requirements_txt(
        entry_point, requirements_fname="requirements-synetune.txt"
    )
    combine_requirements_txt(synetune_requirements_file, benchmark.script)

    combinations = list(itertools.product(method_names, seeds))
    for method, seed in tqdm(combinations):
        tuner_name = f"{method}-{seed}"
        sm_args = sagemaker_estimator_args(
            entry_point=entry_point,
            experiment_tag=args.experiment_tag,
            tuner_name=tuner_name,
            benchmark=benchmark,
        )
        hyperparameters = get_hyperparameters(
            seed, method, experiment_tag, args, benchmark, map_extra_args
        )
        hyperparameters["verbose"] = int(args.verbose)
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {sm_args['checkpoint_s3_uri']}"
        )
        est = sagemaker_estimator[benchmark.framework](**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
