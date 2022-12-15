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
from pathlib import Path
from tqdm import tqdm
import itertools
import os

from syne_tune.try_import import try_import_aws_message

try:
    import boto3
except ImportError:
    print(try_import_aws_message())

from benchmarking.commons.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
)
from benchmarking.commons.hpo_main_sagemaker import parse_args
from benchmarking.commons.utils import (
    message_sync_from_s3,
    basic_cpu_instance_sagemaker_estimator,
    find_or_create_requirements_txt,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from benchmarking.commons.launch_remote_local import get_hyperparameters
from benchmarking.commons.baselines import MethodDefinitions
from syne_tune.util import random_string


def launch_remote(
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the SageMaker back-end. The loop runs over methods selected from
    `methods` and repetitions, both controlled by command line arguments.

    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser, optional
    :param map_extra_args: Maps `args` returned by `parse_args` to dictionary
        for extra argument values. Needed only if `extra_args` given
    """
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    if args.warm_pool:
        print(
            "ATTENTION: At the moment, -warm_pool 1 does not work with remote "
            "launching, please use it with local launching only. Switching it off."
        )
        args.warm_pool = False
    if boto3.Session().region_name is None:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    environment = {"AWS_DEFAULT_REGION": boto3.Session().region_name}
    benchmark = get_benchmark(args, benchmark_definitions, sagemaker_backend=True)
    find_or_create_requirements_txt(entry_point)

    combinations = list(itertools.product(method_names, seeds))
    for method, seed in tqdm(combinations):
        tuner_name = f"{method}-{seed}"
        sm_args = sagemaker_estimator_args(
            entry_point=entry_point,
            experiment_tag=args.experiment_tag,
            tuner_name=tuner_name,
            benchmark=benchmark,
            sagemaker_backend=True,
        )
        sm_args["environment"] = environment
        hyperparameters = get_hyperparameters(
            seed, method, experiment_tag, args, map_extra_args
        )
        hyperparameters["max_failures"] = args.max_failures
        hyperparameters["warm_pool"] = int(args.warm_pool)
        hyperparameters["delete_checkpoints"] = int(args.delete_checkpoints)
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {sm_args['checkpoint_s3_uri']}"
        )
        est = basic_cpu_instance_sagemaker_estimator(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
