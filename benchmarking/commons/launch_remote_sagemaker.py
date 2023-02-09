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
import itertools
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from syne_tune.try_import import try_import_aws_message

try:
    import boto3
except ImportError:
    print(try_import_aws_message())

from benchmarking.commons.hpo_main_common import ExtraArgsType
from benchmarking.commons.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
)
from benchmarking.commons.hpo_main_sagemaker import parse_args
from benchmarking.commons.utils import (
    message_sync_from_s3,
    find_or_create_requirements_txt,
    get_master_random_seed,
)
from syne_tune.remote.estimators import (
    basic_cpu_instance_sagemaker_estimator,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from benchmarking.commons.launch_remote_local import get_hyperparameters
from benchmarking.commons.baselines import MethodDefinitions
from syne_tune.util import random_string


def launch_remote(
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the SageMaker backend. The loop runs over methods selected from
    ``methods`` and repetitions, both controlled by command line arguments.

    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser, optional
    """
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    if boto3.Session().region_name is None:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    environment = {"AWS_DEFAULT_REGION": boto3.Session().region_name}
    benchmark = get_benchmark(args, benchmark_definitions, sagemaker_backend=True)
    master_random_seed = get_master_random_seed(args.random_seed)
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
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            random_seed=master_random_seed,
            args=args,
            extra_args=extra_args,
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
