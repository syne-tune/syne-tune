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
import os
from pathlib import Path
from typing import Optional, List

from syne_tune.try_import import try_import_aws_message

try:
    import boto3
except ImportError:
    print(try_import_aws_message())

from syne_tune.experiments.launchers.hpo_main_common import (
    ExtraArgsType,
    ConfigDict,
    config_from_argparse,
)
from syne_tune.experiments.launchers.hpo_main_sagemaker import (
    SAGEMAKER_BACKEND_EXTRA_PARAMETERS,
)
from syne_tune.experiments.launchers.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
)
from syne_tune.experiments.launchers.utils import (
    message_sync_from_s3,
    find_or_create_requirements_txt,
    get_master_random_seed,
)
from syne_tune.remote.estimators import (
    basic_cpu_instance_sagemaker_estimator,
)
from syne_tune.experiments.launchers.launch_remote_local import (
    _launch_experiment_remotely,
)
from syne_tune.experiments.baselines import MethodDefinitions


def launch_remote(
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    source_dependencies: Optional[List[str]] = None,
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
    :param source_dependencies: If given, these are source dependencies for the
        SageMaker estimator, on top of Syne Tune itself
    :param extra_args: Extra arguments for command line parser, optional
    """
    configuration = config_from_argparse(extra_args, SAGEMAKER_BACKEND_EXTRA_PARAMETERS)
    launch_remote_experiments_sagemaker(
        configuration=configuration,
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=source_dependencies,
    )


def launch_remote_experiments_sagemaker(
    configuration: ConfigDict,
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    source_dependencies: Optional[List[str]],
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the SageMaker backend. The loop runs over methods selected from
    ``methods`` and repetitions.

    :param configuration: ConfigDict with parameters of the benchmark.
            Must contain all parameters from
            hpo_main_sagemaker.LOCAL_SAGEMAKER_BENCHMARK_REQUIRED_PARAMETERS
    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    """
    configuration.check_if_all_paremeters_present(SAGEMAKER_BACKEND_EXTRA_PARAMETERS)
    configuration.expand_base_arguments(SAGEMAKER_BACKEND_EXTRA_PARAMETERS)

    method_names = (
        [configuration.method]
        if configuration.method is not None
        else list(methods.keys())
    )
    if boto3.Session().region_name is None:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    environment = {"AWS_DEFAULT_REGION": boto3.Session().region_name}

    benchmark = get_benchmark(
        configuration, benchmark_definitions, sagemaker_backend=True
    )
    master_random_seed = get_master_random_seed(configuration.random_seed)
    find_or_create_requirements_txt(entry_point)

    extra_sagemaker_hyperparameters = {
        "max_failures": configuration.max_failures,
        "warm_pool": int(configuration.warm_pool),
        "delete_checkpoints": int(configuration.delete_checkpoints),
        "remote_tuning_metrics": int(configuration.remote_tuning_metrics),
    }
    experiment_tag = _launch_experiment_remotely(
        configuration=configuration,
        entry_point=entry_point,
        method_names=method_names,
        benchmark=benchmark,
        master_random_seed=master_random_seed,
        sagemaker_estimator_base_class=basic_cpu_instance_sagemaker_estimator,
        environment=environment,
        extra_sagemaker_hyperparameters=extra_sagemaker_hyperparameters,
        use_sagemaker_backend=True,
        source_dependencies=source_dependencies,
    )

    print("\n" + message_sync_from_s3(experiment_tag))
