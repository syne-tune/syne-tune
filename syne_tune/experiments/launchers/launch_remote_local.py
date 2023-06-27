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
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from tqdm import tqdm

from sagemaker.estimator import EstimatorBase

from syne_tune.experiments.baselines import MethodDefinitions
from syne_tune.experiments.benchmark_definitions import RealBenchmarkDefinition
from syne_tune.experiments.launchers.hpo_main_common import (
    extra_metadata,
    ExtraArgsType,
    ConfigDict,
    DictStrKey,
    config_from_argparse,
)
from syne_tune.experiments.launchers.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
    LOCAL_BACKEND_EXTRA_PARAMETERS,
)
from syne_tune.experiments.launchers.launch_remote_common import (
    sagemaker_estimator_args,
    fit_sagemaker_estimator,
)
from syne_tune.experiments.launchers.utils import (
    filter_none,
    message_sync_from_s3,
    find_or_create_requirements_txt,
    combine_requirements_txt,
    get_master_random_seed,
)
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    add_metric_definitions_to_sagemaker_estimator,
)
from syne_tune.remote.estimators import sagemaker_estimator
from syne_tune.remote.remote_metrics_callback import RemoteTuningMetricsCallback
from syne_tune.util import random_string

logger = logging.getLogger(__name__)


def get_hyperparameters(
    seed: int,
    method: str,
    experiment_tag: str,
    random_seed: int,
    configuration: ConfigDict,
) -> Dict[str, Any]:
    """Compose hyperparameters for SageMaker training job

    :param seed: Seed of repetition
    :param method: Method name
    :param experiment_tag: Tag of experiment
    :param random_seed: Master random seed
    :param configuration: Configuration for the job
    :return: Dictionary of hyperparameters
    """
    hyperparameters = {
        "experiment_tag": experiment_tag,
        "benchmark": configuration.benchmark,
        "method": method,
        "save_tuner": int(configuration.save_tuner),
        "num_seeds": seed + 1,
        "start_seed": seed,
        "random_seed": random_seed,
        "scale_max_wallclock_time": int(configuration.scale_max_wallclock_time),
        "launched_remotely": 1,
    }
    for k in (
        "n_workers",
        "max_wallclock_time",
        "instance_type",
        "max_size_data_for_model",
    ):
        v = getattr(configuration, k)
        if v is not None:
            hyperparameters[k] = v
    hyperparameters.update(
        filter_none(extra_metadata(configuration, configuration.extra_parameters()))
    )
    return hyperparameters


def register_metrics_with_estimator(
    estimator: EstimatorBase, benchmark: RealBenchmarkDefinition
):
    metric_names = RemoteTuningMetricsCallback.get_metric_names(
        config_space=benchmark.config_space,
        resource_attr=benchmark.resource_attr,
    )
    add_metric_definitions_to_sagemaker_estimator(estimator, metric_names)


def launch_remote(
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    source_dependencies: Optional[List[str]] = None,
    extra_args: Optional[ExtraArgsType] = None,
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the local backend. The loop runs over methods selected from ``methods``
    and repetitions, both controlled by command line arguments.

    Combination of ``requirements.txt``: Dependencies for ``entry_point`` are the
    union of Syne Tune dependencies and dependencies of the training script
    (not contained in its SageMaker framework). For the former, we scan
    ``entry_point.parent`` for a file named ``requirements*.txt``. If this is not
    found, we create a default one called ``requirements-synetune.txt``. This is
    then combined with the ``requirements.txt`` file for the training script
    (if any), and the union is written to ``requirements.txt`` in
    ``entry_point.parent``.
    If you like to control the Syne Tune requirements (the default ones are
    ``"extra"``, which can be a lot), place a file ``requirements_synetune.txt`` in
    ``entry_point.parent``.

    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param source_dependencies: If given, these are source dependencies for the
        SageMaker estimator, on top of Syne Tune itself
    :param extra_args: Extra arguments for command line parser, optional
    """
    configuration = config_from_argparse(extra_args, LOCAL_BACKEND_EXTRA_PARAMETERS)
    launch_remote_experiments(
        configuration=configuration,
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=source_dependencies,
    )


def launch_remote_experiments(
    configuration: ConfigDict,
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    source_dependencies: Optional[List[str]],
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the local backend. The loop runs over methods selected from ``methods``
    and repetitions.

    Combination of ``requirements.txt``: Dependencies for ``entry_point`` are the
    union of Syne Tune dependencies and dependencies of the training script
    (not contained in its SageMaker framework). For the former, we scan
    ``entry_point.parent`` for a file named ``requirements*.txt``. If this is not
    found, we create a default one called ``requirements-synetune.txt``. This is
    then combined with the ``requirements.txt`` file for the training script
    (if any), and the union is written to ``requirements.txt`` in
    ``entry_point.parent``.
    If you like to control the Syne Tune requirements (the default ones are
    ``"extra"``, which can be a lot), place a file ``requirements_synetune.txt`` in
    ``entry_point.parent``.

    :param configuration: ConfigDict with parameters of the benchmark.
            Must contain all parameters from
            hpo_main_local.LOCAL_LOCAL_BENCHMARK_REQUIRED_PARAMETERS
    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    """
    configuration.check_if_all_paremeters_present(LOCAL_BACKEND_EXTRA_PARAMETERS)
    configuration.expand_base_arguments(LOCAL_BACKEND_EXTRA_PARAMETERS)

    method_names = (
        [configuration.method]
        if configuration.method is not None
        else list(methods.keys())
    )
    benchmark = get_benchmark(configuration, benchmark_definitions)
    master_random_seed = get_master_random_seed(configuration.random_seed)
    synetune_requirements_file = find_or_create_requirements_txt(
        entry_point, requirements_fname="requirements-synetune.txt"
    )
    combine_requirements_txt(synetune_requirements_file, benchmark.script)
    extra_sagemaker_hyperparameters = {
        "verbose": int(configuration.verbose),
        "remote_tuning_metrics": int(configuration.remote_tuning_metrics),
        "delete_checkpoints": int(configuration.delete_checkpoints),
        "num_gpus_per_trial": int(configuration.num_gpus_per_trial),
    }
    experiment_tag = _launch_experiment_remotely(
        configuration=configuration,
        entry_point=entry_point,
        method_names=method_names,
        benchmark=benchmark,
        master_random_seed=master_random_seed,
        sagemaker_estimator_base_class=sagemaker_estimator[benchmark.framework],
        environment=None,
        extra_sagemaker_hyperparameters=extra_sagemaker_hyperparameters,
        use_sagemaker_backend=False,
        source_dependencies=source_dependencies,
    )

    print("\n" + message_sync_from_s3(experiment_tag))


def _launch_experiment_remotely(
    configuration: ConfigDict,
    entry_point: Path,
    method_names: List[str],
    benchmark: RealBenchmarkDefinition,
    master_random_seed: int,
    sagemaker_estimator_base_class: Callable[[Any], EstimatorBase],
    environment: Optional[DictStrKey],
    extra_sagemaker_hyperparameters: DictStrKey,
    use_sagemaker_backend: bool,
    source_dependencies: Optional[List[str]],
):
    experiment_tag = configuration.experiment_tag
    suffix = random_string(4)
    combinations = list(itertools.product(method_names, configuration.seeds))
    for method, seed in tqdm(combinations):
        tuner_name = f"{method}-{seed}"
        sm_args = sagemaker_estimator_args(
            entry_point=entry_point,
            experiment_tag=configuration.experiment_tag,
            tuner_name=tuner_name,
            benchmark=benchmark,
            sagemaker_backend=use_sagemaker_backend,
            source_dependencies=source_dependencies,
        )
        hyperparameters = get_hyperparameters(
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            random_seed=master_random_seed,
            configuration=configuration,
        )
        hyperparameters.update(extra_sagemaker_hyperparameters)
        if environment is not None:
            sm_args["environment"] = environment
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {sm_args['checkpoint_s3_uri']}"
        )
        est = sagemaker_estimator_base_class(**sm_args)
        if configuration.remote_tuning_metrics:
            register_metrics_with_estimator(est, benchmark)
        fit_sagemaker_estimator(
            backoff_wait_time=configuration.estimator_fit_backoff_wait_time,
            estimator=est,
            job_name=f"{experiment_tag}-{tuner_name}-{suffix}",
            wait=False,
        )
    return experiment_tag
