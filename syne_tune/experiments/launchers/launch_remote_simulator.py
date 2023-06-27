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
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from tqdm import tqdm

from syne_tune.experiments.baselines import MethodDefinitions
from syne_tune.experiments.launchers.hpo_main_common import (
    extra_metadata,
    ExtraArgsType,
    ConfigDict,
    config_from_argparse,
)
from syne_tune.experiments.launchers.hpo_main_simulator import (
    SurrogateBenchmarkDefinitions,
    is_dict_of_dict,
    SIMULATED_BACKEND_EXTRA_PARAMETERS,
    BENCHMARK_KEY_EXTRA_PARAMETER,
)
from syne_tune.experiments.launchers.launch_remote_common import (
    sagemaker_estimator_args,
    fit_sagemaker_estimator,
)
from syne_tune.experiments.launchers.utils import (
    filter_none,
    message_sync_from_s3,
    find_or_create_requirements_txt,
    get_master_random_seed,
)
from syne_tune.remote.estimators import (
    basic_cpu_instance_sagemaker_estimator,
)
from syne_tune.util import random_string


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
        "method": method,
        "save_tuner": int(configuration.save_tuner),
        "random_seed": random_seed,
        "scale_max_wallclock_time": int(configuration.scale_max_wallclock_time),
        "launched_remotely": 1,
    }
    if seed is not None:
        hyperparameters["num_seeds"] = seed + 1
        hyperparameters["start_seed"] = seed
    else:
        hyperparameters["num_seeds"] = configuration.num_seeds
        hyperparameters["start_seed"] = configuration.start_seed
    if configuration.benchmark is not None:
        hyperparameters["benchmark"] = configuration.benchmark
    for k in (
        "n_workers",
        "max_wallclock_time",
        "max_size_data_for_model",
        "fcnet_ordinal",
    ):
        v = getattr(configuration, k)
        if v is not None:
            hyperparameters[k] = v
    hyperparameters.update(
        filter_none(extra_metadata(configuration, configuration.extra_parameters()))
    )
    return hyperparameters


def launch_remote(
    entry_point: Path,
    methods: Dict[str, Any],
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    source_dependencies: Optional[List[str]] = None,
    extra_args: Optional[ExtraArgsType] = None,
    is_expensive_method: Optional[Callable[[str], bool]] = None,
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the simulator backend.

    The loop runs over methods selected from ``methods``. Different repetitions
    (seeds) are run sequentially in the remote job. However, if
    ``is_expensive_method(method_name)`` is true, we launch different remote
    jobs for every seed for this particular method. This is to cater for
    methods which are themselves expensive to run (e.g., involving Gaussian
    process based Bayesian optimization).

    If ``benchmark_definitions`` is a single-level dictionary and no benchmark
    is selected on the command line, then all benchmarks are run sequentially
    in the remote job. However, if ``benchmark_definitions`` is two-level nested,
    we loop over the outer level and start separate remote jobs, each of which
    iterates over its inner level of benchmarks. This is useful if the number
    of benchmarks to iterate over is large.

    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks, can be nested
        (see above)
    :param source_dependencies: If given, these are source dependencies for the
        SageMaker estimator, on top of Syne Tune itself
    :param extra_args: Extra arguments for command line parser, optional
    :param is_expensive_method: See above. The default is a predicative always
        returning False (no method is expensive)
    """
    simulated_backend_extra_parameters = SIMULATED_BACKEND_EXTRA_PARAMETERS.copy()
    if is_dict_of_dict(benchmark_definitions):
        simulated_backend_extra_parameters.append(BENCHMARK_KEY_EXTRA_PARAMETER)
    configuration = config_from_argparse(extra_args, simulated_backend_extra_parameters)
    launch_remote_experiments_simulator(
        configuration=configuration,
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        source_dependencies=source_dependencies,
        is_expensive_method=is_expensive_method,
    )


def launch_remote_experiments_simulator(
    configuration: ConfigDict,
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    source_dependencies: Optional[List[str]],
    is_expensive_method: Optional[Callable[[str], bool]] = None,
):
    """
    Launches sequence of SageMaker training jobs, each running an experiment
    with the simulator backend.

    The loop runs over methods selected from ``methods``. Different repetitions
    (seeds) are run sequentially in the remote job. However, if
    ``is_expensive_method(method_name)`` is true, we launch different remote
    jobs for every seed for this particular method. This is to cater for
    methods which are themselves expensive to run (e.g., involving Gaussian
    process based Bayesian optimization).

    If ``benchmark_definitions`` is a single-level dictionary and no benchmark
    is selected on the command line, then all benchmarks are run sequentially
    in the remote job. However, if ``benchmark_definitions`` is two-level nested,
    we loop over the outer level and start separate remote jobs, each of which
    iterates over its inner level of benchmarks. This is useful if the number
    of benchmarks to iterate over is large.

    :param configuration: ConfigDict with parameters of the benchmark.
            Must contain all parameters from
            hpo_main_simulator.LOCAL_LOCAL_SIMULATED_BENCHMARK_REQUIRED_PARAMETERS
    :param entry_point: Script for running the experiment
    :param methods: Dictionary with method constructors; one is selected from
        command line arguments
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param is_expensive_method: See above. The default is a predicative always
        returning False (no method is expensive)
    """
    if is_expensive_method is None:
        # Nothing is expensive
        def is_expensive_method(method):
            return False

    simulated_backend_extra_parameters = SIMULATED_BACKEND_EXTRA_PARAMETERS.copy()
    if is_dict_of_dict(benchmark_definitions):
        simulated_backend_extra_parameters.append(BENCHMARK_KEY_EXTRA_PARAMETER)
    configuration.check_if_all_paremeters_present(simulated_backend_extra_parameters)
    configuration.expand_base_arguments(simulated_backend_extra_parameters)

    method_names = (
        [configuration.method]
        if configuration.method is not None
        else list(methods.keys())
    )

    nested_dict = is_dict_of_dict(benchmark_definitions)
    experiment_tag = configuration.experiment_tag
    master_random_seed = get_master_random_seed(configuration.random_seed)
    suffix = random_string(4)
    find_or_create_requirements_txt(entry_point)

    combinations = []
    for method in method_names:
        seed_range = configuration.seeds if is_expensive_method(method) else [None]
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
            experiment_tag=configuration.experiment_tag,
            tuner_name=tuner_name,
            source_dependencies=source_dependencies,
        )
        hyperparameters = get_hyperparameters(
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            random_seed=master_random_seed,
            configuration=configuration,
        )
        hyperparameters["verbose"] = int(configuration.verbose)
        hyperparameters["support_checkpointing"] = int(
            configuration.support_checkpointing
        )
        hyperparameters["restrict_configurations"] = int(
            configuration.restrict_configurations
        )
        if benchmark_key is not None:
            hyperparameters["benchmark_key"] = benchmark_key
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {sm_args['checkpoint_s3_uri']}"
        )
        est = basic_cpu_instance_sagemaker_estimator(**sm_args)
        fit_sagemaker_estimator(
            backoff_wait_time=configuration.estimator_fit_backoff_wait_time,
            estimator=est,
            job_name=f"{experiment_tag}-{tuner_name}-{suffix}",
            wait=False,
        )

    print("\n" + message_sync_from_s3(experiment_tag))
