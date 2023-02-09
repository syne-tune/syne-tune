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
from typing import Optional, Callable, Dict, Any

from tqdm import tqdm

from benchmarking.commons.hpo_main_common import extra_metadata, ExtraArgsType
from benchmarking.commons.hpo_main_simulator import (
    parse_args,
    SurrogateBenchmarkDefinitions,
    is_dict_of_dict,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from benchmarking.commons.utils import (
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
    args,
    extra_args: Optional[ExtraArgsType],
) -> Dict[str, Any]:
    """Compose hyperparameters for SageMaker training job

    :param seed: Seed of repetition
    :param method: Method name
    :param experiment_tag: Tag of experiment
    :param random_seed: Master random seed
    :param args: Result from :func:`parse_args`
    :param extra_args: Argument of ``launch_remote``
    :return: Dictionary of hyperparameters
    """
    hyperparameters = {
        "experiment_tag": experiment_tag,
        "method": method,
        "save_tuner": int(args.save_tuner),
        "random_seed": random_seed,
    }
    if seed is not None:
        hyperparameters["num_seeds"] = seed + 1
        hyperparameters["start_seed"] = seed
    else:
        hyperparameters["num_seeds"] = args.num_seeds
        hyperparameters["start_seed"] = args.start_seed
    if args.benchmark is not None:
        hyperparameters["benchmark"] = args.benchmark
    for k in ("n_workers", "max_wallclock_time", "max_size_data_for_model"):
        v = getattr(args, k)
        if v is not None:
            hyperparameters[k] = v
    if extra_args is not None:
        hyperparameters.update(filter_none(extra_metadata(args, extra_args)))
    return hyperparameters


def launch_remote(
    entry_point: Path,
    methods: Dict[str, Any],
    benchmark_definitions: SurrogateBenchmarkDefinitions,
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
    :param extra_args: Extra arguments for command line parser, optional
    :param is_expensive_method: See above. The default is a predicative always
        returning False (no method is expensive)
    """
    if is_expensive_method is None:
        # Nothing is expensive
        def is_expensive_method(method):
            return False

    args, method_names, benchmark_names, seeds = parse_args(
        methods, benchmark_definitions, extra_args
    )
    nested_dict = is_dict_of_dict(benchmark_definitions)
    experiment_tag = args.experiment_tag
    master_random_seed = get_master_random_seed(args.random_seed)
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
        hyperparameters = get_hyperparameters(
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            random_seed=master_random_seed,
            args=args,
            extra_args=extra_args,
        )
        hyperparameters["support_checkpointing"] = int(args.support_checkpointing)
        hyperparameters["verbose"] = int(args.verbose)
        hyperparameters["restrict_configurations"] = int(args.restrict_configurations)
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
