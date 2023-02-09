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
from typing import Optional, Dict, Any

from tqdm import tqdm

from benchmarking.commons.baselines import MethodDefinitions
from benchmarking.commons.hpo_main_common import extra_metadata, ExtraArgsType
from benchmarking.commons.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
    parse_args,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args
from benchmarking.commons.utils import (
    filter_none,
    message_sync_from_s3,
    find_or_create_requirements_txt,
    combine_requirements_txt,
    get_master_random_seed,
)
from syne_tune.remote.estimators import sagemaker_estimator
from syne_tune.util import random_string

logger = logging.getLogger(__name__)


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
        "benchmark": args.benchmark,
        "method": method,
        "save_tuner": int(args.save_tuner),
        "num_seeds": seed + 1,
        "start_seed": seed,
        "random_seed": random_seed,
    }
    for k in (
        "n_workers",
        "max_wallclock_time",
        "instance_type",
        "max_size_data_for_model",
    ):
        v = getattr(args, k)
        if v is not None:
            hyperparameters[k] = v
    if extra_args is not None:
        hyperparameters.update(filter_none(extra_metadata(args, extra_args)))
    return hyperparameters


def launch_remote(
    entry_point: Path,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
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
    :param extra_args: Extra arguments for command line parser, optional
    """
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    benchmark = get_benchmark(args, benchmark_definitions)
    master_random_seed = get_master_random_seed(args.random_seed)

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
            seed=seed,
            method=method,
            experiment_tag=experiment_tag,
            random_seed=master_random_seed,
            args=args,
            extra_args=extra_args,
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
