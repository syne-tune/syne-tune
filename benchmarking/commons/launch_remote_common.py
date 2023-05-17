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
from pathlib import Path
from typing import Optional, Dict, Any

import benchmarking
import syne_tune
from benchmarking.commons.benchmark_definitions.common import BenchmarkDefinition
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
from syne_tune.util import s3_experiment_path


def sagemaker_estimator_args(
    entry_point: Path,
    experiment_tag: str,
    tuner_name: str,
    benchmark: Optional[BenchmarkDefinition] = None,
    sagemaker_backend: bool = False,
) -> Dict[str, Any]:
    """
    Returns SageMaker estimator keyword arguments for remote tuning job.

    Note: We switch off SageMaker profiler and debugger, as both are not needed
    and consume extra resources and may introduce instabilities.

    :param entry_point: Script for running HPO experiment, used for ``entry_point``
        and ``source_dir`` arguments
    :param experiment_tag: Tag of experiment, used to create ``checkpoint_s3_uri``
    :param tuner_name: Name of tuner, used to create ``checkpoint_s3_uri``
    :param benchmark: Benchmark definition, optional
    :param sagemaker_backend: Is remote tuning job running the SageMaker backend?
        If not, it either runs local or simulator backend. Defaults to False
    :return: Keyword arguments for SageMaker estimator
    """
    checkpoint_s3_uri = s3_experiment_path(
        tuner_name=tuner_name, experiment_name=experiment_tag
    )
    if checkpoint_s3_uri[-1] != "/":
        checkpoint_s3_uri += "/"
    max_run = (
        int(1.2 * benchmark.max_wallclock_time) if benchmark is not None else 3600 * 72
    )
    sm_args = dict(
        entry_point=entry_point.name,
        source_dir=str(entry_point.parent),
        checkpoint_s3_uri=checkpoint_s3_uri,
        max_run=max_run,
        role=get_execution_role(),
        dependencies=syne_tune.__path__ + benchmarking.__path__,
        disable_profiler=True,
        debugger_hook_config=False,
    )
    if benchmark is not None and not sagemaker_backend:
        sm_args.update(dict(instance_type=benchmark.instance_type, instance_count=1))
        if benchmark.estimator_kwargs is not None:
            sm_args.update(benchmark.estimator_kwargs)
    return sm_args


REMOTE_LAUNCHING_EXTRA_PARAMETERS = [
    dict(
        name="skip_initial_jobs",
        type=int,
        default=0,
        help="Skip this number of initial jobs which would be launched",
    )
]


def remove_remote_launching_parameters(
    hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    remove_keys = [x["name"] for x in REMOTE_LAUNCHING_EXTRA_PARAMETERS]
    return {k: v for k, v in hyperparameters.items() if k not in remove_keys}
