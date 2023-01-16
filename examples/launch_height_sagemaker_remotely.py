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
"""
This example show how to launch a tuning job that will be executed on Sagemaker rather than on your local machine.
"""
import logging
from pathlib import Path

import numpy as np
from sagemaker.pytorch import PyTorch

from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.remote.estimators import (
    DEFAULT_CPU_INSTANCE_SMALL,
    PYTORCH_LATEST_FRAMEWORK,
    PYTORCH_LATEST_PY_VERSION,
)
from syne_tune.util import repository_root_path

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Here, we specify the training script we want to tune what is reported in the training script
    # We can use the local or sagemaker backend when tuning remotely.
    # Using the local backend means that the remote instance will evaluate the trials locally.
    # Using the sagemaker backend means the remote instance will launch one sagemaker job per trial.
    # For runnig the actual tuning job we will rely on two other example scripts from our repo
    distribute_trials_on_sagemaker = True
    if distribute_trials_on_sagemaker:
        entry_point = Path(__file__).parent / "launch_height_sagemaker.py"
    else:
        entry_point = Path(__file__).parent / "launch_height_local_backend.py"

    max_wallclock_time = 60 * 60  # Run for 60 min

    # SageMaker back-end: Responsible for scheduling trials
    # Each trial is run as a separate SageMaker training job. This is useful for
    # expensive workloads, where all resources of an instance (or several ones)
    # are used for training. On the other hand, training job start-up overhead
    # is incurred for every trial.
    sm_estimator = PyTorch(
        entry_point=str(entry_point.name),
        source_dir=str(entry_point.parent),
        instance_type=DEFAULT_CPU_INSTANCE_SMALL,
        instance_count=1,
        role=get_execution_role(),
        dependencies=[
            str(repository_root_path() / "syne_tune"),
            str(repository_root_path() / "benchmarking"),
        ],
        max_run=max_wallclock_time,
        framework_version=PYTORCH_LATEST_FRAMEWORK,
        py_version=PYTORCH_LATEST_PY_VERSION,
        sagemaker_session=default_sagemaker_session(),
    )
    sm_estimator.fit(wait=False, job_name=f"launch-height-sagemaker-remotely-{np.random.randint(0, 2**31)}")
