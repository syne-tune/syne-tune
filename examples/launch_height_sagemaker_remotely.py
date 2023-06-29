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
from argparse import ArgumentParser

from sagemaker.pytorch import PyTorch

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.config_space import randint
from examples.training_scripts.height_example.train_height import (
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.remote.constants import (
    DEFAULT_CPU_INSTANCE_SMALL,
    PYTORCH_LATEST_FRAMEWORK,
    PYTORCH_LATEST_PY_VERSION,
)
from syne_tune.remote.remote_launcher import RemoteLauncher

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--use_sagemaker_backend", type=int, default=0)
    args = parser.parse_args()
    use_sagemaker_backend = bool(args.use_sagemaker_backend)

    max_steps = 100
    n_workers = 4

    config_space = {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    # We can use the local or sagemaker backend when tuning remotely.
    # Using the local backend means that the remote instance will evaluate the trials locally.
    # Using the sagemaker backend means the remote instance will launch one sagemaker job per trial.
    if use_sagemaker_backend:
        trial_backend = SageMakerBackend(
            sm_estimator=PyTorch(
                instance_type=DEFAULT_CPU_INSTANCE_SMALL,
                instance_count=1,
                framework_version=PYTORCH_LATEST_FRAMEWORK,
                py_version=PYTORCH_LATEST_PY_VERSION,
                entry_point=entry_point,
                role=get_execution_role(),
                max_run=10 * 60,
                base_job_name="hpo-height",
                sagemaker_session=default_sagemaker_session(),
                disable_profiler=True,
                debugger_hook_config=False,
            ),
        )
    else:
        trial_backend = LocalBackend(entry_point=entry_point)

    num_seeds = 1 if use_sagemaker_backend else 2
    for seed in range(num_seeds):
        # Random search without stopping
        scheduler = RandomSearch(
            config_space, mode=METRIC_MODE, metric=METRIC_ATTR, random_seed=seed
        )

        tuner = RemoteLauncher(
            tuner=Tuner(
                trial_backend=trial_backend,
                scheduler=scheduler,
                n_workers=n_workers,
                tuner_name="height-tuning",
                stop_criterion=StoppingCriterion(max_wallclock_time=600),
            ),
            # Extra arguments describing the resource of the remote tuning instance and whether we want to wait
            # the tuning to finish. The instance-type where the tuning job runs can be different than the
            # instance-type used for evaluating the training jobs.
            instance_type=DEFAULT_CPU_INSTANCE_SMALL,
            # We can specify a custom container to use with this launcher with <image_uri=TK>
            # otherwise a sagemaker pre-build will be used
        )

        tuner.run(wait=False)
