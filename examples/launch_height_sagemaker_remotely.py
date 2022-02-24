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

from sagemaker.pytorch import PyTorch

from syne_tune.backend import LocalBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.backend import SageMakerBackend
from syne_tune.config_space import randint
from syne_tune import StoppingCriterion
from syne_tune import Tuner

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = str(
        Path(__file__).parent / "training_scripts" / "height_example" /
        "train_height.py")
    mode = "min"
    metric = "mean_loss"

    # We can use the local or sagemaker backend when tuning remotely.
    # Using the local backend means that the remote instance will evaluate the trials locally.
    # Using the sagemaker backend means the remote instance will launch one sagemaker job per trial.
    distribute_trials_on_sagemaker = False
    if distribute_trials_on_sagemaker:
        trial_backend = SageMakerBackend(
            # we tune a PyTorch Framework from Sagemaker
            sm_estimator=PyTorch(
                entry_point=entry_point,
                instance_type="ml.m5.xlarge",
                instance_count=1,
                role=get_execution_role(),
                max_run=10 * 60,
                framework_version='1.6',
                py_version='py3',
                base_job_name="hpo-height",
            ),
        )
    else:
        trial_backend = LocalBackend(entry_point=entry_point)

    for seed in range(2):
        # Random search without stopping
        scheduler = RandomSearch(
            config_space,
            mode=mode,
            metric=metric,
            random_seed=seed
        )

        tuner = RemoteLauncher(
            tuner=Tuner(
                trial_backend=trial_backend,
                scheduler=scheduler,
                n_workers=n_workers,
                tuner_name="height-tuning",
                stop_criterion=StoppingCriterion(max_wallclock_time=600),
            ),
            # Extra arguments describing the ressource of the remote tuning instance and whether we want to wait
            # the tuning to finish. The instance-type where the tuning job runs can be different than the
            # instance-type used for evaluating the training jobs.
            instance_type='ml.m5.large',
        )

        tuner.run(wait=False)
