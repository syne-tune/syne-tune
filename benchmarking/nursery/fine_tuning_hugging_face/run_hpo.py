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
import logging

from sagemaker.huggingface import HuggingFace

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend, SageMakerBackend
from syne_tune.config_space import randint, loguniform, uniform
from syne_tune.optimizer.baselines import BayesianOptimization
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.experiments import load_experiment
from syne_tune.constants import ST_TUNER_TIME

logging.getLogger().setLevel(logging.INFO)

config_space = {
    "learning_rate": loguniform(1e-6, 1e-4),
    "per_device_train_batch_size": 8,
    "warmup_ratio": uniform(0, 0.5),
    "epochs": 3,
    "weight_decay": uniform(0, 1e-1),
    "adam_beta1": uniform(0.0, 0.9999),
    "adam_beta2": uniform(0.0, 0.9999),
    "adam_epsilon": loguniform(1e-10, 1e-6),
    "max_grad_norm": uniform(0, 2),
}

# Default hyperparameter configuration from HuggingFace as specified in TrainingArguments
default_config = {
    "learning_rate": 5e-5,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
}

entry_point = "multiple_choice_on_swag.py"
run_locally = False

if run_locally:
    trial_backend = LocalBackend(entry_point=entry_point, rotate_gpus=False)
    n_workers = 1  # if we only have a single GPU, we can also only run a single worker
else:
    trial_backend = SageMakerBackend(
        sm_estimator=HuggingFace(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            entry_point=str(entry_point),
            role=get_execution_role(),
            transformers_version="4.26",
            pytorch_version="1.13",
            py_version="py39",
            max_run=10 * 600,
            sagemaker_session=default_sagemaker_session(),
            disable_profiler=True,
            debugger_hook_config=False,
        ),
        metrics_names=["eval_accuracy"],
    )
    n_workers = 4  # runs 4 SageMaker Training Jobs in parallel


tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=BayesianOptimization(
        config_space,
        metric="eval_accuracy",
        mode="max",
        points_to_evaluate=[
            default_config
        ],  # let's start with the default configuration
    ),
    stop_criterion=StoppingCriterion(max_wallclock_time=3600 * 5),
    n_workers=n_workers,  # how many trials are evaluated in parallel
)
tuner.run()


experiment = load_experiment(tuner.name)
results = experiment.results
metric_name = experiment.metadata["metric_names"][0]

best_config = experiment.best_config()

for trial_id, df_trial in results.groupby("trial_id"):
    if trial_id == best_config["trial_id"]:
        plt.plot(
            df_trial[ST_TUNER_TIME],
            df_trial[metric_name],
            marker="o",
            color="red",
            label="best",
            linewidth=3,
        )
    elif trial_id == 0:
        plt.plot(
            df_trial[ST_TUNER_TIME],
            df_trial[metric_name],
            marker="o",
            color="blue",
            label="default",
            linewidth=3,
        )
    else:
        plt.plot(
            df_trial[ST_TUNER_TIME],
            df_trial[metric_name],
            marker="o",
            color="black",
            alpha=0.4,
            linewidth=1,
        )

plt.xlabel("wall-clock time (seconds)")
plt.ylabel(metric_name.replace("_", "-"))
plt.title("Fine-tuning on SWAG")
plt.grid(alpha=0.4)
plt.legend()
plt.show()
