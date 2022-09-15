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
import itertools
from tqdm import tqdm

from sagemaker.pytorch import PyTorch
import syne_tune

from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
from syne_tune.util import s3_experiment_path, random_string


if __name__ == "__main__":
    experiment_name = "glue-4"
    random_seed_offset = 31415627

    dataset_name = "rte"  # GLUE task
    model_type = "bert-base-cased"  # Default model used if not selected
    num_train_epochs = 3  # Maximum number of epochs
    max_runtime = 1800  # Each experiment runs for 30 mins
    instance_type = "ml.g4dn.xlarge"
    # instance_type = "ml.g4dn.12xlarge"
    # Useful if not all experiments could be started:
    skip_initial_experiments = 0

    # Compare selecting the model to fixing it to a default choice:
    model_selection = [False, True]
    # Compare 4 different HPO algorithms (2 multi-fidelity):
    optimizers = ["rs", "bo", "asha", "mobster"]
    # Each setup is repeated 10 times:
    num_runs = 10
    run_ids = list(range(num_runs))
    num_experiments = len(model_selection) * len(optimizers) * len(run_ids)
    # We need 1 GPU for each worker:
    if instance_type == "ml.g4dn.12xlarge":
        n_workers = 4
    else:
        n_workers = 1

    # Loop over all combinations and repetitions
    suffix = random_string(4)
    combinations = list(itertools.product(model_selection, optimizers, run_ids))
    for exp_id, (choose_model, optimizer, run_id) in tqdm(enumerate(combinations)):
        if exp_id < skip_initial_experiments:
            continue
        print(f"Experiment {exp_id} (of {num_experiments})")
        # Make sure that results (on S3) are written to different subdirectories.
        # Otherwise, the SM training job will download many previous results at
        # start
        tuner_name = f"{optimizer}-{choose_model}-{run_id}"
        # Results written to S3 under this path
        checkpoint_s3_uri = s3_experiment_path(
            experiment_name=experiment_name, tuner_name=tuner_name
        )
        print(f"Results stored to {checkpoint_s3_uri}")
        # We use a different seed for each `run_id`
        seed = (random_seed_offset + run_id) % (2**32)

        # Each experiment run is executed as SageMaker training job
        hyperparameters = {
            "run_id": run_id,
            "dataset": dataset_name,
            "model_type": model_type,
            "max_runtime": max_runtime,
            "num_train_epochs": num_train_epochs,
            "n_workers": n_workers,
            "optimizer": optimizer,
            "experiment_name": experiment_name,
            "choose_model": int(choose_model),
            "seed": seed,
        }

        # Pass Syne Tune sources as dependencies
        source_dir = str(Path(__file__).parent)
        entry_point = "hpo_main.py"
        dependencies = syne_tune.__path__ + [source_dir]
        # Latest PyTorch version (1.10):
        est = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type=instance_type,
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            volume_size=125,
            max_run=int(1.25 * max_runtime),
            role=get_execution_role(),
            dependencies=dependencies,
            disable_profiler=True,
            hyperparameters=hyperparameters,
        )

        job_name = f"{experiment_name}-{tuner_name}-{suffix}"
        print(f"Launching {job_name}")
        est.fit(wait=False, job_name=job_name)

    print(
        "\nLaunched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_name)}/ "
        f'~/syne-tune/{experiment_name}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )
