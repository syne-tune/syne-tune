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
# launch_collect_results.py

import argparse
import datetime
import time

from experiment_master_file import experiments_meta_dict

# Note: This uses internal development names to identify the methods.
# These are translated into the names used in the paper in plotting/plotting_helper.py


def main(run_locally):

    optimisers_all = [
        ("BoundingBox", "Transfer"),
        ("ZeroShot", "Transfer"),
        ("Quantiles", "Transfer"),
        ("BayesianOptimization", "Naive"),
        ("WarmBO", "Transfer"),
        ("WarmBOShuffled", "Transfer"),
        ("BoTorchTransfer", "Transfer"),
        ("RandomSearch", "Naive"),
        ("PrevBO", "Transfer"),
        ("PrevNoBO", "Transfer"),
    ]

    optimisers_subset = [
        ("WarmBO", "Transfer"),
        ("WarmBOShuffled", "Transfer"),
        ("RandomSearch", "Naive"),
        ("PrevBO", "Transfer"),
        ("PrevNoBO", "Transfer"),
    ]

    experiments = [
        ("SimOpt", optimisers_all),
        ("XGBoost", optimisers_all),
        ("YAHPO_auc_svm_1220", optimisers_all),
        ("YAHPO_auc_svm_458", optimisers_subset),
        ("YAHPO_auc_aknn_4538", optimisers_subset),
        ("YAHPO_auc_aknn_41138", optimisers_subset),
        ("YAHPO_auc_ranger_4154", optimisers_subset),
        ("YAHPO_auc_ranger_40978", optimisers_subset),
        ("YAHPO_auc_glmnet_375", optimisers_subset),
        ("YAHPO_auc_glmnet_40981", optimisers_subset),
    ]

    seed_start = 0
    seed_end = 49
    points_per_task = 25

    xgboost_res_file_long = (
        "xgboost_experiment_results/random-mnist/aggregated_experiments.json"
    )
    xgboost_res_file_short = "aggregated_experiments.json"

    if run_locally:
        from collect_results import collect_res

        xgboost_res_file = xgboost_res_file_long
        print("Running experiments locally. This will take a while.")
    else:
        from pathlib import Path
        from sagemaker.pytorch import PyTorch
        import boto3
        import sagemaker
        import shutil
        import os

        # local file with configs for running experiments on Sagemaker
        from sagemaker_config import role, profile_name, alias, s3bucket

        session = boto3.Session(profile_name=profile_name)
        sm_session = sagemaker.Session(boto_session=session)

        repo_main_dir = Path(os.path.dirname(os.path.realpath(__file__)))

        source_dir = repo_main_dir / "sagemaker_source_dir_temp_code_to_upload"
        if source_dir.exists():
            shutil.rmtree(str(source_dir))
        source_dir.mkdir(parents=True, exist_ok=True)

        # Only needed for SimOpt
        shutil.copytree(repo_main_dir / "simopt", source_dir / "simopt")

        # Copy required files
        shutil.copy(
            repo_main_dir / "requirements_on_sagemaker.txt",
            source_dir / "requirements.txt",
        )
        shutil.copy(repo_main_dir / "collect_results.py", source_dir)
        shutil.copy(repo_main_dir / "blackbox_helper.py", source_dir)
        shutil.copy(repo_main_dir / "bo_warm_transfer.py", source_dir)
        shutil.copy(repo_main_dir / "backend_definitions_dict.py", source_dir)

        # Only needed for XGBoost
        shutil.copy(repo_main_dir / xgboost_res_file_long, source_dir)
        xgboost_res_file = xgboost_res_file_short

    for experiment, optimiser_set in experiments:
        experiment_meta_data = experiments_meta_dict[experiment]
        backend = experiment_meta_data["backend"]
        for optimiser, optimiser_type in optimiser_set:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            hyperparameters = {
                "seed_start": seed_start,
                "seed_end": seed_end,
                "timestamp": timestamp,
                "points_per_task": points_per_task,
                "optimiser": optimiser,
                "optimiser_type": optimiser_type,
                "backend": backend,
            }

            if backend == "SimOpt":
                hyperparameters["simopt_backend_file"] = experiment_meta_data[
                    "simopt_backend_file"
                ]
            if backend == "XGBoost":
                hyperparameters["xgboost_res_file"] = xgboost_res_file
            if backend == "YAHPO":
                hyperparameters["yahpo_dataset"] = experiment_meta_data["yahpo_dataset"]
                hyperparameters["yahpo_scenario"] = experiment_meta_data[
                    "yahpo_scenario"
                ]

            if run_locally:
                hyperparameters["run_locally"] = True
                collect_res(**hyperparameters)

                print("Finished experiment %s, optimiser %s" % (experiment, optimiser))
            else:
                print(
                    "Starting experiment %s, optimiser %s remotely"
                    % (experiment, optimiser)
                )
                estimator = PyTorch(
                    entry_point="collect_results.py",
                    source_dir=str(source_dir),
                    py_version="py38",
                    framework_version="1.10.0",
                    role=role,
                    instance_type="ml.c5.18xlarge",
                    instance_count=1,
                    disable_profiler=True,
                    sagemaker_session=sm_session,
                    hyperparameters=hyperparameters,
                    output_path="s3://%s/outputs/" % s3bucket,
                )
                estimator.fit(
                    inputs={"ds_path": "s3://%s/datasets/" % s3bucket},
                    wait=False,
                    job_name="%s-collect-res-%s-%s-%s"
                    % (alias, timestamp, optimiser[:11], backend),
                )

            time.sleep(1)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="run experiments locally.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.local)
