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
# launch.py
import argparse
import datetime
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def main(run_locally):

    dataset = "mnist"
    hyp_dict_file = "hyperparameters_file_random_num-1000_seed-13.p"
    num_hyp_pars = 1000

    # Split into multiple runs
    start_ids, end_ids = [], []
    for ii in range(num_hyp_pars // 100):
        start_ids.append(ii * 100)
        end_ids.append((ii + 1) * 100 - 1)
    if num_hyp_pars % 100 != 0:
        start_ids.append((num_hyp_pars // 100) * 100)
        end_ids.append(num_hyp_pars - 1)

    if run_locally:
        from XGBoost_script import evaluate_XGBoost
    else:
        from pathlib import Path
        from sagemaker.pytorch import PyTorch
        import boto3
        import sagemaker

        # local file with configs for running experiments on Sagemaker
        from sagemaker_config import role, profile_name, alias, s3bucket

        session = boto3.Session(profile_name=profile_name)
        sm_session = sagemaker.Session(boto_session=session)

    print("Experiment broken into following sub-experiments by hyperparameter ids.")
    for hyp_id_start, hyp_id_end in zip(start_ids, end_ids):
        print(hyp_id_start, hyp_id_end)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        hyperparameters = {
            "hyp_id_start": hyp_id_start,
            "hyp_id_end": hyp_id_end,
            "hyp_dict_file": hyp_dict_file,
            "timestamp": timestamp,
            "dataset": dataset,
        }

        if run_locally:
            hyperparameters["run_locally"] = True
            evaluate_XGBoost(**hyperparameters)
        else:
            estimator = PyTorch(
                entry_point="XGBoost_script.py",
                source_dir=str(Path(__file__).parent),
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
                job_name="%s-xgboost-%s" % (alias, timestamp),
            )

        time.sleep(1)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_true", help="collect XGBoost evaluations locally."
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.local)
