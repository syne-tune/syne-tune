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
from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_neuralband.baselines import methods, Methods
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


benchmark_names = [
    "fcnet-protein",
    "fcnet-naval",
    "fcnet-parkinsons",
    "fcnet-slice",
    "nas201-cifar10",
    "nas201-cifar100",
    "nas201-ImageNet16-120",
    "lcbench-APSFailure",
    "lcbench-Amazon-employee-access",
    "lcbench-Australian",
    "lcbench-Fashion-MNIST",
    "lcbench-KDDCup09-appetency",
    "lcbench-MiniBooNE",
    "lcbench-adult",
    "lcbench-airlines",
    "lcbench-albert",
    "lcbench-bank-marketing",
    "lcbench-car",
    "lcbench-christine",
    "lcbench-cnae-9",
    "lcbench-connect-4",
    "lcbench-covertype",
    "lcbench-credit-g",
    "lcbench-dionis",
    "lcbench-fabert",
    "lcbench-helena",
    "lcbench-higgs",
    "lcbench-jannis",
    "lcbench-jasmine",
    "lcbench-kc1",
    "lcbench-kr-vs-kp",
    "lcbench-mfeat-factors",
    "lcbench-nomao",
    "lcbench-numerai286",
    "lcbench-phoneme",
    "lcbench-segment",
    "lcbench-shuttle",
    "lcbench-sylvine",
    "lcbench-vehicle",
    "lcbench-volkert",
]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    hash = random_string(4)
    num_seeds = 5
    for method in methods.keys():
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=method, experiment_name=experiment_tag
            ),
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )

        if (
            method == Methods.NeuralBandSH
            or method == Methods.NeuralBandHB
            or method == Methods.MOBSTER
        ):
            for seed in range(num_seeds):
                for benchm in benchmark_names:
                    print(f"{experiment_tag}-{method}-{benchm}-{seed}")
                    sm_args["hyperparameters"] = {
                        "experiment_tag": experiment_tag,
                        "num_seeds": seed,
                        "run_all_seed": 0,
                        "method": method,
                        "benchmark": benchm,
                    }
                    est = PyTorch(**sm_args)
                    est.fit(
                        job_name=f"{experiment_tag}-{method}-{benchm}-{seed}-{hash}",
                        wait=False,
                    )

        elif (
            method == Methods.NeuralBand_UCB
            or method == Methods.NeuralBand_TS
            or method == Methods.NeuralBandEpsilon
        ):
            for seed in range(num_seeds):
                for benchm in benchmark_names:
                    print(f"{experiment_tag}-{method}-{benchm}-{seed}")
                    sm_args["hyperparameters"] = {
                        "experiment_tag": experiment_tag,
                        "num_seeds": seed,
                        "run_all_seed": 0,
                        "method": method,
                        "benchmark": benchm,
                    }
                    est = PyTorch(**sm_args)
                    est.fit(
                        job_name=f"{experiment_tag}-{method}-{benchm}-{seed}-{hash}",
                        wait=False,
                    )

        elif method == Methods.RS:
            print(f"{experiment_tag}-{method}")
            sm_args["hyperparameters"] = {
                "experiment_tag": experiment_tag,
                "num_seeds": num_seeds,
                "run_all_seed": 1,
                "method": method,
            }
            est = PyTorch(**sm_args)
            est.fit(job_name=f"{experiment_tag}-{method}-{hash}", wait=False)
        else:
            for seed in range(num_seeds):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag,
                    "num_seeds": seed,
                    "run_all_seed": 0,
                    "method": method,
                }
                est = PyTorch(**sm_args)
                est.fit(job_name=f"{experiment_tag}-{method}-{seed}-{hash}", wait=False)
