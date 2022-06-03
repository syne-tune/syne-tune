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
Entrypoint script that allows to launch a tuning job remotely.
It loads the tuner from a specified path then runs it.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path

from syne_tune import Tuner


def decode_bool(hp: str):
    # Sagemaker encodes hyperparameters in estimators as literals which are compatible with Python,
    # except for true and false that are respectively encoded as 'True' and 'False'.
    assert hp in ["True", "False"]
    return hp == "True"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tuner_path", type=str, default="tuner/")
    parser.add_argument("--store_logs", type=str, default="False")
    parser.add_argument("--log_level", type=int, default=logging.INFO)
    parser.add_argument("--no_tuner_logging", type=str, default="False")
    args, _ = parser.parse_known_args()

    root = logging.getLogger()
    root.setLevel(args.log_level)

    args.store_logs = decode_bool(args.store_logs)
    args.no_tuner_logging = decode_bool(args.no_tuner_logging)

    tuner_path = Path(args.tuner_path)
    logging.info(f"load tuner from path {args.tuner_path}")
    tuner = Tuner.load(tuner_path)

    # The output of the tuner (results, metadata, tuner state) is written into SageMaker checkpoint directory
    # which is synced regularly by SageMaker so that results are updated continuously
    tuner.tuner_path = Path("/opt/ml/checkpoints/")

    # The logs/checkpoints of trials are persisted to the checkpoint directory only where store_logs is True
    if args.store_logs:
        # inform the backend of the desired path so that logs are persisted
        tuner.trial_backend.set_path(results_root=tuner.tuner_path)
    else:
        # sets a path where logs will not be stored, we use "/opt/ml/input/data/training" as it is mounted
        # on a partition that is larger than "~/"
        path = Path("/opt/ml/input/data/")
        path.mkdir(parents=True, exist_ok=True)
        tuner.trial_backend.set_path(results_root=str(path), tuner_name=tuner.name)

    # Run the tuner on the sagemaker instance. If the simulation back-end is
    # used, this needs a specific callback
    if args.no_tuner_logging == "True":
        logging.getLogger("syne_tune.tuner").setLevel(logging.ERROR)
    logging.info("starting remote tuning")
    tuner.run()
