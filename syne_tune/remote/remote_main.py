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

from syne_tune.tuner import Tuner


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tuner_path', type=str, default="tuner/")
    parser.add_argument('--store_logs', dest='store_logs', action='store_true', default=False)
    parser.add_argument('--log_level', type=int, default=logging.INFO)
    parser.add_argument('--no_tuner_logging', type=str, default='False')
    args, _ = parser.parse_known_args()

    root = logging.getLogger()
    root.setLevel(args.log_level)

    tuner_path = Path(args.tuner_path)
    logging.info(f"load tuner from path {args.tuner_path}")
    tuner = Tuner.load(tuner_path)

    if args.store_logs:
        # inform the backend of the desired path so that logs are persisted
        tuner.backend.set_path(results_root='/opt/ml/checkpoints', tuner_name=tuner.name)

    # Run the tuner on the sagemaker instance. If the simulation back-end is
    # used, this needs a specific callback
    if args.no_tuner_logging == 'True':
        logging.getLogger('syne_tune.tuner').setLevel(logging.ERROR)
    logging.info("starting remote tuning")
    tuner.run()
