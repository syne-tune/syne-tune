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
import json
import logging
from pathlib import Path

from syne_tune import Reporter
from argparse import ArgumentParser

report = Reporter()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(f"--st_checkpoint_dir", type=str)
    args, _ = parser.parse_known_args()

    # gets hyperparameters that are written into {trial_path}/config.json
    # note: only works with LocalBackend for now.
    trial_path = Path(args.st_checkpoint_dir).parent
    with open(Path(args.st_checkpoint_dir).parent / "config.json", "r") as f:
        hyperparameters = json.load(f)

    report(error=hyperparameters["x"] ** 2)
