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
Script for testing copying of checkpoints in SageMaker backend.
"""

import argparse
import logging
import os
import json
from pathlib import Path
from typing import Optional

from syne_tune.constants import ST_CHECKPOINT_DIR
from syne_tune import Reporter


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    result = dict()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            result = json.load(f)
    return result


def save_checkpoint(checkpoint_path: Path, content: dict):
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(content, f)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_id', type=int, required=True)
    parser.add_argument('--value1', type=int, required=True)
    parser.add_argument('--value2', type=int, required=True)

    # convention the path where to serialize and deserialize is given as checkpoint-dir
    parser.add_argument(f'--{ST_CHECKPOINT_DIR}', type=str, default="./")

    args, _ = parser.parse_known_args()

    report = Reporter()
    error_msg = None

    # Try to load checkpoint (may not be present)
    checkpoint1_path = Path(getattr(args, ST_CHECKPOINT_DIR)) / "checkpoint1.json"
    checkpoint2_path = Path(getattr(args, ST_CHECKPOINT_DIR)) / "subdir" / "checkpoint2.json"
    old_checkpoint1 = load_checkpoint(checkpoint1_path)
    old_checkpoint2 = dict()
    if old_checkpoint1:
        root.info(f"Loaded checkpoint {checkpoint1_path}:\n{old_checkpoint1}")
        old_checkpoint2 = load_checkpoint(checkpoint2_path)
        if not old_checkpoint2:
            error_msg = f"Found checkpoint at {checkpoint1_path}, but not at {checkpoint2_path}"
        else:
            root.info(f"Loaded checkpoint {checkpoint2_path}:\n{old_checkpoint2}")

    if error_msg is None:
        # Write new checkpoints
        checkpoint1 = {
            'trial_id': args.trial_id,
            'value1': args.value1,
        }
        for k, v in old_checkpoint1.items():
            checkpoint1['parent_' + k] = v
        save_checkpoint(checkpoint1_path, checkpoint1)
        root.info(f"Wrote checkpoint {checkpoint1_path}:\n{checkpoint1}")
        checkpoint2 = {
            'trial_id': args.trial_id,
            'value2': args.value2,
        }
        for k, v in old_checkpoint2.items():
            checkpoint2['parent_' + k] = v
        save_checkpoint(checkpoint2_path, checkpoint2)
        root.info(f"Wrote checkpoint {checkpoint2_path}:\n{checkpoint2}")
        result = dict(
            trial_id=args.trial_id,
            value1=args.value1,
            value2=args.value2)
        for old_cp, prefix in [(old_checkpoint1, 'parent1_'),
                               (old_checkpoint2, 'parent2_')]:
            for k, v in old_cp.items():
                result[prefix + k] = v
        root.info(f"Report back:\n{result}")
        report(**result)
    else:
        report(error_msg=error_msg)
