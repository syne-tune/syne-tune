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
import time
from typing import Optional, Dict, Any
import json
from pathlib import Path
import os
import numpy as np

from syne_tune import Reporter
from argparse import ArgumentParser

from syne_tune.config_space import randint
from syne_tune.constants import ST_CHECKPOINT_DIR


report = Reporter()


RESOURCE_ATTR = "epoch"

METRIC_ATTR = "mean_loss"

METRIC_MODE = "min"

MAX_RESOURCE_ATTR = "steps"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    with open(checkpoint_path, "r") as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: Path, epoch: int, value: float):
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({"epoch": epoch, "value": value}, f)


def train_height_delta(step: int, width: float, height: float, value: float) -> float:
    """
    For the original example, we have that

    .. math::
       f(t + 1) - f(t) = f(t) \cdot \frac{w}{10 + w \cdot t},

       f(0) = 10 + h / 10

    We implement an incremental version with a stochastic term.

    :param step: Step t, nonnegative int
    :param width: Width w, nonnegative
    :param height: Height h
    :param value: Value :math:`f(t - 1)` if :math:`t > 0`
    :return: New value :math:`f(t)`
    """
    u = 1.0 - 0.1 * np.random.rand()  # uniform(0.9, 1) multiplier
    if step == 0:
        return u * 10 + 0.1 * height
    else:
        return value * (1.0 + u * width / (width * (step - 1) + 10))


def height_config_space(
    max_steps: int, sleep_time: Optional[float] = None
) -> Dict[str, Any]:
    kwargs = {"sleep_time": sleep_time} if sleep_time is not None else dict()
    return {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        **kwargs,
    }


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--" + MAX_RESOURCE_ATTR, type=int)
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--sleep_time", type=float, default=0.1)
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)

    args, _ = parser.parse_known_args()

    width = args.width
    height = args.height
    checkpoint_dir = getattr(args, ST_CHECKPOINT_DIR)
    num_steps = getattr(args, MAX_RESOURCE_ATTR)
    start_step = 0
    value = 0.0
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.json"
        if checkpoint_path.exists():
            state = load_checkpoint(checkpoint_path)
            start_step = state["epoch"]
            value = state["value"]
    else:
        checkpoint_path = None

    for step in range(start_step, num_steps):
        # Sleep first, since results are returned at end of "epoch"
        time.sleep(args.sleep_time)
        # Feed the score back to Syne Tune.
        value = train_height_delta(step, width, height, value)
        epoch = step + 1
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, epoch, value)
        report(
            **{
                "step": step,
                METRIC_ATTR: value,
                RESOURCE_ATTR: epoch,
            }
        )
