"""
Example similar to Raytune, https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/skopt_example.py
"""
import logging
import time
from typing import Any

from syne_tune import Reporter
from argparse import ArgumentParser

from syne_tune.config_space import randint


report = Reporter()


RESOURCE_ATTR = "epoch"

METRIC_ATTR = "mean_loss"

METRIC_MODE = "min"

MAX_RESOURCE_ATTR = "steps"


def train_height(step: int, width: float, height: float) -> float:
    return 100 / (10 + width * step) + 0.1 * height


def height_config_space(
    max_steps: int, sleep_time: float | None = None
) -> dict[str, Any]:
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

    args, _ = parser.parse_known_args()

    width = args.width
    height = args.height
    num_steps = getattr(args, MAX_RESOURCE_ATTR)
    for step in range(num_steps):
        # Sleep first, since results are returned at end of "epoch"
        time.sleep(args.sleep_time)
        # Feed the score back to Syne Tune.
        dummy_score = train_height(step, width, height)
        report(
            **{
                "step": step,
                METRIC_ATTR: dummy_score,
                RESOURCE_ATTR: step + 1,
            }
        )
