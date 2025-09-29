import logging
import time
from typing import Any
from argparse import ArgumentParser

from syne_tune import Reporter
from syne_tune.config_space import randint
from syne_tune.utils import add_config_json_to_argparse, load_config_json


report = Reporter()


TIME_ATTR = "epoch"

METRIC_ATTR = "mean_loss"

METRIC_MODE = "min"

MAX_RESOURCE_ATTR = "steps"


def train_height(step: int, width: float, height: float) -> float:
    return 100 / (10 + width * step) + 0.1 * height


def height_config_space(
    max_steps: int, sleep_time: float | None = None
) -> dict[str, Any]:
    if sleep_time is None:
        sleep_time = 0.1
    return {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
        "sleep_time": sleep_time,
        "list_arg": ["this", "is", "a", "list", 1, 2, 3],
        "dict_arg": {
            "this": 27,
            "is": [1, 2, 3],
            "a": "dictionary",
            "even": {
                "a": 0,
                "nested": 1,
                "one": 2,
            },
        },
    }


def _check_extra_args(config: dict[str, Any]):
    config_space = height_config_space(5)
    for k in ("list_arg", "dict_arg"):
        a, b = config[k], config_space[k]
        assert a == b, (k, a, b)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    # Append required argument(s):
    add_config_json_to_argparse(parser)
    args, _ = parser.parse_known_args()
    # Loads config JSON and merges with ``args``
    config = load_config_json(vars(args))

    # Check that args with complex types have been received correctly
    _check_extra_args(config)
    width = config["width"]
    height = config["height"]
    sleep_time = config["sleep_time"]
    num_steps = config[MAX_RESOURCE_ATTR]
    for step in range(num_steps):
        # Sleep first, since results are returned at end of "epoch"
        time.sleep(sleep_time)
        # Feed the score back to Syne Tune.
        dummy_score = train_height(step, width, height)
        report(
            **{
                "step": step,
                METRIC_ATTR: dummy_score,
                TIME_ATTR: step + 1,
            }
        )
