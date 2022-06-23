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
