"""
Script used for testing checkpointing.
The main reports "nothing" if the checkpoint folder is empty and writes the name given in argument to the checkpoint.
If a checkpoint is present, it reports the content of the checkpointing folder.
"""

import argparse
import logging
import os
from pathlib import Path

from sagemaker_tune.constants import SMT_CHECKPOINT_DIR
from sagemaker_tune.report import Reporter


def load_checkpoint(checkpoint_path: Path):
    with open(checkpoint_path, "r") as f:
        return f.readline()


def save_checkpoint(checkpoint_path: Path, content: str):
    with open(checkpoint_path, "w") as f:
        f.write(content)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    # convention the path where to serialize and deserialize is given as checkpoint-dir
    parser.add_argument(f'--{SMT_CHECKPOINT_DIR}', type=str, default="./")

    args, _ = parser.parse_known_args()

    checkpoint_path = Path(getattr(args, SMT_CHECKPOINT_DIR)) / "checkpoint.txt"
    os.makedirs(checkpoint_path.parent, exist_ok=True)

    if checkpoint_path.exists():
        checkpoint_content = load_checkpoint(checkpoint_path)
    else:
        checkpoint_content = "nothing"

    report = Reporter()
    report(checkpoint_content=checkpoint_content)

    save_checkpoint(checkpoint_path, args.name)
