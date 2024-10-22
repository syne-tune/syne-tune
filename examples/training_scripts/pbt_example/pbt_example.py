import numpy as np
import argparse
import logging
import json
import os
import random
import time

from syne_tune import Reporter
from syne_tune.constants import ST_CHECKPOINT_DIR

report = Reporter()


def pbt_function(config):
    """Toy PBT problem for benchmarking adaptive learning rate.

    The goal is to optimize this trainable's accuracy. The accuracy increases
    fastest at the optimal lr, which is a function of the current accuracy.

    The optimal lr schedule for this problem is the triangle wave as follows.
    Note that many lr schedules for real models also follow this shape:

     best lr
      ^
      |    /\
      |   /  \
      |  /    \
      | /      \
      ------------> accuracy

    In this problem, using PBT with a population of 2-4 is sufficient to
    roughly approximate this lr schedule. Higher population sizes will yield
    faster convergence. Training will not converge without PBT.
    """
    lr = config["lr"]
    checkpoint_dir = config.get(ST_CHECKPOINT_DIR)
    accuracy = 0.0  # end = 1000
    start = 1
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "checkpoint.json"), "r") as f:
            state = json.loads(f.read())
            accuracy = state["acc"]
            start = state["step"]

    midpoint = 100  # lr starts decreasing after acc > midpoint
    q_tolerance = 3  # penalize exceeding lr by more than this multiple
    noise_level = 2  # add gaussian noise to the acc increase
    # triangle wave:
    #  - start at 0.001 @ t=0,
    #  - peak at 0.01 @ t=midpoint,
    #  - end at 0.001 @ t=midpoint * 2,
    for step in range(start, 200):
        if accuracy < midpoint:
            optimal_lr = 0.01 * accuracy / midpoint
        else:
            optimal_lr = 0.01 - 0.01 * (accuracy - midpoint) / midpoint
        optimal_lr = min(0.01, max(0.001, optimal_lr))
        # Compute accuracy increase
        q_err = max(lr, optimal_lr) / min(lr, optimal_lr)
        if q_err < q_tolerance:
            accuracy += (1.0 / q_err) * random.random()
        elif lr > optimal_lr:
            accuracy -= (q_err - q_tolerance) * random.random()
        accuracy += noise_level * np.random.normal()
        accuracy = max(0, accuracy)
        # Save checkpoint
        if checkpoint_dir is not None:
            os.makedirs(os.path.join(checkpoint_dir), exist_ok=True)
            path = os.path.join(checkpoint_dir, "checkpoint.json")
            with open(path, "w") as f:
                f.write(json.dumps({"acc": accuracy, "step": step}))

        report(
            mean_accuracy=accuracy,
            cur_lr=lr,
            training_iteration=step,
            optimal_lr=optimal_lr,  # for debugging
            q_err=q_err,  # for debugging
            # done=accuracy > midpoint * 2  # this stops the training process
        )
        time.sleep(2)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)

    args, _ = parser.parse_known_args()

    params = vars(args)
    pbt_function(params)
