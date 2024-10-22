import logging
import numpy as np

from syne_tune import Reporter
from argparse import ArgumentParser


report = Reporter()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    parser.add_argument("--cost", type=float)

    args, _ = parser.parse_known_args()

    x1 = args.x1
    x2 = args.x2
    cost = args.cost
    r = 6
    objective_value = (
        (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - r) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    cost_value = x2**cost  # the larger x2, the more costly the evaluation
    report(objective=-objective_value, elapsed_time=cost_value)
