import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from sagemaker_tune.report import Reporter

def f(t, theta):
    # Function drawing upper-right circles with radius set to `t` and with center set at
    # (-t, -t). `t` is interpreted as a fidelity and larger `t` corresponds to larger radius and better candidates.
    # The optimal multiobjective solution should select theta uniformly from [0, pi/2].
    return {
        "y1": - t + t * np.cos(theta),
        "y2": - t + t * np.sin(theta),
    }


def plot_function():
    ts = np.linspace(0, 27, num=5)
    thetas = np.linspace(0, 1) * np.pi / 2
    y1s = []
    y2s = []
    for t in ts:
        for theta in thetas:
            res = f(t, theta)
            y1s.append(res["y1"])
            y2s.append(res["y2"])
    plt.scatter(y1s, y2s)
    plt.show()


if __name__ == '__main__':
    # plot_function()
    parser = ArgumentParser()
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--theta', type=float, required=True)
    parser.add_argument('--sleep_time', type=float, required=False, default=0.1)
    args, _ = parser.parse_known_args()

    assert 0 <= args.theta < np.pi / 2
    reporter = Reporter()
    for step in range(args.steps):
        y = f(t=step, theta=args.theta)
        reporter(step=step, **y)
        time.sleep(args.sleep_time)