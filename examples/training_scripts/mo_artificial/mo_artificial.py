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
import time
from argparse import ArgumentParser

import numpy as np

from syne_tune import Reporter

def f(t, theta):
    # Function drawing upper-right circles with radius set to `t` and with center set at
    # (-t, -t). `t` is interpreted as a fidelity and larger `t` corresponds to larger radius and better candidates.
    # The optimal multiobjective solution should select theta uniformly from [0, pi/2].
    return {
        "y1": - t + t * np.cos(theta),
        "y2": - t + t * np.sin(theta),
    }


def plot_function():
    import matplotlib.pyplot as plt
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