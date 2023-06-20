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
from simopt.base import Problem
import numpy as np
from typing import List


def evaluate_problem_price(prob: Problem, xx, rand_gen, reps: int = 1):
    """
    Function that evaluates a feature vector xx on a problem prob,
    using the random number generator rand_gen

    Parameters
        ----------
        prob
            Problem from SimOpt library
        xx
            feature vector of prices, one per item
        rand_gen
            random number generator from MRG32k3a
        reps
            how many simulations to average over
    """
    prob.model.factors["price"] = xx
    utility_raw = np.array(prob.model.factors["c_utility"])

    # Modify the utility factors based on price
    utility_scaled = utility_raw - xx / 2
    prob.model.factors["c_utility"] = list(utility_scaled)

    # Simulate the feature vector performance
    profit = np.mean(
        [prob.model.replicate([rand_gen])[0]["profit"] for _ in range(reps)]
    )

    # Reset the utility factors
    prob.model.factors["c_utility"] = list(utility_raw)
    return profit


def plot_problem(
    ax,
    ii: int,
    results: List[float],
    task_num: int,
    N: int,
    num_cols: int,
    vmin: float,
    vmax: float,
):
    """
    Function to plot 2D optimisation landscape
    """
    # Plot values
    cb = ax.imshow(np.reshape(results, (N, N)).T, origin="lower", vmin=vmin, vmax=vmax)

    # Set title
    ax.set_title("Task %s" % task_num)

    # Add crosses to optimal point(s)
    y_axs, x_axs = np.nonzero(
        np.reshape(results, (N, N)).T == np.max(np.reshape(results, (N, N)).T)
    )
    for jj in range(len(y_axs)):
        ax.scatter(
            x_axs[jj],
            y_axs[jj],
            marker="X",
            c="k",
            s=100,
            edgecolors="white",
            linewidths=1.75,
        )

    # Add y-axis label to leftmost plot(s)
    if ii % num_cols == 0:
        ax.set_ylabel("Item 2 price")

    # Add x-axis label
    ax.set_xlabel("Item 1 price")

    # Set tick values
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([0, 10, 20])

    # Return structure to plot colour bar
    return cb
