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
"""
Example collecting evaluations and using them for transfer learning on a
related task.
"""
from examples.training_scripts.height_example.train_height import (
    height_config_space,
    METRIC_ATTR,
    METRIC_MODE,
)

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import BayesianOptimization, ZeroShotTransfer
from syne_tune.optimizer.schedulers import FIFOScheduler

from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
    BoundingBox,
)

from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    QuantileBasedSurrogateSearcher,
)

import argparse
import copy
import numpy as np
from pathlib import Path


def add_labels(ax, conf_space, title):
    ax.legend()
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_xlim([conf_space["width"].lower - 1, conf_space["width"].upper + 1])
    ax.set_ylim([conf_space["height"].lower - 10, conf_space["height"].upper + 10])
    ax.set_title(title)


def scatter_space_exploration(ax, task_hyps, max_trials, label, color=None):
    ax.scatter(
        task_hyps["width"][:max_trials],
        task_hyps["height"][:max_trials],
        alpha=0.4,
        label=label,
        color=color,
    )


colours = {
    "BayesianOptimization": "C0",
    "BoundingBox": "C1",
    "ZeroShotTransfer": "C2",
    "Quantiles": "C3",
}


def plot_last_task(max_trials, df, label, metric, color):
    max_tr = min(max_trials, len(df))
    plt.scatter(range(max_tr), df[metric][:max_tr], label=label, color=color)
    plt.plot([np.min(df[metric][:ii]) for ii in range(1, max_trials + 1)], color=color)


def filter_completed(df):
    # Filter out runs that didn't finish
    return df[df["status"] == "Completed"].reset_index()


def extract_transferable_evaluations(df, metric, config_space):
    """
    Take a dataframe from a tuner run, filter it and generate
    TransferLearningTaskEvaluations from it
    """
    filter_df = filter_completed(df)

    return TransferLearningTaskEvaluations(
        configuration_space=config_space,
        hyperparameters=filter_df[config_space.keys()],
        objectives_names=[metric],
        # objectives_evaluations need to be of shape
        # (num_evals, num_seeds, num_fidelities, num_objectives)
        # We only have one seed, fidelity and objective
        objectives_evaluations=np.array(filter_df[metric], ndmin=4).T,
    )


def run_scheduler_on_task(entry_point, scheduler, max_trials):
    """
    Take a scheduler and run it for max_trials on the backend specified by entry_point
    Return a dataframe of the optimisation results
    """
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=str(entry_point)),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_num_trials_finished=max_trials),
        n_workers=4,
        sleep_time=0.001,
    )
    tuner.run()

    return tuner.tuning_status.get_dataframe()


def init_scheduler(
    scheduler_str, max_steps, seed, mode, metric, transfer_learning_evaluations
):
    """
    Initialise the scheduler
    """
    kwargs = {
        "metric": metric,
        "config_space": height_config_space(max_steps=max_steps),
        "mode": mode,
        "random_seed": seed,
    }
    kwargs_w_trans = copy.deepcopy(kwargs)
    kwargs_w_trans["transfer_learning_evaluations"] = transfer_learning_evaluations

    if scheduler_str == "BayesianOptimization":
        return BayesianOptimization(**kwargs)

    if scheduler_str == "ZeroShotTransfer":
        return ZeroShotTransfer(use_surrogates=True, **kwargs_w_trans)

    if scheduler_str == "Quantiles":
        return FIFOScheduler(
            searcher=QuantileBasedSurrogateSearcher(**kwargs_w_trans),
            **kwargs,
        )

    if scheduler_str == "BoundingBox":
        kwargs_sched_fun = {key: kwargs[key] for key in kwargs if key != "config_space"}
        kwargs_w_trans[
            "scheduler_fun"
        ] = lambda new_config_space, mode, metric: BayesianOptimization(
            new_config_space,
            **kwargs_sched_fun,
        )
        del kwargs_w_trans["random_seed"]
        return BoundingBox(**kwargs_w_trans)
    raise ValueError("scheduler_str not recognised")


if __name__ == "__main__":

    max_trials = 10
    np.random.seed(1)
    # Use train_height backend for our tests
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    # Collect evaluations on preliminary tasks
    transfer_learning_evaluations = {}
    for max_steps in range(1, 6):
        scheduler = init_scheduler(
            "BayesianOptimization",
            max_steps=max_steps,
            seed=np.random.randint(100),
            mode=METRIC_MODE,
            metric=METRIC_ATTR,
            transfer_learning_evaluations=None,
        )

        print("Optimising preliminary task %s" % max_steps)
        prev_task = run_scheduler_on_task(entry_point, scheduler, max_trials)

        # Generate TransferLearningTaskEvaluations from previous task
        transfer_learning_evaluations[max_steps] = extract_transferable_evaluations(
            prev_task, METRIC_ATTR, scheduler.config_space
        )

    # Collect evaluations on transfer task
    max_steps = 6
    transfer_task_results = {}
    labels = ["BayesianOptimization", "BoundingBox", "ZeroShotTransfer", "Quantiles"]
    for scheduler_str in labels:
        scheduler = init_scheduler(
            scheduler_str,
            max_steps=max_steps,
            seed=max_steps,
            mode=METRIC_MODE,
            metric=METRIC_ATTR,
            transfer_learning_evaluations=transfer_learning_evaluations,
        )
        print("Optimising transfer task using %s" % scheduler_str)
        transfer_task_results[scheduler_str] = run_scheduler_on_task(
            entry_point, scheduler, max_trials
        )

    # Optionally generate plots. Defaults to False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_plots", action="store_true", help="generate optimisation plots."
    )
    args = parser.parse_args()

    if args.generate_plots:
        from syne_tune.try_import import try_import_visual_message

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(try_import_visual_message())

        print("Generating optimisation plots.")
        """ Plot the results on the transfer task """
        for label in labels:
            plot_last_task(
                max_trials,
                transfer_task_results[label],
                label=label,
                metric=METRIC_ATTR,
                color=colours[label],
            )
        plt.legend()
        plt.ylabel(METRIC_ATTR)
        plt.xlabel("Iteration")
        plt.title("Transfer task (max_steps=6)")
        plt.savefig("Transfer_task.png", bbox_inches="tight")

        """ Plot the configs tried for the preliminary tasks """
        fig, ax = plt.subplots()
        for key in transfer_learning_evaluations:
            scatter_space_exploration(
                ax,
                transfer_learning_evaluations[key].hyperparameters,
                max_trials,
                "Task %s" % key,
            )
        add_labels(
            ax,
            scheduler.config_space,
            "Explored locations of BO for preliminary tasks",
        )
        plt.savefig("Configs_explored_preliminary.png", bbox_inches="tight")

        """ Plot the configs tried for the transfer task """
        fig, ax = plt.subplots()

        # Plot the configs tried by the different schedulers on the transfer task
        for label in labels:
            finished_trials = filter_completed(transfer_task_results[label])
            scatter_space_exploration(
                ax, finished_trials, max_trials, label, color=colours[label]
            )

            # Plot the first config tested as a big square
            ax.scatter(
                finished_trials["width"][0],
                finished_trials["height"][0],
                marker="s",
                color=colours[label],
                s=100,
            )

        # Plot the optima from the preliminary tasks as black crosses
        past_label = "Preliminary optima"
        for key in transfer_learning_evaluations:
            argmin = np.argmin(
                transfer_learning_evaluations[key].objective_values(METRIC_ATTR)[
                    :max_trials, 0, 0
                ]
            )
            ax.scatter(
                transfer_learning_evaluations[key].hyperparameters["width"][argmin],
                transfer_learning_evaluations[key].hyperparameters["height"][argmin],
                color="k",
                marker="x",
                label=past_label,
            )
            past_label = None
        add_labels(ax, scheduler.config_space, "Explored locations for transfer task")
        plt.savefig("Configs_explored_transfer.png", bbox_inches="tight")
