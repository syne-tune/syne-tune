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
This is an example on how to use syne-tune in the ask-tell mode.
In this setup the tuning loop and experiments are disentangled. The AskTell Scheduler suggests new configurations
and the users themselves perform experiments to test the performance of each configuration.
Once done, user feeds the result into the Scheduler which uses the data to suggest better configurations.


In some cases, experiments needed for function evaluations can be very complex and require extra orchestration
(example vary from setting up jobs on non-aws clusters to runnig physical lab experiments) in which case this
interface provides all the necessary flexibility
"""
from typing import Dict
import datetime
import logging

import dill
import numpy as np

from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.config_space import uniform
from syne_tune.optimizer.baselines import RandomSearch, BayesianOptimization
from syne_tune.optimizer.scheduler import TrialScheduler


class AskTellScheduler:
    bscheduler: TrialScheduler
    trial_counter: int
    completed_experiments: Dict[int, TrialResult]

    def __init__(self, base_scheduler: TrialScheduler):
        self.bscheduler = base_scheduler
        self.trial_counter = 0
        self.completed_experiments = {}

    def ask(self) -> Trial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self.bscheduler.suggest(self.trial_counter)
        trial = Trial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        self.trial_counter += 1
        return trial

    def tell(self, trial: Trial, experiment_result: Dict[str, float]):
        """
        Feed experiment results back to the Scheduler

        :param trial: Trial that was run
        :param experiment_result: {metric: value} dictionary with experiment results
        """
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.datetime.now(),
        )
        self.bscheduler.on_trial_complete(trial=trial, result=experiment_result)
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metris: str) -> TrialResult:
        """
        Return the best trial according to the provided metric
        """
        if self.bscheduler.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metris],
        )


def target_function(x, noise: bool = True):
    fx = x * x + np.sin(x)
    if noise:
        sigma = np.cos(x) ** 2 + 0.01
        noise = 0.1 * np.random.normal(loc=x, scale=sigma)
        fx = fx + noise

    return fx


def get_objective():
    metric = "mean_loss"
    mode = "min"
    max_iterations = 100
    config_space = {
        "x": uniform(-1, 1),
    }
    return metric, mode, config_space, max_iterations


def plot_objective():
    """
    In this function, we will inspect the objective by plotting the target function
    :return:
    """
    from syne_tune.try_import import try_import_visual_message

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(try_import_visual_message())

    metric, mode, config_space, max_iterations = get_objective()

    plt.set_cmap("viridis")
    x = np.linspace(config_space["x"].lower, config_space["x"].upper, 400)
    fx = target_function(x, noise=False)
    noise = 0.1 * np.cos(x) ** 2 + 0.01

    plt.plot(x, fx, "r--", label="True value")
    plt.fill_between(x, fx + noise, fx - noise, alpha=0.2, fc="r")
    plt.legend()
    plt.grid()
    plt.show()


def tune_with_random_search() -> TrialResult:
    metric, mode, config_space, max_iterations = get_objective()
    scheduler = AskTellScheduler(
        base_scheduler=RandomSearch(config_space, metric=metric, mode=mode)
    )
    for iter in range(max_iterations):
        trial_suggestion = scheduler.ask()
        test_result = target_function(**trial_suggestion.config)
        scheduler.tell(trial_suggestion, {metric: test_result})
    return scheduler.best_trial(metric)


def save_restart_with_gp() -> TrialResult:
    metric, mode, config_space, max_iterations = get_objective()
    scheduler = AskTellScheduler(
        base_scheduler=BayesianOptimization(config_space, metric=metric, mode=mode)
    )
    for iter in range(int(max_iterations / 2)):
        trial_suggestion = scheduler.ask()
        test_result = target_function(**trial_suggestion.config)
        scheduler.tell(trial_suggestion, {metric: test_result})

    # --- The scheduler can be written to disk to pause experiment
    output_path = "scheduler-checkpoint.dill"
    with open(output_path, "wb") as f:
        dill.dump(scheduler, f)

    # --- The Scheduler can be read from disk at a later time to resume experiments
    with open(output_path, "rb") as f:
        scheduler = dill.load(f)

    for iter in range(int(max_iterations / 2)):
        trial_suggestion = scheduler.ask()
        test_result = target_function(**trial_suggestion.config)
        scheduler.tell(trial_suggestion, {metric: test_result})
    return scheduler.best_trial(metric)


def tune_with_gp() -> TrialResult:
    metric, mode, config_space, max_iterations = get_objective()
    scheduler = AskTellScheduler(
        base_scheduler=BayesianOptimization(config_space, metric=metric, mode=mode)
    )
    for iter in range(max_iterations):
        trial_suggestion = scheduler.ask()
        test_result = target_function(**trial_suggestion.config)
        scheduler.tell(trial_suggestion, {metric: test_result})
    return scheduler.best_trial(metric)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARN)
    # plot_objective() # Please uncomment this to plot the objective
    print("Random:", tune_with_random_search())
    print("GP with restart:", save_restart_with_gp())
    print("GP:", tune_with_gp())
