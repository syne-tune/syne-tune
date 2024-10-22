"""
This is an example on how to use syne-tune in the ask-tell mode.
In this setup the tuning loop and experiments are disentangled. The AskTell Scheduler suggests new configurations
and the users themselves perform experiments to test the performance of each configuration.
Once done, user feeds the result into the Scheduler which uses the data to suggest better configurations.


In some cases, experiments needed for function evaluations can be very complex and require extra orchestration
(example vary from setting up jobs on non-aws clusters to running physical lab experiments) in which case this
interface provides all the necessary flexibility
"""
import logging

import dill
import numpy as np

from syne_tune.backend.trial_status import TrialResult
from syne_tune.config_space import uniform
from syne_tune.optimizer.baselines import RandomSearch, BayesianOptimization
from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler


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

    try:
        import matplotlib.pyplot as plt

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

    except ImportError as e:
        logging.debug(e)


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
