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
import datetime
import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import dill
import numpy as np

from examples.launch_ask_tell_scheduler import AskTellScheduler
from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.baselines import RandomSearch, BayesianOptimization, ASHA
from syne_tune.optimizer.scheduler import TrialScheduler, SchedulerDecision


def target_function(x, step: int = None, noise: bool = True):
    fx = x * x + np.sin(x)
    if noise:
        sigma = np.cos(x) ** 2 + 0.01
        noise = 0.1 * np.random.normal(loc=x, scale=sigma)
        fx = fx + noise

    if step is not None:
        fx += step * 0.01

    return fx


def get_objective():
    metric = "mean_loss"
    mode = "min"
    max_iterations = 100
    config_space = {
        "x": uniform(-1, 1),
    }
    return metric, mode, config_space, max_iterations


def tune_with_hyperband() -> TrialResult:
    metric, mode, config_space, max_iterations = get_objective()
    max_steps = 100

    scheduler = AskTellScheduler(
        base_scheduler=ASHA(
            config_space,
            metric=metric,
            resource_attr="epoch",
            max_t=max_steps,
            mode=mode,
        )
    )
    for iter in range(max_iterations):
        trial_suggestion = scheduler.ask()
        final_step, test_result = run_hyperband_step(scheduler, trial_suggestion, max_steps, metric)
        scheduler.tell(trial_suggestion, {metric: test_result, "epoch": final_step})
    return scheduler.best_trial(metric)


def run_hyperband_step(
    scheduler: AskTellScheduler, trial_suggestion: Trial, max_steps: int, metric: str
) -> Tuple[float, float]:
    for step in range(1, max_steps):
        test_result = target_function(**trial_suggestion.config, step=step)
        decision = scheduler.bscheduler.on_trial_result(trial_suggestion, {metric: test_result, "epoch": step})
        if decision == SchedulerDecision.STOP:
            break
    return step, test_result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARN)
    print("Hyperband:", tune_with_hyperband())
