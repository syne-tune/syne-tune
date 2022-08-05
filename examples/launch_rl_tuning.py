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
This launches a local HPO tuning the discount factor of PPO on cartpole.
To run this example, you should have installed dependencies in `requirements.txt`.
"""
import logging
from pathlib import Path

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
import syne_tune.config_space as sp
from syne_tune import Tuner, StoppingCriterion

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(0)
    max_steps = 100
    trial_backend = LocalBackend(
        entry_point=Path(__file__).parent
        / "training_scripts"
        / "rl_cartpole"
        / "train_cartpole.py"
    )

    scheduler = ASHA(
        config_space={
            "gamma": sp.uniform(0.5, 0.99),
            "lr": sp.loguniform(1e-6, 1e-3),
        },
        metric="episode_reward_mean",
        mode="max",
        max_t=100,
        resource_attr="training_iter",
        search_options={"debug_log": False},
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=60)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        # tune for 3 minutes
        stop_criterion=stop_criterion,
        n_workers=2,
    )

    tuner.run()

    tuning_experiment = load_experiment(tuner.name)

    print(f"best result found: {tuning_experiment.best_config()}")

    tuning_experiment.plot()
