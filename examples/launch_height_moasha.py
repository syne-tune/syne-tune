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
Example showing how to tune multiple objectives at once of an artificial function.
"""
import logging
from pathlib import Path

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune import Tuner
from syne_tune.config_space import uniform
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)

    max_steps = 27
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    entry_point = Path(__file__).parent / "training_scripts" / "mo_artificial" / "mo_artificial.py"
    mode = "min"

    np.random.seed(0)
    scheduler = MOASHA(
        max_t=max_steps,
        time_attr="step",
        mode=mode,
        metrics=["y1", "y2"],
        config_space=config_space,
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    stop_criterion = StoppingCriterion(max_wallclock_time=30)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0.5,
    )
    tuner.run()

