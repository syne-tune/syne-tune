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
from pathlib import Path

import numpy as np

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform
from syne_tune.optimizer.baselines import MORandomScalarizationBayesOpt


def main():
    random_seed = 6287623
    # Hyperparameter configuration space
    config_space = {
        "steps": randint(0, 100),
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    metrics = ["y1", "y2"]
    modes = ["min", "min"]

    # Creates a FIFO scheduler with a ``MultiObjectiveMultiSurrogateSearcher``. The
    # latter is configured by one default GP surrogate per objective, and with the
    # ``MultiObjectiveLCBRandomLinearScalarization`` acquisition function.
    scheduler = MORandomScalarizationBayesOpt(
        config_space=config_space,
        metric=metrics,
        mode=modes,
        random_seed=random_seed,
    )

    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "mo_artificial"
        / "mo_artificial.py"
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=entry_point),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=30),
        n_workers=1,  # how many trials are evaluated in parallel
    )
    tuner.run()


if __name__ == "__main__":
    main()
