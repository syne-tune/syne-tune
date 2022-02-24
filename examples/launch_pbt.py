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
import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining
from syne_tune import Tuner
from syne_tune.config_space import loguniform
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    max_trials = 100

    config_space = {
        "lr": loguniform(0.0001, 0.02),
    }

    entry_point = Path(__file__).parent / "training_scripts" / "pbt_example" / "pbt_example.py"
    trial_backend = LocalBackend(entry_point=str(entry_point))

    mode = "max"
    metric = "mean_accuracy"
    time_attr = "training_iteration"
    population_size = 2

    pbt = PopulationBasedTraining(config_space=config_space,
                                  metric=metric,
                                  resource_attr=time_attr,
                                  population_size=population_size,
                                  mode=mode,
                                  max_t=200,
                                  perturbation_interval=1)

    local_tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=pbt,
        stop_criterion=StoppingCriterion(max_wallclock_time=20),
        n_workers=population_size,
        results_update_interval=1
    )

    local_tuner.run()
