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
from syne_tune.config_space import randint

import shutil
from pathlib import Path

from syne_tune import StoppingCriterion
from syne_tune import Tuner
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from syne_tune.util import random_string


def launch_first_tuning(experiment_name: str):
    max_epochs = 100
    metric = "mean_loss"
    mode = "min"
    config_space = {
        "steps": max_epochs,
        "width": randint(0, 10),
        "height": randint(0, 10),
    }

    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    scheduler = ASHA(
        config_space=config_space,
        metric=metric,
        mode=mode,
        max_t=max_epochs,
        search_options={"allow_duplicates": True},
        resource_attr="epoch",
    )

    trial_backend = LocalBackend(entry_point=str(entry_point))

    stop_criterion = StoppingCriterion(
        max_num_trials_started=10,
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=4,
        tuner_name=experiment_name,
        suffix_tuner_name=False,
    )

    tuner.run()


if __name__ == "__main__":
    experiment_name = f"resume-tuning-example-{random_string(5)}"

    # Launch a tuning, tuning results and checkpoints are written to disk
    launch_first_tuning(experiment_name)

    # Later loads an experiment from disk given the experiment name,
    # in particular sets `load_tuner` to True to deserialize the Tuner
    tuning_experiment = load_experiment(experiment_name, load_tuner=True)

    # Copy the tuner as it will be modified when retuning
    shutil.copy(
        tuning_experiment.path / "tuner.dill",
        tuning_experiment.path / "tuner-backup.dill",
    )

    # Update stop criterion to run the tuning a couple more trials than before
    tuning_experiment.tuner.stop_criterion = StoppingCriterion(
        max_num_trials_started=20
    )

    # Define a new config space for instance favoring a new part of the space based on data analysis
    new_config_space = {
        "steps": 100,
        "width": randint(10, 20),
        "height": randint(1, 10),
    }

    # Update scheduler with random searcher to use new configuration space,
    # For now we modify internals, adding a method `update_config_space` to RandomSearcher would be a cleaner option.
    tuning_experiment.tuner.scheduler.config_space = new_config_space
    tuning_experiment.tuner.scheduler.searcher._hp_ranges = make_hyperparameter_ranges(
        new_config_space
    )
    tuning_experiment.tuner.scheduler.searcher.configure_scheduler(
        tuning_experiment.tuner.scheduler
    )

    # Resume the tuning with the modified search space and stopping criterion
    # The scheduler will now explore the updated search space
    tuning_experiment.tuner.run()
