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

import pytest
import itertools

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

from syne_tune import Tuner
from syne_tune import StoppingCriterion
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers.searchers import GridSearcher


def test_grid_scheduler():
    max_steps = 100
    num_workers = 2
    random_seed = 382378624

    searcher = "grid"
    mode = "min"
    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": uniform(-100, 100),
        "sleep_time": 0.001,
    }

    num_samples = {"width":5, "height":10}
    metric = "mean_loss"

    entry_point = (
            Path(__file__).parent.parent
            / "examples"
            / "training_scripts"
            / "height_example"
            / "train_height.py"
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    search_options = {"debug_log": True, "num_init_random": num_workers, "num_samples":num_samples}

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )
    stop_criterion = StoppingCriterion(max_wallclock_time=60)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()


def test_grid_scheduler_categorical():
    max_steps = 100
    num_workers = 2
    random_seed = 382378623

    searcher = "grid"
    mode = "min"
    categorical_config_space = {
        "steps": max_steps,
        "width": choice([1,5,10,15,20]),
        "height": choice([30,40,50,60,70,80]),
        "sleep_time": 0.001,
    }

    metric = "mean_loss"

    entry_point = (
            Path(__file__).parent.parent
            / "examples"
            / "training_scripts"
            / "height_example"
            / "train_height.py"
    )
    trial_backend = LocalBackend(entry_point=str(entry_point))

    search_options = {"debug_log": True, "num_init_random": num_workers}

    myscheduler = FIFOScheduler(
        config_space=categorical_config_space,
        searcher=searcher,
        search_options=search_options,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
    )
    stop_criterion = StoppingCriterion(max_wallclock_time=60)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        sleep_time=0.1,
        n_workers=num_workers,
        stop_criterion=stop_criterion,
    )

    tuner.run()


def test_get_config():
    config_space = {
        "char_attr": choice(["a", "b"]),
        "float_attr": uniform(1, 5),
        "int_attr": randint(10, 40)
    }
    num_samples = {"float_attr": 2, "int_attr": 2}

    all_candidates_on_grid = [
        {"char_attr": "a", "float_attr": 2.0, "int_attr": 17},
        {"char_attr": "a", "float_attr": 2.0, "int_attr": 33},
        {"char_attr": "a", "float_attr": 4.0, "int_attr": 17},
        {"char_attr": "a", "float_attr": 4.0, "int_attr": 33},
        {"char_attr": "b", "float_attr": 2.0, "int_attr": 17},
        {"char_attr": "b", "float_attr": 2.0, "int_attr": 33},
        {"char_attr": "b", "float_attr": 4.0, "int_attr": 17},
        {"char_attr": "b", "float_attr": 4.0, "int_attr": 33}
    ]

    num_valid_config = len(all_candidates_on_grid)
    searcher = GridSearcher(config_space, num_samples=num_samples, metric="accuracy")
    for trial_id in range(num_valid_config):
        # These should get new config
        config = searcher.get_config(trial_id=trial_id)
        assert config in all_candidates_on_grid

