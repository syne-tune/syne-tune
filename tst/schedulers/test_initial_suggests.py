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
import numpy as np
from datetime import datetime
import pytest
import sys

from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
    BORE,
    SyncBOHB,
    BOHB,
    KDE,
    ASHABORE,
)
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


KDE_METHODS = {"BOHB", "SyncBOHB", "KDE"}

MULTIFID_METHODS = {"ASHA", "MOB", "BOHB", "SyncBOHB", "ASHABORE"}


list_schedulers_to_test = [
    ("BO", BayesianOptimization),
    ("ASHA", ASHA),
    ("MOB", MOBSTER),
    ("BORE", BORE),
    ("SyncBOHB", SyncBOHB),
    ("BOHB", BOHB),
    ("KDE", KDE),
    ("ASHABORE", ASHABORE),
]
if sys.version_info >= (3, 8):
    # BoTorch scheduler requires Python 3.8 or later
    from syne_tune.optimizer.baselines import BoTorch

    list_schedulers_to_test.append(
        ("BoTorch", BoTorch),
    )


@pytest.mark.timeout(3)
@pytest.mark.parametrize("name, scheduler_cls", list_schedulers_to_test)
def test_same_initial_suggests(name, scheduler_cls):
    random_seed = 31415927
    np.random.seed(random_seed)
    num_init_random = 5
    metric = "accuracy"
    max_resource_attr = "epochs"
    resource_attr = "epoch"
    config_space = {
        "n_units_1": randint(4, 1024),
        "n_units_2": randint(4, 1024),
        "batch_size": randint(8, 128),
        "dropout_1": uniform(0, 0.99),
        "dropout_2": uniform(0, 0.99),
        "learning_rate": loguniform(1e-6, 1),
        "weight_decay": loguniform(1e-8, 1),
        max_resource_attr: 27,
    }
    schedulers = [("RS", RandomSearch), (name, scheduler_cls)]

    initial_configs = {k: [] for k in ("RS", name)}
    for name, scheduler_cls in schedulers:
        is_multifidelity = name in MULTIFID_METHODS
        if name in KDE_METHODS:
            name_num_init_random = "num_min_data_points"
        elif name in {"BORE", "ASHABORE"}:
            name_num_init_random = "init_random"
        else:
            name_num_init_random = "num_init_random"
        val_num_init = (
            max(num_init_random, 7) if name in KDE_METHODS else num_init_random
        )
        kwargs = dict(
            metric=metric,
            max_resource_attr=max_resource_attr,
            search_options={
                "debug_log": False,
                name_num_init_random: val_num_init,
            },
            random_seed=random_seed,
        )
        if is_multifidelity:
            kwargs["resource_attr"] = resource_attr
        scheduler = scheduler_cls(config_space, **kwargs)
        for trial_id in range(num_init_random):
            suggestion = scheduler.suggest(trial_id)
            assert suggestion.spawn_new_trial_id
            initial_configs[name].append(suggestion.config)
            trial = Trial(
                trial_id=trial_id,
                config=suggestion.config,
                creation_time=datetime.now(),
            )
            scheduler.on_trial_add(trial)
            result = {metric: np.random.randn()}
            if is_multifidelity:
                result[resource_attr] = 1
                scheduler.on_trial_result(trial, result)
            else:
                scheduler.on_trial_complete(trial, result)

    hp_ranges = make_hyperparameter_ranges(config_space)
    for pos in range(num_init_random):
        configs_at_pos = {
            name: configs[pos] for name, configs in initial_configs.items()
        }
        match_strings = {
            name: hp_ranges.config_to_match_string(config)
            for name, config in configs_at_pos.items()
        }
        ms_rs = match_strings["RS"]
        assert all(
            ms == ms_rs for ms in match_strings.values()
        ), f"pos = {pos}:\n{configs_at_pos}"
