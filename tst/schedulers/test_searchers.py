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
import pytest
from datetime import datetime
import itertools
import numpy as np

from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    ASHA,
    HyperTune,
    DyHPO,
    SyncHyperband,
    DEHB,
    BOHB,
    SyncBOHB,
    BORE,
    KDE,
)
from syne_tune.config_space import (
    choice,
    ordinal,
    config_space_size,
    uniform,
)
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


SCHEDULERS = [
    (GridSearch, False),
    (RandomSearch, False),
    (BayesianOptimization, False),
    (ASHA, True),
    (HyperTune, True),
    (DyHPO, True),
    (SyncHyperband, True),
    (DEHB, True),
    (BOHB, True),
    (SyncBOHB, True),
    (BORE, False),
    (KDE, False),
]


DUPLICATES_AND_FAIL = [(False, False), (True, False), (True, True)]


COMBINATIONS = list(itertools.product(SCHEDULERS, DUPLICATES_AND_FAIL[:-1])) + list(
    itertools.product(SCHEDULERS[1:], DUPLICATES_AND_FAIL[-1:])
)


# Does not contain ASHABORE, because >10 secs on CI
# TODO: Dig more, why is ASHABORE more expensive than BORE here?
@pytest.mark.timeout(10)
@pytest.mark.parametrize("tpl1, tpl2", COMBINATIONS)
def test_allow_duplicates_or_not(tpl1, tpl2):
    # If ``trials_fail == True``, we let all trials fail. In that case, corresponding
    # configs are filtered out, even if ``allow_duplicates == True`` (for all searchers
    # except ``GridSearcher``).
    scheduler_cls, multifid = tpl1
    allow_duplicates, trials_fail = tpl2
    random_seed = 31415927
    np.random.seed(random_seed)

    max_resource_attr = "epochs"
    max_resource_val = 5
    metric = "y"
    mode = "min"
    resource_attr = "r"
    config_space = {
        "a": choice(["a", "b", "c"]),  # 3
        "c": ordinal([0.1, 0.2, 0.5], kind="nn"),  # 3
        max_resource_attr: max_resource_val,
    }
    cs_size = config_space_size(config_space)
    assert cs_size == 3 * 3

    if multifid:
        kwargs = dict(resource_attr=resource_attr)
    else:
        kwargs = dict()
    scheduler = scheduler_cls(
        config_space,
        metric=metric,
        mode=mode,
        max_resource_attr=max_resource_attr,
        search_options={
            "allow_duplicates": allow_duplicates,
            "debug_log": False,
        },
        **kwargs,
    )
    trial_id = 0
    trials = dict()
    while trial_id <= cs_size:
        err_msg = (
            f"{scheduler_cls}: trial_id {trial_id}, cs_size {cs_size}, allow_duplicates "
            f"{allow_duplicates}, trials_fail {trials_fail}"
        )
        print(f"suggest: trial_id = {trial_id}")
        suggestion = scheduler.suggest(trial_id)
        if trial_id < cs_size or (allow_duplicates and not trials_fail):
            assert suggestion is not None, err_msg
            if suggestion.spawn_new_trial_id:
                # Start new trial
                print(f"Start new trial: {suggestion.config}")
                assert suggestion.checkpoint_trial_id is None, err_msg
                trial = Trial(
                    trial_id=trial_id,
                    config=suggestion.config,
                    creation_time=datetime.now(),
                )
                scheduler.on_trial_add(trial)
                trials[trial_id] = trial
                trial_id += 1
            else:
                # Resume existing trial
                # Note that we do not implement checkpointing, so resume
                # means start from scratch
                print(f"Resume trial: {suggestion.checkpoint_trial_id}")
                assert suggestion.checkpoint_trial_id is not None, err_msg
                trial = trials[suggestion.checkpoint_trial_id]
            if not trials_fail:
                # Return results
                result = None
                it = None
                for it in range(max_resource_val):
                    result = {
                        metric: np.random.rand(),
                        resource_attr: it + 1,
                    }
                    decision = scheduler.on_trial_result(trial=trial, result=result)
                    if decision != SchedulerDecision.CONTINUE:
                        break
                if it >= max_resource_val - 1:
                    scheduler.on_trial_complete(trial=trial, result=result)
            else:
                # Trial fails
                scheduler.on_trial_error(trial=trial)
        else:
            # Maybe trials are being resumed?
            print(f"OK, I am here. trial_id = {trial_id}")
            while suggestion is not None and suggestion.checkpoint_trial_id is not None:
                print(f"suggest: trial_id = {trial_id}")
                suggestion = scheduler.suggest(trial_id)
            assert suggestion is None, err_msg
            trial_id += 1


# BoTorch fails with:
# botorch.exceptions.errors.ModelFittingError: All attempts to fit the model have failed.
SCHEDULERS = [
    (RandomSearch, False, False),
    (BayesianOptimization, False, True),
    (ASHA, True, False),
    (HyperTune, True, True),
    (SyncHyperband, True, False),
    (BORE, False, False),
    # (BoTorch, False, True),
]


@pytest.mark.timeout(5)
@pytest.mark.parametrize("scheduler_cls, is_multifid, is_bo", SCHEDULERS)
def test_restrict_configurations(scheduler_cls, is_multifid, is_bo):
    random_seed = 31415927
    np.random.seed(random_seed)

    max_resource_attr = "epochs"
    max_resource_val = 5
    metric = "y"
    mode = "min"
    resource_attr = "r"
    config_space = {
        "a": uniform(0.0, 1.0),
        "b": choice(["a", "b", "c"]),
        "c": ordinal([0.1, 0.2, 0.5], kind="nn"),
        max_resource_attr: max_resource_val,
    }
    hp_ranges = make_hyperparameter_ranges(config_space)

    if is_multifid:
        kwargs = dict(resource_attr=resource_attr)
    else:
        kwargs = dict()
    restrict_configurations = [
        {"a": 0.5, "b": "a", "c": 0.2},
        {"a": 0.1, "b": "b", "c": 0.1},
        {"a": 0.3, "b": "c", "c": 0.1},
        {"a": 0.4, "b": "a", "c": 0.5},
        {"a": 0.31, "b": "c", "c": 0.1},
    ]
    search_options = {
        "allow_duplicates": False,
        "debug_log": False,
        "restrict_configurations": restrict_configurations,
    }
    if is_bo:
        search_options["num_init_random"] = 3
    scheduler = scheduler_cls(
        config_space,
        metric=metric,
        mode=mode,
        max_resource_attr=max_resource_attr,
        search_options=search_options,
        **kwargs,
    )
    rc_set = set(
        hp_ranges.config_to_match_string(config) for config in restrict_configurations
    )
    trial_id = 0
    trials = dict()
    while trial_id < len(restrict_configurations):
        err_msg = f"trial_id {trial_id}"
        print(f"suggest: trial_id = {trial_id}")
        suggestion = scheduler.suggest(trial_id)
        assert suggestion is not None, err_msg
        if suggestion.spawn_new_trial_id:
            # Start new trial
            config = suggestion.config
            print(f"Start new trial: {config}")
            assert hp_ranges.config_to_match_string(config) in rc_set, err_msg
            trial = Trial(
                trial_id=trial_id,
                config=config,
                creation_time=datetime.now(),
            )
            scheduler.on_trial_add(trial)
            trials[trial_id] = trial
            trial_id += 1
        else:
            # Resume existing trial
            # Note that we do not implement checkpointing, so resume
            # means start from scratch
            print(f"Resume trial: {suggestion.checkpoint_trial_id}")
            assert suggestion.checkpoint_trial_id is not None, err_msg
            trial = trials[suggestion.checkpoint_trial_id]
        # Return results
        result = None
        it = None
        for it in range(max_resource_val):
            result = {
                metric: np.random.rand(),
                resource_attr: it + 1,
            }
            decision = scheduler.on_trial_result(trial=trial, result=result)
            if decision != SchedulerDecision.CONTINUE:
                break
        if it >= max_resource_val - 1:
            scheduler.on_trial_complete(trial=trial, result=result)
