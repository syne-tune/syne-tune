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
import numpy as np
from typing import Optional

from syne_tune.config_space import randint, choice
from syne_tune.optimizer.schedulers.synchronous import (
    GeometricDifferentialEvolutionHyperbandScheduler,
    DifferentialEvolutionHyperbandScheduler,
)


def _create_scheduler(
    mutation_factor: Optional[float] = None,
    crossover_probability: Optional[float] = None,
) -> DifferentialEvolutionHyperbandScheduler:
    config_space = {
        "a": randint(0, 5),
        "b": choice(["a", "b", "c"]),
    }
    return GeometricDifferentialEvolutionHyperbandScheduler(
        config_space=config_space,
        searcher="random_encoded",
        search_options={"debug_log": False},
        mode="min",
        metric="criterion",
        max_resource_level=9,
        grace_period=1,
        reduction_factor=3,
        resource_attr="epoch",
        random_seed=31415927,
        mutation_factor=mutation_factor,
        crossover_probability=crossover_probability,
    )


class FixedUniformRandomState:
    def __init__(self, fill_value: float):
        self._fill_value = fill_value

    def uniform(self, low: float, high: float, size: int = 1):
        if size == 1:
            return self._fill_value
        else:
            return np.full(shape=(size,), fill_value=self._fill_value)


_de_mutation_parameterizations = [
    [0.5, 0.2],
    [0.75, 0.3],
    [0.1, 0.4],
]


@pytest.mark.parametrize("mutation_factor, fill_value", _de_mutation_parameterizations)
def test_de_mutation(mutation_factor, fill_value):
    scheduler = _create_scheduler(mutation_factor=mutation_factor)
    # Ask for 3 suggestions
    for trial_id in range(3):
        suggestion = scheduler.suggest(trial_id)
        assert suggestion.spawn_new_trial_id
    assert len(scheduler._trial_info) == 3
    assert all(x.level == 1 for x in scheduler._trial_info.values())
    # Control behavior for boundary violations
    scheduler.random_state = FixedUniformRandomState(fill_value)
    mutant = scheduler._de_mutation(list(range(3)))
    encoded_configs = [
        scheduler._trial_info[trial_id].encoded_config for trial_id in range(3)
    ]
    mutant_ours = (
        encoded_configs[1] - encoded_configs[2]
    ) * mutation_factor + encoded_configs[0]
    for i, v in enumerate(mutant_ours):
        if v > 1 or v < 0:
            mutant_ours[i] = fill_value
    assert np.all(
        mutant == mutant_ours
    ), f"mutant = {mutant}\nmutant_ours = {mutant_ours}"


class FixedCrossoverRandomState:
    def __init__(self, rand_vals: np.ndarray, randint_val: int):
        self._rand_vals = rand_vals
        self._randint_val = randint_val

    def rand(self, size: int):
        assert self._rand_vals.size == size
        return self._rand_vals

    def randint(self, start: int, end: int):
        assert start <= self._randint_val < end
        return self._randint_val


_crossover_parameterizations = [
    [0.5, [0.45, 0.7], 1, [True, False]],
    [0.2, [0.45, 0.1], 0, [False, True]],
    [0.1, [0.15, 0.55], 0, [True, False]],
]


@pytest.mark.parametrize(
    "crossover_probability, rand_vals, randint_val, _hp_mask",
    _crossover_parameterizations,
)
def test_crossover(crossover_probability, rand_vals, randint_val, _hp_mask):
    scheduler = _create_scheduler(crossover_probability=crossover_probability)
    # Ask for 2 suggestions
    for trial_id in range(2):
        suggestion = scheduler.suggest(trial_id)
        assert suggestion.spawn_new_trial_id
    assert len(scheduler._trial_info) == 2
    assert all(x.level == 1 for x in scheduler._trial_info.values())
    # Control random draws in _crossover
    scheduler.random_state = FixedCrossoverRandomState(
        rand_vals=np.array(rand_vals), randint_val=randint_val
    )
    target = scheduler._trial_info[0].encoded_config
    mutant = scheduler._trial_info[1].encoded_config
    offspring = scheduler._crossover(mutant, target)
    hp_mask = np.array(rand_vals) < crossover_probability
    if not np.any(hp_mask):
        hp_mask[randint_val] = True
    assert np.all(hp_mask == np.array(_hp_mask))
    offspring_ours = target.copy()
    if hp_mask[0]:
        offspring_ours[0] = mutant[0]
    if hp_mask[1]:
        offspring_ours[1:] = mutant[1:]
    assert np.all(
        offspring == offspring_ours
    ), f"offspring = {offspring}\noffspring_ours = {offspring_ours}"
