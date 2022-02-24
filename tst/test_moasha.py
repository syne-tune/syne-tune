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
from datetime import datetime
from functools import partial

import pytest
import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA, _Bracket
from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import FixedObjectivePriority, \
    LinearScalarizationPriority, NonDominatedPriority
from syne_tune.config_space import randint


def test_bucket():
    b = _Bracket(1, 10, 2, 0)
    assert b.on_result(0, 1, {metric1: 2}) == "CONTINUE"
    assert b.on_result(1, 1, {metric1: 0}) == "CONTINUE"
    assert b.on_result(2, 1, {metric1: 3}) == "STOP"


max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}
time_attr = "step"
metric1 = "mean_loss"
metric2 = "cost"


def make_trial(trial_id: int):
    return Trial(
        trial_id=trial_id,
        config={"steps": 0, "width": trial_id},
        creation_time=datetime.now(),
    )

scheduler_fun = partial(
    MOASHA,
    max_t=max_steps,
    brackets=1,
    reduction_factor=2.0,
    time_attr="step",
    metrics=[metric1, metric2],
    config_space=config_space,
)


@pytest.mark.parametrize("scheduler", [
    scheduler_fun(mode="max", multiobjective_priority=FixedObjectivePriority()),
    scheduler_fun(mode="max", multiobjective_priority=LinearScalarizationPriority()),
    scheduler_fun(mode=["max", "max"], multiobjective_priority=LinearScalarizationPriority()),
    scheduler_fun(mode="max", multiobjective_priority=NonDominatedPriority()),
])
def test_moasha_mode_max(scheduler):
    np.random.seed(0)
    trial1 = make_trial(trial_id=0)
    trial2 = make_trial(trial_id=1)
    trial3 = make_trial(trial_id=2)

    scheduler.on_trial_add(trial=trial1)
    scheduler.on_trial_add(trial=trial2)
    scheduler.on_trial_add(trial=trial3)

    make_metric = lambda x: {time_attr: 1, metric1: x, metric2: x}
    decision1 = scheduler.on_trial_result(trial1, make_metric(2.0))
    decision2 = scheduler.on_trial_result(trial2, make_metric(4.0))
    decision3 = scheduler.on_trial_result(trial3, make_metric(1.0))
    assert decision1 == SchedulerDecision.CONTINUE
    assert decision2 == SchedulerDecision.CONTINUE
    assert decision3 == SchedulerDecision.STOP


@pytest.mark.parametrize("scheduler", [
    scheduler_fun(mode="min", multiobjective_priority=FixedObjectivePriority()),
    scheduler_fun(mode="min", multiobjective_priority=LinearScalarizationPriority()),
    scheduler_fun(mode=["min", "min"], multiobjective_priority=LinearScalarizationPriority()),
    scheduler_fun(mode="min", multiobjective_priority=NonDominatedPriority()),
])
def test_moasha_mode_min(scheduler):
    np.random.seed(0)
    trial1 = make_trial(trial_id=0)
    trial2 = make_trial(trial_id=1)
    trial3 = make_trial(trial_id=2)

    scheduler.on_trial_add(trial=trial1)
    scheduler.on_trial_add(trial=trial2)
    scheduler.on_trial_add(trial=trial3)

    make_metric = lambda x: {time_attr: 1, metric1: x, metric2: x}
    decision1 = scheduler.on_trial_result(trial1, make_metric(4.0))
    decision2 = scheduler.on_trial_result(trial2, make_metric(2.0))
    decision3 = scheduler.on_trial_result(trial3, make_metric(10.0))
    assert decision1 == SchedulerDecision.CONTINUE
    assert decision2 == SchedulerDecision.CONTINUE
    assert decision3 == SchedulerDecision.STOP


@pytest.mark.parametrize("mo_priority,expected_priority", [
    (LinearScalarizationPriority(), [1.0, 4.0, 7.0, 10.0, 13.0]),
    (
        LinearScalarizationPriority(weights=[0.2, 0.4, 0.8]),
        [0.6666666666666666, 2.066666666666667, 3.466666666666667, 4.866666666666667, 6.266666666666667]
    ),
    (NonDominatedPriority(), [0, 1, 2, 3, 4]),
    (NonDominatedPriority(dim=1), [0, 1, 2, 3, 4]),
    (NonDominatedPriority(max_num_samples=10), [0, 1, 2, 3, 4]),
])
def test_multiobjective_priorities(mo_priority, expected_priority):
    num_samples = 5
    num_objectives = 3
    objectives = np.arange(num_samples * num_objectives).reshape((num_samples, num_objectives))

    priorities = mo_priority.__call__(objectives=objectives)
    assert np.allclose(priorities, expected_priority)
    assert priorities.shape == (num_samples,)