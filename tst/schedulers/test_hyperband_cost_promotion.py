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

from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.config_space import randint, uniform
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision


def _make_result(epoch, metric, cost):
    return dict(epoch=epoch, metric=metric, cost=cost)


def _new_trial(trial_id: int, config: dict):
    return Trial(
        trial_id=trial_id,
        config=config,
        creation_time=datetime.now())


def test_cost_offset():
    config_space = {
        'int': randint(1, 2),
        'float': uniform(5.5, 6.5),
        'epochs': 27}
    scheduler = HyperbandScheduler(
        config_space,
        searcher='random',
        metric='metric',
        mode='max',
        resource_attr='epoch',
        type='cost_promotion',
        max_resource_attr='epochs',
        cost_attr='cost')
    # Start 4 trials
    trials = dict()
    for trial_id in range(4):
        trials[trial_id] = _new_trial(
            trial_id, scheduler.suggest(trial_id=trial_id).config)
    # Make sure that 0, 1 are promoted eventually
    decision = scheduler.on_trial_result(
        trials[0], _make_result(1, 0.9, 1.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(0)] == 1.0
    sugg = scheduler.suggest(trial_id=len(trials))
    assert sugg.spawn_new_trial_id
    trials[4] = _new_trial(4, sugg.config)
    decision = scheduler.on_trial_result(
        trials[1], _make_result(1, 0.8, 3.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(1)] == 3.0
    # 1 < 4/3 -> Promote 0
    sugg = scheduler.suggest(trial_id=len(trials))
    assert not sugg.spawn_new_trial_id
    assert sugg.checkpoint_trial_id == 0
    assert sugg.config is not None
    trials[0].config = sugg.config
    decision = scheduler.on_trial_result(
        trials[2], _make_result(1, 0.7, 10.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(2)] == 10.0
    decision = scheduler.on_trial_result(
        trials[0], _make_result(3, 0.95, 3.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(0)] == 4.0
    # 4 < 14/3 -> Promote 1
    sugg = scheduler.suggest(trial_id=len(trials))
    assert not sugg.spawn_new_trial_id
    assert sugg.checkpoint_trial_id == 1
    assert sugg.config is not None
    trials[1].config = sugg.config
    decision = scheduler.on_trial_result(
        trials[1], _make_result(3, 0.85, 4.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(1)] == 7.0
    new_trial_id = len(trials)
    # Nothing can be promoted here (4 > 11/3)
    sugg = scheduler.suggest(trial_id=new_trial_id)
    assert sugg.spawn_new_trial_id
    trials[new_trial_id] = _new_trial(new_trial_id, sugg.config)


# Same scenario as above, but resumed trials start from
# scratch, which should lead to cost offsets being reset
def test_reset_cost_offset():
    config_space = {
        'int': randint(1, 2),
        'float': uniform(5.5, 6.5),
        'epochs': 27}
    scheduler = HyperbandScheduler(
        config_space,
        searcher='random',
        metric='metric',
        mode='max',
        resource_attr='epoch',
        type='cost_promotion',
        max_resource_attr='epochs',
        cost_attr='cost')
    # Start 4 trials
    trials = dict()
    for trial_id in range(4):
        trials[trial_id] = _new_trial(
            trial_id, scheduler.suggest(trial_id=trial_id).config)
    # Make sure that 0, 1 are promoted eventually
    decision = scheduler.on_trial_result(
        trials[0], _make_result(1, 0.9, 1.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(0)] == 1.0
    sugg = scheduler.suggest(trial_id=len(trials))
    assert sugg.spawn_new_trial_id
    trials[4] = _new_trial(4, sugg.config)
    decision = scheduler.on_trial_result(
        trials[1], _make_result(1, 0.8, 3.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(1)] == 3.0
    # 1 < 4/3 -> Promote 0
    sugg = scheduler.suggest(trial_id=len(trials))
    assert not sugg.spawn_new_trial_id
    assert sugg.checkpoint_trial_id == 0
    assert sugg.config is not None
    trials[0].config = sugg.config
    decision = scheduler.on_trial_result(
        trials[2], _make_result(1, 0.7, 10.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(2)] == 10.0
    # trial_id 0 reports for epoch=1, which signals restart
    # This should trigger reset of cost offset
    decision = scheduler.on_trial_result(
        trials[0], _make_result(1, 0.9, 1.5))
    assert decision == SchedulerDecision.CONTINUE
    assert scheduler._cost_offset[str(0)] == 0.0
    decision = scheduler.on_trial_result(
        trials[0], _make_result(2, 0.91, 2.5))
    assert decision == SchedulerDecision.CONTINUE
    assert scheduler._cost_offset[str(0)] == 0.0
    decision = scheduler.on_trial_result(
        trials[0], _make_result(3, 0.95, 3.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(0)] == 3.0
    # 4 < 14/3 -> Promote 1
    sugg = scheduler.suggest(trial_id=len(trials))
    assert not sugg.spawn_new_trial_id
    assert sugg.checkpoint_trial_id == 1
    assert sugg.config is not None
    trials[1].config = sugg.config
    # trial_id 1 reports for epoch=1, which signals restart
    # This should trigger reset of cost offset
    decision = scheduler.on_trial_result(
        trials[1], _make_result(1, 0.8, 2.5))
    assert decision == SchedulerDecision.CONTINUE
    assert scheduler._cost_offset[str(1)] == 0.0
    decision = scheduler.on_trial_result(
        trials[1], _make_result(2, 0.81, 3.5))
    assert decision == SchedulerDecision.CONTINUE
    assert scheduler._cost_offset[str(1)] == 0.0
    decision = scheduler.on_trial_result(
        trials[1], _make_result(3, 0.85, 4.0))
    assert decision == SchedulerDecision.PAUSE
    assert scheduler._cost_offset[str(1)] == 4.0
