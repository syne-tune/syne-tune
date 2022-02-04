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
import pandas as pd
import pytest

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.schedulers.transfer_learning.rush import RUSHScheduler
from syne_tune.search_space import randint


@pytest.fixture()
def config_space():
    return {
        'steps': 10,
        'm': randint(0, 20),
        'b': randint(-100, 100)
    }


@pytest.fixture()
def scheduler(metadata, config_space):
    return RUSHScheduler(config_space=config_space,
                         metric='loss',
                         max_t=10,
                         transfer_learning_evaluations=metadata)


@pytest.fixture()
def points_to_evaluate():
    return [{'m': 1, 'b': -2}]


@pytest.fixture()
def best_config():
    return {'m': 3, 'b': -89, 'steps': 10}


@pytest.fixture()
def metadata(config_space, best_config):
    max_steps = 10
    hp_data = [best_config, {'m': 10, 'b': -10, 'steps': 10}, {'m': 2, 'b': -92, 'steps': 10}]
    hp = pd.DataFrame(hp_data)
    metric = list()
    for i in range(len(hp_data)):
        metric.append([[[s, -s] for s in get_learning_curve(
            hp_data[i]['m'],
            hp_data[i]['b'],
            max_steps,
            n,
        )] for n in [-1.2, 1.1]])

    metrics_names = ['loss', 'gain']
    return {'task': TransferLearningTaskEvaluations(configuration_space=config_space,
                                                    hyperparameters=hp,
                                                    objectives_evaluations=np.array(metric),
                                                    objectives_names=metrics_names)}


@pytest.fixture()
def trial():
    return Trial(trial_id=0, config=dict(), creation_time=0)


def get_learning_curve(m, b, steps, n):
    result = [(m * n) * step + b for step in range(steps)]
    return result


def get_result(loss=0, epoch=1):
    return {
        'loss': loss,
        'epoch': epoch,
    }


def test_given_only_metadata_num_init_config_equals_number_of_tasks(metadata, config_space):
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              transfer_learning_evaluations=metadata)
    assert scheduler._num_init_configs == len(metadata)


def test_given_metadata_and_points_to_evaluate_num_init_config_equals_sum_of_unique_configurations(metadata,
                                                                                                   config_space,
                                                                                                   points_to_evaluate):
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              transfer_learning_evaluations=metadata,
                              points_to_evaluate=points_to_evaluate)
    assert scheduler._num_init_configs == len(metadata) + len(points_to_evaluate)


def test_given_metadata_and_points_to_evaluate_with_overlap_keep_only_unique_configurations(metadata, config_space):
    points_to_eval = metadata['task'].hyperparameters.to_dict('records')
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              transfer_learning_evaluations=metadata,
                              points_to_evaluate=points_to_eval)
    assert scheduler._num_init_configs == len(points_to_eval)


def test_given_metadata_return_best_configurations_per_task(metadata, config_space):
    min_loss = RUSHScheduler._determine_baseline_configurations(config_space, metadata, 'loss', 'min')
    max_gain = RUSHScheduler._determine_baseline_configurations(config_space, metadata, 'gain', 'max')
    assert len(min_loss) == 1
    assert min_loss == max_gain
    assert min_loss[0] == metadata['task'].hyperparameters.to_dict('records')[0]


def test_given_trial_should_be_stopped_return_stop(scheduler, trial):
    assert scheduler._on_trial_result(SchedulerDecision.STOP, trial, dict()) == SchedulerDecision.STOP


def test_given_trial_should_be_continued_and_no_milestone_reached_return_continue(scheduler):
    for trial_id in range(10):
        suggestion = scheduler.suggest(trial_id)
        trial = Trial(trial_id=trial_id, config=suggestion.config, creation_time=None)
        scheduler.on_trial_add(trial=trial)
        loss = 0 if trial_id == 0 else 1
        epoch = 1 if trial_id == 0 else 2
        decision = scheduler._on_trial_result(SchedulerDecision.CONTINUE, trial, get_result(loss=loss, epoch=epoch))
        assert decision == SchedulerDecision.CONTINUE


def test_on_reaching_milestone_update_threshold(scheduler):
    loss = 0.1
    trial_id = 0
    suggestion = scheduler.suggest(trial_id)
    trial = Trial(trial_id=trial_id, config=suggestion.config, creation_time=None)
    scheduler.on_trial_add(trial=trial)
    scheduler.on_trial_result(trial, {'epoch': 1, 'loss': loss})
    assert scheduler._thresholds == {1: loss}


def test_given_trial_not_meeting_threshold_return_stop(scheduler):
    loss = 0.1
    worse_loss = 0.2
    for trial_id in range(10):
        suggestion = scheduler.suggest(trial_id)
        trial = Trial(trial_id=trial_id, config=suggestion.config, creation_time=None)
        scheduler.on_trial_add(trial=trial)
        decision = scheduler.on_trial_result(trial, {'epoch': 1, 'loss': loss if trial_id == 0 else worse_loss})
        assert trial_id == 0 or decision == SchedulerDecision.STOP
