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
from syne_tune.optimizer.schedulers.hyperband_rush import RUSHStoppingRungSystem, RUSHDecider
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.schedulers.transfer_learning.rush import RUSHScheduler
from syne_tune.config_space import randint


@pytest.fixture()
def config_space():
    return {
        'steps': 10,
        'm': randint(0, 20),
        'b': randint(-100, 100)
    }


@pytest.fixture()
def scheduler(metadata, config_space, request):
    return RUSHScheduler(config_space=config_space,
                         metric='loss',
                         max_t=10,
                         type=request.param,
                         transfer_learning_evaluations=metadata)


@pytest.fixture()
def num_threshold_candidates():
    return 2


@pytest.fixture()
def rung_levels():
    return [1, 3, 9, 27, 81]


@pytest.fixture()
def promote_quantiles(rung_levels):
    return [1.0 / 3 for _ in range(len(rung_levels) - 1)] + [1]


@pytest.fixture()
def rung_system(num_threshold_candidates, rung_levels, promote_quantiles):
    rung_system = RUSHStoppingRungSystem(num_threshold_candidates=num_threshold_candidates,
                                         rung_levels=rung_levels,
                                         promote_quantiles=promote_quantiles,
                                         metric='loss',
                                         mode='min',
                                         resource_attr='steps')
    return rung_system


@pytest.fixture()
def decider(num_threshold_candidates, rung_levels):
    decider = RUSHDecider(num_threshold_candidates=num_threshold_candidates, mode='min')
    decider._thresholds = {level: 0 for level in rung_levels if level < 10}
    return decider


@pytest.fixture()
def custom_rush_points():
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


def num_estimated_threshold_candidates(scheduler):
    return scheduler.terminator._rung_systems[0]._decider._num_threshold_candidates


@pytest.mark.parametrize('scheduler_type', ['stopping', 'promotion'])
def test_given_only_metadata_num_threshold_candidates_equals_number_of_tasks(metadata, config_space, scheduler_type):
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              type=scheduler_type,
                              transfer_learning_evaluations=metadata)
    assert num_estimated_threshold_candidates(scheduler) == len(metadata)


@pytest.mark.parametrize('scheduler_type', ['stopping', 'promotion'])
def test_points_of_evaluations_has_no_impact_on__num_threshold_candidates(metadata, config_space, custom_rush_points,
                                                                          scheduler_type):
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              type=scheduler_type,
                              transfer_learning_evaluations=metadata,
                              points_to_evaluate=custom_rush_points)
    assert num_estimated_threshold_candidates(scheduler) == len(metadata)


@pytest.mark.parametrize('scheduler_type', ['stopping', 'promotion'])
def test_given_metadata_and_custom_rush_points_num_threshold_candidates_equals_sum_of_unique_configs(metadata,
                                                                                                     config_space,
                                                                                                     custom_rush_points,
                                                                                                     scheduler_type):
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              type=scheduler_type,
                              transfer_learning_evaluations=metadata,
                              custom_rush_points=custom_rush_points)
    assert num_estimated_threshold_candidates(scheduler) == len(metadata) + len(custom_rush_points)


@pytest.mark.parametrize('scheduler_type', ['stopping', 'promotion'])
def test_given_metadata_and_custom_rush_points_with_overlap_keep_only_unique_configurations(metadata,
                                                                                            config_space,
                                                                                            scheduler_type):
    custom_rush_points = metadata['task'].hyperparameters.to_dict('records')
    scheduler = RUSHScheduler(config_space=config_space,
                              metric='loss',
                              max_t=10,
                              type=scheduler_type,
                              transfer_learning_evaluations=metadata,
                              custom_rush_points=custom_rush_points)
    assert num_estimated_threshold_candidates(scheduler) == len(custom_rush_points)


def test_given_hyperband_indicates_to_discontinue_return_discontinue(decider, num_threshold_candidates):
    assert not decider.task_continues(task_continues=False, trial_id=num_threshold_candidates - 1,
                                      metric_value=-1,
                                      resource=1)


def test_given_metric_better_than_threshold_update_threshold_if_threshold_configuration(decider,
                                                                                        num_threshold_candidates,
                                                                                        rung_levels):
    loss = -1
    for rung_level in rung_levels:
        for trial_id in [num_threshold_candidates, num_threshold_candidates - 1]:
            old_val = decider._thresholds.get(rung_level)
            decider.task_continues(task_continues=True, trial_id=trial_id, metric_value=loss,
                                   resource=rung_level)
            if trial_id == num_threshold_candidates:
                if old_val is None:
                    assert rung_level not in decider._thresholds
                else:
                    assert decider._thresholds[rung_level] == old_val
            else:
                assert decider._thresholds[rung_level] == loss


def test_given_metric_worse_than_threshold_return_discontinue_if_standard_trial(decider, num_threshold_candidates,
                                                                                rung_levels):
    for rung_level in rung_levels[:3]:
        assert not decider.task_continues(task_continues=True, trial_id=num_threshold_candidates,
                                          metric_value=0.1,
                                          resource=rung_level)


@pytest.mark.parametrize('hyperband_decision', [True, False])
def test_given_metric_worse_than_threshold_return_hyperband_decision_if_init_trial(decider,
                                                                                   num_threshold_candidates,
                                                                                   hyperband_decision):
    assert decider.task_continues(task_continues=hyperband_decision,
                                  trial_id=num_threshold_candidates - 1,
                                  metric_value=1, resource=1) is hyperband_decision
