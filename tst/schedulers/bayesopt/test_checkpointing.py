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
import pickle

from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_fifo_searcher_defaults, constrained_gp_fifo_searcher_defaults, cost_aware_gp_fifo_searcher_defaults
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.constrained_gp_fifo_searcher import \
    ConstrainedGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher import \
    CostAwareGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import INTERNAL_METRIC_NAME, \
    INTERNAL_CONSTRAINT_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.comparison_gpy \
    import Ackley, sample_data, assert_equal_candidates,\
    assert_equal_randomstate


def test_pickle_gp_fifo_searcher():
    random_seed = 894623209
    # This data is used below
    _, searcher_options, _ = gp_fifo_searcher_defaults()
    num_data = searcher_options['num_init_random'] + 2
    num_pending = 2
    data = sample_data(Ackley, num_train=num_data + num_pending, num_grid=5)
    # Create searcher1 using default arguments
    searcher_options['config_space'] = data['state'].hp_ranges.config_space
    searcher_options['scheduler'] = 'fifo'
    searcher_options['random_seed'] = random_seed
    reward_attr = 'accuracy'
    searcher_options['metric'] = reward_attr
    searcher_options['debug_log'] = False
    searcher1 = GPFIFOSearcher(**searcher_options)
    # Feed searcher1 with some data
    config_for_trial = data['state'].config_for_trial
    for ev in data['state'].trials_evaluations[:num_data]:
        reward = ev.metrics[INTERNAL_METRIC_NAME]
        map_reward = searcher1.map_reward
        if map_reward is not None:
            reward = map_reward.reverse(reward)
        config = config_for_trial[ev.trial_id]
        searcher1._update(ev.trial_id, config, {reward_attr: reward})
    # Calling next_config is forcing a GP hyperparameter update
    next_config = searcher1.get_config()
    # Register some pending evaluations
    for ev in data['state'].trials_evaluations[-num_pending:]:
        config = config_for_trial[ev.trial_id]
        searcher1.register_pending(ev.trial_id, config=config)
    # Pickle mutable state of searcher1
    pkl_state = pickle.dumps(searcher1.get_state())
    # Clone searcher2 from mutable state
    searcher2 = GPFIFOSearcher(**searcher_options)
    searcher2 = searcher2.clone_from_state(pickle.loads(pkl_state))
    # At this point, searcher1 and searcher2 should be essentially the same
    # Compare model parameters
    params1 = searcher1.model_parameters()
    params2 = searcher2.model_parameters()
    for k, v1 in params1.items():
        v2 = params2[k]
        np.testing.assert_almost_equal(
            np.array([v1]), np.array([v2]), decimal=4)
    # Compare states
    state1 = searcher1.state_transformer.state
    state2 = searcher2.state_transformer.state
    hp_ranges = state1.hp_ranges
    configs1, _ = state1.observed_data_for_metric()
    configs2, _ = state2.observed_data_for_metric()
    assert_equal_candidates(configs1, configs2, hp_ranges, decimal=5)
    eval_targets1 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    eval_targets2 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    np.testing.assert_almost_equal(eval_targets1, eval_targets2, decimal=5)
    assert_equal_candidates(
        state1.pending_configurations(), state2.pending_configurations(),
        hp_ranges, decimal=5)
    # Compare random_state, random_generator state
    assert_equal_randomstate(searcher1.random_state, searcher2.random_state)


def test_pickle_constrained_gp_fifo_searcher():
    random_seed = 894623209
    # This data is used below
    _, searcher_options, _ = constrained_gp_fifo_searcher_defaults()
    num_data = searcher_options['num_init_random'] + 2
    num_pending = 2
    data = sample_data(Ackley, num_train=num_data + num_pending, num_grid=5)
    # Create searcher1 using default arguments
    searcher_options['config_space'] = data['state'].hp_ranges.config_space
    searcher_options['scheduler'] = 'fifo'
    searcher_options['random_seed'] = random_seed
    reward_attr = 'accuracy'
    searcher_options['metric'] = reward_attr
    searcher_options['debug_log'] = False
    constraint_attr = INTERNAL_CONSTRAINT_NAME
    searcher_options['constraint_attr'] = constraint_attr
    searcher1 = ConstrainedGPFIFOSearcher(**searcher_options)
    # Feed searcher1 with some data
    config_for_trial = data['state'].config_for_trial
    for ev in data['state'].trials_evaluations[:num_data]:
        reward = ev.metrics[INTERNAL_METRIC_NAME]
        map_reward = searcher1.map_reward
        if map_reward is not None:
            reward = map_reward.reverse(reward)
        config = config_for_trial[ev.trial_id]
        searcher1._update(
            ev.trial_id, config, {reward_attr: reward, constraint_attr: 1.0})

    # Calling next_config is forcing a GP hyperparameter update
    next_config = searcher1.get_config()
    # Register some pending evaluations
    for ev in data['state'].trials_evaluations[-num_pending:]:
        config = config_for_trial[ev.trial_id]
        searcher1.register_pending(ev.trial_id, config=config)
    # Pickle mutable state of searcher1
    pkl_state = pickle.dumps(searcher1.get_state())
    # Clone searcher2 from mutable state
    searcher2 = ConstrainedGPFIFOSearcher(**searcher_options)
    searcher2 = searcher2.clone_from_state(pickle.loads(pkl_state))
    # At this point, searcher1 and searcher2 should be essentially the same
    # Compare model parameters
    params1 = searcher1.model_parameters()
    params2 = searcher2.model_parameters()
    for output_metric in params1.keys():
        for k, v1 in params1[output_metric].items():
            v2 = params2[output_metric][k]
            np.testing.assert_almost_equal(
                np.array([v1]), np.array([v2]), decimal=4)
    # Compare states
    state1 = searcher1.state_transformer.state
    state2 = searcher2.state_transformer.state
    hp_ranges = state1.hp_ranges
    configs1, _ = state1.observed_data_for_metric()
    configs2, _ = state2.observed_data_for_metric()
    assert_equal_candidates(configs1, configs2, hp_ranges, decimal=5)
    eval_targets1 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    eval_targets2 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    np.testing.assert_almost_equal(eval_targets1, eval_targets2, decimal=5)
    assert_equal_candidates(
        state1.pending_configurations(), state2.pending_configurations(),
        hp_ranges, decimal=5)
    # Compare random_state, random_generator state
    assert_equal_randomstate(searcher1.random_state, searcher2.random_state)


def test_pickle_cost_aware_gp_fifo_searcher():
    random_seed = 894623209
    # This data is used below
    _, searcher_options, _ = cost_aware_gp_fifo_searcher_defaults()
    num_data = searcher_options['num_init_random'] + 2
    num_pending = 2
    data = sample_data(Ackley, num_train=num_data + num_pending, num_grid=5)
    # Create searcher1 using default arguments
    searcher_options['config_space'] = data['state'].hp_ranges.config_space
    searcher_options['scheduler'] = 'fifo'
    searcher_options['random_seed'] = random_seed
    reward_attr = 'accuracy'
    searcher_options['metric'] = reward_attr
    cost_attr = 'elapsed_time'
    searcher_options['cost_attr'] = cost_attr
    searcher_options['debug_log'] = False
    searcher1 = CostAwareGPFIFOSearcher(**searcher_options)
    # Feed searcher1 with some data
    config_for_trial = data['state'].config_for_trial
    for ev in data['state'].trials_evaluations[:num_data]:
        reward = ev.metrics[INTERNAL_METRIC_NAME]
        map_reward = searcher1.map_reward
        if map_reward is not None:
            reward = map_reward.reverse(reward)
        config = config_for_trial[ev.trial_id]
        searcher1._update(
            ev.trial_id, config, {reward_attr: reward, cost_attr: 1.0})

    # Calling next_config is forcing a GP hyperparameter update
    next_config = searcher1.get_config()
    # Register some pending evaluations
    for ev in data['state'].trials_evaluations[-num_pending:]:
        config = config_for_trial[ev.trial_id]
        searcher1.register_pending(ev.trial_id, config=config)
    # Pickle mutable state of searcher1
    pkl_state = pickle.dumps(searcher1.get_state())
    # Clone searcher2 from mutable state
    searcher2 = CostAwareGPFIFOSearcher(**searcher_options)
    searcher2 = searcher2.clone_from_state(pickle.loads(pkl_state))
    # At this point, searcher1 and searcher2 should be essentially the same
    # Compare model parameters
    params1 = searcher1.model_parameters()
    params2 = searcher2.model_parameters()
    for output_metric in params1.keys():
        for k, v1 in params1[output_metric].items():
            v2 = params2[output_metric][k]
            np.testing.assert_almost_equal(
                np.array([v1]), np.array([v2]), decimal=4)
    # Compare states
    state1 = searcher1.state_transformer.state
    state2 = searcher2.state_transformer.state
    hp_ranges = state1.hp_ranges
    configs1, _ = state1.observed_data_for_metric()
    configs2, _ = state2.observed_data_for_metric()
    assert_equal_candidates(configs1, configs2, hp_ranges, decimal=5)
    eval_targets1 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    eval_targets2 = np.array([
        x.metrics[INTERNAL_METRIC_NAME] for x in state1.trials_evaluations])
    np.testing.assert_almost_equal(eval_targets1, eval_targets2, decimal=5)
    assert_equal_candidates(
        state1.pending_configurations(), state2.pending_configurations(),
        hp_ranges, decimal=5)
    # Compare random_state, random_generator state
    assert_equal_randomstate(searcher1.random_state, searcher2.random_state)


if __name__ == "__main__":
    test_pickle_gp_fifo_searcher()