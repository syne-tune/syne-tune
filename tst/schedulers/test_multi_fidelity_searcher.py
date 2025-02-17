import pytest
import numpy as np

from syne_tune.optimizer.schedulers.searchers.multi_fidelity_searcher import (
    IndependentMultiFidelitySearcher,
)
from syne_tune.config_space import choice
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_cls_dict
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


@pytest.mark.parametrize("searcher_cls", searcher_cls_dict)
def test_independent_multi_fidelity_searcher(searcher_cls):
    random_seed = 31415927
    resource_levels = [3] * 20 + [1] * 25 + [9] * 9
    random_state = np.random.RandomState(random_seed)
    hp_cols = ("hp_x0", "hp_x1", "hp_x2")
    config_space = {
        node: choice(
            ["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"]
        )
        for node in hp_cols
    }
    searcher = IndependentMultiFidelitySearcher(
        config_space=config_space,
        points_to_evaluate=[],
        searcher_cls=searcher_cls,
    )
    # Sample data at random (except for ``resource_levels``)
    hp_ranges = make_hyperparameter_ranges(config_space)
    num_data = len(resource_levels)
    trial_ids = list(range(num_data))
    configs = hp_ranges.random_configs(random_state, num_data)
    metric_values = random_state.randn(num_data)
    # Feed data to searcher
    for trial_id, config, resource, metric_val in zip(
        trial_ids, configs, resource_levels, metric_values
    ):
        searcher.on_trial_result(
            trial_id=trial_id,
            config=config,
            metric=metric_val,
            resource_level=resource,
        )

    # check that we have for each resource level one model
    assert len(searcher.models) == 3
