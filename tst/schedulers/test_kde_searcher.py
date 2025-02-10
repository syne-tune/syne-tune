import pytest
import numpy as np

from syne_tune.optimizer.schedulers.searchers.kde import (
    MultiFidelityKernelDensityEstimator,
)
from syne_tune.config_space import choice
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


@pytest.mark.parametrize(
    "resource_levels, top_n_percent, resource_acq",
    [
        (
            [],
            15,
            None,
        ),
        (
            [1, 1, 1, 3, 3, 1, 1],
            15,
            None,
        ),
        (
            [1] * 6 + [3] * 2,
            15,
            None,
        ),
        (
            [3] * 3 + [1] * 19,
            20,
            None,
        ),
        (
            [3] * 3 + [1] * 20,
            20,
            1,
        ),
        (
            [3] * 20 + [1] * 25 + [9] * 9,
            20,
            3,
        ),
        (
            [3] * 3 + [1] * 20,
            80,
            1,
        ),
        (
            [3] * 3 + [1] * 20,
            85,
            None,
        ),
    ],
)
def test_train_kde_multifidelity(resource_levels, top_n_percent, resource_acq):
    random_seed = 31415927
    random_state = np.random.RandomState(random_seed)
    hp_cols = ("hp_x0", "hp_x1", "hp_x2")
    config_space = {
        node: choice(
            ["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"]
        )
        for node in hp_cols
    }
    searcher = MultiFidelityKernelDensityEstimator(
        config_space=config_space,
        points_to_evaluate=[],
        top_n_percent=top_n_percent,
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
    # Test n_good
    num_features = len(hp_cols)

    for model in searcher.models.values():
        assert model.num_min_data_points == num_features

    # check that we have for each resource level one model
    assert len(searcher.models) == np.unique(resource_levels).shape[0]
