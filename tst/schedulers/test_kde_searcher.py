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
    metric = "error"
    resource_attr = "epoch"
    searcher = MultiFidelityKernelDensityEstimator(
        config_space=config_space,
        metric=metric,
        points_to_evaluate=[],
        top_n_percent=top_n_percent,
        resource_attr=resource_attr,
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
        searcher._update(
            trial_id=trial_id,
            config=config,
            result={metric: metric_val, resource_attr: resource},
        )
    # Test n_good
    num_features = len(hp_cols)
    assert searcher.num_min_data_points == num_features
    assert searcher._highest_resource_model_can_fit(num_features) == resource_acq, (
        resource_levels,
        top_n_percent,
        resource_acq,
    )
