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
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    BoxCoxTargetTransform,
    BOXCOX_TARGET_THRES,
)


@pytest.mark.parametrize(
    "boxcox_lambda, compare_func",
    [
        (1.0, lambda y: y - 1.0),
        (0.0, lambda y: np.log(y)),
        (0.5, lambda y: 2.0 * (np.sqrt(y) - 1.0)),
        (-1.0, lambda y: 1.0 - np.reciprocal(y)),
        (-0.5, lambda y: 2.0 * (1.0 - np.reciprocal(np.sqrt(y)))),
        (2.0, lambda y: 0.5 * (np.square(y) - 1.0)),
        (0.0001, lambda y: np.log(y) * (0.00005 * np.log(y) + 1.0)),
        (-0.0001, lambda y: np.log(y) * (-0.00005 * np.log(y) + 1.0)),
    ],
)
def test_boxcox_transform_forward(boxcox_lambda, compare_func):
    random_seed = 3141592123
    random_state = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    num_data = 100

    targets = np.maximum(
        np.exp(random_state.normal(loc=1, scale=1, size=num_data)), BOXCOX_TARGET_THRES
    )
    target_transform = BoxCoxTargetTransform()
    target_transform.collect_params().initialize()
    target_transform.set_boxcox_lambda(boxcox_lambda)
    zvals = target_transform(targets)
    zvals_comp = compare_func(targets)
    np.testing.assert_almost_equal(zvals, zvals_comp, decimal=6)
