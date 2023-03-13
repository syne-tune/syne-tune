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
import autograd.numpy as anp
import numpy as np
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    Warping,
    warpings_for_hyperparameters,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    DATA_TYPE,
    NUMERICAL_JITTER,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    LogarithmScalarEncoding,
    PositiveScalarEncoding,
)
from syne_tune.config_space import uniform, randint, choice, ordinal, finrange
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges


def test_warping_encoding():
    dimension = 5
    coordinate_range = (1, 3)
    warping = Warping(dimension, coordinate_range)
    assert isinstance(warping.encoding, LogarithmScalarEncoding)
    assert warping.encoding.dimension == 2
    assert warping.lower == 1 and warping.upper == 3
    warping = Warping(dimension, coordinate_range, encoding_type="positive")
    assert isinstance(warping.encoding, PositiveScalarEncoding)


def test_warping_default_parameters():
    x = anp.array([0.0, 0.5, 1.0], dtype=DATA_TYPE).reshape((1, -1))
    warping = Warping(dimension=3)
    warping.collect_params().initialize()

    warping_a = warping.encoding.get(warping.power_a_internal.data())
    np.testing.assert_almost_equal(warping_a, anp.ones(3))

    expected = anp.array([NUMERICAL_JITTER, 0.5, 1.0 - NUMERICAL_JITTER]).reshape(
        (1, -1)
    )
    np.testing.assert_almost_equal(warping(x), expected)


def test_warping_with_arbitrary_parameters():
    x = anp.array([0.0, 0.5, 1.0], dtype=DATA_TYPE).reshape((1, -1))
    warping = Warping(dimension=3)
    warping.collect_params().initialize()
    warping.encoding.set(warping.power_a_internal, [2.0, 2.0, 2.0])
    warping.encoding.set(warping.power_b_internal, [0.5, 0.5, 0.5])
    warping_a = warping.encoding.get(warping.power_a_internal.data())
    np.testing.assert_almost_equal(warping_a, [2.0, 2.0, 2.0])
    # In that case (with parameters [2., 0.5]), the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1.0 - anp.sqrt(1.0 - x * x)

    expected = expected_warping(
        anp.array([NUMERICAL_JITTER, 0.5, 1.0 - NUMERICAL_JITTER]).reshape((1, -1))
    )
    np.testing.assert_almost_equal(warping(x), expected)


def test_warping_with_multidimension_and_arbitrary_parameters():
    X = anp.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=DATA_TYPE)
    dimension = 3

    # We transform only the columns {0,2} of the 3-dimensional data X
    warping0 = Warping(dimension, coordinate_range=(0, 1))
    warping2 = Warping(dimension, coordinate_range=(2, 3))
    warping0.collect_params().initialize()
    warping2.collect_params().initialize()

    # We change the warping parameters of the first dimension only
    warping0.set_params({"power_a": 2.0, "power_b": 0.5})

    # The parameters of w2 should be the default ones (as there was no set operations)
    w2_params = warping2.get_params()
    np.testing.assert_almost_equal(
        [w2_params["power_a"], w2_params["power_b"]], [1.0, 1.0]
    )

    # With parameters [2., 0.5], the warping is given by x => 1. - sqrt(1. - x^2)
    def expected_warping(x):
        return 1.0 - anp.sqrt(1.0 - x * x)

    expected_column0 = expected_warping(
        anp.array([1.0 - NUMERICAL_JITTER, 0.5, NUMERICAL_JITTER])
    ).reshape((-1, 1))
    expected_column1 = anp.array([0.5, 1.0, 0.0]).reshape((-1, 1))
    expected_column2 = anp.array(
        [NUMERICAL_JITTER, 0.5, 1.0 - NUMERICAL_JITTER]
    ).reshape((-1, 1))

    np.testing.assert_almost_equal(
        warping0(warping2(X)),
        anp.hstack([expected_column0, expected_column1, expected_column2]),
    )


@pytest.mark.parametrize(
    "config_space, result",
    [
        (
            {
                "a": choice(["a", "b"]),
                "b": choice(["1", "2", "3"]),
                "c": choice([1, 2]),
            },
            [],
        ),
        (
            {
                "a": ordinal([1, 2]),
                "b": uniform(0.5, 1.5),
                "c": finrange(0.125, 1, 5),
            },
            [(0, 3)],
        ),
        (
            {
                "a": ordinal([1, 2]),  # 0
                "b": uniform(0.5, 1.5),  # 1
                "c": choice(["a", "b"]),  # 2
                "d": choice(["1", "2", "3"]),  # 3
                "e": finrange(0.125, 1, 5),  # 6
                "f": randint(0, 25),  # 7
                "g": choice([1, 2]),  # 8
            },
            [(0, 2), (6, 8)],
        ),
        (
            {
                "a": choice(["1", "2", "3"]),  # 0
                "b": ordinal([1, 2]),  # 3
                "c": uniform(0.5, 1.5),  # 4
                "d": choice(["a", "b"]),  # 5
                "e": finrange(0.125, 1, 5),  # 6
                "f": randint(0, 25),  # 7
                "g": choice([1, 2]),  # 8
            },
            [(3, 5), (6, 8)],
        ),
        (
            {
                "a": choice(["1", "2", "3"]),
            },
            [],
        ),
        (
            {
                "a": ordinal([1, 2]),
            },
            [(0, 1)],
        ),
    ],
)
def test_warpings_for_hyperparameters(config_space, result):
    hp_ranges = make_hyperparameter_ranges(config_space)
    warpings = warpings_for_hyperparameters(hp_ranges)
    ranges = [(warping.lower, warping.upper) for warping in warpings]
    assert ranges == result


# This test is based on:
# y = f(x, a, b) <-> 1 - x = f(1 - y, 1/b, 1/a)
def test_warping_and_inverse():
    random_seed = 3141592123
    random_state = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    num_data = 10
    num_iter = 50
    for _ in range(num_iter):
        dimension = random_state.randint(low=2, high=20)
        lower = random_state.randint(low=0, high=dimension)
        if lower < dimension - 1:
            upper = random_state.randint(low=lower + 1, high=dimension + 1)
        else:
            upper = lower + 1
        warping = Warping(dimension, coordinate_range=(lower, upper))
        warping.collect_params().initialize()
        size = upper - lower
        avals = random_state.uniform(low=0.33, high=3, size=size)
        bvals = random_state.uniform(low=0.33, high=3, size=size)
        xmat = random_state.uniform(low=0, high=1, size=(num_data, dimension))
        # y = f(x, a, b)
        warping.encoding.set(warping.power_a_internal, avals)
        warping.encoding.set(warping.power_b_internal, bvals)
        ymat = warping(xmat)
        # 1 - x = f(1 - y, 1/b, 1/a)
        warping.encoding.set(warping.power_a_internal, 1.0 / bvals)
        warping.encoding.set(warping.power_b_internal, 1.0 / avals)
        xmat2 = 1.0 - warping(1.0 - ymat)
        np.testing.assert_almost_equal(xmat, xmat2, decimal=4)


@pytest.mark.parametrize(
    "a, b, func",
    [
        (1, 1, lambda x: x),
        (2, 0.5, lambda x: 1.0 - np.sqrt(1.0 - np.square(x))),
        (0.5, 2, lambda x: 1.0 - np.square(1.0 - np.sqrt(x))),
    ],
)
def test_ab_fixed(a, b, func):
    random_seed = 3141592123
    random_state = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    num_data = 10
    num_iter = 5
    for _ in range(num_iter):
        dimension = random_state.randint(low=2, high=20)
        lower = random_state.randint(low=0, high=dimension)
        if lower < dimension - 1:
            upper = random_state.randint(low=lower + 1, high=dimension + 1)
        else:
            upper = lower + 1
        warping = Warping(dimension, coordinate_range=(lower, upper))
        warping.collect_params().initialize()
        size = upper - lower
        avals = np.full(shape=(size,), fill_value=a)
        bvals = np.full(shape=(size,), fill_value=b)
        xmat = random_state.uniform(low=0, high=1, size=(num_data, dimension))
        warping.encoding.set(warping.power_a_internal, avals)
        warping.encoding.set(warping.power_b_internal, bvals)
        ymat = warping(xmat)
        ymat2 = xmat.copy()
        ymat2[:, lower:upper] = func(xmat[:, lower:upper])
        np.testing.assert_almost_equal(ymat, ymat2, decimal=4)
