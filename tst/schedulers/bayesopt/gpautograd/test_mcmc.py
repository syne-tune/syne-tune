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
import numpy
import autograd.numpy as anp
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd import SliceException
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.slice import (
    SliceSampler,
    slice_sampler_step_out,
    slice_sampler_step_in,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Normal,
    LogNormal,
    Horseshoe,
    Uniform,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    Warping,
    WarpedKernel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import Matern52
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    ScalarMeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gpr_mcmc import (
    GPRegressionMCMC,
    _get_gp_hps,
    _set_gp_hps,
    _create_likelihood,
)


# This is just to make the tests work. In GPRegressionMCMC, lower and upper
# bounds are dealt with through the encoding
def old_log_likelihood(x, distribution, lower=-float("inf"), upper=float("inf")):
    if any(x < lower) or any(x > upper):
        return -float("inf")
    return -distribution(x)


def test_uniform():
    uniform = Uniform(0.0, 1.0)
    lower, upper = 0.0, 1.0
    assert old_log_likelihood(
        anp.array([0.2]), uniform, lower, upper
    ) == old_log_likelihood(anp.array([0.3]), uniform, lower, upper)
    assert old_log_likelihood(anp.array([2.0]), uniform, lower, upper) == -float("inf")
    assert old_log_likelihood(anp.array([-1.0]), uniform, lower, upper) == -float("inf")


def test_normal():
    normal = Normal(0, 1)
    lower, upper = -1e3, 1e3
    assert old_log_likelihood(
        anp.array([0.0]), normal, lower, upper
    ) > old_log_likelihood(anp.array([0.1]), normal, lower, upper)
    assert old_log_likelihood(
        anp.array([0.0]), normal, lower, upper
    ) > old_log_likelihood(anp.array([-0.1]), normal, lower, upper)
    assert old_log_likelihood(anp.array([1e4]), normal, lower, upper) == -float("inf")
    assert old_log_likelihood(anp.array([-1e4]), normal, lower, upper) == -float("inf")


def test_log_normal():
    log_normal = LogNormal(0.0, 1.0)
    lower, upper = 1e-6, 1e9
    assert old_log_likelihood(
        anp.array([1.0]), log_normal, lower, upper
    ) > old_log_likelihood(anp.array([1.1]), log_normal, lower, upper)
    assert old_log_likelihood(
        anp.array([1.0]), log_normal, lower, upper
    ) < old_log_likelihood(anp.array([0.9]), log_normal, lower, upper)
    assert old_log_likelihood(anp.array([1e10]), log_normal, lower, upper) == -float(
        "inf"
    )
    assert old_log_likelihood(anp.array([1e-8]), log_normal, lower, upper) == -float(
        "inf"
    )


def test_horse_shoe():
    horse_shoe = Horseshoe(0.1)
    lower, upper = 1e-6, 1e6
    assert old_log_likelihood(
        anp.array([0.01]), horse_shoe, lower, upper
    ) > old_log_likelihood(anp.array([0.1]), horse_shoe, lower, upper)
    assert old_log_likelihood(anp.array([1e-7]), horse_shoe, lower, upper) == -float(
        "inf"
    )
    assert old_log_likelihood(anp.array([1e7]), horse_shoe, lower, upper) == -float(
        "inf"
    )


def test_slice_normal():
    normal = Normal(0, 1)
    random_state = anp.random.RandomState(0)
    slice = SliceSampler(
        log_density=lambda x: old_log_likelihood(x, normal),
        scale=1.0,
        random_state=random_state,
    )
    samples = slice.sample(anp.array([0.0]), 5000, 1, 1)
    numpy.testing.assert_almost_equal(anp.mean(samples), 0.0, decimal=2)
    numpy.testing.assert_almost_equal(anp.std(samples), 1.0, decimal=2)


def test_slice_step_out():
    normal = Normal(0, 1)

    def sliced_log_density(x):
        return old_log_likelihood(anp.array([x]), normal)

    # the lower and upper bound should has log density smaller than this log_pivot
    log_pivot = sliced_log_density(1.0)
    random_state = anp.random.RandomState(0)
    lower, upper = slice_sampler_step_out(
        log_pivot, 0.1, sliced_log_density, random_state
    )
    assert lower < -1.0 and upper > 1.0

    log_pivot = sliced_log_density(100)
    with pytest.raises(
        SliceException
    ):  # the log_pivot is too small so need > 200 steps
        slice_sampler_step_out(log_pivot, 0.1, sliced_log_density, random_state)


def test_slice_step_in():
    normal = Normal(0.0, 1.0)

    def sliced_log_density(x):
        return old_log_likelihood(anp.array([x]), normal)

    log_pivot = sliced_log_density(
        1.0
    )  # the movement should between [-1., 1.] after step in
    random_state = anp.random.RandomState(0)
    movement = slice_sampler_step_in(
        -20.0, 20.0, log_pivot, sliced_log_density, random_state
    )
    assert -1.0 < movement < 1.0

    with pytest.raises(SliceException):  # when bound is off, should get SliceException
        slice_sampler_step_in(2.0, 10.0, log_pivot, sliced_log_density, random_state)


def test_get_gp_hps():
    mean = ScalarMeanFunction()
    kernel = Matern52(dimension=1)
    warping = Warping(dimension=1, index_to_range={0: (-4.0, 4.0)})
    warped_kernel = WarpedKernel(kernel=kernel, warping=warping)
    likelihood = GaussianProcessMarginalLikelihood(
        kernel=warped_kernel, mean=mean, initial_noise_variance=1e-6
    )
    likelihood.initialize(force_reinit=True)
    likelihood.hybridize()
    hp_values = _get_gp_hps(likelihood)
    # the oder of hps are noise, mean, covariance scale, bandwidth, warping a, warping b
    numpy.testing.assert_array_almost_equal(
        hp_values, anp.array([1e-6, 0.0, 1.0, 1.0, 1.0, 1.0])
    )


def test_set_gp_hps():
    mean = ScalarMeanFunction()
    kernel = Matern52(dimension=1)
    warping = Warping(dimension=1, index_to_range={0: (-4.0, 4.0)})
    warped_kernel = WarpedKernel(kernel=kernel, warping=warping)
    likelihood = GaussianProcessMarginalLikelihood(
        kernel=warped_kernel, mean=mean, initial_noise_variance=1e-6
    )
    likelihood.initialize(force_reinit=True)
    likelihood.hybridize()
    hp_values = anp.array([1e-2, 1.0, 0.5, 0.3, 0.2, 1.1])
    _set_gp_hps(hp_values, likelihood)
    numpy.testing.assert_array_almost_equal(hp_values, _get_gp_hps(likelihood))


def test_create_likelihood():
    def build_kernel():
        kernel = Matern52(dimension=1)
        warping = Warping(dimension=1, index_to_range={0: (-4.0, 4.0)})
        return WarpedKernel(kernel=kernel, warping=warping)

    random_state = anp.random.RandomState(0)
    likelihood1 = _create_likelihood(build_kernel, random_state=random_state)
    likelihood2 = _create_likelihood(build_kernel, random_state=random_state)
    numpy.testing.assert_array_almost_equal(
        _get_gp_hps(likelihood1), _get_gp_hps(likelihood2)
    )


@pytest.mark.skip(reason="Need manual inspection on the plots")
def test_mcmc():
    import matplotlib.pyplot as plt

    anp.random.seed(7)

    def f_n(x):
        noise = anp.random.normal(0.0, 0.25, x.shape[0])
        return 0.1 * anp.power(x, 3) + noise

    def f(x):
        return 0.1 * anp.power(x, 3)

    x_train = anp.concatenate(
        (anp.random.uniform(-4.0, -1.0, 40), anp.random.uniform(1.0, 4.0, 40))
    )
    y_train = f_n(x_train)
    x_test = anp.sort(anp.random.uniform(-4.0, 4.0, 200))

    y_train_np_nd = anp.array(y_train, dtype=anp.float64)
    x_train_np_nd = anp.array(x_train, dtype=anp.float64)
    x_test_np_nd = anp.array(x_test, dtype=anp.float64)

    def build_kernel():
        return WarpedKernel(
            kernel=Matern52(dimension=1),
            warping=Warping(dimension=1, index_to_range={0: (-4.0, 4.0)}),
        )

    model_mcmc = GPRegressionMCMC(build_kernel=build_kernel, random_seed=1)
    data = {"features": x_train_np_nd, "targets": y_train_np_nd}
    model_mcmc.fit(data)
    mcmc_predictions = model_mcmc.predict(x_test_np_nd)

    for mcmc_mean, mcmc_var in mcmc_predictions:
        mcmc_mean, mcmc_std = mcmc_mean, anp.sqrt(mcmc_var)
        plt.figure()
        plt.scatter(x_train, y_train, color="red", label="observations")
        plt.plot(x_test, f(x_test), color="black", label="ground truth")
        plt.plot(x_test, mcmc_mean, color="blue", label="mcmc prediction")
        plt.fill_between(
            x_test, mcmc_mean - 1.96 * mcmc_std, mcmc_mean + 1.96 * mcmc_std, alpha=0.5
        )
        plt.legend()
    plt.show()
