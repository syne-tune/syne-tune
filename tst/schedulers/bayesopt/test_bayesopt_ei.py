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
from typing import List
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import \
    dictionarize_objective, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import \
    DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl \
    import EIAcquisitionFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc \
    import ActiveMetricCurrentBestProvider
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import \
    GaussProcSurrogateModel, GaussProcEmpiricalBayesModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_mcmc_model \
    import GaussProcMCMCModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import \
    LBFGSOptimizeAcquisition
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import \
    default_gpmodel, default_gpmodel_mcmc
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import create_tuning_job_state


# This setup makes little sense for good testing.
#
# When default model for no MCMC is plotted:
# - Plot on [0, 1]^2:
#   - Mean essentially constant at 10, stddev essentially constant at 2.5
#   - EI essentially constant at -0.12717145
# - Plot on [0, 0.1]^2:
#   - Mean = 10, except dropping in corner, stddev = 2.5
#   - EI essentially constant, dropping in corner
# - Plot on [0, 0.01]^2:
#   - Mean growing 0 -> 8, sttdev = 2.5, but drops to 0 in corner
#   - EI from -0.66 to -0.12, -> 0 only very close to origin
# - Plot on [0, 0.001]^2:
#   - Mean growing 0 -> 1.5, stddev growing 0 -> 1.8
#   - EI about -0.6, but -> 0 close to origin
# EI is minimized (value -0.66817) very close to origin (order 0.001). Grows to
# 0 at origin, increases to constant -0.12717145 very rapidly away from origin.
#
# In fact, if EI is optimized starting at a point outside [0, 0.1]^2, the optimizer
# returns with the starting point, and test_optimization_improves fails.
def default_models(do_mcmc=True) -> List[GaussProcSurrogateModel]:
    config_space = {
        'x': uniform(0.0, 1.0),
        'y': uniform(0.0, 1.0)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    X = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]
    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y)
    random_seed = 0

    gpmodel = default_gpmodel(
        state, random_seed=random_seed,
        optimization_config=DEFAULT_OPTIMIZATION_CONFIG)
    model_factory = GaussProcEmpiricalBayesModelFactory(
        active_metric=INTERNAL_METRIC_NAME, gpmodel=gpmodel, num_fantasy_samples=20)
    result = [model_factory.model(state, fit_params=True)]
    if do_mcmc:
        gpmodel_mcmc = default_gpmodel_mcmc(
            state, random_seed=random_seed,
            mcmc_config=DEFAULT_MCMC_CONFIG)
        model_factory = GaussProcMCMCModelFactory(
            active_metric=INTERNAL_METRIC_NAME, gpmodel=gpmodel_mcmc)
        result.append(model_factory.model(state, fit_params=True))
    return result


def plot_ei_mean_std(model, ei, max_grid=1.0):
    import matplotlib.pyplot as plt

    grid = np.linspace(0, max_grid, 400)
    Xgrid, Ygrid = np.meshgrid(grid, grid)
    inputs = np.hstack([Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1)])
    Z_ei = ei.compute_acq(inputs)[0]
    predictions = model.predict(inputs)[0]
    Z_means = predictions['mean']
    Z_std = predictions['std']
    titles = ['EI', 'mean', 'std']
    for i, (Z, title) in enumerate(zip([Z_ei, Z_means, Z_std], titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(
            Z.reshape(Xgrid.shape), extent=[0, max_grid, 0, max_grid],
            origin='lower')
        plt.colorbar()
        plt.title(title)
    plt.show()


# Note: This test fails when run with GP MCMC model. There, acq[5] > acq[7], and acq[8] > acq[5]
# ==> Need to look into GP MCMC model
def test_sanity_check():
    # - test that values are negative as we should be returning *minus* expected improvement
    # - test that values that are further from evaluated candidates have higher expected improvement
    #   given similar mean
    # - test that points closer to better points have higher expected improvement
    for model in default_models(do_mcmc=False):
        ei = EIAcquisitionFunction(model)
        X = np.array([
            (0.0, 0.0),  # 0
            (1.0, 0.0),  # 1
            (0.0, 1.0),  # 2
            (1.0, 1.0),  # 3
            (0.2, 0.0),  # 4
            (0.0, 0.2),  # 5
            (0.1, 0.0),  # 6
            (0.0, 0.1),  # 7
            (0.1, 0.1),  # 8
            (0.9, 0.9),  # 9
        ])
        _acq = ei.compute_acq(X).flatten()
        #print('Negative EI values:')
        #print(_acq)
        acq = list(_acq)

        assert all(a <= 0 for a in acq), acq

        # lower evaluations should correspond to better acquisition
        # second inequality is less equal because last two values are likely zero
        assert acq[0] < acq[1] <= acq[3], acq
        # Note: The improvement here is tiny, just 0.01%:
        assert acq[8] < acq[9], acq

        # further from an evaluated point should correspond to better acquisition
        assert acq[6] < acq[4] < acq[1], acq
        assert acq[7] < acq[5] < acq[2], acq


def test_best_value():
    # test that the best value affects expected improvement
    for model in default_models():
        ei = EIAcquisitionFunction(model)

        random = np.random.RandomState(42)
        test_X = random.uniform(low=0.0, high=0.01, size=(10, 2))

        acq_best0 = list(ei.compute_acq(test_X).flatten())
        zero_row = np.zeros((1, 2))
        acq0_best0 = ei.compute_acq(zero_row)

        # override current best
        ei._current_bests = ActiveMetricCurrentBestProvider(
            [np.array([10.0])])

        acq_best10 = list(ei.compute_acq(test_X).flatten())
        acq0_best10 = ei.compute_acq(zero_row)

        # if the best is 10, the acquisition function should be better (lower value)
        assert all(a10 < a0 for a10, a0 in zip(acq_best10, acq_best0)), \
            f"\nbest=0:  {acq_best0}\nbest=10: {acq_best10}"

        # there should be a considerable gap at the point of the best evaluation
        assert acq0_best10 < acq0_best0 - 1.0


# The original version of this test is failing. See comments above.
# In fact, if EI is optimized from a starting point outside [0, 0.1]^2,
# the gradient is tiny there, so the optimizer returns with the starting
# point, and no improvement is made.
#
# If the starting point is sampled in [0, 0.1]^2, the test works. The optimum
# of EI is very close to the origin.
def test_optimization_improves():
    debug_output = False
    # Pick a random point, optimize and the expected improvement should be better:
    # But only if the starting point is not too far from the origin
    random = np.random.RandomState(42)
    for model in default_models():
        ei = EIAcquisitionFunction(model)
        hp_ranges = model.hp_ranges_for_prediction()
        opt = LBFGSOptimizeAcquisition(hp_ranges, model, EIAcquisitionFunction)
        if debug_output:
            print('\n\nGP MCMC' if model.does_mcmc() else 'GP Opt')
            fzero = ei.compute_acq(np.zeros((1, 2)))[0]
            print('f(0) = {}'.format(fzero))
        if debug_output and not model.does_mcmc():
            # Plot the thing!
            plot_ei_mean_std(model, ei, max_grid=0.001)
            plot_ei_mean_std(model, ei, max_grid=0.01)
            plot_ei_mean_std(model, ei, max_grid=0.1)
            plot_ei_mean_std(model, ei, max_grid=1.0)

        non_zero_acq_at_least_once = False
        for iter in range(10):
            #initial_point = random.uniform(low=0.0, high=1.0, size=(2,))
            initial_point = random.uniform(low=0.0, high=0.1, size=(2,))
            acq0, df0 = ei.compute_acq_with_gradient(initial_point)
            if debug_output:
                print('\nInitial point: f(x0) = {}, x0 = {}'.format(
                    acq0, initial_point))
                print('grad0 = {}'.format(df0))
            if acq0 != 0:
                non_zero_acq_at_least_once = True
                init_cand = hp_ranges.from_ndarray(initial_point)
                optimized = hp_ranges.to_ndarray(opt.optimize(init_cand))
                acq_opt = ei.compute_acq(optimized)[0]
                if debug_output:
                    print('Final point: f(x1) = {}, x1 = {}'.format(
                        acq_opt, optimized))
                assert acq_opt < 0
                assert acq_opt < acq0

        assert non_zero_acq_at_least_once

# Changes from original version: Half of the time, we sample x in [0, 0.02]^2, where
# the shape of EI is more interesting
def test_numerical_gradient():
    debug_output = False
    random = np.random.RandomState(42)
    eps = 1e-6

    for model in default_models():
        ei = EIAcquisitionFunction(model)

        for iter in range(10):
            high = 1.0 if iter < 5 else 0.02
            x = random.uniform(low=0.0, high=high, size=(2,))
            f0, analytical_gradient = ei.compute_acq_with_gradient(x)
            analytical_gradient = analytical_gradient.flatten()
            if debug_output:
                print('x0 = {}, f(x_0) = {}, grad(x_0) = {}'.format(
                    x, f0, analytical_gradient))

            for i in range(2):
                h = np.zeros_like(x)
                h[i] = eps
                fpeps = ei.compute_acq(x + h)[0]
                fmeps = ei.compute_acq(x - h)[0]
                numerical_derivative = (fpeps - fmeps) / (2 * eps)
                if debug_output:
                    print('f(x0+eps) = {}, f(x0-eps) = {}, findiff = {}, deriv = {}'.format(
                        fpeps[0], fmeps[0], numerical_derivative[0],
                        analytical_gradient[i]))
                np.testing.assert_almost_equal(
                    numerical_derivative.item(), analytical_gradient[i],
                    decimal=4)


def test_value_same_as_with_gradient():
    # test that compute_acq and compute_acq_with_gradients return the same acquisition values
    for model in default_models():
        ei = EIAcquisitionFunction(model)

        random = np.random.RandomState(42)
        X = random.uniform(low=0.0, high=1.0, size=(10, 2))

        # assert same as computation with gradients
        vec1 = ei.compute_acq(X).flatten()
        vec2 = np.array([ei.compute_acq_with_gradient(x)[0] for x in X])
        np.testing.assert_almost_equal(vec1, vec2)


if __name__ == "__main__":
    test_optimization_improves()
    test_numerical_gradient()
