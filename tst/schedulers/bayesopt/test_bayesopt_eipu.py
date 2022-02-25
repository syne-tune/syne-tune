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
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import \
    dictionarize_objective, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import \
    DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import \
    EIpuAcquisitionFunction
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


COST_METRIC_NAME = 'cost_metric'


def default_models(metric, do_mcmc=True) -> List[GaussProcSurrogateModel]:
    config_space = {
        'x': uniform(0.0, 1.0),
        'y': uniform(0.0, 1.0)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    X = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    if metric == INTERNAL_METRIC_NAME:
        Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]
    elif metric == COST_METRIC_NAME:
        # Increasing the first hp increases cost
        Y = [{metric: 1.0 + x[0] * 2.0}
             for x in X]
    else:
        raise ValueError(f"{metric} is not a valid metric")
    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y)
    random_seed = 0

    gpmodel = default_gpmodel(
        state, random_seed=random_seed,
        optimization_config=DEFAULT_OPTIMIZATION_CONFIG)
    model_factory = GaussProcEmpiricalBayesModelFactory(
        active_metric=metric, gpmodel=gpmodel, num_fantasy_samples=2)
    result = [model_factory.model(state, fit_params=True)]
    if do_mcmc:
        gpmodel_mcmc = default_gpmodel_mcmc(
            state, random_seed=random_seed,
            mcmc_config=DEFAULT_MCMC_CONFIG)
        model_factory = GaussProcMCMCModelFactory(
            active_metric=metric, gpmodel=gpmodel_mcmc)
        result.append(model_factory.model(state, fit_params=True))
    return result


def plot_ei_mean_std(model, eipu, max_grid=1.0):
    import matplotlib.pyplot as plt

    grid = np.linspace(0, max_grid, 400)
    Xgrid, Ygrid = np.meshgrid(grid, grid)
    inputs = np.hstack([Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1)])
    Z_ei = eipu.compute_acq(inputs)[0]
    predictions = model.predict(inputs)[0]
    Z_means = predictions['mean']
    Z_std = predictions['std']
    titles = ['EIpu', 'mean', 'std']
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
    active_models = default_models(INTERNAL_METRIC_NAME, do_mcmc=False)
    cost_models = default_models(COST_METRIC_NAME, do_mcmc=False)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {INTERNAL_METRIC_NAME: active_model, COST_METRIC_NAME: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)
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
        acq = list(eipu.compute_acq(X).flatten())
        assert all(a <= 0 for a in acq), acq

        # lower evaluations with lower cost should correspond to better acquisition
        # second inequality is less equal because last two values are likely zero
        assert acq[0] < acq[1] <= acq[3], acq
        assert acq[8] < acq[9], acq

        # evaluations with same EI but lower cost give a better EIpu
        assert acq[5] < acq[4]
        assert acq[7] < acq[6]

        # further from an evaluated point should correspond to better acquisition
        assert acq[6] < acq[4] < acq[1], acq
        assert acq[7] < acq[5] < acq[2], acq


def test_best_value():
    # test that the best value affects the cost-aware expected improvement
    active_models = default_models(INTERNAL_METRIC_NAME)
    cost_models = default_models(COST_METRIC_NAME)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {INTERNAL_METRIC_NAME: active_model, COST_METRIC_NAME: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)

        random = np.random.RandomState(42)
        test_X = random.uniform(low=0.0, high=1.0, size=(10, 2))

        acq_best0 = list(eipu.compute_acq(test_X).flatten())
        zero_row = np.zeros((1, 2))
        acq0_best0 = eipu.compute_acq(zero_row)

        # override current best
        eipu._current_bests = ActiveMetricCurrentBestProvider(
            [np.array([30.0])])

        acq_best2 = list(eipu.compute_acq(test_X).flatten())
        acq0_best2 = eipu.compute_acq(zero_row)

        # if the best is only 2 the acquisition function should be better (lower value)
        assert all(a2 < a0 for a2, a0 in zip(acq_best2, acq_best0))

        # there should be a considerable gap at the point of the best evaluation
        assert acq0_best2 < acq0_best0 - 1.0


@pytest.mark.skip("this unit test is skipped to save time")
def test_optimization_improves():
    debug_output = False
    # Pick a random point, optimize and the expected improvement should be better:
    # But only if the starting point is not too far from the origin
    random = np.random.RandomState(42)
    active_models = default_models(INTERNAL_METRIC_NAME)
    cost_models = default_models(COST_METRIC_NAME)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {INTERNAL_METRIC_NAME: active_model, COST_METRIC_NAME: cost_model}
        eipu = EIpuAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)
        hp_ranges = active_model.hp_ranges_for_prediction()
        opt = LBFGSOptimizeAcquisition(
            hp_ranges, models, EIpuAcquisitionFunction, INTERNAL_METRIC_NAME)
        non_zero_acq_at_least_once = False
        initial_point = random.uniform(low=0.0, high=0.1, size=(2,))
        acq0, df0 = eipu.compute_acq_with_gradient(initial_point)
        if debug_output:
            print('\nInitial point: f(x0) = {}, x0 = {}'.format(
                acq0, initial_point))
            print('grad0 = {}'.format(df0))
        if acq0 != 0:
            non_zero_acq_at_least_once = True
            init_cand = hp_ranges.from_ndarray(initial_point)
            optimized = hp_ranges.to_ndarray(opt.optimize(init_cand))
            acq_opt = eipu.compute_acq(optimized)[0]
            if debug_output:
                print('Final point: f(x1) = {}, x1 = {}'.format(
                    acq_opt, optimized))
            assert acq_opt < 0
            assert acq_opt < acq0

        assert non_zero_acq_at_least_once


def test_numerical_gradient():
    debug_output = False
    random = np.random.RandomState(42)
    eps = 1e-6

    active_models = default_models(INTERNAL_METRIC_NAME)
    cost_models = default_models(COST_METRIC_NAME)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {INTERNAL_METRIC_NAME: active_model, COST_METRIC_NAME: cost_model}
        for exponent_cost in [1.0, 0.5, 0.2]:
            eipu = EIpuAcquisitionFunction(
                models, active_metric=INTERNAL_METRIC_NAME,
                exponent_cost=exponent_cost)
            high = 0.02
            x = random.uniform(low=0.0, high=high, size=(2,))
            f0, analytical_gradient = eipu.compute_acq_with_gradient(x)
            analytical_gradient = analytical_gradient.flatten()
            if debug_output:
                print('x0 = {}, f(x_0) = {}, grad(x_0) = {}'.format(
                    x, f0, analytical_gradient))
            for i in range(2):
                h = np.zeros_like(x)
                h[i] = eps
                fpeps = eipu.compute_acq(x+h)[0]
                fmeps = eipu.compute_acq(x-h)[0]
                numerical_derivative = (fpeps - fmeps) / (2 * eps)
                if debug_output:
                    print('f(x0+eps) = {}, f(x0-eps) = {}, findiff = {}, deriv = {}'.format(
                        fpeps, fmeps, numerical_derivative,
                        analytical_gradient[i]))
                np.testing.assert_almost_equal(
                    numerical_derivative, analytical_gradient[i],
                    decimal=4)


def test_value_same_as_with_gradient():
    # test that compute_acq and compute_acq_with_gradients return the same acquisition values
    active_models = default_models(INTERNAL_METRIC_NAME)
    cost_models = default_models(COST_METRIC_NAME)
    for active_model, cost_model in zip(active_models, cost_models):
        models = {INTERNAL_METRIC_NAME: active_model, COST_METRIC_NAME: cost_model}
        for exponent_cost in [1.0, 0.5, 0.2]:
            eipu = EIpuAcquisitionFunction(
                models, active_metric=INTERNAL_METRIC_NAME,
                exponent_cost=exponent_cost)
            random = np.random.RandomState(42)
            X = random.uniform(low=0.0, high=1.0, size=(10, 2))
            # assert same as computation with gradients
            vec1 = eipu.compute_acq(X).flatten()
            vec2 = np.array([eipu.compute_acq_with_gradient(x)[0] for x in X])
            np.testing.assert_almost_equal(vec1, vec2)
