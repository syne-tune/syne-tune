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
    INTERNAL_METRIC_NAME, INTERNAL_CONSTRAINT_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import \
    DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import \
    CEIAcquisitionFunction
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


def _construct_models(X, Y, metric, hp_ranges, do_mcmc, with_pending):
    pending_tuples = [(0.5, 0.5), (0.2, 0.2)] if with_pending else None
    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y,
        pending_tuples=pending_tuples)
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


def default_models(metric, do_mcmc=True, with_pending=False) -> List[GaussProcSurrogateModel]:
    config_space = {
        'x': uniform(0.0, 1.0),
        'y': uniform(0.0, 1.0)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    X = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    # Continuous constraint, such as memory requirement: the larger x[0] the larger the memory footprint, and
    # the feasible region (i.e., Y <= 0) is for x[0] <= 0.5
    Y = [{INTERNAL_METRIC_NAME: np.sum(x) * 10.0,
          INTERNAL_CONSTRAINT_NAME: x[0] * 2.0 - 1.0}
         for x in X]
    result = _construct_models(X, Y, metric, hp_ranges, do_mcmc, with_pending)
    return result


def _build_models_with_and_without_feasible_candidates(do_mcmc, with_pending):
    active_models = default_models(INTERNAL_METRIC_NAME, do_mcmc=do_mcmc, with_pending=with_pending)
    constraint_models = default_models(INTERNAL_CONSTRAINT_NAME, do_mcmc=do_mcmc, with_pending=with_pending)
    active_models_infeasible = default_models_all_infeasible(
        INTERNAL_METRIC_NAME, do_mcmc=do_mcmc, with_pending=with_pending)
    constraint_models_infeasible = default_models_all_infeasible(
        INTERNAL_CONSTRAINT_NAME, do_mcmc=do_mcmc, with_pending=with_pending)

    all_active_models = active_models + active_models_infeasible
    all_constraint_models = constraint_models + constraint_models_infeasible
    return all_active_models, all_constraint_models


def default_models_all_infeasible(metric, do_mcmc=True, with_pending=False) -> List[GaussProcSurrogateModel]:
    config_space = {
        'x': uniform(0.0, 1.0),
        'y': uniform(0.0, 1.0)}
    hp_ranges = make_hyperparameter_ranges(config_space)
    X = [
        (0.5, 0.0),
        (0.6, 0.0),
        (0.7, 0.0),
        (0.8, 0.0),
        (0.9, 0.0),
        (1.0, 0.0),
        (0.5, 1.0),
        (1.0, 1.0),
    ]

    # Continuous constraint, such as memory requirement: the larger x[0] the larger the memory footprint, and
    # There are no feasible points.
    Y = [{INTERNAL_METRIC_NAME: np.sum(x) * 10.0,
          INTERNAL_CONSTRAINT_NAME: x[0] * 2.0 + 0.01}
         for x in X]
    result = _construct_models(X, Y, metric, hp_ranges, do_mcmc, with_pending)
    return result


def plot_ei_mean_std(model, cei, max_grid=1.0):
    import matplotlib.pyplot as plt

    grid = np.linspace(0, max_grid, 400)
    Xgrid, Ygrid = np.meshgrid(grid, grid)
    inputs = np.hstack([Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1)])
    Z_ei = cei.compute_acq(inputs)[0]
    predictions = model.predict(inputs)[0]
    Z_means = predictions['mean']
    Z_std = predictions['std']
    titles = ['CEI', 'mean', 'std']
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
    # - test that this holds both with and without pending candidates/fantasizing

    for with_pending in [False, True]:
        active_models = default_models(INTERNAL_METRIC_NAME, do_mcmc=False, with_pending=with_pending)
        constraint_models = default_models(INTERNAL_CONSTRAINT_NAME, do_mcmc=False, with_pending=with_pending)
        for active_model, constraint_model in zip(active_models, constraint_models):
            models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
            cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)
            X = np.array([
                (0.0, 0.0),  # 0
                (1.0, 0.0),  # 1
                (0.0, 1.0),  # 2
                (1.0, 1.0),  # 3
                (0.4, 0.0),  # 4
                (0.0, 0.4),  # 5
                (0.1, 0.0),  # 6
                (0.0, 0.1),  # 7
                (0.1, 0.1),  # 8
                (0.9, 0.9),  # 9
            ])
            acq = list(cei.compute_acq(X).flatten())
            assert all(a <= 0 for a in acq), acq

            # lower evaluations with lower memory footprint should correspond to better acquisition
            # second inequality is less equal because last two values are likely zero

            assert acq[0] < acq[1] <= acq[3], acq
            assert acq[8] < acq[9], acq

            # evaluations with same EI but lower memory footprint give a better CEI
            assert acq[5] < acq[4]
            assert acq[7] < acq[6]

            # further from an evaluated point should correspond to better acquisition
            assert acq[6] < acq[4] < acq[1], acq
            assert acq[7] < acq[5] < acq[2], acq


def test_no_feasible_candidates():
    # - test that values are negative as we should be returning *minus* expected improvement
    # - test that values that are further from evaluated candidates have higher expected improvement
    #   given similar mean
    # - test that points closer to better points have higher expected improvement
    # - test that this holds both with and without pending candidates/fantasizing

    for with_pending in [False, True]:
        active_models = default_models_all_infeasible(
            INTERNAL_METRIC_NAME, do_mcmc=False, with_pending=with_pending)
        constraint_models = default_models_all_infeasible(
            INTERNAL_CONSTRAINT_NAME, do_mcmc=False, with_pending=with_pending)
        for active_model, constraint_model in zip(active_models, constraint_models):
            models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
            cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)
            X = np.array([
                (1.0, 1.0),  # 0
                (1.0, 0.0),  # 1
                (0.5, 0.0),  # 2
                (0.4, 0.0),  # 3
                (0.3, 0.0),  # 4
                (0.2, 0.0),  # 5
                (0.1, 0.0),  # 6
                (.05, 0.0),  # 7
            ])
            acq = list(cei.compute_acq(X).flatten())

            # the acquisition function should return only non-positive values
            assert all(a <= 0 for a in acq), acq

            # at the evaluated unfeasible candidates, the probability of satisfying the constraint should be zero
            np.testing.assert_almost_equal(acq[0], acq[1])
            np.testing.assert_almost_equal(acq[0], 0.0)
            # evaluations that are further from the infeasible candidates
            # should have a larger probability of satisfying the constraint
            assert acq[2] >= acq[3] >= acq[4] >= acq[5] >= acq[6] >= acq[7], acq


@pytest.mark.skip("this unit test is skipped to save time")
def test_best_value():
    # test that the best value affects the constrained expected improvement
    active_models = default_models(INTERNAL_METRIC_NAME)
    constraint_models = default_models(INTERNAL_CONSTRAINT_NAME)
    for active_model, constraint_model in zip(active_models, constraint_models):
        models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
        cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)

        random = np.random.RandomState(42)
        test_X = random.uniform(low=0.0, high=1.0, size=(10, 2))

        acq_best0 = list(cei.compute_acq(test_X).flatten())
        zero_row = np.zeros((1, 2))
        acq0_best0 = cei.compute_acq(zero_row)

        # override current best
        cei._feasible_best_list = np.array([30])

        acq_best2 = list(cei.compute_acq(test_X).flatten())
        acq0_best2 = cei.compute_acq(zero_row)

        # if the best is only 30 the acquisition function should be better (lower value)
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
    constraint_models = default_models(INTERNAL_CONSTRAINT_NAME)
    for active_model, constraint_model in zip(active_models, constraint_models):
        models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
        cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)
        hp_ranges = active_model.hp_ranges_for_prediction()
        opt = LBFGSOptimizeAcquisition(
            hp_ranges, models, CEIAcquisitionFunction, INTERNAL_METRIC_NAME)
        non_zero_acq_at_least_once = False
        initial_point = random.uniform(low=0.0, high=0.1, size=(2,))
        acq0, df0 = cei.compute_acq_with_gradient(initial_point)
        if debug_output:
            print('\nInitial point: f(x0) = {}, x0 = {}'.format(
                acq0, initial_point))
            print('grad0 = {}'.format(df0))
        if acq0 != 0:
            non_zero_acq_at_least_once = True
            init_cand = hp_ranges.from_ndarray(initial_point)
            optimized = hp_ranges.to_ndarray(opt.optimize(init_cand))
            acq_opt = cei.compute_acq(optimized)[0]
            if debug_output:
                print('Final point: f(x1) = {}, x1 = {}'.format(
                    acq_opt, optimized))
            assert acq_opt < 0
            assert acq_opt < acq0

        assert non_zero_acq_at_least_once


def test_numerical_gradient():
    # test that the analytical gradient computation is correct by comparing to the numerical gradient
    # both when the feasible best exists and when it does not
    debug_output = False
    do_mcmc = False
    random = np.random.RandomState(42)
    eps = 1e-6

    for with_pending in [True, False]:
        all_active_models, all_constraint_models = \
            _build_models_with_and_without_feasible_candidates(do_mcmc, with_pending)

        for active_model, constraint_model in zip(all_active_models, all_constraint_models):
            models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
            cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)

            high = 1.0
            x = random.uniform(low=0.0, high=high, size=(2,))
            f0, analytical_gradient = cei.compute_acq_with_gradient(x)
            analytical_gradient = analytical_gradient.flatten()
            if debug_output:
                print('x0 = {}, f(x_0) = {}, grad(x_0) = {}'.format(
                    x, f0, analytical_gradient))

            for i in range(2):
                h = np.zeros_like(x)
                h[i] = eps
                fpeps = cei.compute_acq(x+h)[0]
                fmeps = cei.compute_acq(x-h)[0]
                numerical_derivative = (fpeps - fmeps) / (2 * eps)
                if debug_output:
                    print('f(x0+eps) = {}, f(x0-eps) = {}, findiff = {}, deriv = {}'.format(
                        fpeps[0], fmeps[0], numerical_derivative[0],
                        analytical_gradient[i]))
                np.testing.assert_almost_equal(
                    numerical_derivative.item(), analytical_gradient[i],
                    decimal=2)


def test_value_same_as_with_gradient():
    # test that compute_acq and compute_acq_with_gradients return the same acquisition values
    # both when the feasible best exists and when it does not
    do_mcmc = False
    for with_pending in [True, False]:
        all_active_models, all_constraint_models = \
            _build_models_with_and_without_feasible_candidates(do_mcmc, with_pending)
        for active_model, constraint_model in zip(all_active_models, all_constraint_models):
            models = {INTERNAL_METRIC_NAME: active_model, INTERNAL_CONSTRAINT_NAME: constraint_model}
            cei = CEIAcquisitionFunction(models, active_metric=INTERNAL_METRIC_NAME)

            random = np.random.RandomState(42)
            X = random.uniform(low=0.0, high=1.0, size=(10, 2))

            # assert same as computation with gradients
            vec1 = cei.compute_acq(X).flatten()
            vec2 = np.array([cei.compute_acq_with_gradient(x)[0] for x in X])
            np.testing.assert_almost_equal(vec1, vec2)


if __name__ == "__main__":
    test_optimization_improves()
    test_numerical_gradient()
