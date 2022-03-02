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

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import dictionarize_objective, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import DEFAULT_MCMC_CONFIG, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model \
    import GaussProcEmpiricalBayesModelFactory, GaussProcSurrogateModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_mcmc_model \
    import GaussProcMCMCModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl \
    import EIAcquisitionFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import default_gpmodel, default_gpmodel_mcmc
from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import create_tuning_job_state, tuples_to_configs
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges


def _simple_hp_ranges() -> HyperparameterRanges:
    return make_hyperparameter_ranges({
        'x': uniform(0.0, 1.0),
        'y': uniform(0.0, 1.0)})


@pytest.fixture(scope='function')
def tuning_job_state() -> TuningJobState:
    hp_ranges = _simple_hp_ranges()
    X = [(0.0, 0.0),
         (1.0, 0.0),
         (0.0, 1.0),
         (1.0, 1.0)]
    Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]
    return create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y)


def _set_seeds(seed=0):
    np.random.seed(seed)


def _make_model_gp_optimize(
        state: TuningJobState, random_seed,
        opt_config=DEFAULT_OPTIMIZATION_CONFIG, num_fantasy_samples=20,
        normalize_targets=True, fit_params=True):
    gpmodel = default_gpmodel(
        state, random_seed, optimization_config=opt_config)
    model_factory = GaussProcEmpiricalBayesModelFactory(
        active_metric=INTERNAL_METRIC_NAME, gpmodel=gpmodel,
        num_fantasy_samples=num_fantasy_samples,
        normalize_targets=normalize_targets)
    model = model_factory.model(state, fit_params=fit_params)
    return model, gpmodel


def _make_model_mcmc(
        state: TuningJobState, random_seed,
        mcmc_config=DEFAULT_MCMC_CONFIG, fit_params=True):
    gpmodel = default_gpmodel_mcmc(
        state, random_seed, mcmc_config=mcmc_config)
    model_factory = GaussProcMCMCModelFactory(
        active_metric=INTERNAL_METRIC_NAME, gpmodel=gpmodel)
    model = model_factory.model(state, fit_params=fit_params)
    return model, gpmodel


def test_gp_fit(tuning_job_state):
    _set_seeds(0)
    hp_ranges = tuning_job_state.hp_ranges
    X = [(0.0, 0.0),
         (1.0, 0.0),
         (0.0, 1.0),
         (1.0, 1.0)]
    Y = [np.sum(x) * 10.0 for x in X]
    X = tuples_to_configs(X, hp_ranges)

    # checks if fitting is running
    random_seed = 0
    model, _ = _make_model_gp_optimize(tuning_job_state, random_seed)

    X_enc = [hp_ranges.to_ndarray(x) for x in X]
    pred_train = model.predict(np.array(X_enc))[0]

    assert np.all(np.abs(pred_train['mean'] - Y) < 1e-1), \
        "in a noiseless setting, mean of GP should coincide closely with outputs at training points"

    X_test = tuples_to_configs([
        (0.2, 0.2),
        (0.4, 0.2),
        (0.1, 0.9),
        (0.5, 0.5)], hp_ranges)
    X_test_enc = [hp_ranges.to_ndarray(x) for x in X_test]

    pred_test = model.predict(np.array(X_test_enc))[0]
    assert np.min(pred_train['std']) < np.min(pred_test['std']), \
        "Standard deviation on un-observed points should be greater than at observed ones"


def test_gp_mcmc_fit(tuning_job_state):
    hp_ranges = make_hyperparameter_ranges({'x': uniform(-4.0, 4.0)})

    def tuning_job_state_mcmc(X, Y) -> TuningJobState:
        Y = [dictionarize_objective(y) for y in Y]
        return create_tuning_job_state(
            hp_ranges=hp_ranges, cand_tuples=X, metrics=Y)

    _set_seeds(0)

    def f(x):
        return 0.1 * np.power(x, 3)

    X = np.concatenate((np.random.uniform(-4., -1., 10), np.random.uniform(1., 4., 10)))
    Y = f(X)
    X_test = np.sort(np.random.uniform(-1., 1., 10))

    X = tuples_to_configs([(x,) for x in X], hp_ranges)
    X_test = tuples_to_configs([(x,) for x in X_test], hp_ranges)

    tuning_job_state = tuning_job_state_mcmc(X, Y)
    # checks if fitting is running
    random_seed = 0
    model, _ = _make_model_mcmc(tuning_job_state, random_seed)

    X = [hp_ranges.to_ndarray(x) for x in X]
    predictions = model.predict(np.array(X))

    Y_std_list = [p['std'] for p in predictions]
    Y_mean_list = [p['mean'] for p in predictions]
    Y_mean = np.mean(Y_mean_list, axis=0)
    Y_std = np.mean(Y_std_list, axis=0)

    assert np.all(np.abs(Y_mean - Y) < 1e-1), \
        "in a noiseless setting, mean of GP should coincide closely with outputs at training points"

    X_test = [hp_ranges.to_ndarray(x) for x in X_test]

    predictions_test = model.predict(np.array(X_test))
    Y_std_test_list = [p['std'] for p in predictions_test]
    Y_std_test = np.mean(Y_std_test_list, axis=0)
    assert np.max(Y_std) < np.min(Y_std_test), \
        "Standard deviation on un-observed points should be greater than at observed ones"


def _compute_acq_with_gradient_many(acq_func, X_test):
    fvals, grads = zip(
        *[acq_func.compute_acq_with_gradient(x_test) for x_test in X_test])
    return np.array(fvals), np.stack(grads, axis=0)


def test_gp_fantasizing():
    """
    Compare whether acquisition function evaluations (values, gradients) with
    fantasizing are the same as averaging them by hand.
    """
    random_seed = 4567
    _set_seeds(random_seed)
    num_fantasy_samples = 10
    num_pending = 5

    hp_ranges = _simple_hp_ranges()
    X = tuples_to_configs([
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0)], hp_ranges)
    num_data = len(X)
    Y = [dictionarize_objective(np.random.randn(1, 1)) for _ in range(num_data)]
    # Draw fantasies. This is done for a number of fixed pending candidates
    # The model parameters are fit in the first iteration, when there are
    # no pending candidates

    # Note: It is important to not normalize targets, because this would be
    # done on the observed targets only, not the fantasized ones, so it
    # would be hard to compare below.
    pending_tuples = [tuple(np.random.rand(2,)) for _ in range(num_pending)]
    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y,
        pending_tuples=pending_tuples)
    model, gpmodel = _make_model_gp_optimize(
        state, random_seed, num_fantasy_samples=num_fantasy_samples,
        normalize_targets=False)
    fantasy_samples = model.fantasy_samples
    # Evaluate acquisition function and gradients with fantasizing
    num_test = 50
    X_test_enc = [
        hp_ranges.to_ndarray(hp_ranges.tuple_to_config(
            tuple(np.random.rand(2,)))) for _ in range(num_test)]
    acq_func = EIAcquisitionFunction(model)
    fvals, grads = _compute_acq_with_gradient_many(acq_func, X_test_enc)
    # Do the same computation by averaging by hand
    fvals_cmp = np.empty((num_fantasy_samples,) + fvals.shape)
    grads_cmp = np.empty((num_fantasy_samples,) + grads.shape)
    X_full = X + state.pending_configurations()
    for it in range(num_fantasy_samples):
        Y_full = Y + [dictionarize_objective(eval.fantasies[INTERNAL_METRIC_NAME][:, it])
                      for eval in fantasy_samples]
        state2 = create_tuning_job_state(
            hp_ranges=hp_ranges, cand_tuples=X_full, metrics=Y_full)
        # We have to skip parameter optimization here
        model_factory = GaussProcEmpiricalBayesModelFactory(
            active_metric=INTERNAL_METRIC_NAME, gpmodel=gpmodel,
            num_fantasy_samples=num_fantasy_samples,
            normalize_targets=False)
        model2 = model_factory.model(state2, fit_params=False)
        acq_func2 = EIAcquisitionFunction(model2)
        fvals_, grads_ = _compute_acq_with_gradient_many(acq_func2, X_test_enc)
        fvals_cmp[it, :] = fvals_
        grads_cmp[it, :] = grads_
    # Comparison
    fvals2 = np.mean(fvals_cmp, axis=0)
    grads2 = np.mean(grads_cmp, axis=0)
    assert np.allclose(fvals, fvals2)
    assert np.allclose(grads, grads2)


def default_models() -> List[GaussProcSurrogateModel]:
    hp_ranges = _simple_hp_ranges()
    X = [(0.0, 0.0),
         (1.0, 0.0),
         (0.0, 1.0),
         (1.0, 1.0),
         (0.0, 0.0),  # same evals are added multiple times to force GP to unlearn prior
         (1.0, 0.0),
         (0.0, 1.0),
         (1.0, 1.0),
         (0.0, 0.0),
         (1.0, 0.0),
         (0.0, 1.0),
         (1.0, 1.0)]
    Y = [dictionarize_objective(np.sum(x) * 10.0) for x in X]

    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=X, metrics=Y)
    random_seed = 0

    return [
        _make_model_gp_optimize(state, random_seed)[0],
        _make_model_mcmc(state, random_seed)[0]]


def test_current_best():
    for model in default_models():
        current_best = model.current_best()[0].item()
        print(current_best)
        assert -0.1 < current_best < 0.1
