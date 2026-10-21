import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import dill
import pytest

from syne_tune.backend.trial_status import Trial

from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.multiobjective import (
    MultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.searchers.random_searcher import (
    MultiObjectiveRandomSearcher,
)
from syne_tune.optimizer.schedulers.multiobjective.expected_hyper_volume_improvement import (
    ExpectedHyperVolumeImprovement,
)
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
    QuantileBasedSurrogateSearcher,
)
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.optimizer.schedulers.transfer_learning.zero_shot import ZeroShotTransfer
from syne_tune.optimizer.baselines import BOHB, CQR, ASHACQR
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
    ConformalQuantileRegression,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_model import (
    TabICLQuantileRegressor,
    TabPFNQuantileRegressor,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedQuantileRegressor,
)


config_space = {
    "steps": 100,
    "x": randint(0, 20),
    "y": uniform(0, 1),
    "z": choice(["a", "b", "c"]),
}

categorical_config_space = {
    "steps": 100,
    "x": choice(["0", "1", "2"]),
    "y": choice([0, 1, 2]),
    "z": choice(["a", "b", "c"]),
}

metric1 = "objective1"
metric2 = "objective2"
time_attr = "step"
max_t = 10
random_seed = 42
mode = "max"


def make_transfer_learning_evaluations(num_evals: int = 10):
    num_seeds = 3
    num_fidelity = 5
    return {
        "dummy-task-1": TransferLearningTaskEvaluations(
            config_space,
            hyperparameters=pd.DataFrame(
                [
                    {
                        k: v.sample() if hasattr(v, "sample") else v
                        for k, v in config_space.items()
                    }
                    for _ in range(10)
                ]
            ),
            objectives_evaluations=np.arange(
                num_evals * num_seeds * num_fidelity * 2
            ).reshape(num_evals, num_seeds, num_fidelity, 2),
            objectives_names=[metric1, metric2],
        ),
        "dummy-task-2": TransferLearningTaskEvaluations(
            config_space,
            hyperparameters=pd.DataFrame(
                [
                    {
                        k: v.sample() if hasattr(v, "sample") else v
                        for k, v in config_space.items()
                    }
                    for _ in range(10)
                ]
            ),
            objectives_evaluations=-np.arange(
                num_evals * num_seeds * num_fidelity * 2
            ).reshape(num_evals, num_seeds, num_fidelity, 2),
            objectives_names=[metric1, metric2],
        ),
    }


transfer_learning_evaluations = make_transfer_learning_evaluations()


list_schedulers_to_test = [
    SingleObjectiveScheduler(
        config_space,
        searcher="random_search",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="bore",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="kde",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleFidelityScheduler(
        config_space,
        searcher=MultiObjectiveRandomSearcher(
            config_space=config_space,
            random_seed=random_seed,
        ),
        metrics=[metric1, metric2],
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="botorch",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="regularized_evolution",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="cqr",
        metric=metric1,
        do_minimize=False,
        random_seed=random_seed,
    ),
    # Multi-objective methods
    SingleFidelityScheduler(
        config_space,
        searcher=ExpectedHyperVolumeImprovement(
            config_space=config_space,
            random_seed=random_seed,
        ),
        metrics=[metric1, metric2],
        do_minimize=False,
        random_seed=random_seed,
    ),
    SingleFidelityScheduler(
        config_space,
        searcher=MultiObjectiveRegularizedEvolution(
            config_space=config_space,
            random_seed=random_seed,
        ),
        metrics=[metric1, metric2],
        do_minimize=False,
        random_seed=random_seed,
    ),
    MOASHA(
        config_space,
        metrics=[metric1, metric2],
        do_minimize=False,
        random_seed=random_seed,
        time_attr=time_attr,
    ),
    # Multi-fidelity methods
    BOHB(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        time_attr=time_attr,
        max_t=max_t,
    ),
    MedianStoppingRule(
        scheduler=SingleObjectiveScheduler(
            config_space,
            searcher="random_search",
            metric=metric1,
            random_seed=random_seed,
        ),
        time_attr=time_attr,
        metric=metric1,
        random_seed=random_seed,
    ),
    AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        searcher="random_search",
        time_attr=time_attr,
    ),
    AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        searcher="bore",
        time_attr=time_attr,
    ),
    AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        searcher="kde",
        time_attr=time_attr,
    ),
    AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        searcher="cqr",
        time_attr=time_attr,
    ),
    BoundingBox(
        scheduler_fun=lambda new_config_space, metric, do_minimize, random_seed: SingleObjectiveScheduler(
            new_config_space,
            searcher="random_search",
            metric=metric,
            random_seed=random_seed,
            do_minimize=do_minimize,
        ),
        do_minimize=False,
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        transfer_learning_evaluations=transfer_learning_evaluations,
    ),
    ZeroShotTransfer(
        do_minimize=False,
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        transfer_learning_evaluations=transfer_learning_evaluations,
    ),
    SingleObjectiveScheduler(
        config_space,
        metric=metric1,
        searcher=QuantileBasedSurrogateSearcher(
            config_space=config_space,
            random_seed=random_seed,
            transfer_learning_evaluations=transfer_learning_evaluations,
        ),
        do_minimize=False,
        random_seed=random_seed,
    ),
]


@pytest.mark.timeout(20)
@pytest.mark.parametrize("scheduler", list_schedulers_to_test)
def test_schedulers_api(scheduler):
    trial_ids = range(6)

    # checks suggestions are properly formatted
    trials = []
    for i in trial_ids:
        suggestion = scheduler.suggest()
        assert all(
            x in suggestion.config.keys() for x in config_space.keys()
        ), "suggestion configuration should contain all keys of config_space."
        trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

    for trial in trials:
        scheduler.on_trial_add(trial=trial)

    # checks results can be transmitted with appropriate scheduling decisions
    make_metric = lambda t, x: {time_attr: t, metric1: x, metric2: -x}
    for i, trial in enumerate(trials):
        for t in range(1, max_t + 1):
            decision = scheduler.on_trial_result(trial, make_metric(t, i))
            assert decision in [
                SchedulerDecision.CONTINUE,
                SchedulerDecision.PAUSE,
                SchedulerDecision.STOP,
            ]

    scheduler.on_trial_error(trials[0])
    for i, trial in enumerate(trials):
        scheduler.on_trial_complete(trial, make_metric(max_t, i))

    # checks serialization
    with tempfile.TemporaryDirectory() as local_path:
        with open(Path(local_path) / "scheduler.dill", "wb") as f:
            dill.dump(scheduler, f)
        with open(Path(local_path) / "scheduler.dill", "rb") as f:
            dill.load(f)

    # Check metadata, metric_names() and metric_mode() are tested above
    meta = scheduler.metadata()
    assert meta["scheduler_name"] == str(scheduler.__class__.__name__)
    assert "config_space" in meta


def _icl_model_params():
    params = []
    for model_type, package in [("tabicl", "tabicl"), ("tabpfn", "tabpfn")]:
        params.append(
            pytest.param(
                model_type,
                marks=pytest.mark.skipif(
                    importlib.util.find_spec(package) is None,
                    reason=f"{package} is not installed",
                ),
                id=model_type,
            )
        )
    return params


#@pytest.mark.timeout(300)
@pytest.mark.skip("too slow")
@pytest.mark.parametrize("model_type", _icl_model_params())
@pytest.mark.parametrize("multi_fidelity", [False, True], ids=["CQR", "ASHACQR"])
def test_icl_schedulers(model_type, multi_fidelity):
    # Smoke test which checks that CQR runs with in-context learning models, the
    # surrogate is fitted once enough observations are available.
    if multi_fidelity:
        scheduler = ASHACQR(
            config_space=config_space,
            metric=metric1,
            random_seed=random_seed,
            time_attr=time_attr,
            max_t=max_t,
            model_type=model_type,
        )
    else:
        scheduler = CQR(
            config_space=config_space,
            metric=metric1,
            random_seed=random_seed,
            model_type=model_type,
        )
    for i in range(8):
        suggestion = scheduler.suggest()
        assert all(x in suggestion.config.keys() for x in config_space.keys())
        trial = Trial(trial_id=i, config=suggestion.config, creation_time=None)
        scheduler.on_trial_add(trial=trial)
        for t in range(1, max_t + 1):
            decision = scheduler.on_trial_result(
                trial, {time_attr: t, metric1: float(i)}
            )
            if decision == SchedulerDecision.STOP:
                break
        scheduler.on_trial_complete(trial, {time_attr: t, metric1: float(i)})

    # checks that the surrogate was fitted with the requested model
    regressor_cls = {
        "tabicl": TabICLQuantileRegressor,
        "tabpfn": TabPFNQuantileRegressor,
    }[model_type]
    searcher = scheduler.searcher
    if multi_fidelity:
        # the CQR searcher is wrapped by LastValueMultiFidelitySearcher
        searcher = searcher.searcher
    assert isinstance(searcher.surrogate_model.quantile_regressor, regressor_cls)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("model_type", _icl_model_params())
def test_icl_conformalized_scheduler(model_type):
    # Smoke test which checks that conformal calibration runs with in-context
    # learning models, calibration starts right away with
    # min_samples_to_conformalize=0.
    scheduler = SingleObjectiveScheduler(
        config_space,
        searcher=ConformalQuantileRegression(
            config_space=config_space,
            random_seed=random_seed,
            model_type=model_type,
            min_samples_to_conformalize=0,
            ncandidates=200,
        ),
        metric=metric1,
        random_seed=random_seed,
    )
    for i in range(8):
        suggestion = scheduler.suggest()
        trial = Trial(trial_id=i, config=suggestion.config, creation_time=None)
        scheduler.on_trial_add(trial=trial)
        scheduler.on_trial_complete(trial, {metric1: float(i)})

    regressor = scheduler.searcher.surrogate_model.quantile_regressor
    assert isinstance(regressor, SymmetricConformalizedQuantileRegressor)
    assert len(regressor.corrections) > 0


@pytest.mark.parametrize(
    "model_type, conformalize, expected",
    [
        ("gradient_boosting", None, 32),
        ("gradient_boosting", False, None),
        ("tabicl", None, None),
        ("tabicl", True, 32),
        ("tabpfn", True, 32),
    ],
)
def test_cqr_conformalize_option(model_type, conformalize, expected):
    cqr = CQR(
        config_space=config_space,
        metric=metric1,
        model_type=model_type,
        conformalize=conformalize,
    )
    assert cqr.searcher.min_samples_to_conformalize == expected
    asha_cqr = ASHACQR(
        config_space=config_space,
        metric=metric1,
        time_attr=time_attr,
        max_t=max_t,
        model_type=model_type,
        conformalize=conformalize,
    )
    assert asha_cqr.searcher.searcher_kwargs["min_samples_to_conformalize"] == expected


@pytest.mark.skipif(
    importlib.util.find_spec("tabpfn") is None, reason="tabpfn is not installed"
)
def test_tabpfn_model_version():
    random_state = np.random.RandomState(0)
    X = random_state.uniform(size=(20, 2))
    y = X.sum(axis=1)
    regressor = TabPFNQuantileRegressor(quantiles=5, model_version="v2.5")
    regressor.fit(X, y)
    assert "v2.5" in str(regressor._regressor.model_path)
    assert regressor.predict(X).results_stacked.shape == (20, 5)
