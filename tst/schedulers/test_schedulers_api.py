import tempfile
from pathlib import Path

import dill
import pytest
# FIXME: Needs Ray to be installed
from ray.tune.schedulers import AsyncHyperBandScheduler
import pandas as pd
import numpy as np

from examples.launch_height_standalone_scheduler import SimpleScheduler
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.baselines import RandomSearch, BayesianOptimization, \
    ASHA, MOBSTER, REA, SyncHyperband, SyncBOHB, SyncMOBSTER, ZeroShotTransfer
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers import FIFOScheduler, MedianStoppingRule, \
    HyperbandScheduler, PopulationBasedTraining, RayTuneScheduler
from syne_tune.optimizer.schedulers.botorch.botorch_searcher import BotorchSearcher
from syne_tune.optimizer.schedulers.multiobjective import MOASHA
from syne_tune.optimizer.schedulers.transfer_learning import \
    TransferLearningTaskEvaluations, BoundingBox, RUSHScheduler
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import \
    QuantileBasedSurrogateSearcher
from syne_tune.config_space import randint, uniform, choice


config_space = {
    "steps": 100,
    "x": randint(0, 20),
    "y": uniform(0, 1),
    "z": choice(["a", "b", "c"]),
}

metric1 = "objective1"
metric2 = "objective2"
resource_attr = 'step'
max_t = 10


def make_ray_skopt():
    from ray.tune.suggest.skopt import SkOptSearch

    ray_searcher = SkOptSearch()
    ray_searcher.set_search_properties(
        mode='min', metric=metric1,
        config=RayTuneScheduler.convert_config_space(config_space)
    )
    return ray_searcher


def make_transfer_learning_evaluations(num_evals: int = 10):
    num_seeds = 3
    num_fidelity = 5
    return {
        "dummy-task-1": TransferLearningTaskEvaluations(
            config_space,
            hyperparameters=pd.DataFrame(
                [{k: v.sample() if hasattr(v, "sample") else v for k, v in config_space.items()} for _ in range(10)]
            ),
            objectives_evaluations=np.arange(num_evals * num_seeds * num_fidelity * 2).reshape(num_evals, num_seeds, num_fidelity, 2),
            objectives_names=[metric1, metric2],
        ),
        "dummy-task-2": TransferLearningTaskEvaluations(
            config_space,
            hyperparameters=pd.DataFrame(
                [{k: v.sample() if hasattr(v, "sample") else v for k, v in config_space.items()} for _ in range(10)]
            ),
            objectives_evaluations=-np.arange(num_evals * num_seeds * num_fidelity * 2).reshape(num_evals, num_seeds, num_fidelity, 2),
            objectives_names=[metric1, metric2],
        ),
    }


transfer_learning_evaluations = make_transfer_learning_evaluations()


@pytest.mark.parametrize("scheduler", [
    FIFOScheduler(config_space, searcher='random', metric=metric1),
    FIFOScheduler(config_space, searcher='bayesopt', metric=metric1),
    FIFOScheduler(config_space, searcher='kde', metric=metric1),
    HyperbandScheduler(config_space, searcher='random', resource_attr=resource_attr, max_t=max_t, metric=metric1),
    HyperbandScheduler(config_space, searcher='bayesopt', resource_attr=resource_attr, max_t=max_t, metric=metric1),
    HyperbandScheduler(
        config_space, searcher='random', type='pasha', max_t=max_t, resource_attr=resource_attr, metric=metric1
    ),
    PopulationBasedTraining(config_space=config_space, metric=metric1, resource_attr=resource_attr, max_t=max_t),
    RayTuneScheduler(
        config_space=config_space,
        ray_scheduler=AsyncHyperBandScheduler(max_t=max_t, time_attr=resource_attr, mode='min', metric=metric1)
    ),
    RayTuneScheduler(
        config_space=config_space,
        ray_scheduler=AsyncHyperBandScheduler(max_t=max_t, time_attr=resource_attr, mode='min', metric=metric1),
        ray_searcher=make_ray_skopt(),
    ),
    SimpleScheduler(config_space=config_space, metric=metric1),
    RandomSearch(config_space=config_space, metric=metric1),
    BayesianOptimization(config_space=config_space, metric=metric1),
    REA(config_space=config_space, metric=metric1, population_size=1, sample_size=2),
    ASHA(config_space=config_space, metric=metric1, resource_attr=resource_attr, max_t=max_t),
    MOBSTER(config_space=config_space, metric=metric1, resource_attr=resource_attr, max_t=max_t),
    # TODO fix me, assert is thrown refusing to take PASHA arguments as valid
    # PASHA(config_space=config_space, metric=metric1, resource_attr=resource_attr, max_t=max_t),
    MOASHA(config_space=config_space, time_attr=resource_attr, metrics=[metric1, metric2]),
    MedianStoppingRule(
        scheduler=FIFOScheduler(config_space, searcher='random', metric=metric1),
        resource_attr=resource_attr, metric=metric1
    ),
    BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: RandomSearch(
            new_config_space,
            points_to_evaluate=[],
            metric=metric,
            mode=mode,
        ),
        mode="min",
        config_space=config_space,
        metric=metric1,
        transfer_learning_evaluations=transfer_learning_evaluations,
    ),
    FIFOScheduler(
        searcher=QuantileBasedSurrogateSearcher(
            mode="min",
            config_space=config_space,
            metric=metric1,
            transfer_learning_evaluations=transfer_learning_evaluations,
        ),
        mode='min',
        config_space=config_space,
        metric=metric1
    ),
    RUSHScheduler(
        resource_attr=resource_attr,
        max_t=max_t,
        mode="min",
        config_space=config_space,
        metric=metric1,
        transfer_learning_evaluations=make_transfer_learning_evaluations(),
    ),
    SyncHyperband(
        config_space=config_space,
        metric=metric1,
        resource_attr=resource_attr,
        max_resource_level=max_t,
        max_resource_attr='steps',
        brackets=3),
    SyncMOBSTER(
        config_space=config_space,
        metric=metric1,
        resource_attr=resource_attr,
        max_resource_level=max_t,
        max_resource_attr='steps',
        brackets=3),
    SyncBOHB(
        config_space=config_space,
        metric=metric1,
        resource_attr=resource_attr,
        max_resource_level=max_t,
        max_resource_attr='steps',
        brackets=3),
    ZeroShotTransfer(
        config_space=config_space,
        metric=metric1,
        transfer_learning_evaluations=transfer_learning_evaluations,
        use_surrogates=True,
    ),
    FIFOScheduler(
        config_space,
        searcher=BotorchSearcher(config_space=config_space, metric=metric1, mode='min'),
        metric=metric1
    ),
])
def test_async_schedulers_api(scheduler):
    trial_ids = range(4)

    if isinstance(scheduler, MOASHA):
        assert scheduler.metric_names() == [metric1, metric2]
    else:
        assert scheduler.metric_names() == [metric1]
    assert scheduler.metric_mode() == "min"

    # checks suggestions are properly formatted
    trials = []
    for i in trial_ids:
        suggestion = scheduler.suggest(i)
        assert all(x in suggestion.config.keys() for x in config_space.keys()), \
            "suggestion configuration should contain all keys of config_space."
        trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

    for trial in trials:
        scheduler.on_trial_add(trial=trial)

    # checks results can be transmitted with appropriate scheduling decisions
    make_metric = lambda t, x: {resource_attr: t, metric1: x, metric2: -x}
    for i, trial in enumerate(trials):
        for t in range(1, max_t + 1):
            decision = scheduler.on_trial_result(trial, make_metric(t, i))
            assert decision in [SchedulerDecision.CONTINUE, SchedulerDecision.PAUSE, SchedulerDecision.STOP]

    scheduler.on_trial_error(trials[0])
    for i, trial in enumerate(trials):
        scheduler.on_trial_complete(trial, make_metric(max_t, i))

    # checks serialization
    with tempfile.TemporaryDirectory() as local_path:
        with open(Path(local_path) / "scheduler.dill", "wb") as f:
            dill.dump(scheduler, f)
        with open(Path(local_path) / "scheduler.dill", "rb") as f:
            dill.load(f)
