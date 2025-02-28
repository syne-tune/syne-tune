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
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule

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
resource_attr = "step"
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
        searcher="random_search",
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
        time_attr=resource_attr
    ),

    MedianStoppingRule(
        scheduler=SingleObjectiveScheduler(
            config_space,
            searcher="random_search",
            metric=metric1,
            random_seed=random_seed,
        ),
        resource_attr=resource_attr,
        metric=metric1,
        random_seed=random_seed,
    ),
    AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        random_seed=random_seed,
        searcher="random_search",
        time_attr=resource_attr,
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
]


@pytest.mark.timeout(20)
@pytest.mark.parametrize("scheduler", list_schedulers_to_test)
def test_schedulers_api(scheduler):
    trial_ids = range(4)

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
    make_metric = lambda t, x: {resource_attr: t, metric1: x, metric2: -x}
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
