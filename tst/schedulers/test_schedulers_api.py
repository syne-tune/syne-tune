import tempfile
from pathlib import Path

import dill
import pytest

from syne_tune.backend.trial_status import Trial

from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

from syne_tune.config_space import randint, uniform, choice

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
