import pytest

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.baselines import BOHB
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.multiobjective import (
    MultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.multiobjective.expected_hyper_volume_improvement import (
    ExpectedHyperVolumeImprovement,
)
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.optimizer.schedulers.searchers.random_searcher import (
    MultiObjectiveRandomSearcher,
)
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

config_space = {
    "steps": 100,
    "x": randint(0, 20),
    "y": uniform(0, 1),
    "z": choice(["a", "b", "c"]),
}
metric1 = "objective1"
metric2 = "objective2"
time_attr = "step"
max_t = 3
num_trials = 10


def single_objective(searcher, **searcher_kwargs):
    return lambda seed: SingleObjectiveScheduler(
        config_space,
        searcher=searcher,
        metric=metric1,
        random_seed=seed,
        searcher_kwargs=searcher_kwargs,
    )


def multi_objective(searcher_cls):
    return lambda seed: SingleFidelityScheduler(
        config_space,
        searcher=searcher_cls(config_space=config_space, random_seed=seed),
        metrics=[metric1, metric2],
        random_seed=seed,
    )


def asha(searcher):
    return lambda seed: AsynchronousSuccessiveHalving(
        config_space=config_space,
        metric=metric1,
        searcher=searcher,
        time_attr=time_attr,
        max_t=max_t,
        brackets=2,
        random_seed=seed,
    )


scheduler_factories = {
    "random_search": single_objective("random_search"),
    "bore": single_objective("bore", init_random=4),
    "bore_de": single_objective(
        "bore", init_random=4, acq_optimizer="de", feval_acq=50
    ),
    "kde": single_objective("kde"),
    "regularized_evolution": single_objective("regularized_evolution"),
    "botorch": single_objective("botorch"),
    "cqr": single_objective("cqr"),
    "mo_random_search": multi_objective(MultiObjectiveRandomSearcher),
    "mo_regularized_evolution": multi_objective(MultiObjectiveRegularizedEvolution),
    "ehvi": multi_objective(ExpectedHyperVolumeImprovement),
    "moasha": lambda seed: MOASHA(
        config_space,
        metrics=[metric1, metric2],
        time_attr=time_attr,
        max_t=max_t,
        brackets=2,
        random_seed=seed,
    ),
    "asha_random_search": asha("random_search"),
    "asha_kde": asha("kde"),
    "bohb": lambda seed: BOHB(
        config_space=config_space,
        metric=metric1,
        time_attr=time_attr,
        max_t=max_t,
        random_seed=seed,
    ),
}


def run_scheduler(make_scheduler, seed: int) -> list:
    scheduler = make_scheduler(seed)
    trace = []
    for trial_id in range(num_trials):
        suggestion = scheduler.suggest()
        trial = Trial(trial_id=trial_id, config=suggestion.config, creation_time=None)
        scheduler.on_trial_add(trial)
        for t in range(1, max_t + 1):
            value = suggestion.config["x"] + suggestion.config["y"] + t
            result = {time_attr: t, metric1: value, metric2: -value}
            decision = scheduler.on_trial_result(trial, result)
            trace.append(decision)
        scheduler.on_trial_complete(trial, result)
        trace.append(suggestion.config)
    return trace


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "make_scheduler",
    list(scheduler_factories.values()),
    ids=list(scheduler_factories.keys()),
)
def test_same_seed_gives_same_suggestions(make_scheduler):
    trace = run_scheduler(make_scheduler, seed=42)
    assert trace == run_scheduler(make_scheduler, seed=42)
    assert trace != run_scheduler(make_scheduler, seed=43)
