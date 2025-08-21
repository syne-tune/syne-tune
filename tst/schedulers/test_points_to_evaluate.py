import pytest

from syne_tune.config_space import randint
from syne_tune.optimizer.baselines import ASHA, ASHACQR
from syne_tune.optimizer.schedulers.multiobjective import (
    MultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

random_seed = 42
metric = "accuracy"
config_space = {
    "a": randint(1, 1000),
    "b": randint(1, 2000),
    "c": 27,
}

points_to_evaluate = [
    {"a": 1, "b": 2, "c": 27},
    {"a": 100, "b": 200, "c": 27},
    {"a": 1000, "b": 2000, "c": 27},
]
list_schedulers_to_test = [
    SingleObjectiveScheduler(
        config_space,
        searcher="random_search",
        metric=metric,
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="bore",
        metric=metric,
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleObjectiveScheduler(
        config_space,
        searcher="kde",
        metric=metric,
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleFidelityScheduler(
        config_space,
        searcher="random_search",
        metrics=[metric],
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleFidelityScheduler(
        config_space,
        searcher="botorch",
        metrics=[metric],
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleFidelityScheduler(
        config_space,
        searcher="regularized_evolution",
        metrics=[metric],
        do_minimize=False,
        random_seed=random_seed,
        searcher_kwargs={"points_to_evaluate": points_to_evaluate},
    ),
    SingleFidelityScheduler(
        config_space,
        searcher=MultiObjectiveRegularizedEvolution(
            config_space=config_space,
            random_seed=random_seed,
            points_to_evaluate=points_to_evaluate,
        ),
        metrics=[metric],
        do_minimize=False,
        random_seed=random_seed,
    ),
    ASHA(
        config_space,
        metric="mean_loss",
        resource_attr="epoch",
        max_t=10,
        points_to_evaluate=points_to_evaluate,
    ),
    ASHACQR(
        config_space,
        metric="mean_loss",
        resource_attr="epoch",
        max_t=10,
        points_to_evaluate=points_to_evaluate,
    ),
]


@pytest.mark.timeout(3)
@pytest.mark.parametrize("scheduler", list_schedulers_to_test)
def test_points_to_evaluate(scheduler):

    # Check that the first points match those defined in points_to_evaluate
    for i in range(len(points_to_evaluate)):
        trial_suggestion = scheduler.suggest()
        assert trial_suggestion.config == points_to_evaluate[i], (
            "Initial point %s does not match listed points_to_evaluate." % i
        )
