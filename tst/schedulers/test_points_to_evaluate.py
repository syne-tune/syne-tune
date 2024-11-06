import pytest
import numpy as np

from syne_tune.config_space import randint
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)


list_schedulers_to_test = [
    ("single_fidelity", SingleFidelityScheduler),
    ("single_objective", SingleObjectiveScheduler),
]


@pytest.mark.timeout(3)
@pytest.mark.parametrize("name, scheduler_cls", list_schedulers_to_test)
def test_points_to_evaluate(name, scheduler_cls):
    random_seed = 42
    np.random.seed(random_seed)
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

    kwargs = {
        "config_space": config_space,
        "random_seed": random_seed,
        "searcher": "random_search",
        "searcher_kwargs": {"points_to_evaluate": points_to_evaluate},
    }

    if scheduler_cls in [SingleObjectiveScheduler]:
        kwargs["metric"] = metric
    else:
        kwargs["metrics"] = [metric]

    scheduler = scheduler_cls(**kwargs)

    # Check that the first points match those defined in points_to_evaluate
    for i in range(len(points_to_evaluate)):
        config = scheduler.suggest()
        config == points_to_evaluate[i], (
            "Initial point %s does not match listed points_to_evaluate." % i
        )
