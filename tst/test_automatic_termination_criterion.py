"""
from syne_tune.stopping_criterions.automatic_termination_criterion import (
    AutomaticTerminationCriterion,
)
from syne_tune.backend.trial_status import Trial, Status
from syne_tune.tuning_status import TuningStatus
from syne_tune.config_space import uniform


def test_automatic_termination():
    metric = "loss"
    mode = "min"
    config_space = {"x": uniform(0, 1)}
    seed = 42
    warm_up = 10
    stop_criterion = AutomaticTerminationCriterion(
        config_space, threshold=0.9, metric=metric,
        mode=mode, seed=seed, warm_up=warm_up,
    )
    status = TuningStatus(metric_names=[metric])

    trial_status_dict = {}
    new_results = []
    for i in range(20):
        x = config_space["x"].sample()
        trial = Trial(trial_id=i, config={"x": x}, creation_time=None)
        trial_status_dict[i] = (trial, Status.completed)
        new_results.append((i, {metric: (x - 0.5) ** 2}))
        status.update(trial_status_dict, new_results)
        if i < warm_up - 1:
            assert stop_criterion(status) is False
    assert stop_criterion(status)
"""
