from datetime import datetime

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.config_space import randint

from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}
time_attr = "step"
metric = "mean_loss"


def make_trial(trial_id: int):
    return Trial(
        trial_id=trial_id,
        config={"steps": 0, "width": trial_id},
        creation_time=datetime.now(),
    )


def test_median_stopping_rule():
    random_seed = 42
    scheduler = MedianStoppingRule(
        scheduler=SingleObjectiveScheduler(
            config_space,
            searcher="random_search",
            metric=metric,
        ),
        resource_attr="step",
        metric=metric,
        random_seed=random_seed,
        grace_population=1,
    )

    trial1 = make_trial(trial_id=0)
    trial2 = make_trial(trial_id=1)
    trial3 = make_trial(trial_id=2)

    scheduler.on_trial_add(trial=trial1)
    scheduler.on_trial_add(trial=trial2)
    scheduler.on_trial_add(trial=trial3)

    make_metric = lambda x: {time_attr: 1, metric: x}
    decision1 = scheduler.on_trial_result(trial1, make_metric(2.0))
    decision2 = scheduler.on_trial_result(trial2, make_metric(1.0))
    decision3 = scheduler.on_trial_result(trial3, make_metric(5.0))
    assert decision1 == SchedulerDecision.CONTINUE
    assert decision2 == SchedulerDecision.CONTINUE
    assert decision3 == SchedulerDecision.STOP
