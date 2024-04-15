from syne_tune.optimizer.schedulers.ask_tell_scheduler import AskTellScheduler
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.config_space import uniform


def test_ask_tell_scheduler():

    config_space = {
        "x": uniform(0, 1),
        "y": uniform(0, 1),
    }
    metric = "test_metric"
    mode = "max"
    max_iterations = 10
    target_function = lambda x, y: x**2 + y**2

    scheduler = AskTellScheduler(base_scheduler=RandomSearch(config_space, metric=metric, mode=mode))
    for iter in range(max_iterations):
        trial_suggestion = scheduler.ask()
        test_result = target_function(**trial_suggestion.config)
        scheduler.tell(trial_suggestion, {metric: test_result})