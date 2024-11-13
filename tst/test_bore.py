import pytest
import numpy as np

from datetime import datetime

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.searchers.bore.bore import Bore
from syne_tune.config_space import randint

max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}
time_attr = "step"
metric1 = "mean_loss"
metric2 = "cost"


def make_trial(trial_id: int):
    return Trial(
        trial_id=trial_id,
        config={"steps": 0, "width": trial_id},
        creation_time=datetime.now(),
    )


list_classifiers = ["xgboost", "logreg", "rf", "mlp"]


@pytest.mark.parametrize("classifier", list_classifiers)
def test_bore_models(classifier):
    searcher = Bore(config_space, classifier=classifier, feval_acq=5)

    for i in range(10):
        config = searcher.suggest()
        searcher.on_trial_result(trial_id=i, config=config, metric=np.random.rand())

    config = searcher.suggest(trial_id=10)
