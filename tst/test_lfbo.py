import pytest
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bore.bore import LFBO
from syne_tune.config_space import randint

max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}

list_classifiers = ["xgboost", "logreg", "rf"]


@pytest.mark.timeout(10)
@pytest.mark.parametrize("classifier", list_classifiers)
def test_lfbo_models(classifier):
    searcher = LFBO(config_space, classifier=classifier, feval_acq=5)

    for i in range(10):
        config = searcher.suggest()
        searcher.on_trial_complete(trial_id=i, config=config, metric=np.random.rand())

    config = searcher.suggest(trial_id=10)
    assert config is not None


def test_lfbo_rejects_mlp():
    with pytest.raises(AssertionError):
        LFBO(config_space, classifier="mlp")
