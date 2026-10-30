from unittest import mock

import pytest

pytest.importorskip("botorch")

from syne_tune.config_space import uniform
from syne_tune.optimizer.schedulers.searchers.botorch import botorch_searcher
from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
    BoTorchSearcher,
)


def _searcher_with_observations():
    searcher = BoTorchSearcher(
        config_space={"x": uniform(0, 1), "y": uniform(0, 1)},
        num_init_random=2,
        random_seed=0,
    )
    for _ in range(3):
        config = searcher.suggest()
        searcher.on_trial_complete(
            trial_id=len(searcher.trial_configs) - 1,
            config=config,
            metric=config["x"] + config["y"],
        )
    return searcher


@pytest.mark.timeout(20)
def test_unexpected_error_falls_back_to_random_config(caplog):
    searcher = _searcher_with_observations()
    with mock.patch.object(
        botorch_searcher, "fit_gpytorch_mll", side_effect=RuntimeError("boom")
    ):
        with caplog.at_level("WARNING", logger=botorch_searcher.__name__):
            config = searcher.suggest()
    assert set(config.keys()) == {"x", "y"}
    assert "sampling randomly" in caplog.text
    assert "boom" in caplog.text


@pytest.mark.timeout(20)
def test_keyboard_interrupt_is_not_swallowed():
    searcher = _searcher_with_observations()
    with mock.patch.object(
        botorch_searcher, "fit_gpytorch_mll", side_effect=KeyboardInterrupt
    ):
        with pytest.raises(KeyboardInterrupt):
            searcher.suggest()
