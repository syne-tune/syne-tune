from datetime import datetime
from typing import Optional, Dict, Tuple, Any
import pytest
import numpy as np

from syne_tune.optimizer.schedulers.legacy_hyperband import LegacyHyperbandScheduler
from syne_tune.config_space import randint, uniform
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.searchers import LegacyRandomSearcher
from syne_tune.optimizer.schedulers.legacy_hyperband_stopping import (
    RungEntry,
    Rung,
    StoppingRungSystem,
)
from syne_tune.optimizer.schedulers.legacy_hyperband_promotion import (
    PromotionRungSystem,
)
from syne_tune.optimizer.schedulers.legacy_hyperband_pasha import PASHARungSystem
from syne_tune.optimizer.schedulers.legacy_hyperband_rush import (
    RUSHPromotionRungSystem,
    RUSHStoppingRungSystem,
)
from syne_tune.optimizer.schedulers.legacy_hyperband_cost_promotion import (
    CostPromotionRungSystem,
)


def _make_result(epoch, metric):
    return dict(epoch=epoch, metric=metric)


def _new_trial(trial_id: int, config: Dict[str, Any]):
    return Trial(trial_id=trial_id, config=config, creation_time=datetime.now())


class MyRandomSearcher(LegacyRandomSearcher):
    def __init__(self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(config_space, metric, points_to_evaluate, **kwargs)
        self._pending_records = []

    def register_pending(
        self, trial_id: str, config: Optional[Dict] = None, milestone=None
    ):
        self._pending_records.append((trial_id, config, milestone))

    def get_pending_records(self):
        result = self._pending_records
        self._pending_records = []
        return result


def _should_be(record: Tuple, trial_id: int, milestone: int, config_none: bool):
    assert record[0] == str(trial_id) and record[2] == milestone, (
        record,
        trial_id,
        milestone,
    )
    assert config_none == (record[1] is None), (record, config_none)


def test_register_pending():
    config_space = {"int": randint(1, 2), "float": uniform(5.5, 6.5), "epochs": 27}
    grace_period = 3
    reduction_factor = 3

    for searcher_data in ("rungs", "all"):
        # We need to plug in a searcher which logs calls to ``register_pending``
        scheduler = LegacyHyperbandScheduler(
            config_space,
            searcher="random",
            metric="metric",
            mode="min",
            resource_attr="epoch",
            max_resource_attr="epochs",
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            searcher_data=searcher_data,
        )
        old_searcher = scheduler.searcher
        new_searcher = MyRandomSearcher(
            old_searcher.config_space, metric=old_searcher._metric
        )
        scheduler._searcher = new_searcher

        # Start 4 trials (0, 1, 2, 3)
        trials = dict()
        for trial_id in range(4):
            trials[trial_id] = _new_trial(
                trial_id, scheduler.suggest(trial_id=trial_id).config
            )
        records = new_searcher.get_pending_records()
        if searcher_data == "rungs":
            assert len(records) == 4, records
            for trial_id, record in enumerate(records):
                _should_be(record, trial_id, grace_period, False)
        else:
            assert len(records) == 4 * grace_period, records
            for i, record in enumerate(records):
                trial_id = i // grace_period
                milestone = (i % grace_period) + 1
                _should_be(record, trial_id, milestone, False)

        # 0, 1, 2 continue, but 3 is stopped
        decision = scheduler.on_trial_result(trials[0], _make_result(grace_period, 1.0))
        assert decision == SchedulerDecision.CONTINUE
        decision = scheduler.on_trial_result(trials[1], _make_result(grace_period, 0.9))
        assert decision == SchedulerDecision.CONTINUE
        decision = scheduler.on_trial_result(trials[2], _make_result(grace_period, 0.8))
        assert decision == SchedulerDecision.CONTINUE
        decision = scheduler.on_trial_result(trials[3], _make_result(grace_period, 1.2))
        assert decision == SchedulerDecision.STOP
        records = new_searcher.get_pending_records()
        if searcher_data == "rungs":
            assert len(records) == 3, records
            milestone = grace_period * reduction_factor
            for trial_id, record in enumerate(records):
                _should_be(record, trial_id, milestone, False)
        else:
            num_per_trial = grace_period * (reduction_factor - 1)
            assert len(records) == 3 * num_per_trial, records
            for i, record in enumerate(records):
                trial_id = i // num_per_trial
                milestone = (i % num_per_trial) + grace_period + 1
                _should_be(record, trial_id, milestone, False)


def test_hyperband_max_t_inference():
    config_space1 = {
        "epochs": 15,
        "max_t": 14,
        "max_epochs": 13,
        "blurb": randint(0, 20),
    }
    config_space2 = {"max_t": 14, "max_epochs": 13, "blurb": randint(0, 20)}
    config_space3 = {"max_epochs": 13, "blurb": randint(0, 20)}
    config_space4 = {
        "epochs": randint(15, 20),
        "max_t": 14,
        "max_epochs": 13,
        "blurb": randint(0, 20),
    }
    config_space5 = {
        "epochs": randint(15, 20),
        "max_t": randint(14, 21),
        "max_epochs": 13,
        "blurb": randint(0, 20),
    }
    config_space6 = {"blurb": randint(0, 20)}
    config_space7 = {
        "epochs": randint(15, 20),
        "max_t": randint(14, 21),
        "max_epochs": randint(13, 22),
        "blurb": randint(0, 20),
    }
    # Fields: (max_t, config_space, final_max_t)
    # If final_max_t is None, an assertion should be raised
    cases = [
        (None, config_space1, 15),
        (None, config_space2, 14),
        (None, config_space3, 13),
        (None, config_space4, 14),
        (None, config_space5, 13),
        (None, config_space6, None),
        (None, config_space7, None),
        (10, config_space1, 10),
        (10, config_space2, 10),
        (10, config_space3, 10),
        (10, config_space4, 10),
        (10, config_space5, 10),
        (10, config_space6, 10),
        (10, config_space7, 10),
    ]

    for max_t, config_space, final_max_t in cases:
        if final_max_t is not None:
            myscheduler = LegacyHyperbandScheduler(
                config_space,
                searcher="random",
                max_t=max_t,
                resource_attr="epoch",
                mode="max",
                metric="accuracy",
            )
            assert final_max_t == myscheduler.max_t
        else:
            with pytest.raises(AssertionError):
                myscheduler = LegacyHyperbandScheduler(
                    config_space,
                    searcher="random",
                    max_t=max_t,
                    resource_attr="epoch",
                    mode="max",
                    metric="accuracy",
                )


@pytest.mark.parametrize(
    "scheduler_type,terminator_cls,does_pause_resume",
    [
        ("stopping", StoppingRungSystem, False),
        ("promotion", PromotionRungSystem, True),
        ("pasha", PASHARungSystem, True),
        ("rush_promotion", RUSHPromotionRungSystem, True),
        ("rush_stopping", RUSHStoppingRungSystem, False),
        ("cost_promotion", CostPromotionRungSystem, True),
    ],
)
def test_hyperband_scheduler_type(scheduler_type, terminator_cls, does_pause_resume):
    config_space = {
        "epochs": 15,
    }
    myscheduler = LegacyHyperbandScheduler(
        config_space,
        type=scheduler_type,
        metric="accuracy",
        mode="max",
        resource_attr="epoch",
        max_resource_attr="epochs",
        cost_attr="cost",
    )
    assert isinstance(myscheduler.terminator._rung_systems[0], terminator_cls)
    assert myscheduler.does_pause_resume() == does_pause_resume


def test_quantile_hyperband_stopping():
    num_runs = 100
    random_seed = 31415938
    random_state = np.random.RandomState(random_seed)
    for num_run in range(num_runs):
        len_data = 2 if num_run == 0 else random_state.randint(low=2, high=200)
        prom_quant = random_state.uniform(low=0.01, high=0.5)
        mode = "min" if random_state.uniform() <= 0.5 else "max"
        metric_vals = random_state.normal(loc=0, scale=1, size=len_data)
        data = [
            RungEntry(trial_id, metric_val)
            for trial_id, metric_val in zip(range(len_data), metric_vals)
        ]
        rung = Rung(level=1, prom_quant=prom_quant, mode=mode, data=data)
        quantile = rung.quantile()
        q = prom_quant if mode == "min" else 1 - prom_quant
        quantile_test = np.quantile(metric_vals, q=q)
        np.testing.assert_almost_equal(quantile, quantile_test)
