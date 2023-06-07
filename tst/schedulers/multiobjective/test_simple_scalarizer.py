# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import datetime
from typing import Callable

import pytest

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)


@pytest.fixture
def config_space():
    return {
        "steps": 100,
        "x": randint(0, 20),
        "y": uniform(0, 1),
        "z": choice(["a", "b", "c"]),
    }


@pytest.fixture
def metric1():
    return "objective1"


@pytest.fixture
def metric2():
    return "objective2"


@pytest.fixture
def resource_attr():
    return "step"


@pytest.fixture
def max_t():
    return 10


@pytest.fixture
def mode():
    return "max"


@pytest.fixture
def make_metric(metric1, metric2):
    return lambda x: {metric1: x, metric2: -x}


@pytest.fixture
def scheduler(config_space, metric1, metric2, mode):
    return LinearScalarizedScheduler(
        config_space=config_space,
        metric=[metric1, metric2],
        mode=[mode, mode],
        scalarization_weights=[1, 1],
        base_scheduler_factory=FIFOScheduler,
        searcher="kde",
    )


def test_scalarization(scheduler: LinearScalarizedScheduler, make_metric: Callable):
    results = make_metric(321)
    scalarized = scheduler._scalarized_metric(results)
    assert scalarized == 321 - 321


def test_results_gathering(scheduler: LinearScalarizedScheduler, make_metric: Callable):
    trialid = 123
    trial = Trial(
        trial_id=trialid,
        config=scheduler.suggest(trialid).config,
        creation_time=datetime.datetime.now(),
    )
    results = make_metric(321)
    scheduler.on_trial_complete(trial, results)

    observed_metric = scheduler.base_scheduler.searcher.y[0]
    assert observed_metric == 0.0

    observed_trial = scheduler.base_scheduler.searcher._from_feature(
        scheduler.base_scheduler.searcher.X[0]
    )
    assert observed_trial == trial.config
