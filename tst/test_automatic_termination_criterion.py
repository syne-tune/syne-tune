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
