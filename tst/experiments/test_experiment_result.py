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
from unittest.mock import Mock

import pandas as pd
import pytest

from syne_tune.experiments import ExperimentResult


@pytest.mark.parametrize(
    "metadata,results,expected_result",
    [
        pytest.param(
            dict(metric_names=["loss"], metric_mode="min"),
            pd.DataFrame({"loss": [0.1, 10]}),
            dict(loss=0.1),
            id="single metric - min",
        ),
        pytest.param(
            dict(metric_names=["loss"], metric_mode="max"),
            pd.DataFrame({"loss": [0.1, 10]}),
            dict(loss=10),
            id="single metric - max",
        ),
        pytest.param(
            dict(metric_names=["loss", "some_metric"], metric_mode=["min", "max"]),
            pd.DataFrame({"loss": [0.1, 10], "some_metric": [0.2, 20]}),
            dict(loss=0.1, some_metric=0.2),
            id="multiple metrics",
        ),
    ],
)
def test_get_best_result(metadata: dict, results: pd.DataFrame, expected_result: dict):
    exp_result = ExperimentResult(
        name="some name", results=results, metadata=metadata, tuner=Mock(), path=Mock()
    )
    assert exp_result.best_config() == expected_result
