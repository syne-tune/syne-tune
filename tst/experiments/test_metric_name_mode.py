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
import pytest

from syne_tune.util import metric_name_mode

metric_names = ["m1", "m2", "m3"]


@pytest.mark.parametrize(
    "metric_mode, query_metric, expected_metric, expected_mode,",
    [
        ("max", "m2", "m2", "max"),
        ("min", "m2", "m2", "min"),
        (["max", "min", "max"], "m2", "m2", "min"),
        (["max", "min", "max"], "m3", "m3", "max"),
        ("max", 1, "m2", "max"),
        ("min", 1, "m2", "min"),
        (["max", "min", "max"], 1, "m2", "min"),
        (["max", "min", "max"], 2, "m3", "max"),
    ],
)
def test_metric_name_mode(metric_mode, query_metric, expected_metric, expected_mode):
    metric_name, metric_mode = metric_name_mode(
        metric_names=metric_names, metric_mode=metric_mode, metric=query_metric
    )
    assert metric_name == expected_metric
    assert metric_mode == expected_mode
