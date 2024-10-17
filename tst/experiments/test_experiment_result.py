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
