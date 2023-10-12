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
