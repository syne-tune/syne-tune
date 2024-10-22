import numpy as np
import pandas as pd

from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import hypervolume_indicator_column_generator


def test_append_hypervolume_indicator():
    reference_point = np.array([2, 2])
    points = np.array([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
    metric_names = ["metric_a", "metric_b"]
    metrics_and_modes = list(zip(metric_names, ["min"] * 2))
    df = pd.DataFrame(points, columns=metric_names)
    tuner_time_values = np.arange(4) * 0.05 + 1.2
    df[ST_TUNER_TIME] = tuner_time_values
    hp_a_values = np.array([1.8, 1.9, 0.5, 0.1])
    df["hp_a"] = hp_a_values
    hp_b_values = ["yes", "no", "no", "yes"]
    df["hp_b"] = hp_b_values

    dataframe_column_generator = hypervolume_indicator_column_generator(
        metrics_and_modes, reference_point=reference_point
    )
    hvi_values = dataframe_column_generator(df).values
    hvi_values_shouldbe = np.array([2.0, 2.75, 3.25, 3.25])
    assert np.allclose(hvi_values, hvi_values_shouldbe)
