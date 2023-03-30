import numpy as np

from syne_tune.optimizer.schedulers.multiobjective.utils import hypervolume


def test_hypervolume():
    ref_point = np.array([2, 2])
    points = np.array([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
    hv = hypervolume(results_array=points, reference_points=ref_point)
    assert hv(points) == 3.25
