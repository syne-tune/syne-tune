import numpy as np

from syne_tune.blackbox_repository.synthetic import Rosenbrock, Michalewicz, Ackley


def test_rosenbrock():
    # The Rosenbrock function has a global minimum of 0 at (1, 1, ..., 1).
    rosenbrock = Rosenbrock(dimension=5)
    configuration = {f"x{i}": 1.0 for i in range(5)}
    result = rosenbrock.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = rosenbrock.objective_function(configuration)
    assert np.isclose(result["y"], 4.0)


def test_michalewicz():
    # The Michalewicz function has a global minimum that depends on the dimension.
    # For d=2, the minimum is approx -1.8013.
    michalewicz = Michalewicz(dimension=2)
    configuration = {"x0": 2.20, "x1": 1.57}
    result = michalewicz.objective_function(configuration)
    assert np.isclose(result["y"], -1.8013, atol=1e-4)


def test_ackley():
    # The Ackley function has a global minimum of 0 at (0, 0, ..., 0).
    ackley = Ackley(dimension=5)
    configuration = {f"x{i}": 0.0 for i in range(5)}
    result = ackley.objective_function(configuration)
    assert np.isclose(result["y"], 0.0)

    # Test another point
    configuration = {f"x{i}": 1.0 for i in range(5)}
    result = ackley.objective_function(configuration)
    assert not np.isclose(result["y"], 0.0)
