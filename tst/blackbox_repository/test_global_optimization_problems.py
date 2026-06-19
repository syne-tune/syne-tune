import unittest
import numpy as np

from syne_tune.blackbox_repository import load_blackbox


class TestSyntheticFunctions(unittest.TestCase):
    def test_rosenbrock(self):
        # Global minimum is at (1, 1, ..., 1) with value 0
        rosenbrock = load_blackbox("rosenbrock_2d")
        config = {"x0": 1.0, "x1": 1.0}
        result = rosenbrock(config)
        self.assertAlmostEqual(result["y"], 0.0, places=4)

    def test_michalewicz(self):
        # Global minimum for 2D is approx -1.8013
        michalewicz = load_blackbox("michalewicz_2d")
        config = {"x0": 2.20, "x1": 1.57}
        result = michalewicz(config)
        self.assertAlmostEqual(result["y"], -1.8013, places=3)

    def test_ackley(self):
        # Global minimum is at (0, 0, ..., 0) with value 0
        ackley = load_blackbox("ackley_2d")
        config = {"x0": 0.0, "x1": 0.0}
        result = ackley(config)
        self.assertAlmostEqual(result["y"], 0.0, places=4)

    def test_branin(self):
        # One of the global minima
        branin = load_blackbox("branin")
        config = {"x0": -np.pi, "x1": 12.275}
        result = branin(config)
        self.assertAlmostEqual(result["y"], 0.397887, places=4)

    def test_hartman3(self):
        # Global minimum
        hartman3 = load_blackbox("hartman3")
        config = {"x0": 0.114614, "x1": 0.555649, "x2": 0.852547}
        result = hartman3(config)
        self.assertAlmostEqual(result["y"], -3.86278, places=4)

    def test_hartman6(self):
        # Global minimum
        hartman6 = load_blackbox("hartman6")
        config = {
            "x0": 0.201690,
            "x1": 0.150011,
            "x2": 0.476874,
            "x3": 0.275332,
            "x4": 0.311652,
            "x5": 0.657300,
        }
        result = hartman6(config)
        self.assertAlmostEqual(result["y"], -3.32237, places=4)

    def test_goldstein_price(self):
        # Global minimum is at (0, -1) with value 3
        goldstein_price = load_blackbox("goldstein_price")
        config = {"x0": 0.0, "x1": -1.0}
        result = goldstein_price(config)
        self.assertAlmostEqual(result["y"], 3.0, places=4)
