import unittest
import numpy as np

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.bbob import (
    BBOBSphere,
    BBOBSeparableEllipsoidal,
    BBOBRastrigin,
    BBOBBuecheRastrigin,
    BBOBAttractiveSector,
    BBOBStepEllipsoidal,
    BBOBRosenbrock,
    BBOBEllipsoidalRotated,
    BBOBDiscus,
    BBOBBentCigar,
    BBOBSharpRidge,
    BBOBDifferentPowers,
    BBOBRastriginRotated,
    BBOBWeierstrass,
    BBOBSchaffers,
    BBOBSchaffersModerate,
    BBOBKatsuura,
    bbob_problem_collection,
)


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


class TestBBOBAtOptimum(unittest.TestCase):
    """
    For most BBOB functions the transformed argument equals zero when x = xopt,
    so f(xopt) = fopt exactly. This test class verifies that invariant for the
    functions where it holds analytically.
    """

    def _check_at_xopt(self, cls, dimension=2, instance=1, places=8):
        f = cls(dimension=dimension, instance=instance)
        config = {f"x{i}": float(f.xopt[i]) for i in range(dimension)}
        result = f(config)
        self.assertAlmostEqual(result["y"], f.fopt, places=places)

    def test_sphere(self):
        self._check_at_xopt(BBOBSphere)

    def test_sphere_5d(self):
        self._check_at_xopt(BBOBSphere, dimension=5)

    def test_separable_ellipsoidal(self):
        self._check_at_xopt(BBOBSeparableEllipsoidal)

    def test_rastrigin(self):
        self._check_at_xopt(BBOBRastrigin)

    def test_bueche_rastrigin(self):
        self._check_at_xopt(BBOBBuecheRastrigin)

    def test_attractive_sector(self):
        self._check_at_xopt(BBOBAttractiveSector)

    def test_step_ellipsoidal(self):
        # The step function rounds z_hat to zero when x = xopt, so f(xopt) = fopt
        self._check_at_xopt(BBOBStepEllipsoidal)

    def test_rosenbrock(self):
        # At xopt: z = 1, Rosenbrock(1,...,1) = 0
        self._check_at_xopt(BBOBRosenbrock)

    def test_ellipsoidal_rotated(self):
        self._check_at_xopt(BBOBEllipsoidalRotated)

    def test_discus(self):
        self._check_at_xopt(BBOBDiscus)

    def test_bent_cigar(self):
        self._check_at_xopt(BBOBBentCigar)

    def test_sharp_ridge(self):
        self._check_at_xopt(BBOBSharpRidge)

    def test_different_powers(self):
        self._check_at_xopt(BBOBDifferentPowers)

    def test_rastrigin_rotated(self):
        self._check_at_xopt(BBOBRastriginRotated)

    def test_weierstrass(self):
        self._check_at_xopt(BBOBWeierstrass, places=6)

    def test_schaffers(self):
        self._check_at_xopt(BBOBSchaffers)

    def test_schaffers_moderate(self):
        self._check_at_xopt(BBOBSchaffersModerate)

    def test_katsuura(self):
        self._check_at_xopt(BBOBKatsuura)


class TestBBOBSmoke(unittest.TestCase):
    """Smoke tests: every registered BBOB entry can be loaded and evaluated."""

    def test_load_and_call_all(self):
        for name, bb in bbob_problem_collection.items():
            with self.subTest(name=name):
                loaded = load_blackbox(name)
                # Evaluate at the lower bound of each dimension
                config = {
                    k: float(v.lower)
                    for k, v in loaded.configuration_space.items()
                }
                result = loaded(config)
                self.assertIn("y", result)
                self.assertTrue(np.isfinite(result["y"]), f"{name} returned non-finite value")
