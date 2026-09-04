"""
Implementation of the 24 noiseless BBOB (Black-Box Optimization Benchmarking) test functions.

The BBOB test suite is part of the COCO benchmarking platform:
    https://numbbo.github.io/coco/testsuites/bbob

Reference:
    Finck, S., Hansen, N., Ros, R., & Auger, A. (2010). Real-Parameter Black-Box
    Optimization Benchmarking 2009: Noiseless Functions Definitions. INRIA RR-6829.

Each function accepts an ``instance`` parameter (default 1) that seeds a pseudo-random
generator to produce the problem-specific shift vector ``xopt``, optimal value ``fopt``,
and rotation matrices ``R`` / ``Q``. Different instances yield distinct but reproducible
problem landscapes while preserving the characteristic difficulty of each function class.
"""

from typing import Any

import numpy as np

from syne_tune.blackbox_repository.blackbox import ObjectiveFunctionResult
from syne_tune.blackbox_repository.blackbox_artificial import BlackboxArtificial
from syne_tune.config_space import uniform


# ---------------------------------------------------------------------------
# Shared transformation helpers
# ---------------------------------------------------------------------------


def _t_osz(x: np.ndarray) -> np.ndarray:
    """Element-wise oscillation transformation T_osz.

    Maps each component via sign(x)*exp(log|x| + 0.049*(sin(c1*log|x|)+sin(c2*log|x|))).
    Returns 0 at exactly x=0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        xhat = np.where(x != 0.0, np.log(np.abs(x)), 0.0)
    c1 = np.where(x > 0.0, 10.0, 5.5)
    c2 = np.where(x > 0.0, 7.9, 3.1)
    return np.where(
        x == 0.0,
        0.0,
        np.sign(x) * np.exp(xhat + 0.049 * (np.sin(c1 * xhat) + np.sin(c2 * xhat))),
    )


def _t_asy(x: np.ndarray, beta: float) -> np.ndarray:
    """Element-wise asymmetric transformation T_asy(beta).

    For x_i > 0: x_i^(1 + beta * i/(n-1) * sqrt(x_i)).  For x_i <= 0: x_i.
    """
    n = len(x)
    if n == 1:
        return (
            x.copy() if x[0] <= 0.0 else x ** (1.0 + beta * np.sqrt(np.maximum(x, 0.0)))
        )
    idx = np.arange(n, dtype=float)
    exponents = 1.0 + beta * idx / (n - 1) * np.sqrt(np.maximum(x, 0.0))
    return np.where(x > 0.0, x**exponents, x)


def _cond_vec(n: int, alpha: float) -> np.ndarray:
    """Diagonal conditioning vector: entry i = alpha^(i / (2*(n-1))).

    Produces entries ranging from 1 (i=0) to sqrt(alpha) (i=n-1).
    """
    if n == 1:
        return np.ones(1)
    return alpha ** (np.arange(n, dtype=float) / (2.0 * (n - 1)))


def _penalty(x: np.ndarray) -> float:
    """Quadratic boundary penalty: sum of max(0, |x_i| - 5)^2."""
    return float(np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2))


def _rotation(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition of a Gaussian matrix.

    Diagonal of R is used to fix the sign so the determinant is +1.
    """
    if n == 1:
        return np.array([[1.0]])
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    return Q * np.sign(np.diag(R))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BBOBBase(BlackboxArtificial):
    """
    Base class for all 24 BBOB benchmark functions.

    Instance-specific parameters are derived from a seeded NumPy ``RandomState``
    so that every ``instance`` value yields a distinct but reproducible problem:

    - ``xopt``: shift vector drawn uniformly from [-4, 4]^d.
    - ``fopt``: optimal function value drawn uniformly from [-1000, 1000].
    - ``R``, ``Q``: random orthogonal matrices.

    :param dimension: Search-space dimensionality.
    :param instance: Problem instance identifier (≥ 1).
    """

    def __init__(self, dimension: int, instance: int = 1):
        configuration_space = {f"x{i}": uniform(-5.0, 5.0) for i in range(dimension)}
        super().__init__(
            dimension=dimension,
            configuration_space=configuration_space,
            objectives_names=["y"],
        )
        self.instance = instance
        rng = np.random.RandomState(instance)
        self.xopt: np.ndarray = rng.uniform(-4.0, 4.0, dimension)
        self.fopt: float = float(rng.uniform(-1000.0, 1000.0))
        self.R: np.ndarray = _rotation(rng, dimension)
        self.Q: np.ndarray = _rotation(rng, dimension)

    def _x(self, configuration: dict[str, Any]) -> np.ndarray:
        return np.array([configuration[f"x{i}"] for i in range(self.dimension)])


# ---------------------------------------------------------------------------
# f1 – f5: Separable functions
# ---------------------------------------------------------------------------


class BBOBSphere(BBOBBase):
    """
    f1: Sphere Function.

    Separable, unimodal, fully symmetric. Global minimum f(xopt) = fopt.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        z = self._x(configuration) - self.xopt
        return {"y": float(np.dot(z, z) + self.fopt)}


class BBOBSeparableEllipsoidal(BBOBBase):
    """
    f2: Separable Ellipsoidal Function.

    Separable, unimodal. Ill-conditioned (condition number 10^6). T_osz applied
    before the axis-aligned ellipsoidal weighting.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        z = _t_osz(self._x(configuration) - self.xopt)
        lam = 10.0 ** (6.0 * np.arange(n) / max(n - 1, 1))
        return {"y": float(np.dot(lam, z**2) + self.fopt)}


class BBOBRastrigin(BBOBBase):
    """
    f3: Rastrigin Function (separable).

    Separable, highly multimodal. T_osz and T_asy applied with mild
    conditioning (factor 10) before the Rastrigin evaluation.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        lam = _cond_vec(n, 10.0)
        z = lam * _t_asy(_t_osz(self._x(configuration) - self.xopt), 0.2)
        return {
            "y": float(
                10.0 * (n - np.sum(np.cos(2.0 * np.pi * z))) + np.dot(z, z) + self.fopt
            )
        }


class BBOBBuecheRastrigin(BBOBBase):
    """
    f4: Büche-Rastrigin Function.

    Separable, highly multimodal. Asymmetric dimension-wise scaling combined
    with T_osz; includes a quadratic boundary penalty.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        z_raw = _t_osz(x - self.xopt)
        # Extra factor 10 for even-indexed dimensions where xopt_i > 0
        s = np.array(
            [
                10.0 ** (i / max(2.0 * (n - 1), 1))
                * (10.0 if (i % 2 == 0 and self.xopt[i] > 0.0) else 1.0)
                for i in range(n)
            ]
        )
        z = s * z_raw
        return {
            "y": float(
                10.0 * (n - np.sum(np.cos(2.0 * np.pi * z)))
                + np.dot(z, z)
                + 100.0 * _penalty(x)
                + self.fopt
            )
        }


class BBOBLinearSlope(BBOBBase):
    """
    f5: Linear Slope.

    Separable. The global optimum lies at the boundary sign(xopt_i)*5 in each
    dimension. The slope is exponentially scaled across dimensions.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        sign_xopt = np.sign(self.xopt)
        s = sign_xopt * 10.0 ** (np.arange(n, dtype=float) / max(n - 1, 1))
        # Clip to boundary when x moves past it in the wrong direction
        z = np.where(x * sign_xopt < -25.0, sign_xopt * 5.0, x)
        return {"y": float(np.sum(5.0 * np.abs(s) - s * z) + self.fopt)}


# ---------------------------------------------------------------------------
# f6 – f9: Low or moderate conditioning
# ---------------------------------------------------------------------------


class BBOBAttractiveSector(BBOBBase):
    """
    f6: Attractive Sector Function.

    Components pointing toward the optimum are amplified by a factor of 100.
    Uses a combined rotation-conditioning matrix M = R * Λ^½(10) * Q.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        lam = _cond_vec(dimension, 10.0)
        self._M: np.ndarray = self.R @ np.diag(lam) @ self.Q

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        z = self._M @ (self._x(configuration) - self.xopt)
        sq = np.where(self.xopt * z > 0.0, (100.0 * z) ** 2, z**2)
        return {"y": float(_t_osz(float(np.sum(sq))) ** 0.9 + self.fopt)}


class BBOBStepEllipsoidal(BBOBBase):
    """
    f7: Step Ellipsoidal Function.

    Combines an ellipsoidal structure with a step-like (round-to-nearest)
    transformation, creating flat regions that impede gradient-based methods.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        self._lam: np.ndarray = _cond_vec(dimension, 10.0)

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        z_hat = self._lam * (self.Q @ (x - self.xopt))
        # Round large components; shrink small ones
        z = np.where(
            np.abs(z_hat) > 0.5,
            np.floor(0.5 + z_hat),
            np.floor(0.5 + 10.0 * z_hat) / 10.0,
        )
        g = self.R @ z
        lam2 = 10.0 ** (2.0 * np.arange(n, dtype=float) / max(n - 1, 1))
        val = 0.1 * max(np.abs(g[0]) / 1e4, float(np.dot(lam2, g**2)))
        return {"y": float(val + _penalty(x) + self.fopt)}


class BBOBRosenbrock(BBOBBase):
    """
    f8: Rosenbrock Function, original.

    Classic banana-shaped valley. Linear shift so that the optimum is at xopt.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = self._x(configuration)
        scale = max(1.0, np.sqrt(self.dimension) / 8.0)
        z = scale * (x - self.xopt) + 1.0
        val = np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2)
        return {"y": float(val + self.fopt)}


class BBOBRosenbrockRotated(BBOBBase):
    """
    f9: Rosenbrock Function, rotated.

    Like f8 but rotation is applied to break separability; the optimum is no
    longer at xopt but at R^T * (0.5/scale) * 1.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        scale = max(1.0, np.sqrt(n) / 8.0)
        z = scale * (self.R @ x) + 0.5
        val = np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2)
        return {"y": float(val + self.fopt)}


# ---------------------------------------------------------------------------
# f10 – f14: High conditioning, unimodal
# ---------------------------------------------------------------------------


class BBOBEllipsoidalRotated(BBOBBase):
    """
    f10: Ellipsoidal Function (rotated).

    Non-separable, unimodal. High condition number (10^6). T_osz applied
    after rotation by R.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        z = _t_osz(self.R @ (self._x(configuration) - self.xopt))
        lam = 10.0 ** (6.0 * np.arange(n, dtype=float) / max(n - 1, 1))
        return {"y": float(np.dot(lam, z**2) + self.fopt)}


class BBOBDiscus(BBOBBase):
    """
    f11: Discus Function.

    The first variable carries 10^6 times the weight of the remaining ones.
    Unimodal; T_osz applied after rotation.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        z = _t_osz(self.R @ (self._x(configuration) - self.xopt))
        return {"y": float(1e6 * z[0] ** 2 + np.dot(z[1:], z[1:]) + self.fopt)}


class BBOBBentCigar(BBOBBase):
    """
    f12: Bent Cigar Function.

    First variable has much smaller weight (10^6 conditioning). T_asy applied
    between two rotations R so the asymmetry interacts with both.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        z = self.R @ _t_asy(self.R @ (self._x(configuration) - self.xopt), 0.5)
        return {"y": float(z[0] ** 2 + 1e6 * np.dot(z[1:], z[1:]) + self.fopt)}


class BBOBSharpRidge(BBOBBase):
    """
    f13: Sharp Ridge Function.

    A single sharp ridge (linear in the transverse norm) makes gradient
    following difficult. Uses M = R * Λ^½(10) * Q.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        lam = _cond_vec(dimension, 10.0)
        self._M: np.ndarray = self.R @ np.diag(lam) @ self.Q

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        z = self._M @ (self._x(configuration) - self.xopt)
        ridge = (
            100.0 * np.sqrt(float(np.dot(z[1:], z[1:]))) if self.dimension > 1 else 0.0
        )
        return {"y": float(z[0] ** 2 + ridge + self.fopt)}


class BBOBDifferentPowers(BBOBBase):
    """
    f14: Different Powers Function.

    Each variable is raised to a different power (2 to 6), creating very
    different curvature across dimensions. Unimodal.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        z = self.R @ (self._x(configuration) - self.xopt)
        exponents = 2.0 + 4.0 * np.arange(n, dtype=float) / max(n - 1, 1)
        return {"y": float(np.sqrt(np.sum(np.abs(z) ** exponents)) + self.fopt)}


# ---------------------------------------------------------------------------
# f15 – f19: Multimodal with adequate structure
# ---------------------------------------------------------------------------


class BBOBRastriginRotated(BBOBBase):
    """
    f15: Rastrigin Function (multimodal / rotated).

    Non-separable variant of f3: two rotations (R, Q) and T_asy applied
    before the Rastrigin formula, conditioning factor 10.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        self._lam: np.ndarray = _cond_vec(dimension, 10.0)

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        y = self.R @ (self._x(configuration) - self.xopt)
        z = self.R @ (self._lam * _t_asy(self.Q @ _t_osz(y), 0.2))
        return {
            "y": float(
                10.0 * (n - np.sum(np.cos(2.0 * np.pi * z))) + np.dot(z, z) + self.fopt
            )
        }


class BBOBWeierstrass(BBOBBase):
    """
    f16: Weierstrass Function.

    Continuous but intentionally non-differentiable at many points.
    Conditioning factor 100, two rotations, T_osz.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        self._lam: np.ndarray = _cond_vec(dimension, 100.0)
        k = np.arange(12)
        self._ak: np.ndarray = 0.5**k
        self._bk: np.ndarray = 3.0**k
        # Baseline: value of the inner sum at z_i = 0, used so f(xopt) = fopt
        self._f0: float = float(np.sum(self._ak * np.cos(np.pi * self._bk)))

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        y = self.R @ (x - self.xopt)
        z = self.R @ (self._lam * (self.Q @ _t_osz(y)))
        per_dim = np.array(
            [
                float(np.sum(self._ak * np.cos(2.0 * np.pi * self._bk * (z[i] + 0.5))))
                for i in range(n)
            ]
        )
        val = 10.0 * (np.mean(per_dim) - self._f0) ** 3
        val += (10.0 / n**2) * _penalty(x)
        return {"y": float(val + self.fopt)}


class BBOBSchaffers(BBOBBase):
    """
    f17: Schaffer's F7 Function.

    Non-separable, multimodal. Oscillating ridges created by a sqrt and
    sinusoidal term. Conditioning factor 10.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        self._lam: np.ndarray = _cond_vec(dimension, 10.0)

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = self._x(configuration)
        z = self._lam * (self.Q @ _t_asy(self.R @ (x - self.xopt), 0.5))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        inner = np.sqrt(s) * (1.0 + np.sin(50.0 * s**0.2) ** 2)
        val = float(np.mean(inner)) ** 2
        return {"y": float(val + 10.0 * _penalty(x) + self.fopt)}


class BBOBSchaffersModerate(BBOBBase):
    """
    f18: Schaffer's F7, moderately ill-conditioned.

    Same structure as f17 but with conditioning factor 1000 instead of 10,
    significantly increasing difficulty.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        self._lam: np.ndarray = _cond_vec(dimension, 1000.0)

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = self._x(configuration)
        z = self._lam * (self.Q @ _t_asy(self.R @ (x - self.xopt), 0.5))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        inner = np.sqrt(s) * (1.0 + np.sin(50.0 * s**0.2) ** 2)
        val = float(np.mean(inner)) ** 2
        return {"y": float(val + 10.0 * _penalty(x) + self.fopt)}


class BBOBGriewankRosenbrock(BBOBBase):
    """
    f19: Composite Griewank-Rosenbrock Function F8F2.

    Combines Rosenbrock-like nearest-variable coupling with a Griewank cosine
    term, creating a moderately multimodal landscape. No xopt shift is used;
    the rotation R determines the optimum location.
    """

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        scale = max(1.0, np.sqrt(n) / 8.0)
        z = scale * (self.R @ x) + 0.5
        # Rosenbrock-style coupling terms
        s = 100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
        griewank = s**2 / 4000.0 - np.cos(s) + 1.0
        val = 10.0 * float(np.mean(griewank))
        return {"y": float(val + self.fopt)}


# ---------------------------------------------------------------------------
# f20 – f24: Multimodal with weak structure
# ---------------------------------------------------------------------------


class BBOBSchwefel(BBOBBase):
    """
    f20: Schwefel Function.

    Multimodal with deceptive structure; many good local optima near the domain
    boundary. The classical Schwefel peak is at ≈ 420.97, which is reached via
    a 100x scaling.

    xopt components are fixed at ±0.5*4.2096874637 (signs random per instance)
    so the 100x-scaled optimum falls exactly on the Schwefel peak.
    """

    _SCHWEFEL_MAG = 0.5 * 4.2096874637

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        # Override xopt: random signs, fixed magnitude
        rng_s = np.random.RandomState(instance + 100000)
        signs = np.where(rng_s.randint(0, 2, self.dimension) == 0, -1.0, 1.0)
        self.xopt = signs * self._SCHWEFEL_MAG

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        # Step 1: sign-flip based on xopt
        x_hat = 2.0 * np.sign(self.xopt) * x
        # Step 2: neighbor coupling (z_hat)
        z_hat = np.empty(n)
        z_hat[0] = x_hat[0]
        for i in range(1, n):
            z_hat[i] = x_hat[i] + 0.25 * (x_hat[i - 1] - 2.0 * np.abs(self.xopt[i - 1]))
        # Steps 3–5: shift up, condition, shift down, scale by 100
        lam = _cond_vec(n, 10.0)
        w = z_hat + 2.0 * np.abs(self.xopt)
        v = lam * w
        u = v - 2.0 * np.abs(self.xopt)
        z = 100.0 * u
        # Raw Schwefel with penalty for |z_i| > 500
        z_pen = float(np.sum(np.maximum(0.0, np.abs(z) - 500.0) ** 2))
        val = 0.01 * (
            z_pen + 418.9828872724339 - float(np.mean(z * np.sin(np.sqrt(np.abs(z)))))
        )
        return {"y": float(val + _penalty(x) + self.fopt)}


class _GallagherBase(BBOBBase):
    """
    Shared implementation for Gallagher's Gaussian peak functions (f21, f22).

    The rotation R from BBOBBase is applied to the input. Peak locations,
    per-peak widths, and weights are drawn from a separate RNG seeded by
    ``instance + 200000`` to avoid interference with R / Q.
    """

    def __init__(self, dimension: int, n_peaks: int, instance: int = 1):
        super().__init__(dimension, instance)
        rng = np.random.RandomState(instance + 200000)
        n = dimension
        # Peak weights: highest peak has weight 10; others linearly spaced in [1.1, 9.1]
        if n_peaks > 2:
            self._w = np.concatenate(
                [
                    [10.0],
                    1.1 + 8.0 * np.arange(1, n_peaks) / (n_peaks - 2),
                ]
            )
        else:
            self._w = np.array([10.0, 1.1] if n_peaks == 2 else [10.0])
        self._w = self._w[:n_peaks]
        # Peak locations in the rotated space
        self._y: np.ndarray = rng.uniform(-5.0, 5.0, (n_peaks, n))
        # Per-peak conditioning exponent and resulting sigma vectors
        cond_exp = np.linspace(-5.0, 5.0, n_peaks)
        rng.shuffle(cond_exp)
        self._sigma: np.ndarray = np.array(
            [_cond_vec(n, 10.0) * (10.0 ** (cond_exp[k] / 2.0)) for k in range(n_peaks)]
        )

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        tx = self.R @ x  # rotated input (peak locations are in this space)
        peaks = np.array(
            [
                self._w[k]
                * np.exp(-np.sum(((tx - self._y[k]) / self._sigma[k]) ** 2) / (2.0 * n))
                for k in range(len(self._w))
            ]
        )
        val = float(_t_osz(10.0 - float(np.max(peaks)))) ** 2 + _penalty(x)
        return {"y": float(val + self.fopt)}


class BBOBGallagher101(_GallagherBase):
    """
    f21: Gallagher's Gaussian 101-me Peaks Function.

    101 Gaussian peaks of varying height and width; one dominant global peak
    with 100 competing local ones.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, n_peaks=101, instance=instance)


class BBOBGallagher21(_GallagherBase):
    """
    f22: Gallagher's Gaussian 21-hi Peaks Function.

    Like f21 but with only 21 peaks, making the global structure slightly
    easier to identify while the local peaks are more pronounced.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, n_peaks=21, instance=instance)


class BBOBKatsuura(BBOBBase):
    """
    f23: Katsuura Function.

    Fractal-like landscape based on a product of perturbed Weierstrass-style
    sums. Conditioning factor 100, two rotations.
    """

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        lam = _cond_vec(dimension, 100.0)
        self._M: np.ndarray = self.R @ np.diag(lam) @ self.Q

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        z = self._M @ (x - self.xopt)
        two_pow = 2.0 ** np.arange(1, 33)
        product = 1.0
        for i in range(n):
            frac = np.abs(two_pow * z[i] - np.round(two_pow * z[i]))
            inner = float(np.sum(frac / two_pow))
            product *= (1.0 + (i + 1) * inner) ** (10.0 / n**1.2)
        val = (10.0 / n**2) * (product - 1.0)
        return {"y": float(val + _penalty(x) + self.fopt)}


class BBOBLunacekBiRastrigin(BBOBBase):
    """
    f24: Lunacek bi-Rastrigin Function.

    Two competing global optima: one sphere-like and one on the boundary.
    Highly deceptive; multimodal oscillation layered on top.
    """

    _MU0 = 2.5
    _D_PARAM = 1.0  # the "d" in the Lunacek formula (not dimension)

    def __init__(self, dimension: int, instance: int = 1):
        super().__init__(dimension, instance)
        n = dimension
        s = 1.0 - 0.5 / (np.sqrt(n + 20.0) - 4.1)
        self._s = s
        self._mu1 = -np.sqrt((self._MU0**2 - self._D_PARAM) / s)
        lam = _cond_vec(dimension, 100.0)
        self._M: np.ndarray = self.R @ np.diag(lam) @ self.Q

    def _objective_function(
        self,
        configuration: dict[str, Any],
        fidelity: dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        n = self.dimension
        x = self._x(configuration)
        # Sign-scaled input; optimum of sum1 at x_hat = mu0 (all components)
        x_hat = 2.0 * np.sign(self.xopt) * x
        z = self._M @ (x_hat - self._MU0)
        sum1 = float(np.dot(x_hat - self._MU0, x_hat - self._MU0))
        sum2 = float(np.dot(x_hat - self._mu1, x_hat - self._mu1))
        sum3 = float(np.sum(np.cos(2.0 * np.pi * z)))
        pen = float(np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2))
        val = (
            min(sum1, self._D_PARAM * n + self._s * sum2)
            + 10.0 * (n - sum3)
            + 1e4 * pen
        )
        return {"y": float(val + self.fopt)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Pre-instantiated BBOB functions for dimensions 2, 5, and 10 (instance=1).
#: Accessible via ``load_blackbox(name)`` after merging into
#: ``global_optimization_problem_collection``.
bbob_problem_collection: dict[str, BBOBBase] = {}

_BBOB_DIMS = [2, 5, 10]
_BBOB_CLASSES: list[tuple[str, type]] = [
    ("f01_sphere", BBOBSphere),
    ("f02_separable_ellipsoidal", BBOBSeparableEllipsoidal),
    ("f03_rastrigin", BBOBRastrigin),
    ("f04_bueche_rastrigin", BBOBBuecheRastrigin),
    ("f05_linear_slope", BBOBLinearSlope),
    ("f06_attractive_sector", BBOBAttractiveSector),
    ("f07_step_ellipsoidal", BBOBStepEllipsoidal),
    ("f08_rosenbrock", BBOBRosenbrock),
    ("f09_rosenbrock_rotated", BBOBRosenbrockRotated),
    ("f10_ellipsoidal_rotated", BBOBEllipsoidalRotated),
    ("f11_discus", BBOBDiscus),
    ("f12_bent_cigar", BBOBBentCigar),
    ("f13_sharp_ridge", BBOBSharpRidge),
    ("f14_different_powers", BBOBDifferentPowers),
    ("f15_rastrigin_rotated", BBOBRastriginRotated),
    ("f16_weierstrass", BBOBWeierstrass),
    ("f17_schaffers", BBOBSchaffers),
    ("f18_schaffers_moderate", BBOBSchaffersModerate),
    ("f19_griewank_rosenbrock", BBOBGriewankRosenbrock),
    ("f20_schwefel", BBOBSchwefel),
    ("f21_gallagher101", BBOBGallagher101),
    ("f22_gallagher21", BBOBGallagher21),
    ("f23_katsuura", BBOBKatsuura),
    ("f24_lunacek_bi_rastrigin", BBOBLunacekBiRastrigin),
]

for _dim in _BBOB_DIMS:
    for _fname, _cls in _BBOB_CLASSES:
        bbob_problem_collection[f"bbob_{_fname}_{_dim}d"] = _cls(_dim)
