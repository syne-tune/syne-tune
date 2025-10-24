from typing import Any, Dict, List

import numpy as np

from syne_tune.blackbox_repository.blackbox import Blackbox, ObjectiveFunctionResult
from syne_tune.config_space import uniform


class SyntheticFunction(Blackbox):
    """
    Base class for synthetic blackbox functions. These are defined by a formula and
    are cheap to evaluate. They are mainly used for testing and to prototype new
    algorithms.
    """

    def __init__(
        self,
        dimension: int,
        configuration_space: Dict[str, Any] | None = None,
        objectives_names: List[str] | None = None,
    ):
        if configuration_space is None:
            configuration_space = {
                f"x{i}": uniform(-5.0, 10.0) for i in range(dimension)
            }
        if objectives_names is None:
            objectives_names = ["y"]
        super().__init__(
            configuration_space=configuration_space,
            objectives_names=objectives_names,
        )
        self.dimension = dimension

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        raise NotImplementedError


class Rosenbrock(SyntheticFunction):
    """
    The Rosenbrock function is a non-convex function used as a performance test
    problem for optimization algorithms.
    See https://www.sfu.ca/~ssurjano/rosen.html for details.
    """

    def __init__(self, dimension: int, lower_bound: float = -5.0, upper_bound: float = 10.0):
        self.configuration_space = {f'x{i}': uniform(lower_bound, upper_bound) for i in range(dimension)}
        super().__init__(dimension=dimension,
                         configuration_space=self.configuration_space,
                         objectives_names=["y"])

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        val = np.sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (x[:-1] - 1.0) ** 2.0
        )
        return {self.objectives_names[0]: float(val)}


class Michalewicz(SyntheticFunction):
    """
    The Michalewicz function is a multimodal function with d! local minima.
    See https://www.sfu.ca/~ssurjano/michal.html for details.
    """

    def __init__(self, dimension: int, m: float = 10.0):
        self.configuration_space = {
            f"x{i}": uniform(0, np.pi) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.m = m

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        i = np.arange(1, self.dimension + 1)
        val = -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi)) ** (2 * self.m))
        return {self.objectives_names[0]: float(val)}


class Ackley(SyntheticFunction):
    """
    The Ackley function is a multimodal function with a global optimum surrounded
    by many local optima.
    See https://www.sfu.ca/~ssurjano/ackley.html for details.
    """

    def __init__(self, dimension: int):
        self.configuration_space = {
            f"x{i}": uniform(-32.768, 32.768) for i in range(dimension)
        }
        super().__init__(
            dimension=dimension,
            configuration_space=self.configuration_space,
            objectives_names=["y"],
        )
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Dict | None = None,
        seed: int | None = None,
    ) -> ObjectiveFunctionResult:
        x = np.array([configuration[f"x{i}"] for i in range(self.dimension)])
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(self.c * x)))
        val = term1 + term2 + self.a + np.exp(1)
        return {self.objectives_names[0]: float(val)}
