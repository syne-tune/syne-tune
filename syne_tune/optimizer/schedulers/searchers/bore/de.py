from typing import Callable

import numpy as np

STRATEGIES = (
    "rand1",
    "best1",
    "rand2",
    "best2",
    "currenttobest1",
    "randtobest1",
)


class DifferentialevolutionOptimizer:
    """
    Differential evolution with binomial crossover, used by BORE to optimize
    the acquisition function.

    :param f: Function to minimize, maps a point of shape ``(dimensions,)``
        to a scalar
    :param lower: Lower bounds, shape ``(dimensions,)``
    :param upper: Upper bounds, shape ``(dimensions,)``
    :param fevals: Total budget of function evaluations
    :param strategy: Mutation strategy, one of :const:`STRATEGIES`.
        Defaults to "best1"
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        lower: np.ndarray,
        upper: np.ndarray,
        fevals: int,
        strategy: str = "best1",
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"strategy = {strategy} not supported, must be one of {STRATEGIES}"
            )
        self.f = f

        self.lower_bound = lower
        self.upper_bound = upper

        self.mut = 0.5
        self.crossp = 0.5

        self.popsize = 10
        self.its = fevals // self.popsize - 1
        self.dimensions = len(self.lower_bound)

        self.de_pop = []
        self.fitness = []
        self.fbest = np.inf
        self.idxbest = 1
        self.strategy = strategy

    def evolve(self, j):
        best_idv = self.de_pop[self.idxbest]
        current_idv = self.de_pop[j]

        # perform mutation operation
        if self.strategy == "rand1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.de_pop[np.random.choice(idxs, 3, replace=False)]

            # Step 3.1: Perform mutation and checking
            temp = r1 + self.mut * (r2 - r3)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "best1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.de_pop[np.random.choice(idxs, 2, replace=False)]
            temp = best_idv + self.mut * (r1 - r2)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "rand2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4, r5 = self.de_pop[np.random.choice(idxs, 5, replace=False)]
            temp = r1 + self.mut * (r1 - r2) + self.mut * (r3 - r4)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "best2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4 = self.de_pop[np.random.choice(idxs, 4, replace=False)]
            temp = best_idv + self.mut * (r1 - r2) + self.mut * (r3 - r4)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "currenttobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.de_pop[np.random.choice(idxs, 2, replace=False)]
            temp = (
                current_idv + self.mut * (best_idv - current_idv) + self.mut * (r1 - r2)
            )
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "randtobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.de_pop[np.random.choice(idxs, 3, replace=False)]
            temp = r1 + self.mut * (best_idv - r1) + self.mut * (r2 - r3)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        # perform binomial crossover operation
        cross_points = np.random.rand(self.dimensions) < self.crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        ui = np.where(cross_points, vi, current_idv)

        return ui

    def run(self):

        traj = []

        # Step1: initialization
        rand_temp = np.random.rand(self.popsize, self.dimensions)
        diff = np.fabs(self.lower_bound - self.upper_bound)
        self.de_pop = self.lower_bound + rand_temp * diff

        # Step 2: population evaluation
        for j in range(self.popsize):
            ftemp = self.f(self.de_pop[j])
            self.fitness.append(ftemp)

            if ftemp < self.fbest:
                self.fbest = ftemp
                self.idxbest = j
            traj.append(self.fbest)

        # Step 3: Start evolutionary search
        for i in range(self.its):
            for j in range(self.popsize):
                ui = self.evolve(j)

                fit = self.f(ui)

                # Step3.5: Perform Selection
                if fit < self.fitness[j]:
                    self.fitness[j] = fit
                    self.de_pop[j] = ui
                    if fit < self.fitness[self.idxbest]:
                        self.idxbest = j

                traj.append(self.fitness[self.idxbest])

        return self.de_pop[self.idxbest].copy(), traj
