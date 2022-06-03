# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np


class DifferentialevolutionOptimizer:
    def __init__(self, f, lower, upper, fevals, strategy="best1", bin=1):

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
        self.fbest = np.float("inf")
        self.idxbest = 1
        self.strategy = strategy
        self.bin = bin

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

        # perform crossover operation
        if self.bin == 1:
            cross_points = np.random.rand(self.dimensions) < self.crossp

            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dimensions)] = True
            ui = np.where(cross_points, vi, current_idv)

        else:
            i = 0
            ui = []
            fill_point = np.random.randint(0, self.dimensions)
            while i < self.dimensions and np.random.rand(0, 1) < self.crossp:
                ui[fill_point] = vi[fill_point]
                fill_point = (fill_point + 1) % self.dimensions
                i += 1

        return ui

    def run(self):

        traj = []

        # Step1: initialization
        rand_temp = np.random.rand(self.popsize, self.dimensions)
        diff = np.fabs(self.lower_bound - self.upper_bound)
        self.de_pop = self.lower_bound + rand_temp * diff

        # Step 2: population evaluation
        best = None
        for j in range(self.popsize):
            ftemp = self.f(self.de_pop[j])
            self.fitness.append(ftemp)

            if ftemp < self.fbest:
                self.fbest = ftemp
                self.idxbest = j
                best = self.de_pop[j]
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
                        best = ui

                traj.append(self.fitness[self.idxbest])

        return best, traj
