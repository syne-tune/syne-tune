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
from typing import Optional, List

import numpy as np
from syne_tune.optimizer.schedulers.multiobjective.non_dominated_priority import (
    nondominated_sort,
)


class MOPriority:
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        :param metrics: name of the objectives, optional if not passed anonymous names are created when seeing the
        first objectives to rank.
        """
        self.metrics = metrics

    def __call__(self, objectives: np.array) -> np.array:
        """
        :param objectives: that should be argsorted with shape (num_samples, num_objectives)
        :return: a vector with shape (num_samples,) that gives priority for the different elements (lower elements
        are picked first).
        """
        num_samples, num_objectives = objectives.shape
        if self.metrics is None:
            # set anonymous metric names
            self.metrics = [f"metric-{i}" for i in range(num_objectives)]
        assert num_objectives == len(self.metrics)
        return self.priority_unsafe(objectives=objectives)

    def priority_unsafe(self, objectives: np.array) -> np.array:
        raise NotImplementedError()


class LinearScalarizationPriority(MOPriority):
    def __init__(
        self, metrics: Optional[List[str]] = None, weights: Optional[np.array] = None
    ):
        """
        A simple multiobjective scalarization strategy that do a weighed sum to assign a priority to the objectives.
        :param metrics:
        :param weights:
        """
        super(LinearScalarizationPriority, self).__init__(metrics=metrics)
        if weights is None:
            # uniform weights by default
            self.weights = np.ones(1)
        else:
            if metrics is not None:
                assert len(weights) == len(metrics)
            self.weights = weights

        # makes multiplication convenient with batch of samples
        self.weights = np.expand_dims(self.weights, 0)

    def priority_unsafe(self, objectives: np.array) -> np.array:
        weighted_objectives = (objectives * self.weights).mean(axis=-1)
        return weighted_objectives


class FixedObjectivePriority(MOPriority):
    def __init__(self, metrics: Optional[List[str]] = None, dim: Optional[int] = None):
        """
        Optimizes a fixed objective, the first one by default.
        :param metrics:
        :param dim: dimension of the objective to optimize, first one by default.
        """
        super(FixedObjectivePriority, self).__init__(metrics=metrics)
        self.dim = dim if dim is not None else 0

    def priority_unsafe(self, objectives: np.array) -> np.array:
        return objectives[:, self.dim]


class NonDominatedPriority(MOPriority):
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        dim: Optional[int] = 0,
        max_num_samples: Optional[int] = None,
    ):
        """
        A non-dominated sort strategy that uses an epsilon-net strategy instead of crowding distance proposed in:

        A multi-objective perspective on jointly tuning hardware and hyperparameters
        David Salinas, Valerio Perrone, Cedric Archambeau and Olivier Cruchant
        NAS workshop, ICLR2021.

        :param metrics:
        :param dim: The objective to prefer when ranking items within the Pareto front and picking the first
        element. If `None`, the first element is chosen randomly.
        :param max_num_samples: The maximum number of samples that should be returned.
        When this is `None`, all items are sorted (less efficient), if you have a large number of samples but only want
        the top k indices, set this to k for efficiency.
        """
        super(NonDominatedPriority, self).__init__(metrics=metrics)
        self.dim = dim
        self.max_num_samples = max_num_samples

    def priority_unsafe(self, objectives: np.array) -> np.array:
        return np.array(
            nondominated_sort(
                X=objectives, dim=self.dim, max_items=self.max_num_samples
            )
        )
