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
import logging

import numpy as np

from typing import Optional, List, Union, Dict, Any

from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.config_space import Domain, Float, Integer, Categorical

from pymoo.algorithms.moo.nsga2 import NSGA2 as PYMOONSGA2
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableGA,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice, Binary
from pymoo.core.variable import Integer as PyMOOInteger


class MultiObjectiveMixedVariableProblem(ElementwiseProblem):
    def __init__(self, n_obj, config_space, **kwargs):
        vars = {}

        for hp in config_space:
            if isinstance(hp, Categorical):
                vars[hp] = Choice(options=config_space[hp].categories)
            elif isinstance(hp, Integer):
                vars[hp] = PyMOOInteger(
                    bounds=(config_space[hp].lower, config_space[hp].upper)
                )
            elif isinstance(hp, Float):
                vars[hp] = Real(bounds=(config_space[hp].lower, config_space[hp].upper))

        super().__init__(vars=vars, n_obj=n_obj, n_ieq_constr=0, **kwargs)


class NSGA2(StochasticSearcher):
    """


    :param mode: Mode to use for the metric given, can be "min" or "max",
        defaults to "min"
    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    """

    def __init__(
        self,
        config_space,
        metric: List[str],
        mode: Union[List[str], str],
        points_to_evaluate: Optional[List[dict]] = None,
        pop_size: int = 100,
        **kwargs,
    ):
        super(NSGA2, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        if isinstance(mode, str):
            modes = [mode] * len(metric)
        else:
            modes = mode

        xl = []
        xu = []
        self.hp_names = []
        is_mixed_variable = False
        for hp_name, hp in config_space.items():
            if isinstance(hp, Domain):
                self.hp_names.append(hp_name)
                if isinstance(hp, Categorical) or isinstance(hp, Integer):
                    is_mixed_variable = True
                elif isinstance(hp, Float):
                    xl.append(hp.lower)
                    xu.append(hp.upper)

                else:
                    raise Exception(
                        f"Type {type(hp)} for hyperparameter {hp_name} "
                        f"is not support for NSGA-2."
                    )

        if is_mixed_variable:
            self.problem = MultiObjectiveMixedVariableProblem(
                config_space=config_space, n_var=len(self.hp_names), n_obj=len(metric)
            )
            self.algorithm = PYMOONSGA2(
                pop_size=pop_size,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(
                    eliminate_duplicates=MixedVariableDuplicateElimination()
                ),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            )
        else:
            self.problem = Problem(n_obj=len(metric), n_var=len(xl), xl=xl, xu=xu)
            self.algorithm = PYMOONSGA2(pop_size=pop_size)
        self.algorithm.setup(problem=self.problem, verbose=True)

        self.current_population = self.algorithm.ask()
        self.current_individual = 0
        self.observed_values = []

    def _update(self, trial_id: str, config: dict, result: dict):
        results = {}
        for mode, metric in zip(self._mode, self._metric):
            value = result[metric]
            if mode == "max":
                value *= -1
            results[metric] = value

        self.observed_values.append(list(result.values()))

    def get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_initial_config` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """
        if self.current_individual >= len(self.current_population):
            logging.info("Update population")

            static = StaticProblem(self.problem, F=np.array(self.observed_values))
            Evaluator().eval(static, self.current_population)
            # self.algorithm.evaluator.eval(self.problem, self.current_population)
            self.algorithm.tell(infills=self.current_population)

            self.current_population = self.algorithm.ask()
            self.observed_values = []
            self.current_individual = 0

        individual = self.current_population[self.current_individual]

        self.current_individual += 1
        config = {hp: individual.x[i] for i, hp in enumerate(self.hp_names)}
        return config


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from syne_tune.config_space import uniform, randint, choice

    # config_space = {"x0": uniform(0, 1), "x1": uniform(0, 1), "x2": randint(1, 100)}
    config_space = {"x0": uniform(0, 1), "x1": uniform(0, 1)}
    pop_size = 50
    method = NSGA2(
        config_space, metric=["f0", "f1"], mode=["min", "min"], pop_size=pop_size
    )
    f = plt.figure(dpi=200)
    color = 0
    for i in range(300):
        config = method.get_config()
        f0 = (0.5 - config["x0"]) ** 2
        f1 = (0.5 - config["x1"]) ** 2
        if i % pop_size == 0:
            x = [element.x[0] for element in method.current_population]
            y = [element.x[1] for element in method.current_population]
            plt.scatter(x, y, color=f"C{color}")
            color += 1
        method.on_trial_result(
            trial_id=i, config=config, result={"f0": f0, "f1": f1}, update=True
        )
        print(i, config, f0, f1)
    plt.show()
