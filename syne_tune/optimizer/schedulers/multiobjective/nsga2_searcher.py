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

from typing import Optional, List, Union, Dict, Any

from syne_tune.optimizer.schedulers.searchers import StochasticSearcher
from syne_tune.config_space import Domain, Float, Integer, Categorical, FiniteRange
from syne_tune.optimizer.schedulers.random_seeds import generate_random_seed
from syne_tune.try_import import try_import_moo_message

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.evaluator import Evaluator
    from pymoo.problems.static import StaticProblem
    from pymoo.core.mixed import (
        MixedVariableMating,
        MixedVariableSampling,
        MixedVariableDuplicateElimination,
    )
    from pymoo.core.variable import Real as PyMOOReal
    from pymoo.core.variable import Choice as PyMOOChoice
    from pymoo.core.variable import Integer as PyMOOInteger
except ImportError:
    print(try_import_moo_message())


def _create_multiobjective_problem(config_space: Dict[str, Any], n_obj: int, **kwargs):
    # This needs to be an inner class, since ``Problem`` can only be imported
    # with ``moo`` dependencies. We want this module to be importable even if
    # ``moo`` dependencies are not present: only creating a ``NSGA2Searcher``
    # object should fail in this case.
    class _MultiObjectiveMixedVariableProblem(Problem):
        def __init__(self, n_obj: int, config_space: Dict[str, Any], **kwargs):
            vars = {}

            for hp_name, hp in config_space.items():
                if isinstance(hp, Domain):
                    if isinstance(hp, Categorical):
                        vars[hp_name] = PyMOOChoice(options=hp.categories)
                    elif isinstance(hp, Integer):
                        vars[hp_name] = PyMOOInteger(bounds=(hp.lower, hp.upper - 1))
                    elif isinstance(hp, FiniteRange):
                        vars[hp_name] = PyMOOInteger(bounds=(0, hp.size - 1))
                    elif isinstance(hp, Float):
                        vars[hp_name] = PyMOOReal(bounds=(hp.lower, hp.upper))
                    else:
                        raise Exception(
                            f"Type {type(hp)} of hyperparameter {hp_name} is not supported!"
                        )

            super().__init__(vars=vars, n_obj=n_obj, n_ieq_constr=0, **kwargs)

    return _MultiObjectiveMixedVariableProblem(
        n_obj=n_obj, config_space=config_space, **kwargs
    )


# TODO:
# - ``points_to_evaluate`` should be used, to initialize the population
# - Do we want the first K configs to be selected at random, so the
#   behavior is the same to other searchers?
class NSGA2Searcher(StochasticSearcher):
    """
    This is a wrapper around the NSGA-2 [1] implementation of pymoo [2].

        | [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan.
        | A fast and elitist multiobjective genetic algorithm: nsga-II.
        | Trans. Evol. Comp, 6(2):182–197, April 2002.

        | [2] J. Blank and K. Deb
        | pymoo: Multi-Objective Optimization in Python
        | IEEE Access, 2020

    :param config_space: Configuration space
    :param metric: Name of metric passed to :meth:`~update`. Can be obtained from
        scheduler in :meth:`~configure_scheduler`. In the case of multi-objective optimization,
         metric is a list of strings specifying all objectives to be optimized.
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :param mode: Should metric be minimized ("min", default) or maximized
        ("max"). In the case of multi-objective optimization, mode can be a list defining for
        each metric if it is minimized or maximized
    :param population_size: Size of the population
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        points_to_evaluate: Optional[List[dict]] = None,
        population_size: int = 20,
        **kwargs,
    ):
        super(NSGA2Searcher, self).__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        if isinstance(mode, str):
            self._mode = [mode] * len(metric)
        else:
            self._mode = mode

        self.hp_names = []
        for hp_name, hp in config_space.items():
            if isinstance(hp, Domain):
                self.hp_names.append(hp_name)
                assert type(hp) in [
                    Categorical,
                    Integer,
                    Float,
                    FiniteRange,
                ], f"Type {type(hp)} for hyperparameter {hp_name} is not supported."

        self.problem = _create_multiobjective_problem(
            config_space=config_space, n_obj=len(metric), n_var=len(self.hp_names)
        )
        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            seed=generate_random_seed(self.random_state),
        )
        self.algorithm.setup(
            problem=self.problem, termination=("n_eval", 2**32 - 1), verbose=False
        )

        self.current_population = self.algorithm.ask()
        self.current_individual = 0
        self.observed_values = dict()

    def _update(self, trial_id: str, config: dict, result: dict):
        observed_metrics = list()
        for mode, metric in zip(self._mode, self._metric):
            value = result[metric]
            if mode == "max":
                value *= -1
            observed_metrics.append(value)

        self.observed_values[trial_id] = observed_metrics

        if len(self.observed_values.keys()) == len(self.current_population):
            func_values = np.array(list(self.observed_values.values()))
            static = StaticProblem(self.problem, F=func_values)
            Evaluator().eval(static, self.current_population)
            self.algorithm.tell(infills=self.current_population)

            self.current_population = self.algorithm.ask()
            self.observed_values = dict()
            self.current_individual = 0

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
            raise Exception(
                "It seems that some configurations are sill pending, while querying a new configuration. "
                "Note that NSGA-2 does not support asynchronous scheduling. To avoid this behaviour, "
                "make sure to set num_workers = 1."
            )
        else:
            individual = self.current_population[self.current_individual]

        self.current_individual += 1
        config = {}
        for hp_name, hp in self.config_space.items():
            if isinstance(hp, Domain):
                if isinstance(hp, FiniteRange):
                    config[hp_name] = hp.values[individual.x[hp_name]]
                else:
                    config[hp_name] = individual.x[hp_name]
        return config
