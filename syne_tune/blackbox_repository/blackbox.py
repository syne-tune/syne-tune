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
from numbers import Number

import pandas as pd
from typing import Optional, Callable, List, Tuple, Union, Dict, Any
import numpy as np


ObjectiveFunctionResult = Union[Dict[str, float], np.ndarray]


class Blackbox:
    """
    Interface designed to be compatible with

        | HPOBench
        | https://github.com/automl/HPOBench

    :param configuration_space: Configuration space of blackbox.
    :param fidelity_space: Fidelity space for blackbox, optional.
    :param objectives_names: Names of the metrics, by default consider all
        metrics prefixed by ``"metric_"`` to be metrics
    """

    def __init__(
        self,
        configuration_space: Dict[str, Any],
        fidelity_space: Optional[dict] = None,
        objectives_names: Optional[List[str]] = None,
    ):
        self.configuration_space = configuration_space
        self.fidelity_space = fidelity_space
        self.objectives_names = objectives_names

    def objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Union[dict, Number] = None,
        seed: Optional[int] = None,
    ) -> ObjectiveFunctionResult:
        """Returns an evaluation of the blackbox.

        First perform data check and then call :meth:`~_objective_function` that
        should be overriden in the child class.

        :param configuration: configuration to be evaluated, should belong to
            :attr:`configuration_space`
        :param fidelity: not passing a fidelity is possible if either the blackbox
            does not have a fidelity space or if it has a single fidelity in its
            fidelity space. In the latter case, all fidelities are returned in
            form of a tensor with shape ``(num_fidelities, num_objectives)``.
        :param seed: Only used if the blackbox defines multiple seeds
        :return: dictionary of objectives evaluated or tensor with shape
            ``(num_fidelities, num_objectives)`` if no fidelity was given.
        """
        self._check_keys(config=configuration, fidelity=fidelity)
        if self.fidelity_space is None:
            assert fidelity is None
        else:
            if fidelity is None:
                assert (
                    len(self.fidelity_space) == 1
                ), "not passing a fidelity is only supported when only one fidelity is present."

        if isinstance(fidelity, Number):
            # allows to call
            # ``objective_function(configuration=..., fidelity=2)``
            # instead of
            # ``objective_function(configuration=..., {'num_epochs': 2})``
            fidelity_names = list(self.fidelity_space.keys())
            assert (
                len(fidelity_names) == 1
            ), "passing numeric value is only possible when there is a single fidelity in the fidelity space."
            fidelity = {fidelity_names[0]: fidelity}

        # todo check configuration/fidelity matches their space
        return self._objective_function(
            configuration=configuration,
            fidelity=fidelity,
            seed=seed,
        )

    def _objective_function(
        self,
        configuration: Dict[str, Any],
        fidelity: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> ObjectiveFunctionResult:
        """Override this method to provide your benchmark function.

        :param configuration: configuration to be evaluated, should belong to
            :attr:`configuration_space`
        :param fidelity: not passing a fidelity is possible if either the blackbox
            does not have a fidelity space or if it has a single fidelity in its
            fidelity space. In the latter case, all fidelities are returned in
            form of a tensor with shape ``(num_fidelities, num_objectives)``.
        :param seed: Only used if the blackbox defines multiple seeds
        :return: dictionary of objectives evaluated or tensor with shape
            ``(num_fidelities, num_objectives)`` if no fidelity was given.
        """
        pass

    def __call__(self, *args, **kwargs) -> ObjectiveFunctionResult:
        return self.objective_function(*args, **kwargs)

    def _check_keys(self, config, fidelity):
        if isinstance(fidelity, dict):
            for key in fidelity.keys():
                assert key in self.fidelity_space.keys(), (
                    f'The key "{key}" passed as fidelity is not present in the fidelity space keys: '
                    f"{self.fidelity_space.keys()}"
                )
        if isinstance(config, dict):
            for key in config.keys():
                assert key in self.configuration_space.keys(), (
                    f'The key "{key}" passed in the configuration is not present in the configuration space keys: '
                    f"{self.configuration_space.keys()}"
                )

    def hyperparameter_objectives_values(
        self, predict_curves: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        If ``predict_curves`` is False, the shape of ``X`` is
        ``(num_evals * num_seeds * num_fidelities, num_hps + 1)``, the shape of ``y``
        is ``(num_evals * num_seeds * num_fidelities, num_objectives)``.
        This can be reshaped to ``(num_fidelities, num_seeds, num_evals, *)``.
        The final column of ``X`` is the fidelity value (only a single fidelity
        attribute is supported).

        If ``predict_curves`` is True, the shape of ``X`` is
        ``(num_evals * num_seeds, num_hps)``, the shape of ``y`` is
        ``(num_evals * num_seeds, num_fidelities * num_objectives)``. The latter
        can be reshaped to ``(num_seeds, num_evals, num_fidelities,
        num_objectives)``.

        :return: a tuple of two dataframes ``(X, y)``, where ``X`` contains
            hyperparameters values and ``y`` contains objective values, this is
            used when fitting a surrogate model.
        """
        pass

    @property
    def fidelity_values(self) -> Optional[np.array]:
        """
        :return: Fidelity values; or None if the blackbox has none
        """
        return None

    def fidelity_name(self) -> str:
        """
        Can only be used for blackboxes with a single fidelity attribute.

        :return: Name of fidelity attribute (must be single one)
        """
        assert len(self.fidelity_space) == 1, "Only supported for single fidelity"
        return next(iter(self.fidelity_space.keys()))

    def configuration_space_with_max_resource_attr(
        self, max_resource_attr: str
    ) -> Dict[str, Any]:
        """
        It is best practice to have one attribute in the configuration space to
        represent the maximum fidelity value used for evaluation (e.g., the
        maximum number of epochs).

        :param max_resource_attr: Name of new attribute for maximum resource
        :return: Configuration space augmented by the new attribute
        """
        assert len(self.fidelity_space) == 1, "Only supported for single fidelity"
        max_resource_value = int(max(self.fidelity_values))
        assert max_resource_attr not in self.configuration_space, (
            f"max_resource_attr = '{max_resource_attr}' must not be a key in "
            f"configuration_space ({list(self.configuration_space.keys())})"
        )
        return dict(
            self.configuration_space,
            **{max_resource_attr: max_resource_value},
        )


def from_function(
    configuration_space: Dict[str, Any],
    eval_fun: Callable,
    fidelity_space: Optional[dict] = None,
    objectives_names: Optional[List[str]] = None,
) -> Blackbox:
    """
    Helper to create a blackbox from a function, useful for test or to wrap-up
    real blackbox functions.

    :param configuration_space: Configuration space for blackbox
    :param eval_fun: Function that returns dictionary of objectives given
        configuration and fidelity
    :param fidelity_space: Fidelity space for blackbox
    :param objectives_names: Objectives returned by blackbox
    :return: Resulting blackbox wrapping ``eval_fun``
    """

    class BB(Blackbox):
        def __init__(self):
            super(BB, self).__init__(
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                objectives_names=objectives_names,
            )

        def objective_function(
            self,
            configuration: Dict[str, Any],
            fidelity: Optional[dict] = None,
            seed: Optional[int] = None,
        ) -> ObjectiveFunctionResult:
            return eval_fun(configuration, fidelity, seed)

    return BB()
