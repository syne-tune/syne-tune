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
from collections import OrderedDict
from itertools import product
from typing import Optional, List, Dict, Union, Any

import numpy as np

from syne_tune.config_space import (
    Float,
    Integer,
    Categorical,
    FiniteRange,
    Domain,
    config_space_size,
)
from syne_tune.optimizer.schedulers.searchers import (
    StochasticSearcher,
    StochasticAndFilterDuplicatesSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges

logger = logging.getLogger(__name__)


class RandomSearcher(StochasticAndFilterDuplicatesSearcher):
    """
    Searcher which randomly samples configurations to try next.

    Additional arguments on top of parent class :class:`StochasticAndFilterDuplicatesSearcher`:

    :param debug_log: If ``True``, debug log printing is activated.
        Logs which configs are chosen when, and which metric values are
        obtained. Defaults to ``False``
    :param resource_attr: Optional. Key in ``result`` passed to :meth:`_update`
        for resource value (for multi-fidelity schedulers)
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: Union[List[str], str],
        points_to_evaluate: Optional[List[dict]] = None,
        debug_log: Union[bool, DebugLogPrinter] = False,
        resource_attr: Optional[str] = None,
        allow_duplicates: Optional[bool] = None,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            restrict_configurations=restrict_configurations,
            **kwargs,
        )
        self._resource_attr = resource_attr
        # Debug log printing (switched off by default)
        if isinstance(debug_log, bool):
            if debug_log:
                self._debug_log = DebugLogPrinter()
            else:
                self._debug_log = None
        else:
            assert isinstance(
                debug_log, DebugLogPrinter
            ), f"debug_log = {debug_log} must either be bool or DebugLogPrinter"
            self._debug_log = debug_log

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.multi_fidelity import (
            MultiFidelitySchedulerMixin,
        )

        super().configure_scheduler(scheduler)
        # If the scheduler is multi-fidelity, we want to know the resource
        # attribute, this is used for ``debug_log``
        if isinstance(scheduler, MultiFidelitySchedulerMixin):
            self._resource_attr = scheduler.resource_attr

    def _get_config(self, **kwargs) -> Optional[dict]:
        """Sample a new configuration at random

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :param trial_id: Optional. Used for ``debug_log``
        :return: New configuration, or None
        """
        new_config = self._next_initial_config()
        if new_config is None:
            new_config = self._get_random_config()
        if new_config is not None:
            if self._debug_log is not None:
                trial_id = kwargs.get("trial_id")
                self._debug_log.start_get_config("random", trial_id=trial_id)
                self._debug_log.set_final_config(new_config)
                # All get_config debug log info is only written here
                self._debug_log.write_block()
        else:
            msg = (
                "Failed to sample a configuration not already chosen "
                + f"before. Exclusion list has size {len(self._excl_list)}."
            )
            cs_size = self._excl_list.configspace_size
            if cs_size is not None:
                msg += f" Configuration space has size {cs_size}."
            logger.warning(msg)
        return new_config

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        if self._debug_log is not None:
            if self._resource_attr is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(result[self._resource_attr])
                trial_id = trial_id + f":{resource}"
            msg = f"Update for trial_id {trial_id}: "
            if isinstance(self._metric, list):
                parts = [f"{name} = {result[name]:.3f}" for name in self._metric]
                msg += ",".join(parts)
            else:
                msg += f"{result[self._metric]:.3f}"
            logger.info(msg)

    def clone_from_state(self, state: Dict[str, Any]):
        new_searcher = RandomSearcher(
            self.config_space,
            metric=self._metric,
            points_to_evaluate=[],
            debug_log=self._debug_log,
            allow_duplicates=self._allow_duplicates,
        )
        new_searcher._resource_attr = self._resource_attr
        new_searcher._restore_from_state(state)
        return new_searcher

    @property
    def debug_log(self):
        return self._debug_log


DEFAULT_NSAMPLE = 5


class GridSearcher(StochasticSearcher):
    """Searcher that samples configurations from an equally spaced grid over config_space.

    It first evaluates configurations defined in points_to_evaluate and then
    continues with the remaining points from the grid.

    Additional arguments on top of parent class :class:`StochasticSearcher`.

    :param num_samples: Dictionary, optional. Number of samples per
        hyperparameter. This is required for hyperparameters of type float,
        optional for integer hyperparameters, and will be ignored for
        other types (categorical, scalar). If left unspecified, a default
        value of :const:`DEFAULT_NSAMPLE` will be used for float parameters, and
        the smallest of :const:`DEFAULT_NSAMPLE` and integer range will be used
        for integer parameters.
    :param shuffle_config: If ``True`` (default), the order of configurations
        suggested after those specified in ``points_to_evaluate`` is
        shuffled. Otherwise, the order will follow the Cartesian product
        of the configurations.
    :param allow_duplicates: If `True`, :meth:`get_config` may return the same
        configuration more than once. Defaults to `False`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        num_samples: Optional[Dict[str, int]] = None,
        shuffle_config: bool = True,
        allow_duplicates: bool = False,
        **kwargs,
    ):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        k = "restrict_configurations"
        if kwargs.get(k) is not None:
            logger.warning(f"{k} is not supported")
            del kwargs[k]
        self._validate_config_space(config_space, num_samples)
        self._hp_ranges = make_hyperparameter_ranges(config_space)

        if not isinstance(shuffle_config, bool):
            shuffle_config = True
        self._shuffle_config = shuffle_config
        self._generate_all_candidates_on_grid()
        self._next_index = 0
        self._allow_duplicates = allow_duplicates
        self._all_initial_configs = ExclusionList(self._hp_ranges)

    def _validate_config_space(
        self, config_space: Dict[str, Any], num_samples: Optional[Dict[str, int]]
    ):
        """
        Validates ``config_space`` from two aspects: first, that all
        hyperparameters are of acceptable types (i.e. float, integer,
        categorical). Second, ``num_samples`` is specified for float
        hyperparameters. ``num_samples`` for categorical variables are ignored
        as all of their values is used. ``num_samples`` for integer variables
        is optional, if specified it will be used and will be capped at their
        range length.

        :param config_space: Configuration space
        :param num_samples: Number of samples for each hyperparameter. Only
            required for float hyperparameters
        """
        if num_samples is None:
            num_samples = dict()
        self.num_samples = num_samples
        for hp, hp_range in config_space.items():
            # num_sample is required for float hp. If not specified default DEFAULT_NSAMPLE is used.
            if isinstance(hp_range, Float):
                if hp not in self.num_samples:
                    self.num_samples[hp] = DEFAULT_NSAMPLE
                    logger.warning(
                        f"Number of samples is required for '{hp}'. By default, "
                        f"{DEFAULT_NSAMPLE} is set as number of samples"
                    )
            # num_sample for integer hp must be capped at length of the range when specified. Otherwise,
            # minimum of default value DEFAULT_NSAMPLE and the integer range is used.
            if isinstance(hp_range, Integer):
                if hp in self.num_samples:
                    if self.num_samples[hp] > len(hp_range):
                        self.num_samples[hp] = min(len(hp_range), DEFAULT_NSAMPLE)
                        logger.info(
                            f"Number of samples for '{hp}' is larger than its "
                            "range. We set it to the minimum of the default "
                            f"number of samples (i.e. {DEFAULT_NSAMPLE}) and "
                            f"its range length (i.e. {len(hp_range)})."
                        )
                else:
                    self.num_samples[hp] = min(len(hp_range), DEFAULT_NSAMPLE)
            # num_samples is ignored for categorical hps.
            if isinstance(hp_range, Categorical) or isinstance(hp_range, FiniteRange):
                if hp in self.num_samples:
                    logger.info(
                        'number of samples for categorical variable "{}" is ignored.'.format(
                            hp
                        )
                    )

    def _generate_all_candidates_on_grid(self):
        """
        Generates all configurations to be evaluated by placing a regular,
        equally spaced grid over the configuration space.
        """
        hp_keys = []
        hp_values = []
        # adding categorical, finiteRange, scalar parameters
        for hp, hp_range in reversed(list(self.config_space.items())):
            if isinstance(hp_range, Float) or isinstance(hp_range, Integer):
                continue
            if isinstance(hp_range, Categorical):
                hp_keys.append(hp)
                values = list(OrderedDict.fromkeys(hp_range.categories))
                hp_values.append(values)
            elif isinstance(hp_range, FiniteRange):
                hp_keys.append(hp)
                hp_values.append(hp_range.values)
            elif not isinstance(hp_range, Domain):
                hp_keys.append(hp)
                hp_values.append([hp_range])

        # adding float, integer parameters
        for hpr in self._hp_ranges._hp_ranges:
            if hpr.name not in hp_keys:
                _hpr_nsamples = self.num_samples[hpr.name]
                _normalized_points = [
                    (idx + 0.5) / _hpr_nsamples for idx in range(_hpr_nsamples)
                ]
                _hpr_points = [
                    hpr.from_ndarray(np.array([point])) for point in _normalized_points
                ]
                _hpr_points = list(set(_hpr_points))
                hp_keys.append(hpr.name)
                hp_values.append(_hpr_points)

        self.hp_keys = hp_keys
        self.hp_values_combinations = list(product(*hp_values))

        if self._shuffle_config:
            self.random_state.shuffle(self.hp_values_combinations)

    def get_config(self, **kwargs) -> Optional[dict]:
        """Select the next configuration from the grid.

        This is done without replacement, so previously returned configs are
        not suggested again.

        :return: A new configuration that is valid, or None if no new config
            can be suggested. The returned configuration is a dictionary that
            maps hyperparameters to its values.
        """

        new_config = self._next_initial_config()
        if new_config is None:
            new_config = self._next_candidate_on_grid()
        else:
            self._all_initial_configs.add(new_config)
        if new_config is None:
            msg = "All the configurations have already been evaluated."
            cs_size = config_space_size(self.config_space)
            if cs_size is not None:
                msg += f" Configuration space has size {cs_size}"
            logger.warning(msg)
        return new_config

    def _next_candidate_on_grid(self) -> Optional[dict]:
        """
        :return: Next configuration from the set of grid candidates
            or None if no candidate is left.
        """

        num_combinations = len(self.hp_values_combinations)
        candidate = None
        while candidate is None and self._next_index < num_combinations:
            candidate = dict(
                zip(self.hp_keys, self.hp_values_combinations[self._next_index])
            )
            self._next_index += 1
            if self._all_initial_configs.contains(candidate):
                candidate = None
            if self._allow_duplicates and self._next_index == num_combinations:
                # Another round through the grid. It is important to reset
                # ``_all_initial_configs`` to empty, so the initial configs can be
                # suggested again in the second round
                self._next_index = 0
                self._all_initial_configs = ExclusionList(self._hp_ranges)
        return candidate

    def get_state(self) -> Dict[str, Any]:
        state = dict(
            super().get_state(),
            next_index=self._next_index,
            all_initial_configs=self._all_initial_configs.get_state(),
        )
        return state

    def clone_from_state(self, state: Dict[str, Any]):
        new_searcher = GridSearcher(
            config_space=self.config_space,
            num_samples=self.num_samples,
            metric=self._metric,
            shuffle_config=self._shuffle_config,
        )
        new_searcher._restore_from_state(state)
        return new_searcher

    def _restore_from_state(self, state: Dict[str, Any]):
        super()._restore_from_state(state)
        self._next_index = state["next_index"]
        self._all_initial_configs = ExclusionList(self._hp_ranges)
        self._all_initial_configs.clone_from_state(state["all_initial_configs"])

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        pass
