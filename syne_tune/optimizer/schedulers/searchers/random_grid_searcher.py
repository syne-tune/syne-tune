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
from typing import Optional, List, Dict

import numpy as np

from syne_tune.config_space import (
    Float,
    Integer,
    Categorical,
    FiniteRange,
    Domain,
    config_space_size,
)
from syne_tune.optimizer.schedulers.searchers.searcher import (
    SearcherWithRandomSeed,
    sample_random_configuration,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    ExclusionList,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges

logger = logging.getLogger(__name__)


class RandomSearcher(SearcherWithRandomSeed):
    """
    Searcher which randomly samples configurations to try next.

    Additional arguments on top of parent class :class:`SearcherWithRandomSeed`:

    :param debug_log: If ``True``, debug log printing is activated.
        Logs which configs are chosen when, and which metric values are
        obtained. Defaults to ``False``
    :type debug_log: bool, optional
    :param resource_attr: Optional. Key in ``result`` passed to :meth:`_update`
        for resource value (for multi-fidelity schedulers)
    :type resource_attr: str, optional
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        self._hp_ranges = make_hyperparameter_ranges(config_space)
        self._resource_attr = kwargs.get("resource_attr")
        # Used to avoid returning the same config more than once:
        self._excl_list = ExclusionList.empty_list(self._hp_ranges)
        # Debug log printing (switched off by default)
        debug_log = kwargs.get("debug_log", False)
        if isinstance(debug_log, bool):
            if debug_log:
                self._debug_log = DebugLogPrinter()
            else:
                self._debug_log = None
        else:
            assert isinstance(debug_log, DebugLogPrinter)
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

    def get_config(self, **kwargs) -> Optional[dict]:
        """Sample a new configuration at random

        This is done without replacement, so previously returned configs are
        not suggested again.

        :param trial_id: Optional. Used for ``debug_log``
        :return: New configuration, or None
        """
        new_config = self._next_initial_config()
        if new_config is None:
            new_config = sample_random_configuration(
                hp_ranges=self._hp_ranges,
                random_state=self.random_state,
                exclusion_list=self._excl_list,
            )
        if new_config is not None:
            self._excl_list.add(new_config)  # Should not be suggested again
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

    def _update(self, trial_id: str, config: dict, result: dict):
        if self._debug_log is not None:
            metric_val = result[self._metric]
            if self._resource_attr is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(result[self._resource_attr])
                trial_id = trial_id + f":{resource}"
            msg = f"Update for trial_id {trial_id}: metric = {metric_val:.3f}"
            logger.info(msg)

    def clone_from_state(self, state: dict):
        new_searcher = RandomSearcher(
            self.config_space,
            metric=self._metric,
            points_to_evaluate=[],
            debug_log=self._debug_log,
        )
        new_searcher._resource_attr = self._resource_attr
        new_searcher._restore_from_state(state)
        return new_searcher

    @property
    def debug_log(self):
        return self._debug_log


DEFAULT_NSAMPLE = 5


class GridSearcher(SearcherWithRandomSeed):
    """Searcher that samples configurations from an equally spaced grid over config_space.

    It first evaluates configurations defined in points_to_evaluate and then
    continues with the remaining points from the grid.

    Additional arguments on top of parent class :class:`SearcherWithRandomSeed`.

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
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        num_samples: Optional[Dict[str, int]] = None,
        shuffle_config: bool = True,
        **kwargs,
    ):
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, **kwargs
        )
        self._validate_config_space(config_space, num_samples)
        self._hp_ranges = make_hyperparameter_ranges(config_space)

        if not isinstance(shuffle_config, bool):
            shuffle_config = True
        self._shuffle_config = shuffle_config
        self._generate_all_candidates_on_grid()
        self._next_index = 0

    def _validate_config_space(self, config_space: dict, num_samples: dict):
        """
        Validates config_space from two aspects: first, that all hyperparameters
        are of acceptable types (i.e. float, integer, categorical). Second, num_samples is specified
        for float hyperparameters. Num_samples for categorical variables are ignored as all of their
        values is used in GridSearch. Num_samples for integer variables is optional, if specified it
        will be used and will be capped at their range length.
        Args:
            config_space: The configuration space that defines search space grid.
            num_samples: Number of samples for each hyperparameters. Only required for float hyperparameters.
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
                        "number of samples is required for {}. By default, {} is set as number of samples".format(
                            hp, DEFAULT_NSAMPLE
                        )
                    )
            # num_sample for integer hp must be capped at length of the range when specified. Otherwise,
            # minimum of default value DEFAULT_NSAMPLE and the interger range is used.
            if isinstance(hp_range, Integer):
                if hp in self.num_samples:
                    if self.num_samples[hp] > len(hp_range):
                        self.num_samples[hp] = min(len(hp_range), DEFAULT_NSAMPLE)
                        logger.info(
                            'number of samples for "{}" is larger than its range. We set it to the minimum of the default number of samples (i.e. {}) and its range length (i.e. {}).'.format(
                                hp, DEFAULT_NSAMPLE, len(hp_range)
                            )
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
            values = self._next_candidate_on_grid()
            if values is not None:
                new_config = dict(zip(self.hp_keys, values))
        if new_config is not None:
            # Write debug log for the config
            entries = ["{}: {}".format(k, v) for k, v in new_config.items()]
            msg = "\n".join(entries)
            trial_id = kwargs.get("trial_id")
            if trial_id is not None:
                msg = "get_config[grid] for trial_id {}\n".format(trial_id) + msg
            else:
                msg = "get_config[grid]: \n".format(trial_id) + msg
            logger.debug(msg)
        else:
            msg = "All the configurations has already been evaluated."
            cs_size = config_space_size(self.config_space)
            if cs_size is not None:
                msg += " Configuration space has size".format(cs_size)
            logger.warning(msg)
        return new_config

    def _next_candidate_on_grid(self) -> Optional[tuple]:
        """
        :return: Next configuration from the set of grid candidates
            or None if no candidate is left.
        """

        if self._next_index < len(self.hp_values_combinations):
            candidate = self.hp_values_combinations[self._next_index]
            self._next_index += 1
            return candidate
        else:
            # No more candidates
            return None

    def get_state(self) -> dict:
        state = dict(
            super().get_state(),
            _next_index=self._next_index,
        )
        return state

    def clone_from_state(self, state: dict):
        new_searcher = GridSearcher(
            config_space=self.config_space,
            num_samples=self.num_samples,
            metric=self._metric,
            shuffle_config=self._shuffle_config,
        )
        new_searcher._restore_from_state(state)
        return new_searcher

    def _restore_from_state(self, state: dict):
        super()._restore_from_state(state)
        self._next_index = state["_next_index"]

    def _update(self, trial_id: str, config: dict, result: dict):
        pass
