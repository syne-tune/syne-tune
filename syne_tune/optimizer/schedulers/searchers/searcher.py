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
from typing import Optional, List, Tuple, Dict
from collections import OrderedDict

from syne_tune.config_space import (
    Domain,
    config_space_size,
    is_log_space,
    Categorical,
    Float,
    Integer,
    FiniteRange,
    Ordinal,
    OrdinalNearestNeighbor,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Hyperparameter,
    Configuration,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    ExclusionList,
)
from itertools import product

logger = logging.getLogger(__name__)


DEFAULT_NSAMPLE = 5


def _impute_default_config(
    default_config: Configuration, config_space: dict
) -> Configuration:
    """Imputes missing values in ``default_config`` by mid-point rule

    For numerical types, the mid-point in the range is chosen (in normal
    or log). For :class:`~syne_tune.config_space.Categorical`, we pick the first
    entry of ``categories``.

    :param default_config: Configuration to be imputed
    :param config_space: Configuration space
    :return: Imputed configuration. If ``default_config`` has entries with
        values not being :class:`~syne_tune.config_space.Domain`, they are not
        included
    """
    new_config = dict()
    for name, hp_range in config_space.items():
        if isinstance(hp_range, Domain):
            if name in default_config:
                new_config[name] = _default_config_value(default_config, hp_range, name)
            else:
                new_config[name] = _non_default_config(hp_range)
    return new_config


def _default_config_value(
    default_config: Configuration, hp_range: Domain, name: str
) -> Hyperparameter:
    # Check validity
    # Note: For ``FiniteRange``, the value is mapped to
    # the closest one in the range
    val = hp_range.cast(default_config[name])
    if isinstance(hp_range, Categorical):
        assert val in hp_range.categories, (
            f"default_config[{name}] = {val} is not in "
            + f"categories = {hp_range.categories}"
        )
    else:
        assert hp_range.lower <= val <= hp_range.upper, (
            f"default_config[{name}] = {val} is not in "
            + f"[{hp_range.lower}, {hp_range.upper}]"
        )
    return val


def _non_default_config(hp_range: Domain) -> Hyperparameter:
    if isinstance(hp_range, Categorical):
        if not isinstance(hp_range, Ordinal):
            # For categorical: Pick first entry
            return hp_range.categories[0]
        if not isinstance(hp_range, OrdinalNearestNeighbor):
            # For non-NN ordinal: Pick middle entry
            num_cats = len(hp_range)
            return hp_range.categories[num_cats // 2]
        # Nearest neighbour ordinal: Treat as numerical
        lower = float(hp_range.categories[0])
        upper = float(hp_range.categories[-1])
    else:
        lower = float(hp_range.lower)
        upper = float(hp_range.upper)
    # Mid-point: Arithmetic or geometric
    if not is_log_space(hp_range):
        midpoint = 0.5 * (upper + lower)
    else:
        midpoint = np.exp(0.5 * (np.log(upper) + np.log(lower)))
    # Casting may involve rounding to nearest value in
    # a finite range
    midpoint = hp_range.cast(midpoint)
    lower = hp_range.value_type(lower)
    upper = hp_range.value_type(upper)
    midpoint = np.clip(midpoint, lower, upper)
    return midpoint


def _to_tuple(config: dict, keys: List) -> Tuple:
    return tuple(config[k] for k in keys)


def _sorted_keys(config_space: dict) -> List[str]:
    return sorted(k for k, v in config_space.items() if isinstance(v, Domain))


def impute_points_to_evaluate(
    points_to_evaluate: Optional[List[dict]], config_space: dict
) -> List[dict]:
    """
    Transforms ``points_to_evaluate`` argument to
    :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`. Each
    config in the list can be partially specified, or even be an empty dict.
    For each hyperparameter not specified, the default value is determined
    using a midpoint heuristic. Also, duplicate entries are filtered out.
    If None (default), this is mapped to ``[dict()]``, a single default config
    determined by the midpoint heuristic. If ``[]`` (empty list), no initial
    configurations are specified.

    :param points_to_evaluate: Argument to
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param config_space: Configuration space
    :return: List of fully specified initial configs
    """
    if points_to_evaluate is None:
        points_to_evaluate = [dict()]
    # Impute and filter out duplicates
    result = []
    excl_set = set()
    keys = _sorted_keys(config_space)
    for point in points_to_evaluate:
        config = _impute_default_config(point, config_space)
        config_tpl = _to_tuple(config, keys)
        if config_tpl not in excl_set:
            result.append(config)
            excl_set.add(config_tpl)
    return result


class BaseSearcher:
    """
    Base class of searchers, which are components of schedulers responsible for
    implementing :meth:`~get_config`.

    :param config_space: Configuration space
    :param metric: Name of metric passed to :meth:`~update`. Can be obtained from
        scheduler in :meth:`~configure_scheduler`
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        self.config_space = config_space
        assert metric is not None, "Argument 'metric' is required"
        self._metric = metric
        self._points_to_evaluate = impute_points_to_evaluate(
            points_to_evaluate, config_space
        )

    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        :param scheduler: Scheduler the searcher is used with.
        :type scheduler: :class:`~syne_tune.optimizer.schedulers.TrialScheduler`
        """
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        if isinstance(scheduler, FIFOScheduler):
            self._metric = scheduler.metric

    def _next_initial_config(self) -> Optional[dict]:
        """
        :return: Next entry from remaining ``points_to_evaluate`` (popped
            from front), or None
        """
        if self._points_to_evaluate:
            return self._points_to_evaluate.pop(0)
        else:
            return None  # No more initial configs

    def get_config(self, **kwargs) -> Optional[dict]:
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
        raise NotImplementedError

    def on_trial_result(self, trial_id: str, config: dict, result: dict, update: bool):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param result: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param update: Should surrogate model be updated?
        """
        if update:
            self._update(trial_id, config, result)

    def _update(self, trial_id: str, config: dict, result: dict):
        """Update surrogate model with result

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param result: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        raise NotImplementedError

    def register_pending(
        self,
        trial_id: str,
        config: Optional[dict] = None,
        milestone: Optional[int] = None,
    ):
        """
        Signals to searcher that evaluation for trial has started, but not yet
        finished, which allows model-based searchers to register this evaluation
        as pending.

        :param trial_id: ID of trial to be registered as pending evaluation
        :param config: If ``trial_id`` has not been registered with the
            searcher, its configuration must be passed here. Ignored
            otherwise.
        :param milestone: For multi-fidelity schedulers, this is the next
            rung level the evaluation will attend, so that model registers
            ``(config, milestone)`` as pending.
        """
        pass

    def remove_case(self, trial_id: str, **kwargs):
        """Remove data case previously appended by :meth:`_update`

        For searchers which maintain the dataset of all cases (reports) passed
        to update, this method allows to remove one case from the dataset.

        :param trial_id: ID of trial whose data is to be removed
        :param kwargs: Extra arguments, optional
        """
        pass

    def evaluation_failed(self, trial_id: str):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        pass

    def cleanup_pending(self, trial_id: str):
        """Removes all pending evaluations for trial ``trial_id``.

        This should be called after an evaluation terminates. For various
        reasons (e.g., termination due to convergence), pending candidates
        for this evaluation may still be present.

        :param trial_id: ID of trial whose pending evaluations should be cleared
        """
        pass

    def dataset_size(self):
        """
        :return: Size of dataset a model is fitted to, or 0 if no model is
            fitted to data
        """
        return 0

    def model_parameters(self):
        """
        :return: Dictionary with current model (hyper)parameter values if
            this is supported; otherwise empty
        """
        return dict()

    def get_state(self) -> dict:
        """
        Together with :meth:`clone_from_state`, this is needed in order to
        store and re-create the mutable state of the searcher.
        The state returned here must be pickle-able.

        :return: Pickle-able mutable state of searcher
        """
        return {"points_to_evaluate": self._points_to_evaluate}

    def clone_from_state(self, state: dict):
        """
        Together with :meth:`get_state`, this is needed in order to store and
        re-create the mutable state of the searcher.

        Given state as returned by :meth:`get_state`, this method combines the
        non-pickle-able part of the immutable state from self with state
        and returns the corresponding searcher clone. Afterwards, ``self`` is
        not used anymore.

        :param state: See above
        :return: New searcher object
        """
        raise NotImplementedError

    def _restore_from_state(self, state: dict):
        self._points_to_evaluate = state["points_to_evaluate"].copy()

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        """
        Some subclasses support writing a debug log, using
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log.DebugLogPrinter`.
        See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
        for an example.

        :return: ``debug_log`` object`` or None (not supported)
        """
        return None


def extract_random_seed(**kwargs) -> (int, dict):
    key = "random_seed_generator"
    if kwargs.get(key) is not None:
        random_seed = kwargs[key]()
    else:
        key = "random_seed"
        if kwargs.get(key) is not None:
            random_seed = kwargs[key]
        else:
            random_seed = 31415927
            key = None
    _kwargs = {k: v for k, v in kwargs.items() if k != key}
    return random_seed, _kwargs


class SearcherWithRandomSeed(BaseSearcher):
    """
    Base class of searchers which use random decisions. Creates the
    ``random_state`` member, which must be used for all random draws.

    Making proper use of this interface allows us to run experiments with
    control of random seeds, e.g. for paired comparisons or integration testing.

    Additional arguments on top of parent class :class:`BaseSearcher`.

    :param random_seed_generator: If given, random seed is drawn from there
    :type random_seed_generator: :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`, optional
    :param random_seed: Used if ``random_seed_generator`` is not given.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate
        )
        random_seed, _ = extract_random_seed(**kwargs)
        self.random_state = np.random.RandomState(random_seed)

    def get_state(self) -> dict:
        state = dict(super().get_state(), random_state=self.random_state.get_state())
        return state

    def _restore_from_state(self, state: dict):
        super()._restore_from_state(state)
        self.random_state.set_state(state["random_state"])


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

    MAX_RETRIES = 100

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
        from syne_tune.optimizer.schedulers import HyperbandScheduler
        from syne_tune.optimizer.schedulers.synchronous import (
            SynchronousHyperbandScheduler,
        )

        super().configure_scheduler(scheduler)
        # If the scheduler is Hyperband, we want to know the resource
        # attribute, this is used for ``debug_log``
        if isinstance(scheduler, (HyperbandScheduler, SynchronousHyperbandScheduler)):
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
            if not self._excl_list.config_space_exhausted():
                for _ in range(self.MAX_RETRIES):
                    _config = self._hp_ranges.random_config(self.random_state)
                    if not self._excl_list.contains(_config):
                        new_config = _config
                        break

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
