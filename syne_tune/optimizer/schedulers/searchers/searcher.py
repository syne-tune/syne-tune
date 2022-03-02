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
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

from syne_tune.config_space import Domain, is_log_space, Categorical
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log \
    import DebugLogPrinter
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import ExclusionList

__all__ = ['BaseSearcher',
           'SearcherWithRandomSeed',
           'RandomSearcher',
           'impute_points_to_evaluate',
           'extract_random_seed']

logger = logging.getLogger(__name__)


def _impute_default_config(default_config, config_space):
    new_config = dict()
    for name, hp_range in config_space.items():
        if isinstance(hp_range, Domain):
            if name not in default_config:
                if isinstance(hp_range, Categorical):
                    # For categorical: Pick first entry
                    new_config[name] = hp_range.categories[0]
                else:
                    lower, upper = float(hp_range.lower), float(hp_range.upper)
                    if not is_log_space(hp_range):
                        midpoint = 0.5 * (upper + lower)
                    else:
                        midpoint = np.exp(0.5 * (
                                np.log(upper) + np.log(lower)))
                    # Casting may involve rounding to nearest value in
                    # a finite range
                    midpoint = hp_range.cast(midpoint)
                    midpoint = np.clip(
                        midpoint, hp_range.lower, hp_range.upper)
                    new_config[name] = midpoint
            else:
                # Check validity
                # Note: For `FiniteRange`, the value is mapped to
                # the closest one in the range
                val = hp_range.cast(default_config[name])
                if isinstance(hp_range, Categorical):
                    assert val in hp_range.categories, \
                        f"default_config[{name}] = {val} is not in " +\
                        f"categories = {hp_range.categories}"
                else:
                    assert hp_range.lower <= val <= hp_range.upper, \
                        f"default_config[{name}] = {val} is not in " +\
                        f"[{hp_range.lower}, {hp_range.upper}]"
                new_config[name] = val
    return new_config


def _to_tuple(config: Dict, keys: List) -> Tuple:
    return tuple(config[k] for k in keys)


def _sorted_keys(config_space: Dict) -> List[str]:
    return sorted(k for k, v in config_space.items() if isinstance(v, Domain))


def impute_points_to_evaluate(
        points_to_evaluate: Optional[List[Dict]],
        config_space: Dict) -> List[Dict]:
    """
    Transforms `points_to_evaluate` argument to `BaseSearcher`. Each config in
    the list can be partially specified, or even be an empty dict. For each
    hyperparameter not specified, the default value is determined using a
    midpoint heuristic. Also, duplicate entries are filtered out.
    If None (default), this is mapped to [dict()], a single default config
    determined by the midpoint heuristic. If [] (empty list), no initial
    configurations are specified.

    :param points_to_evaluate:
    :param config_space:
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


class BaseSearcher(ABC):
    """Base Searcher (virtual class to inherit from if you are creating a custom Searcher).

    Parameters
    ----------
    config_space : Dict
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    metric : str
        Name of metric passed to update.
    points_to_evaluate : List[Dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
    """
    def __init__(
            self, config_space, metric, points_to_evaluate=None):
        self.config_space = config_space
        assert metric is not None, "Argument 'metric' is required"
        self._metric = metric
        self._points_to_evaluate = impute_points_to_evaluate(
            points_to_evaluate, config_space)

    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        The implementation here sets _metric for schedulers which specify it.

        Args:
            scheduler: TaskScheduler
                Scheduler the searcher is used with.

        """
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        if isinstance(scheduler, FIFOScheduler):
            self._metric = scheduler.metric

    def _next_initial_config(self) -> Optional[Dict]:
        if self._points_to_evaluate:
            return self._points_to_evaluate.pop(0)
        else:
            return None  # No more initial configs

    @abstractmethod
    def get_config(self, **kwargs):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new
        configuration.

        Note: Query `_next_initial_config` for initial configs to return first.

        Args:
        kwargs:
            Extra information may be passed from scheduler to searcher
        returns: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """
        pass

    def on_trial_result(
            self, trial_id: str, config: Dict, result: Dict, update: bool):
        """Inform searcher about result

        The scheduler passes every result. If `update` is True, the searcher
        should update its surrogate model (if any), otherwise `result` is an
        intermediate result not modelled.

        The default implementation calls self._update if `update` is True. It
        can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id:
        :param config:
        :param result:
        :param update: Should surrogate model be updated?
        """
        if update:
            self._update(trial_id, config, result)

    @abstractmethod
    def _update(self, trial_id: str, config: Dict, result: Dict):
        """Update surrogate model with result

        :param trial_id:
        :param config:
        :param result:
        """
        pass

    def register_pending(
            self, trial_id: str, config: Optional[Dict] = None,
            milestone=None):
        """
        Signals to searcher that evaluation for trial has started, but not
        yet finished, which allows model-based searchers to register this
        evaluation as pending.
        For multi-fidelity schedulers, milestone is the next milestone the
        evaluation will attend, so that model registers (config, milestone)
        as pending.
        The configuration for the trial has to be passed in `config` for a
        new trial, which the searcher has not seen before. If the trial is
        already registered with th searcher, `config` is ignored.
        """
        pass

    def remove_case(self, trial_id: str, **kwargs):
        """Remove data case previously appended by update

        For searchers which maintain the dataset of all cases (reports) passed
        to update, this method allows to remove one case from the dataset.
        """
        pass

    def evaluation_failed(self, trial_id: str):
        """
        Called by scheduler if an evaluation job for a trial failed. The
        searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).
        """
        pass

    def cleanup_pending(self, trial_id: str):
        """
        Removes all pending candidates whose configuration is equal to
        `config`.
        This should be called after an evaluation terminates. For various
        reasons (e.g., termination due to convergence), pending candidates
        for this evaluation may still be present.

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
        Together with clone_from_state, this is needed in order to store and
        re-create the mutable state of the searcher.

        The state returned here must be pickle-able.

        :return: Pickle-able mutable state of searcher
        """
        return {'points_to_evaluate': self._points_to_evaluate}

    @abstractmethod
    def clone_from_state(self, state: dict):
        """
        Together with get_state, this is needed in order to store and
        re-create the mutable state of the searcher.

        Given state as returned by get_state, this method combines the
        non-pickle-able part of the immutable state from self with state
        and returns the corresponding searcher clone. Afterwards, self is
        not used anymore.

        :param state: See above
        :return: New searcher object
        """
        pass

    def _restore_from_state(self, state: dict):
        self._points_to_evaluate = state['points_to_evaluate'].copy()

    @property
    def debug_log(self):
        """
        Some BaseSearcher subclasses support writing a debug log, using
        DebugLogPrinter. See RandomSearcher for an example.

        :return: DebugLogPrinter; or None (not supported)
        """
        return None


def extract_random_seed(kwargs: dict) -> (int, dict):
    key = 'random_seed_generator'
    if kwargs.get(key) is not None:
        random_seed = kwargs[key]()
    else:
        key = 'random_seed'
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
    `random_state` member, which must be used for all random draws.

    Making proper use of this interface allows us to run experiments
    with control of random seeds, e.g. for paired comparisons or
    integration testing.

    Extra parameters
    ----------------
    random_seed_generator : RandomSeedGenerator (optional)
        If given, the random_seed for `random_state` is obtained from there,
        otherwise `random_seed` is used
    random_seed : int (optional)
        This is used if `random_seed_generator` is not given.

    """
    def __init__(
            self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(
            config_space, metric=metric, points_to_evaluate=points_to_evaluate)
        random_seed, _ = extract_random_seed(kwargs)
        self.random_state = np.random.RandomState(random_seed)

    def get_state(self) -> dict:
        state = dict(
            super().get_state(),
            random_state=self.random_state.get_state())
        return state

    def _restore_from_state(self, state: dict):
        super()._restore_from_state(state)
        self.random_state.set_state(state['random_state'])


class RandomSearcher(SearcherWithRandomSeed):
    """Searcher which randomly samples configurations to try next.

    Extra parameters
    ----------------
    debug_log : bool
        If True (default), debug log printing is activated. Logs which
        configs are chosen when, and which metric values are obtained.

    """
    MAX_RETRIES = 100

    def __init__(self, config_space, metric, points_to_evaluate=None, **kwargs):
        super().__init__(
            config_space, metric, points_to_evaluate, **kwargs)
        self._hp_ranges = make_hyperparameter_ranges(config_space)
        self._resource_attr = kwargs.get('resource_attr')
        self._excl_list = ExclusionList.empty_list(self._hp_ranges)
        # Debug log printing (switched on by default)
        debug_log = kwargs.get('debug_log', True)
        if isinstance(debug_log, bool):
            if debug_log:
                self._debug_log = DebugLogPrinter()
            else:
                self._debug_log = None
        else:
            assert isinstance(debug_log, DebugLogPrinter)
            self._debug_log = debug_log

    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        The implementation here sets _metric for schedulers which
        specify it.

        Args:
            scheduler: TaskScheduler
                Scheduler the searcher is used with.

        """
        from syne_tune.optimizer.schedulers.hyperband import \
            HyperbandScheduler

        super().configure_scheduler(scheduler)
        # If the scheduler is Hyperband, we want to know the resource
        # attribute, this is used for debug_log
        if isinstance(scheduler, HyperbandScheduler):
            self._resource_attr = scheduler._resource_attr

    def get_config(self, **kwargs):
        """Sample a new configuration at random
        This is done without replacement, so previously returned configs are
        not suggested again.

        Returns
        -------
        A new configuration that is valid, or None if no new config can be
        suggested.
        """
        trial_id = kwargs.get('trial_id')
        if self._debug_log is not None:
            self._debug_log.start_get_config('random', trial_id=trial_id)
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
                self._debug_log.set_final_config(new_config)
                # All get_config debug log info is only written here
                self._debug_log.write_block()
        else:
            msg = "Failed to sample a configuration not already chosen " + \
                  f"before. Exclusion list has size {len(self._excl_list)}."
            cs_size = self._excl_list.configspace_size
            if cs_size is not None:
                msg += f" Configuration space has size {cs_size}."
            logger.warning(msg)
        return new_config

    def _update(self, trial_id: str, config: Dict, result: Dict):
        if self._debug_log is not None:
            metric_val = result[self._metric]
            if self._resource_attr is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(result[self._resource_attr])
                trial_id = trial_id + ':{}'.format(resource)
            msg = f"Update for trial_id {trial_id}: metric = {metric_val:.3f}"
            logger.info(msg)

    def clone_from_state(self, state: dict):
        new_searcher = RandomSearcher(
            self.config_space, metric=self._metric, debug_log=self._debug_log)
        new_searcher._resource_attr = self._resource_attr
        new_searcher._restore_from_state(state)
        return new_searcher

    @property
    def debug_log(self):
        return self._debug_log
