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
from typing import Optional, List, Tuple, Dict, Any, Union

from syne_tune.config_space import (
    Domain,
    is_log_space,
    Categorical,
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

logger = logging.getLogger(__name__)


def _impute_default_config(
    default_config: Configuration, config_space: Dict[str, Any]
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


def _to_tuple(config: Dict[str, Any], keys: List) -> Tuple:
    return tuple(config[k] for k in keys)


def _sorted_keys(config_space: Dict[str, Any]) -> List[str]:
    return sorted(k for k, v in config_space.items() if isinstance(v, Domain))


def impute_points_to_evaluate(
    points_to_evaluate: Optional[List[Dict[str, Any]]], config_space: Dict[str, Any]
) -> List[Dict[str, Any]]:
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
    implementing :meth:`get_config`.

    .. note::
       This is an abstract base class. In order to implement a new searcher, try to
       start from
       :class:`~syne_tune.optimizer.scheduler.searcher.StochasticAndFilterDuplicatesSearcher`
       or :class:`~syne_tune.optimizer.scheduler.searcher.StochasticSearcher`,
       which implement generally useful properties.

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
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: Union[List[str], str],
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        mode: Union[List[str], str] = "min",
    ):
        self.config_space = config_space
        assert metric is not None, "Argument 'metric' is required"
        self._metric = metric
        self._points_to_evaluate = impute_points_to_evaluate(
            points_to_evaluate, config_space
        )
        self._mode = mode

    def configure_scheduler(self, scheduler):
        """
        Some searchers need to obtain information from the scheduler they are
        used with, in order to configure themselves.
        This method has to be called before the searcher can be used.

        :param scheduler: Scheduler the searcher is used with.
        :type scheduler: :class:`~syne_tune.optimizer.schedulers.TrialScheduler`
        """
        if hasattr(scheduler, "metric"):
            self._metric = getattr(scheduler, "metric")
        if hasattr(scheduler, "mode"):
            self._mode = getattr(scheduler, "mode")

    def _next_initial_config(self) -> Optional[Dict[str, Any]]:
        """
        :return: Next entry from remaining ``points_to_evaluate`` (popped
            from front), or None
        """
        if self._points_to_evaluate:
            return self._points_to_evaluate.pop(0)
        else:
            return None  # No more initial configs

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
        raise NotImplementedError

    def on_trial_result(
        self,
        trial_id: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        update: bool,
    ):
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

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        """Update surrogate model with result

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param result: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        raise NotImplementedError

    def register_pending(
        self,
        trial_id: str,
        config: Optional[Dict[str, Any]] = None,
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

    def get_state(self) -> Dict[str, Any]:
        """
        Together with :meth:`clone_from_state`, this is needed in order to
        store and re-create the mutable state of the searcher.
        The state returned here must be pickle-able.

        :return: Pickle-able mutable state of searcher
        """
        return {"points_to_evaluate": self._points_to_evaluate}

    def clone_from_state(self, state: Dict[str, Any]):
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

    def _restore_from_state(self, state: Dict[str, Any]):
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
