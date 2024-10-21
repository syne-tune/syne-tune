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
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


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
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
    ):
        self.config_space = config_space
        if points_to_evaluate is None:
            self.points_to_evaluate = []

    def _next_initial_config(self) -> Optional[Dict[str, Any]]:
        """
        :return: Next entry from remaining ``points_to_evaluate`` (popped
            from front), or None
        """
        if self.points_to_evaluate:
            return self.points_to_evaluate.pop(0)
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
        observation: float,
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
        :param observation: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param update: Should surrogate model be updated?
        """
        if update:
            self._update(trial_id, config, observation)

    def _update(self, trial_id: str, config: Dict[str, Any], observation: float):
        """Update surrogate model with result

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param observation: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
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
        return {"points_to_evaluate": self.points_to_evaluate}

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
