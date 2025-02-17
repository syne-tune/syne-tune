import logging
from collections import OrderedDict

from typing import Dict, Any, Optional, List, Union

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_cls_dict

logger = logging.getLogger(__name__)


class IndependentMultiFidelitySearcher(BaseSearcher):
    """
    Searcher for the multi-fidelity setting which fits independent models for each
    resource level as as proposed by Falkner et al.

    | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    | S. Falkner and A. Klein and F. Hutter
    | Proceedings of the 35th International Conference on Machine Learning

    # TODO: Update docstrings

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
        searcher_cls: Optional[Union[str, SingleObjectiveBaseSearcher]] = "kde",
        points_to_evaluate: Optional[List[dict]] = None,
        random_seed: Optional[int] = None,
        searcher_kwargs: dict[str, Any] = None,
    ):
        super().__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

        if searcher_kwargs is None:
            self.searcher_kwargs = dict()
        else:
            self.searcher_kwargs = searcher_kwargs

        if isinstance(searcher_cls, str):
            assert searcher_cls in searcher_cls_dict
            self.searcher_cls = searcher_cls_dict.get(searcher_cls)
        else:
            self.searcher_cls = searcher_cls
        self.models = OrderedDict()

    def initialize_model(self):
        return self.searcher_cls(
            config_space=self.config_space,
            random_seed=self.random_seed,
            **self.searcher_kwargs,
        )

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_points_to_evaluate` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """
        suggestion = self._next_points_to_evaluate()
        if suggestion is not None:
            return suggestion

        if len(self.models) == 0:
            return {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in self.config_space.items()
            }

        highest_observed_resource = next(reversed(self.models))
        return self.models[highest_observed_resource].suggest()

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
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
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
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
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """

        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )
