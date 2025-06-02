import logging
from collections import defaultdict

from typing import Dict, Any, Optional, List, Union

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_dict

logger = logging.getLogger(__name__)


class IndependentMultiFidelitySearcher(BaseSearcher):
    """
    Searcher for the multi-fidelity setting which fits independent models for each
    resource level as proposed by Falkner et al.

    | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    | S. Falkner and A. Klein and F. Hutter
    | Proceedings of the 35th International Conference on Machine Learning

    :param config_space: Configuration space
    :param searcher: Searcher to sample configurations on each rung level.
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order).
    :param random_seed: Seed used to initialize the random number generators.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        searcher: Optional[Union[str, SingleObjectiveBaseSearcher]] = "kde",
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

        if isinstance(searcher, str):
            assert searcher in searcher_dict
            self.searcher = searcher_dict.get(searcher)
        else:
            self.searcher = searcher

        self.searchers = defaultdict(
            lambda: self.searcher(
                config_space=self.config_space,
                random_seed=self.random_seed,
                **self.searcher_kwargs,
            )
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

        # in case we have not seen observations, we have no searcher and sample at random
        if len(self.searchers) == 0:
            return {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in self.config_space.items()
            }

        highest_observed_resource = next(reversed(self.searchers))
        return self.searchers[highest_observed_resource].suggest()

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
    ):
        """
        Updates the model with the latest results of a trial at a specific resource level.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param resource_level: Resource level where the metric was observed from.
        """
        self.searchers[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
    ):
        """
        Updates the model with the final results of a completed trial at a specific resource level.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param resource_level: Resource level where the metric was observed from.
        """
        self.searchers[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )
