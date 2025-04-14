import logging
from typing import Optional, List, Dict, Any

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)

logger = logging.getLogger(__name__)


def sample_random_config(config_space):
    return {
        k: v.sample() if hasattr(v, "sample") else v for k, v in config_space.items()
    }


class RandomSearcher(SingleObjectiveBaseSearcher):
    """
    Sample hyperparameter configurations uniformly at random from the given configuration space.

    :param config_space: The configuration space to sample from.
    :param points_to_evaluate: A list of configurations to evaluate initially (in the given order).
    :param random_seed: Seed used to initialize the random number generators.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[dict]] = None,
        random_seed: int = None,
    ):
        super().__init__(
            config_space, points_to_evaluate=points_to_evaluate, random_seed=random_seed
        )

    def suggest(self) -> Optional[dict]:
        """Sample a new configuration at random

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :return: New configuration, or None
        """
        new_config = self._next_points_to_evaluate()
        if new_config is None:
            new_config = sample_random_config(self.config_space)
        return new_config


class MultiObjectiveRandomSearcher(BaseSearcher):
    """
    Searcher which randomly samples configurations to try next.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[dict]] = None,
        random_seed: int = None,
    ):
        super().__init__(
            config_space, points_to_evaluate=points_to_evaluate, random_seed=random_seed
        )

    def suggest(self) -> Optional[dict]:
        """Sample a new configuration at random

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :return: New configuration, or None
        """
        new_config = self._next_points_to_evaluate()
        if new_config is None:
            new_config = sample_random_config(self.config_space)
        return new_config
