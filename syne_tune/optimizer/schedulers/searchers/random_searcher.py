import logging
from typing import Optional, List, Dict, Any

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

logger = logging.getLogger(__name__)


class RandomSearcher(BaseSearcher):
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

    def suggest(self, **kwargs) -> Optional[dict]:
        """Sample a new configuration at random

        If ``allow_duplicates == False``, this is done without replacement, so
        previously returned configs are not suggested again.

        :return: New configuration, or None
        """
        new_config = self._next_initial_config()
        if new_config is None:
            new_config = {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in self.config_space.items()
            }
        return new_config