from typing import Dict, Optional, List
import logging

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_impl import (
    HyperparameterRangesImpl,
)

logger = logging.getLogger(__name__)


def make_hyperparameter_ranges(
    config_space: Dict,
    name_last_pos: Optional[str] = None,
    value_for_last_pos=None,
    active_config_space: Optional[Dict] = None,
    prefix_keys: Optional[List[str]] = None,
) -> HyperparameterRanges:
    """Default method to create :class:`HyperparameterRanges` from ``config_space``

    :param config_space: Configuration space
    :param name_last_pos: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param value_for_last_pos: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param active_config_space: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :param prefix_keys: See :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`, optional
    :return: New object
    """
    hp_ranges = HyperparameterRangesImpl(
        config_space,
        name_last_pos=name_last_pos,
        value_for_last_pos=value_for_last_pos,
        active_config_space=active_config_space,
        prefix_keys=prefix_keys,
    )
    return hp_ranges
