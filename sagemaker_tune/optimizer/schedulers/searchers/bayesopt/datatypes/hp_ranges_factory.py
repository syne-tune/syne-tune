from typing import Dict
import logging

from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl \
    import HyperparameterRangesImpl

logger = logging.getLogger(__name__)


def make_hyperparameter_ranges(
        config_space: Dict, name_last_pos: str = None,
        value_for_last_pos=None) -> HyperparameterRanges:
    hp_ranges = HyperparameterRangesImpl(
        config_space, name_last_pos, value_for_last_pos)
    return hp_ranges
