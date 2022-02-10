from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.hyperband_stopping import StoppingRungSystem


class RUSHStoppingRungSystem(StoppingRungSystem):
    """ TODO: description
    """
    def __init__(self, rung_levels, promote_quantiles, metric, mode, resource_attr, **kwargs):
        super().__init__(rung_levels, promote_quantiles, metric, mode, resource_attr)


class RUSHPromotionRungSystem(PromotionRungSystem):
    """ TODO: description
    """
    def __init__(self, rung_levels, promote_quantiles, metric, mode, resource_attr, max_t, **kwargs):
        super().__init__(rung_levels, promote_quantiles, metric, mode, resource_attr, max_t)
