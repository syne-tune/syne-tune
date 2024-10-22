from typing import Optional
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gpr_mcmc import (
    GPRegressionMCMC,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    GaussProcEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import ConfigurationFilter

logger = logging.getLogger(__name__)


class GaussProcMCMCEstimator(GaussProcEstimator):
    """
    We support pending evaluations via fantasizing. Note that state does
    not contain the fantasy values, but just the pending configs. Fantasy
    values are sampled here.

    We draw one fantasy sample per MCMC sample here. This could be extended
    by sampling ``> 1`` fantasy samples for each MCMC sample.

    :param gpmodel: GPRegressionMCMC model
    :param active_metric: Name of the metric to optimize.
    :param normalize_targets: Normalize target values in
        ``state.trials_evaluations``?
    """

    def __init__(
        self,
        gpmodel: GPRegressionMCMC,
        active_metric: str = INTERNAL_METRIC_NAME,
        normalize_targets: bool = True,
        debug_log: Optional[DebugLogPrinter] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        hp_ranges_for_prediction: Optional[HyperparameterRanges] = None,
    ):
        super().__init__(
            gpmodel=gpmodel,
            active_metric=active_metric,
            normalize_targets=normalize_targets,
            debug_log=debug_log,
            filter_observed_data=filter_observed_data,
            hp_ranges_for_prediction=hp_ranges_for_prediction,
        )

    def get_params(self):
        return dict()  # Model has no parameters to be fit

    def set_params(self, param_dict):
        pass  # Model has no parameters to fit

    def _get_num_fantasy_samples(self) -> int:
        return self._gpmodel.number_samples

    def _num_samples_for_fantasies(self) -> int:
        assert not self._gpmodel.multiple_targets()  # Sanity check
        return 1
