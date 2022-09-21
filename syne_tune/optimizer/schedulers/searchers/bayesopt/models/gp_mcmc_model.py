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
from typing import Optional
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gpr_mcmc import (
    GPRegressionMCMC,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    GaussProcModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import ConfigurationFilter

logger = logging.getLogger(__name__)


class GaussProcMCMCModelFactory(GaussProcModelFactory):
    def __init__(
        self,
        gpmodel: GPRegressionMCMC,
        active_metric: str = INTERNAL_METRIC_NAME,
        normalize_targets: bool = True,
        profiler: Optional[SimpleProfiler] = None,
        debug_log: Optional[DebugLogPrinter] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        hp_ranges_for_prediction: Optional[HyperparameterRanges] = None,
    ):
        """
        We support pending evaluations via fantasizing. Note that state does
        not contain the fantasy values, but just the pending configs. Fantasy
        values are sampled here.

        We draw one fantasy sample per MCMC sample here. This could be extended
        by sampling >1 fantasy samples for each MCMC sample.

        :param gpmodel: GPRegressionMCMC model
        :param active_metric: Name of the metric to optimize.
        :param normalize_targets: Normalize target values in
            state.trials_evaluations?

        """
        super().__init__(
            gpmodel=gpmodel,
            active_metric=active_metric,
            normalize_targets=normalize_targets,
            profiler=profiler,
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
