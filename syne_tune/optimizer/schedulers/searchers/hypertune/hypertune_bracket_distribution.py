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
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    GaussProcModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model import (
    HyperTuneIndependentGPModel,
    HyperTuneJointGPModel,
)
from syne_tune.optimizer.schedulers.searchers.bracket_distribution import (
    DefaultHyperbandBracketDistribution,
)
from syne_tune.optimizer.scheduler import TrialScheduler

logger = logging.getLogger(__name__)


class HyperTuneBracketDistribution(DefaultHyperbandBracketDistribution):
    """
    Represents the adaptive distribution over brackets [w_k].
    """

    def __init__(self):
        super().__init__()
        self._previous_distribution = None
        self._searcher = None

    def configure(self, scheduler: TrialScheduler):
        from syne_tune.optimizer.schedulers import HyperbandScheduler
        from syne_tune.optimizer.schedulers.searchers import GPMultiFidelitySearcher

        super().configure(scheduler)
        assert isinstance(scheduler, HyperbandScheduler)
        self._searcher = scheduler.searcher
        assert isinstance(self._searcher, GPMultiFidelitySearcher)

    def __call__(self) -> np.ndarray:
        distribution = super().__call__()
        model_factory = self._searcher.state_transformer.model_factory
        err_msg = (
            "Hyper-Tune requires GaussProcModelFactory model factory with "
            "HyperTuneIndependentGPModel or HyperTuneJointGPModel model. Use "
            "searcher = 'hypertune'"
        )
        assert isinstance(model_factory, GaussProcModelFactory), err_msg
        gpmodel = model_factory.gpmodel
        assert isinstance(
            gpmodel, (HyperTuneIndependentGPModel, HyperTuneJointGPModel)
        ), err_msg
        ht_distribution = gpmodel.hypertune_bracket_distribution()
        if ht_distribution is not None:
            # The Hyper-Tune distribution may not be over all brackets.
            # In that case, we keep the tail of the default distribution
            ht_size = ht_distribution.size
            if ht_size == distribution.size:
                distribution = ht_distribution.copy()
            else:
                assert ht_size < distribution.size, (ht_size, distribution.size)
                distribution = distribution.copy()
                mass_old_head = np.sum(distribution[:ht_size])
                distribution[:ht_size] = ht_distribution * mass_old_head
            if self._searcher.debug_log is not None and (
                self._previous_distribution is None
                or np.any(distribution != self._previous_distribution)
            ):
                logger.info(
                    "New distribution over brackets (ht_size = "
                    f"{ht_size}):\n{distribution}\nNew ensemble distribution:"
                    f"\n{gpmodel.hypertune_ensemble_distribution()}"
                )
                self._previous_distribution = distribution
        return distribution
