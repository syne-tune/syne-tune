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
import logging

from syne_tune.optimizer.schedulers.searchers import GPMultiFidelitySearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    hypertune_searcher_factory,
    hypertune_searcher_defaults,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.hypertune.hypertune_bracket_distribution import (
    HyperTuneBracketDistribution,
)

logger = logging.getLogger(__name__)


class HyperTuneSearcher(GPMultiFidelitySearcher):
    """
    Implements Hyper-Tune as extension of :class:`GPMultiFidelitySearcher`, see
    :class:`HyperTuneIndependentGPModel` for references.

    Two modifications:

    * New brackets are sampled from a model-based distribution [w_k]
    * The acquisition function is fed with predictive means and variances from
        a mixture over rung level distributions, weighted by [theta_k]
    """

    def __init__(self, config_space, **kwargs):
        super().__init__(config_space, **kwargs)
        self._previous_distribution = None

    def configure_scheduler(self, scheduler):
        super().configure_scheduler(scheduler)
        # Overwrite default bracket distribution by adaptive one:
        scheduler.bracket_distribution = HyperTuneBracketDistribution()

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *hypertune_searcher_defaults(), dict_name="search_options"
        )
        kwargs_int = hypertune_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _hp_ranges_for_prediction(self):
        """
        Different to :class:`GPMultiFidelitySearcher`, we need non-extended
        configs here.
        """
        return self.hp_ranges

    def _postprocess_config(self, config: dict) -> dict:
        """
        Different to :class:`GPMultiFidelitySearcher`, we need non-extended
        configs here.
        """
        return config
