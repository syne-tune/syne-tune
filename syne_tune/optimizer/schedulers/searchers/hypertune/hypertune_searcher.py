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
from typing import Dict, Any
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
    Implements Hyper-Tune as extension of
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`,
    see
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model.HyperTuneIndependentGPModel`
    for references. Two modifications:

    * New brackets are sampled from a model-based distribution :math:`[w_k]`
    * The acquisition function is fed with predictive means and variances from
      a mixture over rung level distributions, weighted by :math:`[\theta_k]`

    It is *not* recommended to create :class:`HyperTuneSearcher` searcher
    objects directly, but rather to create
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` objects with
    ``searcher="hypertune"``, and passing arguments here in ``search_options``.
    This will use the appropriate functions from
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`` to
    create components in a consistent way.

    The following arguments of the parent class are not relevant here, and are
    ignored: ``gp_resource_kernel``, ``resource_acq``, ``issm_gamma_one``,
    ``expdecay_normalize_inputs``.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`:

    :param model: Selects surrogate model (learning curve model) to be used.
        Choices are:

        * "gp_multitask": GP multi-task surrogate model
        * "gp_independent" (default): Independent GPs for each rung level,
          sharing an ARD kernel

        The default is "gp_independent" (as in the Hyper-Tune paper), which
        is different to the default in :class:`GPMultiFidelitySearcher` (which
        is "gp_multitask"). "gp_issm", "gp_expdecay" not supported here.

    :type model: str, optional
    :param hypertune_distribution_num_samples: Parameter for estimating the
        distribution, given by :math:`[\theta_k]`. Defaults to 50
    :type hypertune_distribution_num_samples: int, optional
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
            kwargs, *hypertune_searcher_defaults(kwargs), dict_name="search_options"
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

    def _postprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Different to :class:`GPMultiFidelitySearcher`, we need non-extended
        configs here.
        """
        return config
