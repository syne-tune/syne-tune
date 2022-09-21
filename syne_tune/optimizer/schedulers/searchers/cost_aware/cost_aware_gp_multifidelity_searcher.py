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
import logging

from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher import (
    GPMultiFidelitySearcher,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    cost_aware_gp_multifidelity_searcher_defaults,
    cost_aware_gp_multifidelity_searcher_factory,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import decode_state
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    ModelStateTransformer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost_fifo_model import (
    CostSurrogateModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
    INTERNAL_COST_NAME,
)

logger = logging.getLogger(__name__)

__all__ = ["CostAwareGPMultiFidelitySearcher"]


class MultiModelGPMultiFidelitySearcher(GPMultiFidelitySearcher):
    """
    Superclass for multi-model extensions of :class:`GPMultiFidelitySearcher`.
    We first call `ModelBasedSearcher._create_internal` passing factory and
    skip_optimization predicate for the `INTERNAL_METRIC_NAME` model, then
    replace the state transformer by a multi-model one.

    """

    def _call_create_internal(self, kwargs_int):
        output_model_factory = kwargs_int.pop("output_model_factory")
        output_skip_optimization = kwargs_int.pop("output_skip_optimization")
        kwargs_int["model_factory"] = output_model_factory[INTERNAL_METRIC_NAME]
        kwargs_int["skip_optimization"] = output_skip_optimization[INTERNAL_METRIC_NAME]
        super()._call_create_internal(kwargs_int)
        # Replace `state_transformer`
        init_state = self.state_transformer.state
        self.state_transformer = ModelStateTransformer(
            model_factory=output_model_factory,
            init_state=init_state,
            skip_optimization=output_skip_optimization,
        )


class CostAwareGPMultiFidelitySearcher(MultiModelGPMultiFidelitySearcher):
    """
    Gaussian process-based cost-aware multi-fidelity hyperparameter
    optimization (to be used with `HyperbandScheduler`). The searcher requires
    a cost metric, which is given by `cost_attr`.

    The acquisition function used here is the same as in
    :class:`GPMultiFidelitySearcher`, but expected improvement (EI) is replaced
    by EIpu (see :class:`EIpuAcquisitionFunction`).

    Cost values are read from each report and cost is modeled as c(x, r), the
    cost model being given by `kwargs['cost_model']`.

    """

    def __init__(self, config_space, metric, **kwargs):
        assert kwargs.get("cost_attr") is not None, (
            "This searcher needs a cost attribute. Please specify its "
            + "name in search_options['cost_attr']"
        )
        super().__init__(config_space, metric, **kwargs)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs,
            *cost_aware_gp_multifidelity_searcher_defaults(),
            dict_name="search_options"
        )
        kwargs_int = cost_aware_gp_multifidelity_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _fix_resource_attribute(self, **kwargs):
        if self.resource_for_acquisition is not None:
            super()._fix_resource_attribute(**kwargs)
            fixed_resource = self.config_space_ext.hp_ranges_ext.value_for_last_pos
        else:
            # Cost at r_max
            fixed_resource = self.config_space_ext.resource_attr_range[1]
        cost_model_factory = self.state_transformer.model_factory[INTERNAL_COST_NAME]
        assert isinstance(cost_model_factory, CostSurrogateModelFactory)
        cost_model_factory.set_fixed_resource(fixed_resource)

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        output_skip_optimization = state["skip_optimization"]
        output_model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = CostAwareGPMultiFidelitySearcher(
            **self._new_searcher_kwargs_for_clone(),
            output_model_factory=output_model_factory,
            init_state=init_state,
            output_skip_optimization=output_skip_optimization,
            config_space_ext=self.config_space_ext,
            resource_for_acquisition=self.resource_for_acquisition
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
