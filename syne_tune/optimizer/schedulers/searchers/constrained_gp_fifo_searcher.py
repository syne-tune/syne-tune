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
from typing import Dict
import logging

from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher \
    import MultiModelGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    constrained_gp_fifo_searcher_defaults, constrained_gp_fifo_searcher_factory
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    decode_state
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import CandidateEvaluation, INTERNAL_METRIC_NAME, INTERNAL_CONSTRAINT_NAME

logger = logging.getLogger(__name__)

__all__ = ['ConstrainedGPFIFOSearcher']


class ConstrainedGPFIFOSearcher(MultiModelGPFIFOSearcher):
    """
    Gaussian process-based constrained hyperparameter optimization (to be used with a FIFO scheduler).

    The searcher requires a constraint metric, which is given by `constraint_attr`.

    """

    def __init__(self, configspace, **kwargs):
        assert kwargs.get('constraint_attr') is not None, \
            "This searcher needs a constraint attribute. Please specify its " +\
            "name in search_options['constraint_attr']"
        super().__init__(configspace, **kwargs)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *constrained_gp_fifo_searcher_defaults(),
            dict_name='search_options')
        kwargs_int = constrained_gp_fifo_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        k = 'constraint_attr'
        kwargs_int[k] = kwargs.get(k)
        return kwargs_int

    def _call_create_internal(self, **kwargs_int):
        self._constraint_attr = kwargs_int.pop('constraint_attr')
        super()._call_create_internal(**kwargs_int)

    def _update(self, trial_id: str, config: Dict, result: Dict):
        # We can call the superclass method, because
        # `state_transformer.label_candidate` can be called two times
        # with parts of the metrics
        super()._update(trial_id, config, result)
        # Get constraint metric
        assert self._constraint_attr in result, \
            f"Constraint metric {self._constraint_attr} not included in " +\
            "reported result. Make sure your evaluation function reports it."
        constr_val = float(result[self._constraint_attr])
        metrics = {INTERNAL_CONSTRAINT_NAME: constr_val}
        self.state_transformer.label_candidate(CandidateEvaluation(
            candidate=config, metrics=metrics))
        if self.debug_log is not None:
            logger.info(f"constraint_val = {constr_val}")

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state['state'], self.hp_ranges)
        skip_optimization = state['skip_optimization']
        # Call internal constructor
        new_searcher = ConstrainedGPFIFOSearcher(
            configspace=None,
            hp_ranges=self.hp_ranges,
            random_seed=self.random_seed,
            output_model_factory=self.state_transformer._model_factory,
            constraint_attr=self._constraint_attr,
            acquisition_class=self.acquisition_class,
            map_reward=self.map_reward,
            init_state=init_state,
            local_minimizer_class=self.local_minimizer_class,
            output_skip_optimization=skip_optimization,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            cost_attr=self._cost_attr)
        self._clone_from_state_common(new_searcher, state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
