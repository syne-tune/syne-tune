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

from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher import \
    GPMultiFidelitySearcher
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils \
    import decode_state
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gpiss_model \
    import GaussProcAdditiveModelFactory

logger = logging.getLogger(__name__)

__all__ = ['GPSyncMultiFidelitySearcher']


class GPSyncMultiFidelitySearcher(GPMultiFidelitySearcher):
    """

    Variant of :class:`GPMultiFidelitySearcher` to be used with synchronous
    Hyperband.

    Additional Parameters
    ---------------------
    batch_size : int
        Batch size of synchronous scheduler
    num_init_candidates_for_batch : int
        While the first `suggest` of a batch uses `num_init_candidates`, this
        value is used for all the others. If not given, `num_init_candidates`
        is used for all `suggest` calls.

    See Also
    --------
    GPMultiFidelitySearcher
    """
    def _copy_kwargs_to_kwargs_int(self, kwargs_int: Dict, kwargs: Dict):
        super()._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        k = 'batch_size'
        batch_size = kwargs.get(k)
        assert batch_size is not None and int(batch_size) == batch_size \
               and batch_size > 1, \
            f"Argument {k} must be given (int >= 2)"
        kwargs_int[k] = batch_size
        kwargs_int['num_initial_candidates_for_batch'] = \
            kwargs.get('num_init_candidates_for_batch')

    def _call_create_internal(self, kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        k = 'batch_size'
        self.batch_size = kwargs_int.pop(k)
        k = 'num_initial_candidates_for_batch'
        self.num_initial_candidates_for_batch = kwargs_int.pop(k)
        self._num_suggest_calls = 0
        super()._call_create_internal(kwargs_int)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.synchronous.hyperband import \
            SynchronousHyperbandScheduler

        assert isinstance(scheduler, SynchronousHyperbandScheduler), \
            "This searcher requires SynchronousHyperbandScheduler scheduler"
        self._metric = scheduler.metric
        self._resource_attr = scheduler._resource_attr
        model_factory = self.state_transformer.model_factory
        if isinstance(model_factory, GaussProcAdditiveModelFactory):
            assert scheduler.searcher_data == 'all', \
                "For an additive Gaussian learning curve model (model=" +\
                "'gp_issm' or model='gp_expdecay' in search_options), you " +\
                "need to set searcher_data='all' when creating the " +\
                "SynchronousHyperbandScheduler"

    def _num_initial_candidates(self) -> int:
        ninit_for_batch = self.num_initial_candidates_for_batch
        if ninit_for_batch is None or \
                self._num_suggest_calls % self.batch_size == 0:
            return self.num_initial_candidates
        else:
            return ninit_for_batch

    def get_config(self, **kwargs) -> Configuration:
        config = super().get_config(**kwargs)
        if config is not None:
            self._num_suggest_calls += 1
        return config

    def get_state(self):
        return dict(
            super().get_state(),
            num_suggest_calls=self._num_suggest_calls)

    def _restore_from_state(self, state: dict):
        super()._restore_from_state(state)
        self._num_suggest_calls = state['num_suggest_calls']

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state['state'], self._hp_ranges_in_state())
        skip_optimization = state['skip_optimization']
        model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = GPSyncMultiFidelitySearcher(
            configspace=self.configspace,
            metric=self._metric,
            clone_from_state=True,
            hp_ranges=self.hp_ranges,
            configspace_ext=self.configspace_ext,
            model_factory=model_factory,
            acquisition_class=self.acquisition_class,
            map_reward=self.map_reward,
            resource_for_acquisition=self.resource_for_acquisition,
            init_state=init_state,
            local_minimizer_class=self.local_minimizer_class,
            skip_optimization=skip_optimization,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            cost_attr=self._cost_attr,
            resource_attr=self._resource_attr,
            batch_size=self.batch_size,
            num_initial_candidates_for_batch=self.num_initial_candidates_for_batch)
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
