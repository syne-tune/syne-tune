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

from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_multifidelity_searcher_defaults, gpiss_multifidelity_searcher_factory
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    decode_state
from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher \
    import GPMultiFidelitySearcher

logger = logging.getLogger(__name__)

__all__ = ['GPISSMultiFidelitySearcher']


class GPISSMultiFidelitySearcher(GPMultiFidelitySearcher):
    """
    Supports asynchronous multi-fidelity hyperparameter optimization, in the
    style of Hyperband or BOHB. Here, a joint GP-ISS surrogate model is fit to
    observations made at all resource levels.

    Parameters
    ----------
    configspace : Dict
        Configuration space. Constant parameters are filtered out
    metric : str
        Name of reward attribute reported by evaluation function
    points_to_evaluate: List[Dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
    resource_attr : str
        Name of resource attribute in reports, equal to `resource_attr` of
        scheduler
    debug_log : bool (default: False)
        If True, both searcher and scheduler output an informative log, from
        which the configs chosen and decisions being made can be traced.
    cost_attr : str (optional)
        Name of cost attribute in data obtained from reporter (e.g., elapsed
        training time). Needed only by cost-aware searchers.
    random_seed : int
        Seed for pseudo-random number generator used.
    num_init_random : int
        See :class:`GPFIFOSearcher`
    num_init_candidates : int
        See :class:`GPFIFOSearcher`
    initial_scoring : str
        See :class:`GPFIFOSearcher`
    opt_nstarts : int
        See :class:`GPFIFOSearcher`
    opt_maxiter : int
        See :class:`GPFIFOSearcher`
    opt_warmstart : bool
        See :class:`GPFIFOSearcher`
    opt_verbose : bool
        See :class:`GPFIFOSearcher`
    opt_skip_init_length : int
        See :class:`GPFIFOSearcher`
    opt_skip_period : int
        See `:class:GPFIFOSearcher`
    map_reward : str or MapReward
        See :class:`GPFIFOSearcher`
    opt_skip_num_max_resource : bool
        Parameter for hyperparameter fitting, skip predicate. If True, and
        number of observations above `opt_skip_init_length`, fitting is done
        only when there is a new datapoint at r = max_t, and skipped otherwise.
    issm_gamma_one : bool
        If True, the gamma parameter of the ISSM is fixed to 1, otherwise it
        is optimized over.

    See Also
    --------
    GPFIFOSearcher
    GPMultiFidelitySearcher
    """
    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_multifidelity_searcher_defaults(),
            dict_name='search_options')
        # Extra arguments not parsed in factory
        kwargs_int = gpiss_multifidelity_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _process_kwargs_int(self, kwargs_int):
        # Set configspace_ext in model_factory (was not done at construction)
        kwargs_int['model_factory'].set_configspace_ext(self.configspace_ext)
        del kwargs_int['resource_attr_range']

    def _hp_ranges_for_prediction(self):
        """
        Use normal configurations (without resource attribute) in predictions
        and optimizing acquisitions.
        """
        return self.hp_ranges

    def _fix_resource_attribute(self, **kwargs):
        """
        We search over normal configs here, so fixing the resource attribute
        is not needed.
        """
        pass

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(
            state['state'], self.configspace_ext.hp_ranges_ext)
        skip_optimization = state['skip_optimization']
        # Call internal constructor
        new_searcher = GPISSMultiFidelitySearcher(
            configspace=None,
            hp_ranges=self.hp_ranges,
            resource_attr_range=self.configspace_ext.resource_attr_range,
            random_seed=self.random_seed,
            model_factory=self.state_transformer._model_factory,
            acquisition_class=self.acquisition_class,
            map_reward=self.map_reward,
            init_state=init_state,
            local_minimizer_class=self.local_minimizer_class,
            skip_optimization=skip_optimization,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            cost_attr = self._cost_attr,
            resource_attr=self._resource_attr)
        self._clone_from_state_common(new_searcher, state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
