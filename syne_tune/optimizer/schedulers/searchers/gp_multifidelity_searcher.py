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
from typing import Optional, Dict
import logging

from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_multifidelity_searcher_factory, gp_multifidelity_searcher_defaults
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher \
    import ModelBasedSearcher, create_initial_candidates_scorer, decode_state
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    ResourceForAcquisitionMap
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import PendingEvaluation, Configuration, MetricValues, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm \
    import BayesianOptimizationAlgorithm
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector \
    import DuplicateDetectorIdentical

logger = logging.getLogger(__name__)

__all__ = ['GPMultiFidelitySearcher']


class GPMultiFidelitySearcher(ModelBasedSearcher):
    """Gaussian process Bayesian optimization for Hyperband scheduler

    This searcher must be used with `HyperbandScheduler`. It provides a novel
    combination of Bayesian optimization, based on a Gaussian process surrogate
    model, with Hyperband scheduling. In particular, observations across
    resource levels are modelled jointly. It is created along with the
    scheduler, using `searcher='bayesopt'`:

    Most of `GPFIFOSearcher` comments apply here as well.
    In multi-fidelity HPO, we optimize a function f(x, r), x the configuration,
    r the resource (or time) attribute. The latter must be a positive integer.
    In most applications, `resource_attr` == 'epoch', and the resource is the
    number of epochs already trained.

    We model the function f(x, r) jointly over all resource levels r at which
    it is observed (but see `searcher_data` in `HyperbandScheduler`). The kernel
    and mean function of our surrogate model are over (x, r). The surrogate
    model is selected by `gp_resource_kernel`. More details about the supported
    kernels is in:

        Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter and Neural Architecture Search
        https://arxiv.org/abs/2003.10865

    The acquisition function (EI) which is optimized in `get_config`, is obtained
    by fixing the resource level r to a value which is determined depending on
    the current state. If `resource_acq` == 'bohb', r is the largest value
    <= max_t, where we have seen >= dimension(x) metric values. If
    `resource_acq` == 'first', r is the first milestone which config x would
    reach when started.

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
    num_fantasy_samples : int
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
    gp_resource_kernel : str
        Surrogate model over criterion function f(x, r), x the config, r the
        resource. Note that x is encoded to be a vector with entries in [0, 1],
        and r is linearly mapped to [0, 1], while the criterion data is
        normalized to mean 0, variance 1. The reference above provides details
        on the models supported here. For the exponential decay kernel, the
        base kernel over x is Matern 5/2 ARD.
        Values are 'matern52' (Matern 5/2 ARD kernel over [x, r]),
        'matern52-res-warp' (Matern 5/2 ARD kernel over [x, r], with additional
        warping on r),
        'exp-decay-sum' (exponential decay kernel, with delta=0. This is the
        additive kernel from Freeze-Thaw Bayesian Optimization),
        'exp-decay-delta1' (exponential decay kernel, with delta=1),
        'exp-decay-combined' (exponential decay kernel, with delta in [0, 1]
        a hyperparameter).
    resource_acq : str
        Determines how the EI acquisition function is used (see above).
        Values: 'bohb', 'first'
    opt_skip_num_max_resource : bool
        Parameter for hyperparameter fitting, skip predicate. If True, and
        number of observations above `opt_skip_init_length`, fitting is done
        only when there is a new datapoint at r = max_t, and skipped otherwise.

    See Also
    --------
    GPFIFOSearcher
    """
    def __init__(self, configspace, **kwargs):
        if configspace is not None:
            super().__init__(
                configspace, metric=kwargs.get('metric'),
                points_to_evaluate=kwargs.get('points_to_evaluate'))
            kwargs['configspace'] = configspace
            kwargs_int = self._create_kwargs_int(kwargs)
        else:
            # Internal constructor, bypassing the factory
            kwargs_int = kwargs.copy()
        self.configspace_ext = ExtendedConfiguration(
            kwargs_int['hp_ranges'],
            resource_attr_key=kwargs_int['resource_attr'],
            resource_attr_range=kwargs_int['resource_attr_range'])
        self._process_kwargs_int(kwargs_int)
        self._call_create_internal(**kwargs_int)
        if self.debug_log is not None:
            # Configure DebugLogPrinter
            self.debug_log.set_configspace_ext(self.configspace_ext)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_multifidelity_searcher_defaults(),
            dict_name='search_options')
        kwargs_int = gp_multifidelity_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _process_kwargs_int(self, kwargs_int):
        self.resource_for_acquisition = kwargs_int.pop(
            'resource_for_acquisition')
        assert isinstance(self.resource_for_acquisition,
                          ResourceForAcquisitionMap)
        del kwargs_int['resource_attr_range']

    def _call_create_internal(self, **kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        self._create_internal(**kwargs_int)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import \
            HyperbandScheduler

        assert isinstance(scheduler, HyperbandScheduler), \
            "This searcher requires HyperbandScheduler scheduler"
        super().configure_scheduler(scheduler)
        self._resource_attr = scheduler._resource_attr

    def _hp_ranges_in_state(self):
        return self.configspace_ext.hp_ranges_ext

    def _metric_val_update(
            self, config: Dict, crit_val: float, result: Dict) -> MetricValues:
        resource = result[self._resource_attr]
        return {str(resource): crit_val}

    def _trial_id_string(self, trial_id: str, result: Dict):
        """
        For multi-fidelity, we also want to output the resource level
        """
        return f"{trial_id}:{result[self._resource_attr]}"

    def register_pending(self, config: Configuration, milestone=None):
        """
        Registers config as pending for resource level milestone. This means
        the corresponding evaluation task is running and should reach that
        level later, when update is called for it.

        :param config:
        :param milestone:
        """
        assert milestone is not None, \
            "This searcher works with a multi-fidelity scheduler only"
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        config_ext = self.configspace_ext.get(config, milestone)
        if config_ext not in state.pending_candidates:
            pos_cand = state.pos_of_config(config)
            if pos_cand is not None:
                active_metric_for_config = state.candidate_evaluations[
                    pos_cand].metrics.get(INTERNAL_METRIC_NAME)
                if active_metric_for_config is not None \
                        and str(milestone) in active_metric_for_config:
                    values = list(active_metric_for_config.items())
                    error_msg = f"""
                    This configuration at milestone {milestone} is already registered as labeled:
                       Position of labeled candidate: {pos_cand}
                       Label values: {values}
                    """
                    assert False, error_msg
            self.state_transformer.append_candidate(config_ext)

    def _fix_resource_attribute(self, **kwargs):
        """
        Determines target resource level r at which the current call of
        `get_config` operates. This is done based on
        `resource_for_acquisition`. This resource level is then set in
        `configspace_ext.hp_ranges_ext.value_for_last_pos`. This does the
        job for GP surrogate models. But if in subclasses, other surrogate
        models are involved, they need to get informed separately (see
        :class:`CostAwareGPMultiFidelitySearcher` for an example).

        :param kwargs:
        :return:
        """
        state = self.state_transformer.state
        # BO should only search over configs at resource level
        # target_resource
        if state.candidate_evaluations:
            target_resource = self.resource_for_acquisition(state, **kwargs)
        else:
            # Any valid value works here:
            target_resource = self.configspace_ext.resource_attr_range[0]
        self.configspace_ext.hp_ranges_ext.value_for_last_pos = target_resource
        if self.debug_log is not None:
            self.debug_log.append_extra(
                f"Score values computed at target_resource = {target_resource}")

    def _get_config_modelbased(self, exclusion_candidates, **kwargs) -> \
            Optional[Configuration]:
        config = None
        # Obtain current SurrogateModel from state transformer. Based on
        # this, the BO algorithm components can be constructed
        if self.do_profile:
            self.profiler.push_prefix('getconfig')
            self.profiler.start('all')
            self.profiler.start('gpmodel')
        # Note: Asking for the model triggers the posterior computation
        model = self.state_transformer.model()
        if self.do_profile:
            self.profiler.stop('gpmodel')
        # Select and fix target resource attribute
        self._fix_resource_attribute(**kwargs)
        # Create BO algorithm
        initial_candidates_scorer = create_initial_candidates_scorer(
            self.initial_scoring, model, self.acquisition_class,
            self.random_state)
        # We search over the same type of configs (normal or extended) which
        # we predict for
        local_optimizer = self.local_minimizer_class(
            hp_ranges=self._hp_ranges_for_prediction(),
            model=model,
            acquisition_class=self.acquisition_class,
            active_metric=INTERNAL_METRIC_NAME)
        bo_algorithm = BayesianOptimizationAlgorithm(
            initial_candidates_generator=self.random_generator,
            initial_candidates_scorer=initial_candidates_scorer,
            num_initial_candidates=self.num_initial_candidates,
            local_optimizer=local_optimizer,
            pending_candidate_state_transformer=None,
            exclusion_candidates=exclusion_candidates,
            num_requested_candidates=1,
            greedy_batch_selection=False,
            duplicate_detector=DuplicateDetectorIdentical(),
            profiler=self.profiler,
            sample_unique_candidates=False,
            debug_log=self.debug_log)
        # Next candidate decision
        _config = bo_algorithm.next_candidates()
        if len(_config) > 0:
            # If _config[0] is normal, nothing is removed
            config = self.configspace_ext.remove_resource(_config[0])
        if self.do_profile:
            self.profiler.stop('all')
            self.profiler.pop_prefix()  # getconfig
        return config

    def evaluation_failed(self, config: Configuration):
        # Remove all pending evaluations for config
        self.cleanup_pending(config)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        # We need to create an extended config by appending a resource
        # attribute. Its value does not matter, because of how the blacklist
        # is created
        lowest_attr_value = self.configspace_ext.resource_attr_range[0]
        config_ext = self.configspace_ext.get(config, lowest_attr_value)
        self.state_transformer.mark_candidate_failed(config_ext)

    def cleanup_pending(self, config: Configuration):
        """
        Removes all pending candidates whose configuration (i.e., lacking the
        resource attribute) is equal to config.
        This should be called after an evaluation terminates. For various
        reasons (e.g., termination due to convergence), pending candidates
        for this evaluation may still be present.
        It is also called for a failed evaluation.

        :param config: See above
        """
        def filter_pred(x: PendingEvaluation) -> bool:
            x_dct = self.configspace_ext.remove_resource(x.candidate)
            return x_dct != config

        self.state_transformer.filter_pending_evaluations(filter_pred)

    def remove_case(self, config, **kwargs):
        resource = kwargs[self._resource_attr]
        self.state_transformer.remove_observed_case(config, key=str(resource))

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(
            state['state'], self.configspace_ext.hp_ranges_ext)
        skip_optimization = state['skip_optimization']
        # Call internal constructor
        new_searcher = GPMultiFidelitySearcher(
            configspace=None,
            hp_ranges=self.hp_ranges,
            resource_attr_range=self.configspace_ext.resource_attr_range,
            random_seed=self.random_seed,
            model_factory=self.state_transformer._model_factory,
            map_reward=self.map_reward,
            acquisition_class=self.acquisition_class,
            resource_for_acquisition=self.resource_for_acquisition,
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
