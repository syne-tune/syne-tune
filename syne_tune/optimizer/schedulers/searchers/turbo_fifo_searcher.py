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
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import copy
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    decode_state
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_turbo_fifo_searcher_defaults, gp_turbo_fifo_searcher_factory
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes \
    import SurrogateOutputModel, SurrogateModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl_new \
    import HyperparameterRangesImplNew
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import RandomStatefulCandidateGenerator, ExclusionList

logger = logging.getLogger(__name__)

__all__ = ['TuRBOFIFOSearcher']


class TuRBOHyperparameterRanges(HyperparameterRangesImplNew):
    def __init__(self, config_space: Dict,
                 ndarray_bounds: List[Tuple[float, float]]):
        super().__init__(config_space)
        assert self.ndarray_size == len(ndarray_bounds), \
            f"ndarray_size = {self.ndarray_size} != {len(ndarray_bounds)} = len(ndarray_bounds)"
        self._ndarray_bounds = copy.copy(ndarray_bounds)
        self._random_config_offset = np.array(
            [lower for lower, _ in ndarray_bounds])
        self._random_config_multiplier = np.array(
            [upper - lower for lower, upper in ndarray_bounds])

    def _random_config(self, random_state: RandomState) -> Configuration:
        enc_config = random_state.rand(self.ndarray_size) * \
               self._random_config_multiplier + self._random_config_offset
        return self.from_ndarray(enc_config)

    def get_ndarray_bounds(self) -> List[Tuple[float, float]]:
        return self._ndarray_bounds


class RandomCandidateGeneratorWithFallback(RandomStatefulCandidateGenerator):
    def __init__(self, hp_ranges: HyperparameterRanges,
                 random_state: np.random.RandomState,
                 fallback_candidates: List[Configuration]):
        super().__init__(hp_ranges, random_state)
        self._fallback_candidates = fallback_candidates

    def generate_candidates_en_bulk(
            self, num_cands: int, exclusion_list=None) -> List[Configuration]:
        result = super().generate_candidates_en_bulk(num_cands, exclusion_list)
        if result:
            return result
        else:
            logger.warning(
                "Failed to sample any initial candidates. Using fallback "
                "set instead")
            return self._fallback_candidates


class TuRBOFIFOSearcher(GPFIFOSearcher):
    """
    Implements asynchronous (non-batch) variant of TuRBO-1 algorithm, proposed
    in

        Eriksson et al
        Scalable Global Optimization via Local Bayesian Optimization
        NeurIPS 32 (2019)

    Normally, the acquisition function is optimized over the feasible space
    `[0, 1]^d`, where `d` is the dimension of the encoding space. In essence,
    TuRBO-1 maintains and adapts a trust region `R`, and the optimization is
    run over the intersection of `R` and `[0, 1]^d`.

    `R` is a rectangle centered at the current incumbent, whose sidelengths
    are proportional to the lenghtscales (or bandwidths) of the ARD kernel of
    the surrogate model. The volume of `R` is controlled by TuRBO-1, using some
    heuristics. It is `power(basic_sidelength, d)`.

    We can hook essentially all changes into `_get_config_modelbased`, and
    in particular into `_get_config_modelbased_prepare_bo`. In order to
    replace `[0, 1]^d` by its intersection with the trust region, we use a
    modified `hp_ranges` in `_get_config_modelbased_prepare_bo`, which affects
    both the `initial_candidates_generator` and `local_optimizer`, so they
    respect the new tighter bounds.

    The update of `basic_sidelength` based on tracking successes and failures
    is also hooked in there. It may be simpler to hook this into `_update`,
    but we take the success/failure decision based on posterior means, which
    are not available there.

    Additional parameters
    ---------------------
    sidelength_init : float
        Initial value `L_init` for basic side length `L`
    sidelength_min : float
        Minimum value `L_min` for basic side length `L`
    sidelength_max : float
        Maximum value `L_max` for basic side length `L`
    threshold_success : int
        Value for `tau_succ`. If this many successes are logged in a row, the
        basic sidelength is increased
    threshold_failure : int
        Value for `tau_fail`. If this many failures are logged in a row, the
        basic sidelength is decreased

    """

    def __init__(self, config_space, metric, **kwargs):
        super().__init__(config_space, metric, **kwargs)
        self.basic_sidelength = self.sidelength_init
        # Contains `(trial_id, incumbent_trial_id)` for trials proposed in
        # `get_config`. Here, `incumbent_trial_id` is the ID of the incumbent
        # when the decision was taken. This is needed in order to determine
        # failure or success later on.
        self._running_trials = []
        # Counters for failures (pos 0), successes (pos 1)
        self._status_counters = [0, 0]
        self._thresholds = [self.threshold_failure, self.threshold_success]
        # See `_is_trust_region_too_narrow`
        self._fallback_initial_candidates = None
        # TODO: This is a temporary solution. Ultimately, all searchers should
        # use the new code
        self.hp_ranges = HyperparameterRangesImplNew(
            config_space=self.hp_ranges.config_space)

    @property
    def counter_success(self) -> int:
        return self._status_counters[1]

    @property
    def counter_failure(self) -> int:
        return self._status_counters[0]

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_turbo_fifo_searcher_defaults(),
            dict_name='search_options')
        kwargs_int = gp_turbo_fifo_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, _kwargs)
        return kwargs_int

    def _copy_kwargs_to_kwargs_int(self, kwargs_int: Dict, kwargs: Dict):
        super()._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        for k in ('sidelength_init', 'sidelength_min', 'sidelength_max',
                  'threshold_success', 'threshold_failure'):
            kwargs_int[k] = kwargs[k]

    def _call_create_internal(self, kwargs_int):
        self.sidelength_init = kwargs_int.pop('sidelength_init')
        self.sidelength_min = kwargs_int.pop('sidelength_min')
        self.sidelength_max = kwargs_int.pop('sidelength_max')
        self.threshold_success = kwargs_int.pop('threshold_success')
        self.threshold_failure = kwargs_int.pop('threshold_failure')
        assert 0 < self.sidelength_min < self.sidelength_init < self.sidelength_max
        assert self.threshold_success > 0 and self.threshold_failure > 0
        super()._call_create_internal(kwargs_int)

    def _process_running_trials(
            self, model: SurrogateOutputModel,
            exclusion_candidates: ExclusionList):
        """
        Called by `_get_config_modelbased_prepare_bo` before bounds are
        updated. For each entry `_running_trials` with data in the state,
        we make a decision on success/failure. This leads to an update of
        the counters, and possibly an update of `basic_sidelength`, according
        to the TuRBO-1 heuristics.

        :param model: Current fitted surrogate model
        """
        state = self.state_transformer.state
        new_running_trials = []
        for trial_id, incumbent_trial_id in self._running_trials:
            if any(ev.trial_id == trial_id for ev in state.trials_evaluations):
                self._process_running_trial(
                    model, trial_id, incumbent_trial_id, exclusion_candidates)
            else:
                new_running_trials.append((trial_id, incumbent_trial_id))
        self._running_trials = new_running_trials

    @staticmethod
    def _counter_name(is_success: bool) -> str:
        return 'success' if is_success else 'failure'

    def _process_running_trial(
            self, model: SurrogateOutputModel, trial_id: str,
            incumbent_trial_id: str, exclusion_candidates: ExclusionList):
        # Determine success/failure by comparing posterior means. If there are
        # fantasy samples, we average the means over that
        state = self.state_transformer.state
        config = state.config_for_trial[trial_id]
        incumbent_config = state.config_for_trial[incumbent_trial_id]
        predictions = model.predict_candidates((config, incumbent_config))
        assert len(predictions) == 1, \
            "MCMC treatment of surrogate model parameters not supported"
        predictions = predictions[0]['mean']
        if predictions.ndim == 1:
            predictions = predictions.reshape((-1, 1))
        assert predictions.shape[0] == 2, \
            f"Internal error: predictions.shape = {predictions.shape}"
        predictions = np.mean(predictions, axis=1).reshape((-1,))
        is_success = predictions[0] < predictions[1]
        counter_pos = int(is_success)
        name_status = self._counter_name(is_success)
        logger.info(
            f"TuRBO: Comparing poster mean of trial_id {trial_id} versus "
            f"incumbent trial_id {incumbent_trial_id}: {predictions[0]} "
            f"vs {predictions[1]} [{name_status}]")
        if self._status_counters[counter_pos] == 0:
            self._status_counters[1 - counter_pos] = 0
        self._status_counters[counter_pos] += 1
        logger.info(
            f"New counter values: success = {self.counter_success}, "
            f"failure = {self.counter_failure}")
        new_basic_sidelength = self.basic_sidelength
        if self._status_counters[counter_pos] >= self._thresholds[counter_pos]:
            # Modify basic sidelength
            if is_success:
                new_basic_sidelength = min(
                    2 * new_basic_sidelength, self.sidelength_max)
                self._fallback_initial_candidates = None  # reset
            else:
                new_basic_sidelength = new_basic_sidelength / 2
                if new_basic_sidelength < self.sidelength_min:
                    new_basic_sidelength = self.sidelength_init
                if new_basic_sidelength < self.basic_sidelength and \
                        self._is_trust_region_too_narrow(
                            model, new_basic_sidelength, exclusion_candidates):
                    new_basic_sidelength = self.basic_sidelength
                    logger.info("Reject new sidelength as too narrow, stick "
                                "with existing one")
            # Reset counters (even if basic sidelength stays the same)
            self._status_counters = [0, 0]
            logger.info(
                f"Threshold for {name_status} is reached: Update "
                f"basic_sidelength to {new_basic_sidelength} (from "
                f"{self.basic_sidelength})")
        self.basic_sidelength = new_basic_sidelength

    def _is_trust_region_too_narrow(
            self, model: SurrogateOutputModel,
            new_basic_sidelength: int,
            exclusion_candidates: ExclusionList) -> bool:
        """
        Checks whether trust region with sidelength `new_basic_sidelength` is
        too narrow. This is done by sampling sampling initial candidates in
        the same way as in `BayesianOptimizationAlgorithm._get_next_candidates`.
        If this fails to return at least 5 candidates, the new sidelength
        should be rejected. Otherwise, the initial candidates obtained this
        way are stored in `_fallback_initial_candidates`. If, further down
        in `_get_config_modelbased`, the sampler fails to return candidates,
        we can fall back to this list.

        :param model: Current fitted surrogate model
        :param new_basic_sidelength: Sidelength to test
        :return: Is it too narrow? Should be rejected then
        """
        basic_sidelength = self.basic_sidelength  # copy
        self.basic_sidelength = new_basic_sidelength
        ndarray_bounds, _ = self._bounds_with_trust_region(
            model, verbose=False)
        self.basic_sidelength = basic_sidelength  # restore
        hp_ranges = TuRBOHyperparameterRanges(
            config_space=self._hp_ranges_in_state().config_space,
            ndarray_bounds=ndarray_bounds)
        random_generator = RandomStatefulCandidateGenerator(
            hp_ranges, random_state=self.random_state)
        initial_candidates = random_generator.generate_candidates_en_bulk(
                self.num_initial_candidates,
                exclusion_list=exclusion_candidates)
        unique_candidates = set(hp_ranges.config_to_match_string(config)
                                for config in initial_candidates)
        if len(unique_candidates) < 5:
            return True
        logger.info(
            f"Storing fallback_initial_candidates ({len(unique_candidates)} unique entries)")
        self._fallback_initial_candidates = initial_candidates
        return False

    def _bounds_with_trust_region(
            self, model: SurrogateOutputModel,
            verbose: bool = True) -> (List[Tuple[float, float]], int):
        """
        Determine the bounds for
        the feasible set of acquisition function optimization. This is the
        intersection of the global feasible set with the current trust region.
        The latter is a hyper-rectangle centered at the current incumbent input
        vector, and its sidelengths are proportional to ARD lengthscales.

        :param model: Current fitted surrogate model
        :return: New bounds for AF optimization, incumbent_trial_id
        """
        hp_ranges = self._hp_ranges_in_state()
        dim = hp_ranges.ndarray_size
        params = self.state_transformer.model_factory.get_params()
        if dim > 1:
            log_sidelengths = np.array(
                [-np.log(float(params[f"kernel_inv_bw{i}"]))
                 for i in range(dim)])
            log_sidelengths -= np.mean(log_sidelengths)
        else:
            log_sidelengths = np.zeros(1)
        log_sidelengths += np.log(self.basic_sidelength)
        # Retrieve incumbent config (whose encoded vector is the center of
        # the trust region)
        incumbent_trial_id = model.current_best_trial_id()
        assert len(incumbent_trial_id) == 1, \
            "MCMC treatment of surrogate model parameters not supported"
        incumbent_trial_id = str(incumbent_trial_id[0])
        incumbent_config = self.state_transformer.state.config_for_trial[
            incumbent_trial_id]
        incumbent_input = hp_ranges.to_ndarray(
            incumbent_config).reshape((-1,))
        half_sidelengths = np.exp(log_sidelengths) * 0.5
        tr_lowers = incumbent_input - half_sidelengths
        tr_uppers = incumbent_input + half_sidelengths
        # Return intersection with global bounds
        new_bounds = [
            (max(lower, tr_lower), min(upper, tr_upper))
            for (lower, upper), tr_lower, tr_upper in zip(
                hp_ranges.get_ndarray_bounds(), tr_lowers, tr_uppers)]
        if verbose:
            # DEBUG
            outer_parts = [f"_bounds_with_trust_region: basic_sidelength = {self.basic_sidelength}"]
            msg_parts = [f"{i:2d}:({lower:.2f},{upper:.2f})"
                         for i, (lower, upper) in enumerate(new_bounds)]
            col_size = 5
            offset = 0
            sub_list = []
            for part in msg_parts:
                sub_list.append(part)
                offset += 1
                if offset == col_size:
                    outer_parts.append(' '.join(sub_list))
                    offset = 0
                    sub_list = []
            if offset > 0:
                outer_parts.append(' '.join(sub_list))
            logger.info('\n'.join(outer_parts))
        return new_bounds, incumbent_trial_id

    def _get_config_modelbased_prepare_bo(
            self, model: SurrogateOutputModel,
            exclusion_candidates: ExclusionList,
            hp_ranges: Optional[HyperparameterRanges] = None,
            **kwargs) -> dict:
        assert isinstance(model, SurrogateModel), \
            "Multi-output models are not supported"
        trial_id = kwargs.get('trial_id')
        assert trial_id is not None, \
            "trial_id must be passed to get_config"
        # TuRBO logic to update basic sidelength:
        self._process_running_trials(model, exclusion_candidates)
        # Determine effective bounds and use `hp_ranges` which respects these
        ndarray_bounds, incumbent_trial_id = self._bounds_with_trust_region(
            model)
        hp_ranges = TuRBOHyperparameterRanges(
            config_space=self._hp_ranges_in_state().config_space,
            ndarray_bounds=ndarray_bounds)
        # Append new entry here, even if it is not 100% sure that the new
        # trial will be run. This does not matter. Whatever happens, the
        # `trial_id` will not be used again
        self._running_trials.append((trial_id, incumbent_trial_id))
        result = super()._get_config_modelbased_prepare_bo(
            model=model, hp_ranges=hp_ranges,
            exclusion_candidates=exclusion_candidates)
        if self._fallback_initial_candidates is not None:
            # Replace `random_generator` by one which uses the fallback in
            # case of a failure
            random_generator = RandomCandidateGeneratorWithFallback(
                hp_ranges=hp_ranges,
                random_state=self.random_state,
                fallback_candidates=self._fallback_initial_candidates)
            result['initial_candidates_generator'] = random_generator
        return result

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state['state'], self._hp_ranges_in_state())
        skip_optimization = state['skip_optimization']
        model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = TuRBOFIFOSearcher(
            **self._new_searcher_kwargs_for_clone(),
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
            sidelength_init=self.sidelength_init,
            sidelength_min=self.sidelength_min,
            sidelength_max=self.sidelength_max,
            threshold_success=self.threshold_success,
            threshold_failure=self.threshold_failure)
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
