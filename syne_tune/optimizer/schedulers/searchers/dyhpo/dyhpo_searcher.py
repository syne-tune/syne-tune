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
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers import (
    BaseSearcher,
    GPMultiFidelitySearcher,
)
from syne_tune.optimizer.schedulers.searchers.model_based_searcher import (
    create_initial_candidates_scorer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BaseSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)

# DEBUG:
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
)

logger = logging.getLogger(__name__)


KEY_NEW_CONFIGURATION = "new_configuration"


INTERNAL_KEY = "RESERVED_KEY_31415927"


# DEBUG:
def _debug_print_info(
    resource: int, scores: List[float], acq_function: EIAcquisitionFunction
):
    msg_parts = [
        f"Summary scores [resource = {resource}]",
        f"  Min:    {np.min(scores):.2e}",
        f"  Median: {np.median(scores):.2e}",
        f"  Max:    {np.max(scores):.2e}",
        f"  Num:    {len(scores)}",
    ]
    if acq_function._debug_data is not None:
        msg_parts.append("Summary inputs to EI:\n" + acq_function.debug_stats_message())
    print("\n".join(msg_parts))


class MyGPMultiFidelitySearcher(GPMultiFidelitySearcher):
    """
    This wrapper is for convenience, to avoid having to depend on internal
    concepts of
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        assert (
            INTERNAL_KEY not in config_space
        ), f"Key {INTERNAL_KEY} must not be used in config_space (reserved)"
        super().__init__(config_space, **kwargs)
        self._debug_log_copy = None
        # DEBUG
        # self.acquisition_class = (
        #     self.acquisition_class, {"debug_collect_stats": True}
        # )

    def _get_config_modelbased(
        self, exclusion_candidates: ExclusionList, **kwargs
    ) -> Optional[Configuration]:
        # Allows us to call :meth:`get_config` in
        # :meth:`score_paused_trials_and_new_configs`. If this returns the dummy
        # here, we are ready to score configs there, otherwise we are still in
        # the initial phase and return the config.
        # This trick is needed in order to make ``debug_log`` work. Otherwise, the
        # rest of ``get_config`` tries to output a block for this dummy return.
        self._debug_log_copy = self._debug_log
        self._debug_log = None  # No more debug logging for rest of ``get_config``
        return {INTERNAL_KEY: "dummy"}

    def _extended_configs_from_paused(
        self, paused_trials: List[Tuple[str, int, int]], state: TuningJobState
    ) -> List[Dict[str, Any]]:
        return [
            self.config_space_ext.get(state.config_for_trial[trial_id], resource)
            for trial_id, _, resource in paused_trials
        ]

    def _extended_configs_new(
        self,
        num_new: int,
        min_resource: int,
        exclusion_candidates: ExclusionList,
        random_generator: CandidateGenerator,
    ) -> List[Dict[str, Any]]:
        return [
            self.config_space_ext.get(config, min_resource)
            for config in random_generator.generate_candidates_en_bulk(
                num_new, exclusion_list=exclusion_candidates
            )
        ]

    def _scores_for_resource(
        self,
        resource: int,
        start: int,
        end: int,
        model: BaseSurrogateModel,
        sorted_ind: np.ndarray,
        configs_all: List[Dict[str, Any]],
    ) -> (List[float], np.ndarray):
        def my_filter_observed_data(config: Configuration) -> bool:
            return self.config_space_ext.get_resource(config) == resource

        # If there is data at level ``resource``, the incumbent in EI
        # should only be computed over this. If there is no data at level
        # ``resource``, the incumbent is computed over all data
        state = model.state
        if state.num_observed_cases(resource=resource) > 0:
            filter_observed_data = my_filter_observed_data
        else:
            filter_observed_data = None
        model.set_filter_observed_data(filter_observed_data)
        candidates_scorer = create_initial_candidates_scorer(
            initial_scoring="acq_func",
            model=model,
            acquisition_class=self.acquisition_class,
            random_state=self.random_state,
        )
        ind_for_resource = sorted_ind[start:end]
        scores_for_resource = candidates_scorer.score(
            [configs_all[pos] for pos in ind_for_resource]
        )
        return scores_for_resource, ind_for_resource

    def score_paused_trials_and_new_configs(
        self,
        paused_trials: List[Tuple[str, int, int]],
        min_resource: int,
        new_trial_id: str,
        skip_optimization: bool,
    ) -> Dict[str, Any]:
        """
        See :meth:`DynamicHPOSearcher.score_paused_trials_and_new_configs`.
        If ``skip_optimization == True``, this is passed to the posterior state
        computation, and refitting of the surrogate model is skipped. Otherwise,
        nothing is passed, so the built-in ``skip_optimization`` logic is used.
        """
        # Test whether we are still at the beginning, where we always return
        # new configs from ``points_to_evaluate`` or drawn at random.
        # Note: We need to pass ``trial_id`` to :meth:`get_config`, which is why
        # we need ``new_trial_id``.
        config = self.get_config(trial_id=new_trial_id)
        if config is None or INTERNAL_KEY not in config:
            return {"config": config}
        # Restore ``debug_log`` (see :meth:``_get_config_modelbased``)
        self._debug_log = self._debug_log_copy
        self._debug_log_copy = None
        # Note that at this point, if debug logging is active, ``get_config``
        # has called ``debug_log.start_get_config("BO", new_trial_id)``
        exclusion_candidates = self._get_exclusion_candidates(
            skip_observed=self._allow_duplicates,
        )
        random_generator = self._create_random_generator()

        # Collect all extended configs to be scored
        state = self.state_transformer.state
        assert (
            not state.hp_ranges.is_attribute_fixed()
        ), "Internal error: state.hp_ranges.is_attribute_fixed() must not be True"
        configs_paused = self._extended_configs_from_paused(paused_trials, state)
        num_new = max(self.num_initial_candidates, len(paused_trials))
        configs_new = self._extended_configs_new(
            num_new, min_resource, exclusion_candidates, random_generator
        )
        num_new = len(configs_new)  # Can be less than before
        configs_all = configs_paused + configs_new
        resources_all = [x[2] for x in paused_trials] + ([min_resource] * num_new)
        num_all = len(resources_all)
        if num_all == 0:
            # Very unlikely, but can happen (if config space is exhausted)
            logger.warning(
                "Cannot score any configurations (no paused trials, no new "
                "configurations)."
            )
            return {"config": None}

        # Score all extended configurations :math:`(x, r)` with
        # :math:`EI(x | r)`, expected improvement at level :math:`r`. Note that
        # the incumbent is the best value at the same level, not overall. This
        # is why we compute the score values in chunks for the same :math:`r`
        # value
        if skip_optimization:
            kwargs = dict(skip_optimization=skip_optimization)
        else:
            kwargs = dict()
        # Note: Asking for the model triggers the posterior computation
        model = self.state_transformer.model(**kwargs)
        # Note: ``model.state`` can have fewer observations than
        # ``self.state_transformer.state`` used above, because the former can
        # be filtered down
        resources_all = np.array(resources_all)
        sorted_ind = np.argsort(resources_all)
        resources_all = resources_all[sorted_ind]
        # Find positions where resource value changes
        change_pos = (
            [0]
            + list(np.flatnonzero(resources_all[:-1] != resources_all[1:]) + 1)
            + [num_all]
        )
        scores = np.empty((num_all,))
        for start, end in zip(change_pos[:-1], change_pos[1:]):
            resource = resources_all[start]
            assert resources_all[end - 1] == resource
            scores_for_resource, ind_for_resource = self._scores_for_resource(
                resource=resource,
                start=start,
                end=end,
                model=model,
                sorted_ind=sorted_ind,
                configs_all=configs_all,
            )
            scores[ind_for_resource] = scores_for_resource
            # DEBUG:
            # _debug_print_info(resource, scores_for_resource, candidates_scorer)

        # Pick the winner
        best_ind = np.argmin(scores)
        if best_ind >= len(paused_trials) or self.debug_log is not None:
            config = self._postprocess_config(configs_all[best_ind])
            if self.debug_log is not None:
                self.debug_log.set_final_config(config)
                self.debug_log.write_block()
        else:
            config = None
        msg = (
            f"*** Scored {len(paused_trials)} paused and {num_new} new configurations. "
        )
        if best_ind < len(paused_trials):
            trial_id, pos, _ = paused_trials[best_ind]
            logger.debug(msg + f"Winner is paused, trial_id = {trial_id}")
            return {"trial_id": trial_id, "pos": pos}
        else:
            logger.debug(msg + f"Winner is new, trial_id = {new_trial_id}")
            return {"config": config}


class DynamicHPOSearcher(BaseSearcher):
    """
    Supports model-based decisions in the DyHPO algorithm proposed by Wistuba
    etal (see
    :class:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DyHPORungSystem`).

    It is *not* recommended to create :class:`DynamicHPOSearcher` searcher
    objects directly, but rather to create
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` objects with
    ``searcher="dyhpo"`` and ``type="dyhpo"``, and passing arguments here in
    ``search_options``. This will use the appropriate functions from
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`` to
    create components in a consistent way.

    This searcher is special, in that it contains a searcher of type
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    Also, its model-based scoring is not triggered by :meth:`get_config`, but
    rather when the scheduler tries to find a trial which can be promoted. At
    this point, :meth:`score_paused_trials_and_new_configs` is called, which
    scores all paused trials along with new configurations. Depending on who
    is the best scorer, a paused trial is resumed, or a trial with a new
    configuration is started. Since all the work is already done in
    :meth:`score_paused_trials_and_new_configs`, the implementation of
    :meth:`get_config` becomes trivial. See also
    :class:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DyHPORungSystem`.
    Extra points:

    * The number of new configurations scored in
      :meth:`score_paused_trials_and_new_configs` is the maximum of
      ``num_init_candidates`` and the number of paused trials scored as well
    * The parameters of the surrogate model are not refit in every call of
      :meth:`score_paused_trials_and_new_configs`, but only when in the last
      recent call, a new configuration was chosen as top scorer. The aim is
      to do refitting in a similar frequency to MOBSTER, where decisions on
      whether to resume a trial are not done in a model-based way.

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` and
     ``type="dyhpo"``. It has the same constructor parameters as
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    Of these, the following are not used, but need to be given valid values:
    ``resource_acq``, ``initial_scoring``, ``skip_local_optimization``.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "min"
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, mode=mode
        )
        if "searcher_int" in kwargs:
            self._searcher_int = kwargs["searcher_int"]
        else:
            assert (
                kwargs.get("model") != "gp_independent"
            ), "model='gp_independent' is not supported"
            self._searcher_int = MyGPMultiFidelitySearcher(
                config_space,
                metric=metric,
                points_to_evaluate=points_to_evaluate,
                **kwargs,
            )
        self._previous_winner_new_trial = True

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers import HyperbandScheduler

        self._searcher_int.configure_scheduler(scheduler)
        err_msg = (
            "This searcher requires HyperbandScheduler scheduler with type='dyhpo'"
        )
        assert isinstance(scheduler, HyperbandScheduler), err_msg
        assert scheduler.scheduler_type == "dyhpo", err_msg

    def get_config(self, **kwargs) -> Optional[dict]:
        assert (
            KEY_NEW_CONFIGURATION in kwargs
        ), f"Internal error: '{KEY_NEW_CONFIGURATION}' argument must be given"
        return kwargs[KEY_NEW_CONFIGURATION]

    def on_trial_result(
        self,
        trial_id: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        update: bool,
    ):
        self._searcher_int.on_trial_result(trial_id, config, result, update)

    def register_pending(
        self,
        trial_id: str,
        config: Optional[dict] = None,
        milestone: Optional[int] = None,
    ):
        self._searcher_int.register_pending(trial_id, config, milestone)

    def remove_case(self, trial_id: str, **kwargs):
        self._searcher_int.remove_case(trial_id, **kwargs)

    def evaluation_failed(self, trial_id: str):
        self._searcher_int.evaluation_failed(trial_id)

    def cleanup_pending(self, trial_id: str):
        self._searcher_int.cleanup_pending(trial_id)

    def dataset_size(self):
        return self._searcher_int.dataset_size()

    def model_parameters(self):
        return self._searcher_int.model_parameters()

    def score_paused_trials_and_new_configs(
        self,
        paused_trials: List[Tuple[str, int, int]],
        min_resource: int,
        new_trial_id: str,
    ) -> Dict[str, Any]:
        """
        This method computes acquisition scores for a number of extended
        configs :math:`(x, r)`. The acquisition score :math:`EI(x | r)` is
        expected improvement (EI) at resource level :math:`r`. Here, the
        incumbent used in EI is the best value attained at level :math:`r`,
        or the best value overall if there is no data yet at that level.
        There are two types of configs being scored:

        * Paused trials: Passed by ``paused_trials`` as tuples
          ``(trial_id, resource)``, where ``resource`` is the level to be
          attained by the trial if it was resumed
        * New configurations drawn at random. For these, the score is EI
          at :math:`r` equal to ``min_resource``

        We return a dictionary. If a paused trial wins, its ``trial_id`` is
        returned with key "trial_id". If a new configuration wins, this
        configuration is returned with key "config".

        Note: As long as the internal searcher still returns configs from
        ``points_to_evaluate`` or drawn at random, this method always returns
        this config with key "config". Scoring and considering paused trials
        is only done afterwards.

        :param paused_trials: See above. Can be empty
        :param min_resource: Smallest resource level
        :param new_trial_id: ID of new trial to be started in case a new
            configuration wins
        :return: Dictionary, see above
        """
        result = self._searcher_int.score_paused_trials_and_new_configs(
            paused_trials=paused_trials,
            min_resource=min_resource,
            new_trial_id=new_trial_id,
            skip_optimization=not self._previous_winner_new_trial,
        )
        self._previous_winner_new_trial = "config" in result
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "searcher_int": self._searcher_int.get_state(),
            "previous_winner_new_trial": self._previous_winner_new_trial,
        }

    def _restore_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError

    def clone_from_state(self, state: Dict[str, Any]):
        searcher_int = self._searcher_int.clone_from_state(state["searcher_int"])
        return DynamicHPOSearcher(
            self.config_space,
            metric=self._metric,
            mode=self._mode,
            searcher_int=searcher_int,
        )

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return self._searcher_int.debug_log
