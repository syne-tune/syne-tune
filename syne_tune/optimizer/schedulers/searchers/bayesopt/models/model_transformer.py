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
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Callable, Union
import logging
import copy

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes \
    import SurrogateModel, SurrogateOutputModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt \
    import SkipOptimizationPredicate, NeverSkipPredicate
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log \
    import DebugLogPrinter
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration, PendingEvaluation, CandidateEvaluation, \
    dictionarize_objective, INTERNAL_METRIC_NAME
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler

logger = logging.getLogger(__name__)


def _assert_same_keys(dict1, dict2):
    assert set(dict1.keys()) == set(dict2.keys()), \
        f'{list(dict1.keys())} and {list(dict2.keys())} need to be the same keys. '


class TransformerModelFactory(ABC):
    """
    Interface for model factories used in :class:`ModelStateTransformer`. A model
    factory provides access to tunable model parameters, and `model` creates
    :class:`SurrogateModel` instances.

    """
    @abstractmethod
    def get_params(self) -> Dict:
        """
        :return: Current tunable model parameters
        """
        pass

    @abstractmethod
    def set_params(self, param_dict: Dict):
        """
        :param param_dict: New model parameters
        """
        pass

    @abstractmethod
    def model(self, state: TuningJobState, fit_params: bool) -> SurrogateModel:
        """
        Creates a `SurrogateModel` based on data in `state`. This involves
        fitting model parameters if `fit_params` is True. Otherwise, the current
        model parameters are not changed (so may be stale, given that `state`
        has changed). The idea is that often, model fitting is much more
        expensive than just creating the final `SurrogateModel` (posterior
        state). It then makes sense to partly work with stale model parameters.

        :param state: Current data model parameters are to be fit on, and the
            posterior state is to be computed from
        :param fit_params: See above
        :return: SurrogateModel, wrapping the posterior state for predictions
        """
        pass

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return None

    @property
    def profiler(self) -> Optional[SimpleProfiler]:
        return None


# Convenience types allowing for multi-output HPO. These are used for methods that work both in the standard case
# of a single output model and in the multi-output case
TransformerOutputModelFactory = Union[TransformerModelFactory, Dict[str, TransformerModelFactory]]

SkipOptimizationOutputPredicate = Union[SkipOptimizationPredicate, Dict[str, SkipOptimizationPredicate]]


class ModelStateTransformer(object):
    """
    This class maintains the :class:`TuningJobState` alongside an HPO
    experiment, and manages the reaction to changes of this state.
    In particular, it provides a :class:`SurrogateModel` on demand, which
    encapsulates the GP posterior.

    The state transformer is generic, it uses :class:`TransformerModelFactory`
    for anything specific to the model type.

    `skip_optimization` is a predicate depending on the state, determining
    what is done at the next recent call of `model`. If False, the model
    parameters are refit, otherwise the current ones are not changed (which
    is usually faster, but risks stale-ness).

    We also track the observed data `state.candidate_evaluations`. If this
    did not change since the last recent `model` call, we do not refit the
    model parameters. This is based on the assumption that model parameter
    fitting only depends on `state.candidate_evaluations` (observed data),
    not on other fields (e.g., pending evaluations).

    Note that model_factory and skip_optimization can also be a dictionary mapping
    output names to models. In that case, the state is shared but the models for each
    output metric are updated independently.

    """
    def __init__(
            self, model_factory: TransformerOutputModelFactory,
            init_state: TuningJobState,
            skip_optimization: Optional[SkipOptimizationOutputPredicate] = None):
        self._use_single_model = False
        if isinstance(model_factory, TransformerModelFactory):
            self._use_single_model = True
        if not self._use_single_model:
            assert isinstance(model_factory, Dict), f'{model_factory} is not an instance of TransformerModelFactory. ' \
                                                    f'It is assumed that we are in the multi-output case and that it ' \
                                                    f'must be a Dict. No other types are supported. '
            _assert_same_keys(model_factory, skip_optimization)
            # Default: Always refit model parameters for each output model
            if skip_optimization is None:
                skip_optimization = {
                    output_name: NeverSkipPredicate()
                    for output_name in model_factory.keys()}
            else:
                assert isinstance(skip_optimization, Dict), \
                    f'{skip_optimization} must be a Dict, consistently ' \
                    f'with {model_factory}.'
                skip_optimization = {
                    output_name: skip_optimization[output_name] \
                        if skip_optimization.get(output_name) is not None \
                        else NeverSkipPredicate()
                    for output_name in model_factory.keys()}
            # debug_log is shared by all output models
            self._debug_log = next(iter(model_factory.values())).debug_log
        else:
            if skip_optimization is None:
                # Default: Always refit model parameters
                skip_optimization = NeverSkipPredicate()
            assert isinstance(skip_optimization, SkipOptimizationPredicate)
            self._debug_log = model_factory.debug_log
            # Make model_factory and skip_optimization single-key dictionaries
            # for convenience, so that we can treat the single model and multi-model case in the same way
            model_factory = dictionarize_objective(model_factory)
            skip_optimization = dictionarize_objective(skip_optimization)
        self._model_factory = model_factory
        self.skip_optimization = skip_optimization
        self._state = copy.copy(init_state)
        # SurrogateOutputModel computed on demand
        self._model: SurrogateOutputModel = None
        # Observed data for which model parameters were re-fit most
        # recently, separately for each model
        self._num_evaluations = {
            output_name: 0 for output_name in model_factory.keys()}

    @property
    def state(self) -> TuningJobState:
        return self._state

    @property
    def model_factory(self) -> TransformerOutputModelFactory:
        if self._use_single_model:
            return next(iter(self._model_factory.values()))
        else:
            return self._model_factory

    def model(self, **kwargs) -> SurrogateOutputModel:
        """
        If skip_optimization is given, it overrides the self.skip_optimization
        predicate.

        :return: SurrogateModel for current state in the standard single model case;
                 in the multi-model case, it returns a dictionary mapping output names
                 to SurrogateModel instances for current state (shared across models).
        """
        if self._model is None:
            skip_optimization = kwargs.get('skip_optimization')
            self._compute_model(skip_optimization=skip_optimization)
        if self._use_single_model:
            # in this case output_models is a single-key dictionary,
            # so we extract the only value it contains
            model = next(iter(self._model.values()))
        else:
            model = self._model
        return model

    def get_params(self):
        params = {output_name: output_model.get_params()
                  for output_name, output_model in self._model_factory.items()}
        if self._use_single_model:
            assert len(params) == 1
            # in this case params is a single-key dictionary,
            # so we extract the only value it contains
            params = next(iter(params.values()))
        return params

    def set_params(self, param_dict):
        if self._use_single_model:
            param_dict = dictionarize_objective(param_dict)
        _assert_same_keys(self._model_factory, param_dict)
        for output_name in self._model_factory:
            self._model_factory[output_name].set_params(param_dict[output_name])

    def append_candidate(self, candidate: Configuration):
        """
        Appends new pending candidate to the state.

        :param candidate: New pending candidate

        """
        self._model = None  # Invalidate
        self._state.pending_evaluations.append(PendingEvaluation(candidate))

    # Note: Comparison by x.candidate == candidate does not work
    # here.
    def _find_candidate(self, candidate: Configuration, lst: List):
        def _to_tuple(config):
            return self._state.hp_ranges.config_to_tuple(config)

        cand_tpl = _to_tuple(candidate)
        try:
            pos = next(
                i for i, x in enumerate(lst)
                if _to_tuple(x.candidate) == cand_tpl)
        except StopIteration:
            pos = -1
        return pos

    def drop_pending_candidate(self, candidate: Configuration) -> bool:
        """
        Drop pending candidate from state. If the candidate is not listed as
        pending, nothing is done

        :param candidate: Configuration to be dropped
        :return: Was pending candidate dropped?

        """
        pos = self._find_candidate(candidate, self._state.pending_evaluations)
        if pos != -1:
            self._model = None  # Invalidate
            self._state.pending_evaluations.pop(pos)
            if self._debug_log is not None:
                deb_msg = "[ModelStateTransformer.drop_candidate]\n"
                deb_msg += ("- len(pending_evaluations) afterwards = {}\n".format(
                    len(self.state.pending_evaluations)))
                logger.info(deb_msg)
        return (pos != -1)

    def remove_observed_case(
            self, config: Configuration,
            metric_name: str = INTERNAL_METRIC_NAME, key: str = None):
        pos = self._state.pos_of_config(config)
        assert pos is not None, \
            f"config {config} not contained in state.candidate_evaluations"
        metrics = self._state.candidate_evaluations[pos].metrics
        assert metric_name in metrics, \
            f"state.candidate_evaluations entry for config {config} does not " +\
            f"contain metric {metric_name}"
        if key is None:
            del metrics[metric_name]
        else:
            metric_vals = metrics[metric_name]
            assert isinstance(metric_vals, dict) and key in metric_vals, \
                f"state.candidate_evaluations entry for config {config} " +\
                f"and metric {metric_name} does not contain case for " +\
                f"key {key}"
            del metric_vals[key]

    def _label_candidate_info_message(self, config):
        trial_id = self._debug_log.trial_id(config)
        if trial_id is not None:
            logger.info(
                f"Labeling {trial_id} without finding a pending evaluation")
        else:
            logger.info(
                "Labeling this config without finding a pending evaluation:\n" +\
                str(config))

    def label_candidate(self, data: CandidateEvaluation):
        """
        Adds observed data for a candidate. The config may already exist in
        the state, in which case the new labels are appended. Otherwise, a
        new config is appended.
        Note that `data.metrics` can be partial. Its information is used to
        update the entry for `data.candidate` if already present.
        If the candidate was pending before, it is removed as pending
        candidate.

        :param data: New labeled candidate

        """
        # Drop pending candidate if it exists
        config = data.candidate
        metric_vals = data.metrics.get(INTERNAL_METRIC_NAME)
        if metric_vals is not None:
            resource_attr_name = self._state.hp_ranges.name_last_pos
            if resource_attr_name is not None:
                assert isinstance(metric_vals, dict), \
                    f"Metric {INTERNAL_METRIC_NAME} must be dict-valued"
                for resource in metric_vals.keys():
                    config_ext = dict(
                        config, **{resource_attr_name: int(resource)})
                    if not self.drop_pending_candidate(config_ext) \
                            and self._debug_log is not None:
                        self._label_candidate_info_message(config_ext)
            elif not self.drop_pending_candidate(config) \
                    and self._debug_log is not None:
                self._label_candidate_info_message(config)
        # Assign / append new labels
        metrics = self._state.metrics_for_config(config)
        for name, new_labels in data.metrics.items():
            if name not in metrics or not isinstance(new_labels, dict):
                metrics[name] = new_labels
            else:
                metrics[name].update(new_labels)
        self._model = None  # Invalidate

    def filter_pending_evaluations(
            self, filter_pred: Callable[[PendingEvaluation], bool]):
        """
        Filters state.pending_evaluations with filter_pred.

        :param filter_pred Filtering predicate

        """
        new_pending_evaluations = list(filter(
            filter_pred, self._state.pending_evaluations))
        if len(new_pending_evaluations) != len(self._state.pending_evaluations):
            if self._debug_log is not None:
                deb_msg = "[ModelStateTransformer.filter_pending_evaluations]\n"
                deb_msg += ("- from len {} to {}".format(
                    len(self.state.pending_evaluations), len(new_pending_evaluations)))
                logger.info(deb_msg)
            self._model = None  # Invalidate
            del self._state.pending_evaluations[:]
            self._state.pending_evaluations.extend(new_pending_evaluations)

    def mark_candidate_failed(self, candidate: Configuration):
        self._state.failed_candidates.append(candidate)

    def _compute_model(self, skip_optimization=None):
        if skip_optimization is None:
            skip_optimization = dict()
            for output_name, output_skip_optimization in self.skip_optimization.items():
                skip_optimization[output_name] = output_skip_optimization(self._state)
        elif self._use_single_model:
            skip_optimization = dictionarize_objective(skip_optimization)
        if self._debug_log is not None:
            for output_name, skip_opt in skip_optimization.items():
                if skip_opt:
                    logger.info(f"Skipping the refitting of model parameters for {output_name}")

        _assert_same_keys(skip_optimization, self._model_factory)
        output_models = dict()
        for output_name, output_skip_optimization in skip_optimization.items():
            fit_params = not output_skip_optimization
            if fit_params:
                # Did the labeled data really change since the last recent refit?
                # If not, skip the refitting
                num_evaluations = self._state.num_observed_cases(output_name)
                if num_evaluations == self._num_evaluations[output_name]:
                    fit_params = False
                    if self._debug_log is not None:
                        logger.info(
                            f"Skipping the refitting of model parameters for {output_name}, "
                            f"since the labeled data did not change since the last recent fit")
                else:
                    # Model will be refitted: Update
                    self._num_evaluations[output_name] = num_evaluations
            output_models[output_name] = self._model_factory[output_name].model(
                state=self._state, fit_params=fit_params)
        self._model = output_models
