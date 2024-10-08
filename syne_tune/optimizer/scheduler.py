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
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import logging
import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import (
    non_constant_hyperparameter_keys,
    cast_config_values,
    config_space_to_json_dict,
)
#from syne_tune.optimizer.schedulers.searchers.searcher_base import BaseSearcher
from syne_tune.util import dump_json_with_numpy

logger = logging.getLogger(__name__)


class SchedulerDecision:
    """
    Possible return values of :meth:`TrialScheduler.on_trial_result`, signals the
    tuner how to proceed with the reporting trial.

    The difference between :const:`PAUSE` and :const:`STOP` is important. If a
    trial is stopped, it cannot be resumed afterwards. Its checkpoints may be
    deleted. If a trial is paused, it may be resumed in the future, and its
    most recent checkpoint should be retained.
    """

    CONTINUE = "CONTINUE"  #: Status for continuing trial execution
    PAUSE = "PAUSE"  #: Status for pausing trial execution
    STOP = "STOP"  #: Status for stopping trial execution


@dataclass
class TrialSuggestion:
    """Suggestion returned by :meth:`TrialScheduler.suggest`

    :param spawn_new_trial_id: Whether a new ``trial_id`` should be used.
    :param checkpoint_trial_id: Checkpoint of this trial ID should
        be used to resume from. If ``spawn_new_trial_id`` is ``False``, then the
        trial ``checkpoint_trial_id`` is resumed with its previous checkpoint.
    :param config: The configuration which should be evaluated.
    """

    spawn_new_trial_id: bool = True
    checkpoint_trial_id: Optional[int] = None
    config: Optional[dict] = None

    def __post_init__(self):
        if self.spawn_new_trial_id:
            assert (
                self.checkpoint_trial_id is not None or self.config is not None
            ), "Cannot start a new trial without specifying a checkpoint or a config."
        else:
            assert (
                self.checkpoint_trial_id is not None
            ), "A trial-id must be passed to resume a trial."

    @staticmethod
    def start_suggestion(
        config: Dict[str, Any], checkpoint_trial_id: Optional[int] = None
    ) -> "TrialSuggestion":
        """Suggestion to start new trial

        :param config: Configuration to use for the new trial.
        :param checkpoint_trial_id: Use checkpoint of this trial
            when starting the new trial (otherwise, it is started from
            scratch).
        :return: A trial decision that consists in starting a new trial (which
            would receive a new trial-id).
        """
        return TrialSuggestion(
            spawn_new_trial_id=True,
            config=config,
            checkpoint_trial_id=checkpoint_trial_id,
        )

    @staticmethod
    def resume_suggestion(
        trial_id: int, config: Optional[dict] = None
    ) -> "TrialSuggestion":
        """Suggestion to resume a paused trial

        :param trial_id: ID of trial to be resumed (from its checkpoint)
        :param config: Configuration to use for resumed trial
        :return: A trial decision that consists in resuming trial ``trial-id``
            with ``config`` if provided, or the previous configuration used if
            not provided.
        """
        return TrialSuggestion(
            spawn_new_trial_id=False,
            config=config,
            checkpoint_trial_id=trial_id,
        )

    def __str__(self):
        res = f"config {self.config}"
        if self.checkpoint_trial_id is not None:
            res += f" using from trial's checkpoint {self.checkpoint_trial_id}"
        return res


class TrialScheduler:
    """
    Schedulers maintain and drive the logic of an experiment, making decisions
    which configs to evaluate in new trials, and which trials to stop early.

    Some schedulers support pausing and resuming trials. In this case, they
    also drive the decision when to restart a paused trial.

    :param config_space: Configuration space for evaluation function
    :type config_space: Dict[str, Any]
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :type searcher: str or
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param metric: Name of metric to optimize, key in results obtained via
        ``on_trial_result``. For multi-objective schedulers, this can also be a
        list
    :type metric: str or List[str]
    :param mode: "min" if ``metric`` is minimized, "max" if ``metric`` is
        maximized, defaults to "min". This can also be a list if ``metric`` is
        a list
    :type mode: str or List[str], optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list
        can be partially specified, or even be an empty dict. For each
        hyperparameter not specified, the default value is determined using
        a midpoint heuristic.
        If not given, this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
        Note: If ``searcher`` is of type :class:`BaseSearcher`,
        ``points_to_evaluate`` must be set there.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param time_keeper: This will be used for timing here (see
        ``_elapsed_time``). The time keeper has to be started at the beginning
        of the experiment. If not given, we use a local time keeper here,
        which is started with the first call to :meth:`_suggest`. Can also be set
        after construction, with :meth:`set_time_keeper`.
        Note: If you use
        :class:`~syne_tune.backend.simulator_backend.SimulatorBackend`, you need
        to pass its ``time_keeper`` here.
    :type time_keeper: :class:`~syne_tune.backend.time_keeper.TimeKeeper`,
        optional
    """

    def __init__(self, config_space: Dict[str, Any],
                 metric: str,
                 mode: str = 'min',
                 searcher: Any = None,
                 points_to_evaluate=None,
                 random_seed: int = None,
                 **kwargs
                 ):
        if points_to_evaluate is None:
            self.points_to_evaluate = []
        else:
            self.points_to_evaluate = points_to_evaluate
        self.config_space = config_space
        self.metric = metric
        self.mode = mode

        if searcher is None:
            from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import RandomSearcher
            self.searcher = RandomSearcher(config_space=config_space, metric=metric)
        else:
            self.searcher = searcher
        if random_seed is None:
            self.random_seed = np.random.randint(0, 2**31 - 1)
        else:
            self.random_seed = random_seed
        self._hyperparameter_keys = set(non_constant_hyperparameter_keys(config_space))

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        """Returns a suggestion for a new trial, or one to be resumed

        This method returns ``suggestion`` of type :class:`TrialSuggestion` (unless
        there is no config left to explore, and None is returned).

        If ``suggestion.spawn_new_trial_id`` is ``True``, a new trial is to be
        started with config ``suggestion.config``. Typically, this new trial
        is started from scratch. But if ``suggestion.checkpoint_trial_id`` is
        given, the trial is to be (warm)started from the checkpoint written
        for the trial with this ID. The new trial has ID ``trial_id``.

        If ``suggestion.spawn_new_trial_id`` is ``False``, an existing and currently
        paused trial is to be resumed, whose ID is
        ``suggestion.checkpoint_trial_id``. If this trial has a checkpoint, we
        start from there. In this case, ``suggestion.config`` is optional. If not
        given (default), the config of the resumed trial does not change.
        Otherwise, its config is overwritten by ``suggestion.config`` (see
        :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
        ``type="promotion"`` for an example why this can be useful).

        Apart from the HP config, additional fields can be appended to the
        dict, these are passed to the trial function as well.

        :param trial_id: ID for new trial to be started (ignored if existing
            trial to be resumed)
        :return: Suggestion for a trial to be started or to be resumed, see
            above. If no suggestion can be made, None is returned
        """
        
 #       if self.time_keeper is None:
 #           self.time_keeper = RealTimeKeeper()
 #           self.time_keeper.start_of_time()

        # Ask searcher for config of new trial to start
        #extra_kwargs["elapsed_time"] = self._elapsed_time()
        trial_id = str(trial_id)
        config = self.searcher.get_config( trial_id=trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            #config = self._on_config_suggest(config, trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(self._postprocess_config(config))
        return config

    def _postprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processes a config as returned by a searcher

        * Adding parameters which are constant, therefore do not feature
          in the config space of the searcher
        * Casting values to types (float, int, str) according to ``config_space``
          value types

        :param config: Config returned by searcher
        :return: Post-processed config
        """
        new_config = self.config_space.copy()
        new_config.update(cast_config_values(config, config_space=self.config_space))
        return new_config

    def _preprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-processes a config before passing it to a searcher

        * Removing parameters which are constant in ``config_space`` (these do
          not feature in the config space used by the searcher)
        * Casting values to types (float, int, str) according to
          ``config_space`` value types

        :param config: Config coming from the tuner
        :return: Pre-processed config, can be passed to searcher
        """
        return cast_config_values(
            {k: v for k, v in config.items() if k in self._hyperparameter_keys},
            config_space=self.config_space,
        )


    def on_trial_add(self, trial: Trial):
        """Called when a new trial is added to the trial runner.

        Additions are normally triggered by ``suggest``.

        :param trial: Trial to be added
        """
        pass

    def on_trial_error(self, trial: Trial):
        trial_id = str(trial.trial_id)
        self.searcher.evaluation_failed(trial_id)
        logger.warning(f"trial_id {trial_id}: Evaluation failed!")

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """Called on each intermediate result reported by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of :const:`SchedulerDecision.CONTINUE`,
        :const:`SchedulerDecision.PAUSE`, or :const:`SchedulerDecision.STOP`.
        This will only be called when the trial is currently running.

        :param trial: Trial for which results are reported
        :param result: Result dictionary
        :return: Decision what to do with the trial
        """
        config = self._preprocess_config(trial.config)
        self.searcher.on_trial_result(str(trial.trial_id), config, result=result, update=False)
        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Notification for the completion of trial.

        Note that :meth:`on_trial_result` is called with the same result before.
        However, if the scheduler only uses one final report from each
        trial, it may ignore :meth:`on_trial_result` and just use ``result`` here.

        :param trial: Trial which is completing
        :param result: Result dictionary
        """
        config = self._preprocess_config(trial.config)
        self.searcher.on_trial_result(
            str(trial.trial_id), config, result=result, update=True
        )

    def on_trial_remove(self, trial: Trial):
        """Called to remove trial.

        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call :meth:`on_trial_complete`.

        :param trial: Trial to be removed
        """
        pass

    def metric_names(self) -> List[str]:
        """
        :return: List of metric names. The first one is the target
            metric optimized over, unless the scheduler is a genuine
            multi-objective metric (for example, for sampling the Pareto front)
        """
        return self.metric if isinstance(self.metric, list) else [self.metric]

    def metric_mode(self) -> Union[str, List[str]]:
        """
        :return: "min" if target metric is minimized, otherwise "max".
            Here, "min" should be the default. For a genuine multi-objective
            scheduler, a list of modes is returned
        """
        return self.mode

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        return {
            "metric_names": self.metric_names(),
            "metric_mode": self.metric_mode(),
            "scheduler_name": str(self.__class__.__name__),
            "config_space": config_space_json,
        }

    def is_multiobjective_scheduler(self) -> bool:
        """
        Return True if a scheduler is multi-objective.
        """
        return True if (isinstance(self.metric, list) and len(self.metric) > 1) else False
