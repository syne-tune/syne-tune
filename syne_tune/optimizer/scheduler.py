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
from typing import Optional, List
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import non_constant_hyperparameter_keys, cast_config_values

logger = logging.getLogger(__name__)


class SchedulerDecision:
    CONTINUE = "CONTINUE"  #: Status for continuing trial execution
    PAUSE = "PAUSE"  #: Status for pausing trial execution
    STOP = "STOP"  #: Status for stopping trial execution


@dataclass
class TrialSuggestion:
    """Suggestion returned by a scheduler.
    :param spawn_new_trial_id: whether a new trial-id should be used.
    :param checkpoint_trial_id: the checkpoint of the trial-id that should be used.
        If `spawn_new_trial_id` is False, then the trial `checkpoint_trial_id` is
        resumed with its previous checkpoint.
    :param config: the configuration that should be evaluated.
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
        config: dict, checkpoint_trial_id: Optional[int] = None
    ) -> "TrialSuggestion":
        """
        :param config: configuration to use for the new trial.
        :param checkpoint_trial_id: if given, then the checkpoint folder of the
            corresponding trial is used when starting the new trial.
        :return: a trial decision that consists in starting a new trial (which
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
        """
        :param trial_id:
        :param config:
        :return: a trial decision that consists in resuming trial `trial-id`
            with `config` if provided or the previous configuration used if
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

    Note that Ray Tune distributes these decisions between schedulers and
    searchers (see :class:`RayTuneScheduler`).
    """

    def __init__(self, config_space: dict):
        self.config_space = config_space
        self._hyperparameter_keys = set(non_constant_hyperparameter_keys(config_space))

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        """Returns a suggestion for a new trial, or one to be resumed

        This method returns `suggestion` of type `TrialSuggestion` (unless
        there is no config left to explore, and None is returned).

        If `suggestion.spawn_new_trial_id` is True, a new trial is to be
        started with config `suggestion.config`. Typically, this new trial
        is started from scratch. But if `suggestion.checkpoint_trial_id` is
        given, the trial is to be (warm)started from the checkpoint written
        for the trial with this ID. The new trial has ID `trial_id`.

        If `suggestion.spawn_new_trial_id` is False, an existing and currently
        paused trial is to be resumed, whose ID is
        `suggestion.checkpoint_trial_id`. If this trial has a checkpoint, we
        start from there. In this case, `suggestion.config` is optional. If not
        given (default), the config of the resumed trial does not change.
        Otherwise, its config is overwritten by `suggestion.config` (see
        :class:`HyperbandScheduler` with type 'promotion' for an example why
        this can be useful).

        Apart from the HP config, additional fields can be appended to the
        dict, these are passed to the trial function as well.

        :param trial_id: ID for new trial to be started (ignored if existing
            trial to be resumed)
        :return: Suggestion for a trial to be started or to be resumed, see
            above
        """
        ret_val = self._suggest(trial_id)
        if ret_val is not None:
            assert isinstance(ret_val, TrialSuggestion)
            if ret_val.config is not None:
                ret_val = TrialSuggestion(
                    spawn_new_trial_id=ret_val.spawn_new_trial_id,
                    checkpoint_trial_id=ret_val.checkpoint_trial_id,
                    config=self._postprocess_config(ret_val.config),
                )
        return ret_val

    def _postprocess_config(self, config: dict) -> dict:
        """
        Post-processes a config as returned by a searcher. This involves:
        - Adding parameters which are constant, therefore do not feature
            in the config space of the searcher
        - Casting values to types (float, int, str) according to config_space
            value types

        :param config: Config returned by searcher
        :return: Post-processed config
        """
        new_config = self.config_space.copy()
        new_config.update(cast_config_values(config, config_space=self.config_space))
        return new_config

    def _preprocess_config(self, config: dict) -> dict:
        """
        Pre-processes a config before passing it to a searcher. This involves:
        - Removing parameters which are constant in the config space (these do
            not feature in the config space used by the searcher)
        - Casting values to types (float, int, str) according to config_space
            value types

        :param config:
        :return: Pre-processed config, can be passed to searcher
        """
        return cast_config_values(
            {k: v for k, v in config.items() if k in self._hyperparameter_keys},
            config_space=self.config_space,
        )

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        """
        Implements `suggest`, except for basic postprocessing of config.
        Note that the config returned here may also contain values for constant
        parameters in the config space. If so, these values take precedence.
        See :class:`HyperbandScheduler` with `type = 'promotion'` for an
        example how this is used.
        """
        raise NotImplementedError()

    def on_trial_add(self, trial: Trial):
        """Called when a new trial is added to the trial runner.

        Additions are normally triggered by `suggest`.
        """
        pass

    def on_trial_error(self, trial: Trial):
        """Notification for the error of trial."""
        pass

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        """Called on each intermediate result returned by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of CONTINUE, PAUSE, and STOP. This will only be called when the
        trial is in the RUNNING state.

        :param trial:
        :param result:
        :return: trial_decision
        """
        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial: Trial, result: dict):
        """Notification for the completion of trial."""
        pass

    def on_trial_remove(self, trial: Trial):
        """Called to remove trial.
        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call `on_trial_complete`."""
        pass

    def metric_names(self) -> List[str]:
        """
        :return: List of metric names. The first one is the target
            metric optimized over
        """
        raise NotImplementedError()

    def metric_mode(self) -> str:
        """
        :return: 'min' if target metric is minimized, otherwise 'max', 'min' is the default in all schedulers.
        """
        if hasattr(self, "mode"):
            return self.mode
        else:
            raise NotImplementedError()
