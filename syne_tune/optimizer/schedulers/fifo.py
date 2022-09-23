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
from typing import Optional, List
import logging
import os
import numpy as np

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Categorical,
    String,
    Boolean,
    assert_no_invalid_options,
    Integer,
)
from syne_tune.optimizer.schedulers.random_seeds import RandomSeedGenerator
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.backend.time_keeper import TimeKeeper, RealTimeKeeper
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values

__all__ = ["FIFOScheduler", "ResourceLevelsScheduler"]

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "checkpoint",
    "resume",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_t",
    "max_resource_attr",
    "time_keeper",
}

_DEFAULT_OPTIONS = {
    "searcher": "random",
    "resume": False,
    "mode": "min",
}

_CONSTRAINTS = {
    "checkpoint": String(),
    "resume": Boolean(),
    "metric": String(),
    "mode": Categorical(choices=("min", "max")),
    "random_seed": Integer(0, 2**32 - 1),
    "max_t": Integer(1, None),
    "max_resource_attr": String(),
}


class ResourceLevelsScheduler(TrialScheduler):
    def _infer_max_resource_level_getval(self, name):
        if name in self.config_space and name not in self._hyperparameter_keys:
            return self.config_space[name]
        else:
            return None

    def _infer_max_resource_level(
        self, max_resource_level: Optional[int], max_resource_attr: Optional[str]
    ):
        """
        Helper to infer `max_resource_level` if not explicitly given.

        :param max_resource_level: Value explicitly provided, or None
        :param max_resource_attr: Name of max resource attribute in
            `config_space` (optional)
        :return:
        """
        inferred_max_t = None
        names = ("epochs", "max_t", "max_epochs")
        if max_resource_attr is not None:
            names = (max_resource_attr,) + names
        for name in names:
            inferred_max_t = self._infer_max_resource_level_getval(name)
            if inferred_max_t is not None:
                break
        if max_resource_level is not None:
            if inferred_max_t is not None and max_resource_level != inferred_max_t:
                logger.warning(
                    f"max_resource_level = {max_resource_level} is different "
                    f"from the value {inferred_max_t} inferred from "
                    "config_space"
                )
        else:
            # It is OK if max_resource_level cannot be inferred
            if inferred_max_t is not None:
                logger.info(
                    f"max_resource_level = {inferred_max_t}, as inferred "
                    "from config_space"
                )
            max_resource_level = inferred_max_t
        return max_resource_level


class FIFOScheduler(ResourceLevelsScheduler):
    r"""Simple scheduler that just runs trials in submission order.

    Parameters
    ----------
    config_space: dict
        Configuration space for trial evaluation function
    searcher : str or BaseSearcher
        Searcher (get_config decisions). If str, this is passed to
        searcher_factory along with search_options.
    search_options : dict
        If searcher is str, these arguments are passed to searcher_factory.
    checkpoint : str
        If filename given here, a checkpoint of scheduler (and searcher) state
        is written to file every time a job finishes.
        Note: May not be fully supported by all searchers.
    resume : bool
        If True, scheduler state is loaded from checkpoint, and experiment
        starts from there.
        Note: May not be fully supported by all searchers.
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max', default to 'min'.
    points_to_evaluate: list[dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
        Note: If `searcher` is BaseSearcher, points_to_evaluate must be set
        there.
    random_seed : int
        Master random seed. Generators used in the scheduler or searcher are
        seeded using `RandomSeedGenerator`. If not given, the master random
        seed is drawn at random here.
    time_keeper : TimeKeeper
        If passed, this will be used for timing here (see `_elapsed_time`). The
        time keeper has to be started at the beginning of the experiment. If not
        given, we use a local time keeper here, which is started with the first
        call to `_suggest`.
        Can also be set after construction, with `set_time_keeper`.
        NOTE: If you use :class:`SimulatorBackend`, you need to pass its
        `time_keeper` here.
    max_t : int (optional)
        Maximum resource (see resource_attr) to be used for a job. Mandatory
        for multi-fidelity scheduling, and for fine-grained cost-aware
        searchers.
        Note: If this is not given, we try to infer its value from `config_space`,
        checking `config_space['epochs']`, `config_space['max-t']`, and
        `config_space['max-epochs']`. If `max_resource_attr` is given, we use
        the value `config_space[max_resource_attr]`. But if `max_t` is given
        here, it takes precedence.
    max_resource_attr : str (optional)
        Key name in config for fixed attribute containing the maximum resource.
        Mandatory for promotion-based multi-fidelity scheduling (see
        :class:`HyperbandScheduler`, type 'promotion'). If given here, it is
        used to infer `max_t` if not given.

    """

    def __init__(self, config_space: dict, **kwargs):
        super().__init__(config_space)
        # Check values and impute default values
        assert_no_invalid_options(kwargs, _ARGUMENT_KEYS, name="FIFOScheduler")
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        metric = kwargs.get("metric")
        assert metric is not None, (
            "Argument 'metric' is mandatory. Pass the name of the metric "
            + "reported by your training script, which you'd like to "
            + "optimize, and use 'mode' to specify whether it should "
            + "be minimized or maximized"
        )
        self.metric = metric
        self.mode = kwargs["mode"]
        self.max_resource_attr = kwargs.get("max_resource_attr")
        # Setting max_t (if not provided as argument -> self.max_t)
        # This value can often be obtained from config_space. We check these
        # attributes (in order): epochs, max_t, max_epochs.
        # In any case, the max_t argument takes precedence. If it is None, we use
        # the one inferred from config_space.
        self.max_t = self._infer_max_resource_level(
            kwargs.get("max_t"), self.max_resource_attr
        )
        # Generator for random seeds
        random_seed = kwargs.get("random_seed")
        if random_seed is None:
            random_seed = np.random.randint(0, 2**32)
        logger.info(f"Master random_seed = {random_seed}")
        self.random_seed_generator = RandomSeedGenerator(random_seed)
        # Generate searcher
        searcher = kwargs["searcher"]
        if isinstance(searcher, str):
            search_options = kwargs.get("search_options")
            if search_options is None:
                search_options = dict()
            else:
                search_options = search_options.copy()
            search_options.update(
                {
                    "config_space": self.config_space.copy(),
                    "metric": self.metric,
                    "points_to_evaluate": kwargs.get("points_to_evaluate"),
                    "mode": kwargs["mode"],
                    "random_seed_generator": self.random_seed_generator,
                }
            )
            if self.max_t is not None:
                search_options["max_epochs"] = self.max_t
            # Subclasses may extend `search_options`
            search_options = self._extend_search_options(search_options)
            # Adjoin scheduler info to search_options, if not already done by
            # subclass (via `_extend_search_options`)
            if "scheduler" not in search_options:
                search_options["scheduler"] = "fifo"
            self.searcher: BaseSearcher = searcher_factory(searcher, **search_options)
        else:
            assert isinstance(searcher, BaseSearcher)
            self.searcher: BaseSearcher = searcher

        checkpoint = kwargs.get("checkpoint")
        self._checkpoint = checkpoint
        self._start_time = None  # Will be set at first `suggest`
        self._searcher_initialized = False
        # Resume experiment from checkpoint?
        if kwargs["resume"]:
            assert checkpoint is not None, "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                raise NotImplementedError()
                # TODO: Need load
                # self.load_state_dict(load(checkpoint))
            else:
                msg = f"checkpoint path {checkpoint} is not available for resume."
                logger.exception(msg)
                raise FileExistsError(msg)
        # Time keeper
        time_keeper = kwargs.get("time_keeper")
        if time_keeper is not None:
            self.set_time_keeper(time_keeper)
        else:
            self.time_keeper = None

    def set_time_keeper(self, time_keeper: TimeKeeper):
        """
        Allows to assign the time keeper after instruction. This is possible
        only if it was not assigned there already, and the experiment has
        not yet started.
        """
        assert self.time_keeper is None, "Time keeper has already been assigned"
        assert isinstance(
            time_keeper, TimeKeeper
        ), "Argument must be of type TimeKeeper"
        self.time_keeper = time_keeper

    def _extend_search_options(self, search_options: dict) -> dict:
        return search_options

    def _initialize_searcher(self):
        if not self._searcher_initialized:
            self.searcher.configure_scheduler(self)
            self._searcher_initialized = True

    def save(self, checkpoint=None):
        """Save Checkpoint"""
        if checkpoint is None:
            checkpoint = self._checkpoint
        if checkpoint is not None:
            raise NotImplementedError()
            # TODO: Need mkdir, save
            # mkdir(os.path.dirname(checkpoint))
            # save(self.state_dict(), checkpoint)

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        self._initialize_searcher()
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        # For pause/resume schedulers: Can a paused trial be promoted?
        promote_trial_id, extra_kwargs = self._promote_trial()
        if promote_trial_id is not None:
            promote_trial_id = int(promote_trial_id)
            return TrialSuggestion.resume_suggestion(
                trial_id=promote_trial_id, config=extra_kwargs
            )
        # Ask searcher for config of new trial to start
        extra_kwargs["elapsed_time"] = self._elapsed_time()
        trial_id = str(trial_id)
        config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = self._on_config_suggest(config, trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(config)
        return config

    def _on_config_suggest(self, config: dict, trial_id: str, **kwargs) -> dict:
        # We register the config here, not in `on_trial_add`. While this risks
        # registering a config which is not successfully started, this is the
        # right thing to do for batch suggestions. There, `suggest` is called
        # multiple times in a row, and the batch trials are started together.
        # If we did not register pending configs after being suggested (but
        # before getting started), fantasizing would not be used for them.
        self.searcher.register_pending(trial_id=trial_id, config=config)
        if self.searcher.debug_log is not None:
            # For log outputs:
            config = dict(config, trial_id=trial_id)
        return config

    def _promote_trial(self) -> (Optional[str], Optional[dict]):
        """
        Has to be implemented by pause/resume schedulers.
        If a trial can be promoted, its trial_id is returned, otherwise None.

        The second return argument, extra_kwargs, plays different roles
        depending on the first return argument:
        - If trial_id = None (no promotion): extra_kwargs are args to be
            passed to `get_config` call of searcher.
        - If trial_id not None (promotion): extra_kwargs may be None or a dict.
            If a dict, extra_kwargs is used to update the config of the
            trial to be promoted. In this case, `FIFOScheduler.suggest` will
            return the tuple (trial_id, extra_kwargs).

        :return: trial_id, extra_kwargs
        """
        return None, dict()

    def _elapsed_time(self):
        """
        :return: Time elapsed since start of experiment (see 'run')
        """
        assert self.time_keeper is not None, "Experiment has not been started yet"
        return self.time_keeper.time()

    def on_trial_error(self, trial: Trial):
        self._initialize_searcher()
        trial_id = str(trial.trial_id)
        self.searcher.evaluation_failed(trial_id)
        if self.searcher.debug_log is not None:
            logger.info(f"trial_id {trial_id}: Evaluation failed!")

    def _check_key_of_result(self, result: dict, key: str):
        assert key in result, (
            "Your training evaluation function needs to report values "
            + f"for the key {key}:\n   report({key}=..., ...)"
        )

    def _check_result(self, result: dict):
        self._check_key_of_result(result, self.metric)

    # Not doing much. Note the result at the end of the trial run is
    # passed to `on_trial_complete`
    def on_trial_result(self, trial: Trial, result: dict) -> str:
        self._check_result(result)
        trial_id = str(trial.trial_id)
        trial_decision = SchedulerDecision.CONTINUE
        if len(result) == 0:
            # An empty dict should just be skipped
            if self.searcher.debug_log is not None:
                logger.info(f"trial_id {trial_id}: Skipping empty result")
        else:
            config = self._preprocess_config(trial.config)
            self.searcher.on_trial_result(trial_id, config, result=result, update=False)
            # Extra info in debug mode
            log_msg = f"trial_id {trial_id} (metric = {result[self.metric]:.3f}"
            for k, is_float in (("epoch", False), ("elapsed_time", True)):
                if k in result:
                    if is_float:
                        log_msg += f", {k} = {result[k]:.2f}"
                    else:
                        log_msg += f", {k} = {result[k]}"
            log_msg += f"): decision = {trial_decision}"
            logger.debug(log_msg)
        return trial_decision

    def on_trial_complete(self, trial: Trial, result: dict):
        if len(result) > 0:
            self._initialize_searcher()
            config = self._preprocess_config(trial.config)
            self.searcher.on_trial_result(
                str(trial.trial_id), config, result=result, update=True
            )

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
