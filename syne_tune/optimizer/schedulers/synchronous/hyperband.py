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
from typing import Optional, Dict, List
from dataclasses import dataclass
import logging
import numpy as np

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager \
    import SynchronousHyperbandBracketManager
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system \
    import RungSystemsPerBracket
from syne_tune.optimizer.scheduler import TrialSuggestion, \
    SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import ResourceLevelsScheduler
from syne_tune.backend.trial_status import Trial
from syne_tune.search_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults, Categorical, String, \
    assert_no_invalid_options, Integer
from syne_tune.optimizer.schedulers.random_seeds import \
    RandomSeedGenerator
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import \
    searcher_factory

__all__ = ['SynchronousHyperbandScheduler']

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    'searcher', 'search_options', 'metric', 'mode', 'points_to_evaluate',
    'random_seed', 'max_resource_attr', 'resource_attr', 'batch_size',
    'searcher_data'}

_DEFAULT_OPTIONS = {
    'searcher': 'random',
    'mode': 'min',
    'resource_attr': 'epoch',
    'searcher_data': 'rungs',
}

_CONSTRAINTS = {
    'metric': String(),
    'mode': Categorical(choices=('min', 'max')),
    'random_seed': Integer(0, 2 ** 32 - 1),
    'max_resource_attr': String(),
    'resource_attr': String(),
    'batch_size': Integer(1, None),
    'searcher_data': Categorical(('rungs', 'all')),
}


@dataclass
class JobResultWriteBack:
    bracket_id: int
    pos: int
    milestone: int


@dataclass
class JobQueueEntry:
    trial_id: Optional[int]
    write_back: JobResultWriteBack


ERROR_MESSAGE = \
    "In order to use SynchronousHyperbandScheduler, you need to create Tuner " +\
    "with asynchronous_scheduling=False, and make sure that n_workers is the " +\
    "same as batch_size passed to SynchronousHyperbandScheduler."


class SynchronousHyperbandScheduler(ResourceLevelsScheduler):
    """
    Synchronous Hyperband. If W is the number of workers, jobs are scheduled in
    batches of size W. A new batch is scheduled only once all workers are free
    and have returned their final results.

    We use a FIFO queue of maximum size W to react to `suggest` calls, and
    cycle through two phases: suggest and collect (results). The suggest phase
    is started by a `suggest` call when the queue is empty. We query
    `bracket_manager.next_jobs` for W jobs ands fill the queue with these,
    then process and remove the first entry (which leads to a trial being
    started or resumed). Subsequent W - 1 `suggest` calls process and remove
    further entries. Once the queue is empty, we switch into the collect phase.

    In the collect phase, results are processed in `on_trial_result` calls.
    They can come in in any order, but must correspond to jobs for the current
    batch. Only results for the relevant (milestone) levels are used. During
    this phase, the `bracket_to_results` argument for
    `bracket_manager.on_results` is assembled. Once this is complete, this
    method is called, and we switch back to the suggest phase. Any `suggest`
    call during the collect phase leads to an exception.

    The current implementation uses a :class:`RandomSearcher`.
    TODO: Support model-based searchers.

    Parameters
    ----------
    config_space : dict
        Configuration space for trial evaluation function
    bracket_rungs : RungSystemsPerBracket
        Determines rung level systems for each bracket, see
        :class:`SynchronousHyperbandBracketManager`
    searcher : str
        Selects searcher. Passed to `searcher_factory`
    search_options : dict
        Passed to `searcher_factory`
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max'
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
    max_resource_attr : str
        Key name in config for fixed attribute containing the maximum resource.
        If given, trials need not be stopped, which can run more efficiently.
    resource_attr : str
        Name of resource attribute in result's obtained via `on_trial_result`.
        Note: The type of resource must be int.
    batch_size : int
        Jobs are scheduled in batches of this size. All jobs in a batch need
        to have finished before the next scheduling decision is taken. Must be
        equal to `n_workers` in :class:`Tuner`.
    searcher_data : str
        Relevant only if a model-based searcher is used.
        Example: For NN tuning and `resource_attr == epoch', we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:
        - 'rungs' (default): Only results at rung levels. Cheapest
        - 'all': All results. Most expensive
        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to 'all'.

    """
    def __init__(
            self, config_space: Dict,
            bracket_rungs: RungSystemsPerBracket, **kwargs):
        super().__init__(config_space)
        self._create_internal(bracket_rungs, **kwargs)

    def _create_internal(
            self, bracket_rungs: RungSystemsPerBracket, **kwargs):
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name='SynchronousHyperbandScheduler')
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS,
            dict_name='scheduler_options')
        self.metric = kwargs.get('metric')
        assert self.metric is not None, \
            "Argument 'metric' is mandatory. Pass the name of the metric " +\
            "reported by your training script, which you'd like to " +\
            "optimize, and use 'mode' to specify whether it should " +\
            "be minimized or maximized"
        self.mode = kwargs['mode']
        self.max_resource_attr = kwargs.get('max_resource_attr')
        self._resource_attr = kwargs['resource_attr']
        self._batch_size = kwargs.get('batch_size')
        assert self._batch_size is not None, \
            "Argument 'batch_size' mandatory, must be equal to n_workers of Tuner"
        # Generator for random seeds
        random_seed = kwargs.get('random_seed')
        if random_seed is None:
            random_seed = np.random.randint(0, 2 ** 32)
        logger.info(f"Master random_seed = {random_seed}")
        self.random_seed_generator = RandomSeedGenerator(random_seed)
        # Generate searcher
        searcher = kwargs['searcher']
        assert isinstance(searcher, str), \
            f"searcher must be of type string, but has type {type(searcher)}"
        search_options = kwargs.get('search_options')
        if search_options is None:
            search_options = dict()
        else:
            search_options = search_options.copy()
        search_options.update({
            'configspace': self.config_space.copy(),
            'metric': self.metric,
            'points_to_evaluate': kwargs.get('points_to_evaluate'),
            'scheduler_mode': kwargs['mode'],
            'random_seed_generator': self.random_seed_generator,
            'resource_attr': self._resource_attr,
            'batch_size': self._batch_size,
            'scheduler': 'hyperband_synchronous'})
        if searcher == 'bayesopt':
            # We need `max_epochs` in this case
            max_epochs = self._infer_max_resource_level(
                max_resource_level=None, max_resource_attr=self.max_resource_attr)
            assert max_epochs is not None, \
                "If searcher='bayesopt', need to know the maximum resource " +\
                "level. Please provide max_resource_attr argument."
            search_options['max_epochs'] = max_epochs
        self.searcher: BaseSearcher = searcher_factory(
            searcher, **search_options)
        # Bracket manager
        self.bracket_manager = SynchronousHyperbandBracketManager(
            bracket_rungs, mode=self.mode)
        self.searcher_data = kwargs['searcher_data']
        # Queue of jobs to be processed by `_suggest` calls
        self._job_queue = []
        # Current phase ('suggest', 'collect')
        self._phase = 'suggest'  # Not yet started
        self._num_collected = None
        # Maps trial_id to write_back info, to be used in 'collect' phase
        self._trial_to_write_back = dict()
        # Builds argument for `bracket_manager.on_results` call
        self._bracket_to_results = None
        # Maps trial_id (active) to config
        self._trial_to_config = dict()

    def batch_size(self) -> int:
        return self._batch_size

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        assert self._phase == 'suggest', \
            "Cannot process _suggest in collect phase.\n" + ERROR_MESSAGE +\
            f"\nbracket_to_results = {self._bracket_to_results}"
        if not self._job_queue:
            # Queue empty: Fetch new batch of jobs
            self._bracket_to_results = dict()
            for bracket_id, (trials, milestone) \
                    in self.bracket_manager.next_jobs(self._batch_size):
                for pos, promote_trial_id in enumerate(trials):
                    write_back = JobResultWriteBack(
                        bracket_id=bracket_id, pos=pos, milestone=milestone)
                    self._job_queue.append(JobQueueEntry(
                        trial_id=promote_trial_id, write_back=write_back))
                # Will be written in `_suggest_next_job_in_queue`:
                self._bracket_to_results[bracket_id] = [None] * len(trials)
            logger.info(
                f"_suggest: New batch of {self._batch_size} jobs:\n" +
                '\n'.join([str(x) for x in self._job_queue]))
        if len(self._job_queue) == 1:
            # Final job in queue: Switch to collect phase
            self._phase = 'collect'
            self._num_collected = 0
        return self._suggest_next_job_in_queue(trial_id=trial_id)

    def _suggest_next_job_in_queue(
            self, trial_id: int) -> Optional[TrialSuggestion]:
        job = self._job_queue.pop(0)
        write_back = job.write_back
        suggestion = None
        if job.trial_id is not None:
            # Paused trial to be resumed (`trial_id` passed in is ignored)
            trial_id = job.trial_id
            _config = self._trial_to_config[str(trial_id)]
            if self.max_resource_attr is not None:
                config = dict(
                    _config, **{self.max_resource_attr: write_back.milestone})
            else:
                config = _config
            suggestion = TrialSuggestion.resume_suggestion(
                trial_id=trial_id, config=config)
        else:
            # New trial to be started (id is `trial_id` passed in)
            config = self.searcher.get_config(trial_id=str(trial_id))
            if config is not None:
                config = cast_config_values(config, self.config_space)
                if self.max_resource_attr is not None:
                    config[self.max_resource_attr] = write_back.milestone
                self._trial_to_config[str(trial_id)] = config
                suggestion = TrialSuggestion.start_suggestion(config)
        if suggestion is not None:
            results = self._bracket_to_results[write_back.bracket_id]
            assert results[write_back.pos] is None, \
                (trial_id, write_back, results[write_back.pos])
            results[write_back.pos] = (trial_id, None)
            self._trial_to_write_back[str(trial_id)] = write_back
        return suggestion

    def _on_trial_result(
            self, trial: Trial, result: Dict,
            call_searcher: bool = True) -> str:
        if self._phase == 'collect':
            trial_id = str(trial.trial_id)
            write_back = self._trial_to_write_back.get(trial_id)
            assert write_back is not None, \
                f"Trial trial_id = {trial_id} is not pending. " +\
                f"_trial_to_writeback = {self._trial_to_write_back}"
            assert self.metric in result, \
                f"Result for trial_id {trial_id} does not contain " +\
                f"'{self.metric}' field"
            metric_val = float(result[self.metric])
            assert self._resource_attr in result, \
                f"Result for trial_id {trial_id} does not contain " +\
                f"'{self._resource_attr}' field"
            resource = int(result[self._resource_attr])
            milestone = write_back.milestone
            trial_decision = SchedulerDecision.CONTINUE
            if resource >= milestone:
                job_result = self._bracket_to_results[write_back.bracket_id]
                pos = write_back.pos
                assert job_result[pos][0] == trial.trial_id, \
                    (write_back, job_result[pos], trial.trial_id)
                if resource == milestone:
                    job_result[pos] = (trial.trial_id, metric_val)
                    trial_decision = SchedulerDecision.PAUSE
                    self._num_collected += 1
                    if self._num_collected == self._batch_size:
                        # All results have been collected
                        self.bracket_manager.on_results(self._bracket_to_results)
                        self._phase = 'suggest'
                        self._trial_to_write_back = dict()
                else:
                    assert job_result[pos][1] is not None, \
                        f"Trial trial_id {trial_id}: Obtained result for " +\
                        f"resource = {resource}, but not for {milestone}. " +\
                        "Training script must not skip rung levels!"
            if call_searcher:
                update = self.searcher_data == 'all' or resource == milestone
                self.searcher.on_trial_result(
                    trial_id=trial_id, config=self._trial_to_config[trial_id],
                    result=result, update=update)
        else:
            # We may receive results in 'suggest' phase, because trials
            # were not properly paused. These results are simply ignored
            logger.warning(
                f"Received result for trial_id {trial.trial_id} in suggest " +
                f"phase. This result will be ignored:\n{result}")
            trial_decision = SchedulerDecision.STOP
        return trial_decision

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        return self._on_trial_result(trial, result, call_searcher=True)

    def on_trial_error(self, trial: Trial):
        """
        Given the `trial` is currently pending, we send a result at its
        milestone for metric value NaN. Such trials are ranked after all others
        and will most likely not be promoted.

        """
        assert self._phase == 'collect', \
            "Cannot process on_trial_error in suggest phase.\n" + ERROR_MESSAGE +\
            f"\n_job_queue = {self._job_queue}"
        trial_id = str(trial.trial_id)
        self.searcher.evaluation_failed(trial_id)
        write_back = self._trial_to_write_back.get(trial_id)
        if write_back is not None:
            # Reaction to a failed trial is to pass a NaN metric value for
            # its milestone
            result = {
                self._resource_attr: write_back.milestone,
                self.metric: np.NAN}
            self._on_trial_result(trial, result, call_searcher=False)

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
