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
from typing import Optional, Dict, Any
import logging

from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.optimizer.scheduler import TrialSuggestion, TrialScheduler
from syne_tune.optimizer.schedulers.random_seeds import (
    RandomSeedGenerator,
    generate_random_seed,
)
from syne_tune.backend.trial_status import Trial

logger = logging.getLogger(__name__)


class TrialSchedulerWithSearcher(TrialScheduler):
    """
    Base class for trial schedulers which have a
    :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher` member
    ``searcher``. This searcher has a method
    :meth:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher.configure_scheduler`
    which has to be called before the searcher is first used.

    We also collect common code here:

    * Determine ``max_resource_level`` if not explicitly given
    * Master seed, :attr:`random_seed_generator`
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        super().__init__(config_space)
        self._searcher_initialized = False
        # Generator for random seeds
        random_seed = kwargs.get("random_seed")
        if random_seed is None:
            random_seed = generate_random_seed()
        logger.info(f"Master random_seed = {random_seed}")
        self.random_seed_generator = RandomSeedGenerator(random_seed)

    @property
    def searcher(self) -> Optional[BaseSearcher]:
        raise NotImplementedError

    def _initialize_searcher(self):
        """Callback to initialize searcher based on scheduler, if not done already"""
        if not self._searcher_initialized:
            if self.searcher is not None:
                self.searcher.configure_scheduler(self)
            self._searcher_initialized = True

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        self._initialize_searcher()
        return super().suggest(trial_id)

    def on_trial_error(self, trial: Trial):
        self._initialize_searcher()
        if self.searcher is not None:
            trial_id = str(trial.trial_id)
            self.searcher.evaluation_failed(trial_id)
            if self.searcher.debug_log is not None:
                logger.info(f"trial_id {trial_id}: Evaluation failed!")

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        self._initialize_searcher()
        if self.searcher is not None:
            config = self._preprocess_config(trial.config)
            self.searcher.on_trial_result(
                str(trial.trial_id), config, result=result, update=True
            )

    def _infer_max_resource_level_getval(self, name):
        if name in self.config_space and name not in self._hyperparameter_keys:
            return self.config_space[name]
        else:
            return None

    def _infer_max_resource_level(
        self, max_resource_level: Optional[int], max_resource_attr: Optional[str]
    ):
        """Infer ``max_resource_level`` if not explicitly given.

        :param max_resource_level: Value explicitly provided, or None
        :param max_resource_attr: Name of max resource attribute in
            ``self.config_space`` (optional)
        :return: Inferred value for ``max_resource_level``
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
