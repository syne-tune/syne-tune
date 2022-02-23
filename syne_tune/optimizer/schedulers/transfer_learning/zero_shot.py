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
from typing import Dict

import pandas as pd

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations, TransferLearningMixin


logger = logging.getLogger(__name__)


class ZeroShotTransfer(TransferLearningMixin, BaseSearcher):
    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric: str,
            mode: str = 'min',
            sort_transfer_learning_evaluations: bool = True
    ) -> None:
        """
        A zero-shot transfer hyperparameter optimization method which selects configurations that minimize the average
        rank obtained on historic metadata (transfer_learning_evaluations).

        Reference: Sequential Model-Free Hyperparameter Tuning.
        Martin Wistuba, Nicolas Schilling, Lars Schmidt-Thieme.
        IEEE International Conference on Data Mining (ICDM) 2015.

        :param config_space: configuration space for trial evaluation function.
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param metric: objective name to optimize, must be present in transfer learning evaluations.
        :param mode: whether to minimize (min) or maximize (max)
        :param sort_transfer_learning_evaluations: use False if the hyperparameters for each task in
        transfer_learning_evaluations are already in the same order. If set to True, hyperparameters are sorted.
        """
        super().__init__(config_space=config_space, configspace=config_space,
                         transfer_learning_evaluations=transfer_learning_evaluations, metric=metric,
                         metric_names=[metric], mode=mode)
        self._mode = mode
        warning_message = 'This searcher assumes that each hyperparameter configuration occurs in all tasks. '
        scores = list()
        hyperparameters = None
        for task_name, task_data in transfer_learning_evaluations.items():
            assert hyperparameters is None or task_data.hyperparameters.shape == hyperparameters.shape, warning_message
            hyperparameters = task_data.hyperparameters
            if sort_transfer_learning_evaluations:
                hyperparameters = task_data.hyperparameters.sort_values(list(task_data.hyperparameters.columns))
            idx = hyperparameters.index.values
            scores.append(task_data.objective_values(metric).mean(axis=1)[idx, -1])
        logger.warning(warning_message + 'If this is not the case, this searcher fails without a warning.')
        if not sort_transfer_learning_evaluations:
            hyperparameters = hyperparameters.copy()
        hyperparameters.reset_index(drop=True, inplace=True)
        self._hyperparameters = hyperparameters
        self._scores = pd.DataFrame(scores)
        self._ranks = self._update_ranks()

    def get_config(self, **kwargs) -> Dict:
        if self._ranks.shape[1] == 0:
            return None
        best_idx = self._ranks.mean(axis=0).idxmin()
        self._ranks.clip(upper=self._ranks[best_idx], axis=0, inplace=True)
        self._scores.drop(columns=best_idx, inplace=True)
        best_config = self._hyperparameters.loc[best_idx]
        self._hyperparameters.drop(index=best_idx, inplace=True)
        if self._ranks.std(axis=1).sum() == 0:
            self._ranks = self._update_ranks()
        return best_config.to_dict()

    def _update_ranks(self) -> pd.DataFrame:
        return ((-1 if self._mode == 'max' else 1) * self._scores).rank(axis=1)

    def _update(self, trial_id: str, config: Dict, result: Dict) -> None:
        pass

    def clone_from_state(self, state: dict):
        raise NotImplementedError()
