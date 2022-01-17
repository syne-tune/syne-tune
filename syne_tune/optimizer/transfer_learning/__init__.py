import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from syne_tune.optimizer.scheduler import TrialScheduler


@dataclass
class TransferLearningTaskEvaluations:
    hyperparameters: pd.DataFrame
    metrics: np.array


class TransferLearningScheduler(TrialScheduler):
    def __init__(
            self,
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            metric_names: List[str],
    ):
        super(TransferLearningScheduler, self).__init__(config_space=config_space)
        for task, evals in transfer_learning_evaluations.items():
            for key in config_space.keys():
                assert key in evals.hyperparameters.columns
            assert len(metric_names) == evals.metrics.shape[1]
        self._metric_names = metric_names

    def metric_names(self) -> List[str]:
        return self._metric_names
