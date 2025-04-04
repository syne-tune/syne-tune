import copy
import logging

import numpy as np
from typing import Optional, List, Dict, Any


try:
    from torch import Tensor
    import torch
    from botorch.utils.transforms import normalize
except ImportError as e:
    logging.debug(e)

import syne_tune.config_space as sp
from syne_tune.optimizer.legacy_baselines import BoTorch
from syne_tune.optimizer.schedulers.searchers.botorch.legacy_botorch_searcher import (
    LegacyBoTorchSearcher,
)
from syne_tune.optimizer.legacy_baselines import _create_searcher_kwargs

from syne_tune.optimizer.schedulers.transfer_learning import (
    LegacyTransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)


def parse_value(val):
    if isinstance(val, np.int64):
        return int(val)
    else:
        return val


def configs_from_df(df) -> List[dict]:
    return [
        {key: parse_value(df[key][ii]) for key in df.keys()} for ii in range(len(df))
    ]


class BoTorchTransfer(BoTorch):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        new_task_id: Any,
        random_seed: Optional[int] = None,
        encode_tasks_ordinal: bool = False,
        **kwargs,
    ):

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["transfer_learning_evaluations"] = transfer_learning_evaluations
        searcher_kwargs["new_task_id"] = new_task_id
        searcher_kwargs["encode_tasks_ordinal"] = encode_tasks_ordinal
        super(BoTorch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=BoTorchTransferSearcher(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class BoTorchTransferSearcher(LegacyBoTorchSearcher):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        new_task_id: Any,
        points_to_evaluate: Optional[List[dict]] = None,
        allow_duplicates: bool = False,
        num_init_random: int = 0,
        encode_tasks_ordinal: bool = False,
        **kwargs,
    ):
        self._transfer_learning_evaluations = transfer_learning_evaluations
        self._new_task_id = new_task_id
        self._transfer_task_order = sorted(transfer_learning_evaluations.keys())
        if encode_tasks_ordinal:
            self._task_space = sp.ordinal(
                list(transfer_learning_evaluations.keys()) + [new_task_id], kind="equal"
            )
        else:
            self._task_space = sp.choice(
                list(transfer_learning_evaluations.keys()) + [new_task_id]
            )
        self._ext_config_space = copy.deepcopy(config_space)
        self._ext_config_space["task"] = self._task_space
        self._ext_hp_ranges = make_hyperparameter_ranges(
            self._ext_config_space,
            name_last_pos="task",
            value_for_last_pos=self._new_task_id,
        )
        self._ext_hp_ranges_for_bounds = make_hyperparameter_ranges(
            self._ext_config_space,
            name_last_pos="task",
        )
        super(BoTorchTransferSearcher, self).__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            **kwargs,
        )
        # Want to sample a few observations at random to start with
        self.num_minimum_observations = num_init_random
        self.num_minimum_observations += len(self.objectives())

    def dataset_size(self):
        num_this_task = len(self.trial_observations)
        num_other_tasks = np.sum(
            [
                len(self._transfer_learning_evaluations[key].hyperparameters)
                for key in self._transfer_learning_evaluations
            ]
        )
        return num_this_task + num_other_tasks

    def objectives(self):
        this_task = super().objectives()
        all_tasks = [this_task]
        for key in self._transfer_task_order:
            obs = self._transfer_learning_evaluations[key].objective_values(
                self._metric
            )
            all_tasks.append(obs.flatten())
        return np.hstack(all_tasks)

    def _extend_config_with_task(self, config: dict, task_val: str) -> dict:
        ext_config = copy.deepcopy(config)
        ext_config["task"] = task_val
        return ext_config

    def _config_to_feature_matrix(
        self, configs: List[dict], task_val: Optional[str] = None
    ) -> Tensor:
        if task_val is None:
            task_val = self._new_task_id
        bounds = Tensor(self._ext_hp_ranges_for_bounds.get_ndarray_bounds()).T
        ext_inp_configs = [
            self._extend_config_with_task(config, task_val)
            if "task" not in config.keys()
            else config
            for config in configs
        ]
        X = Tensor(
            np.array(
                [self._ext_hp_ranges.to_ndarray(config) for config in ext_inp_configs]
            )
        )
        X_normalized = normalize(X, bounds)
        I, J = X.shape
        for jj in range(J):
            for ii in range(I):
                if torch.isnan(X_normalized[ii, jj]):
                    X_normalized[:, jj] = X[:, jj]  # Avoid nan values
        return X_normalized

    def _get_gp_bounds(self):
        return Tensor(self._ext_hp_ranges.get_ndarray_bounds()).T

    def _config_from_ndarray(self, candidate) -> dict:
        return self._ext_hp_ranges.from_ndarray(candidate)

    def _configs_with_results(self) -> List[dict]:
        # List of configs for which we have results
        this_task = [
            self._extend_config_with_task(config, self._new_task_id)
            for config in super()._configs_with_results()
        ]
        all_tasks = this_task
        for key in self._transfer_task_order:
            task_configs = [
                self._extend_config_with_task(config, key)
                for config in configs_from_df(
                    self._transfer_learning_evaluations[key].hyperparameters
                )
            ]
            all_tasks += task_configs
        return all_tasks
