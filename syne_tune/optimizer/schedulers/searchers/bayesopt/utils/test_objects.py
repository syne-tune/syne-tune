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
# Could eventually remove this code: Is this needed in unit tests?

"""
Object definitions that are used for testing.
"""

from typing import Iterator, Tuple, Dict, List, Optional, Union
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Hyperparameter,
    Configuration,
)
from syne_tune.config_space import loguniform, randint, choice, uniform
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    PendingEvaluation,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    MCMCConfig,
    OptimizationConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression import (
    GaussianProcessRegression,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gpr_mcmc import (
    GPRegressionMCMC,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    Matern52,
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    kernel_with_warping,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList


def build_kernel(state: TuningJobState, do_warping: bool = False) -> KernelFunction:
    hp_ranges = state.hp_ranges
    dims = hp_ranges.ndarray_size
    kernel = Matern52(dims, ARD=True)
    if do_warping:
        kernel = kernel_with_warping(kernel, hp_ranges)
    return kernel


def default_gpmodel(
    state: TuningJobState, random_seed: int, optimization_config: OptimizationConfig
) -> GaussianProcessRegression:
    return GaussianProcessRegression(
        kernel=build_kernel(state),
        optimization_config=optimization_config,
        random_seed=random_seed,
    )


def default_gpmodel_mcmc(
    state: TuningJobState, random_seed: int, mcmc_config: MCMCConfig
) -> GPRegressionMCMC:
    return GPRegressionMCMC(
        build_kernel=lambda: build_kernel(state),
        mcmc_config=mcmc_config,
        random_seed=random_seed,
    )


class RepeatedCandidateGenerator(CandidateGenerator):
    """Generates candidates from a fixed set. Used to test the deduplication logic."""

    def __init__(self, n_unique_candidates: int):
        self.config_space = {
            "a": uniform(0, n_unique_candidates),
            "b": randint(0, n_unique_candidates),
            "c": choice([f"value_{i}" for i in range(n_unique_candidates)]),
        }
        self.hp_ranges = make_hyperparameter_ranges(self.config_space)
        self.all_unique_candidates = [
            {"a": 1.0 * j, "b": j, "c": f"value_{j}"}
            for j in range(n_unique_candidates)
        ]

    def generate_candidates(self) -> Iterator[Configuration]:
        i = 0
        while True:
            i += 1
            yield self.all_unique_candidates[i % len(self.all_unique_candidates)]


# Example black box function, with adjustable location of global minimum.
# Potentially could catch issues with optimizer, e.g. if the optimizer
# ignoring somehow candidates on the edge of search space.
# A simple quadratic function is used.
class Quadratic3d:
    def __init__(self, local_minima, active_metric, metric_names):
        # local_minima: point where local_minima is located
        self.local_minima = np.array(local_minima).astype("float")
        self.local_minima[0] = np.log10(self.local_minima[0])
        self.active_metric = active_metric
        self.metric_names = metric_names

    @property
    def search_space(self):
        config_space = {
            "x": loguniform(1.0, 100.0),
            "y": randint(0, 2),
            "z": choice(["0.0", "1.0", "2.0"]),
        }
        return make_hyperparameter_ranges(config_space)

    @property
    def f_min(self):
        return 0.0

    def __call__(self, candidate):
        p = np.array([float(hp) for hp in candidate])
        p[0] = np.log10(p[0])
        return dictionarize_objective(np.sum((self.local_minima - p) ** 2))


def tuples_to_configs(
    config_tpls: List[Tuple[Hyperparameter, ...]], hp_ranges: HyperparameterRanges
) -> List[Configuration]:
    """
    Many unit tests write configs as tuples.

    """
    return [hp_ranges.tuple_to_config(x) for x in config_tpls]


def create_exclusion_set(
    candidates_tpl, hp_ranges: HyperparameterRanges, is_dict: bool = False
) -> ExclusionList:
    """
    Creates exclusion list from set of tuples.

    """
    if not is_dict:
        candidates_tpl = tuples_to_configs(candidates_tpl, hp_ranges)
    return ExclusionList(hp_ranges=hp_ranges, configurations=candidates_tpl)


TupleOrDict = Union[tuple, dict]


def create_tuning_job_state(
    hp_ranges: HyperparameterRanges,
    cand_tuples: List[TupleOrDict],
    metrics: List[Dict],
    pending_tuples: Optional[List[TupleOrDict]] = None,
    failed_tuples: Optional[List[TupleOrDict]] = None,
) -> TuningJobState:
    """
    Builds ``TuningJobState`` from basics, where configs are given as tuples or
    as dicts.

    NOTE: We assume that all configs in the different lists are different!

    """
    if cand_tuples and isinstance(cand_tuples[0], tuple):
        configs = tuples_to_configs(cand_tuples, hp_ranges)
    else:
        configs = cand_tuples
    trials_evaluations = [
        TrialEvaluations(trial_id=str(trial_id), metrics=y)
        for trial_id, y in enumerate(metrics)
    ]
    pending_evaluations = None
    if pending_tuples is not None:
        sz = len(configs)
        extra = len(pending_tuples)
        if pending_tuples and isinstance(pending_tuples[0], tuple):
            extra_configs = tuples_to_configs(pending_tuples, hp_ranges)
        else:
            extra_configs = pending_tuples
        configs.extend(extra_configs)
        pending_evaluations = [
            PendingEvaluation(trial_id=str(trial_id))
            for trial_id in range(sz, sz + extra)
        ]
    failed_trials = None
    if failed_tuples is not None:
        sz = len(configs)
        extra = len(failed_tuples)
        if failed_tuples and isinstance(failed_tuples[0], tuple):
            extra_configs = tuples_to_configs(failed_tuples, hp_ranges)
        else:
            extra_configs = failed_tuples
        configs.extend(extra_configs)
        failed_trials = [str(x) for x in range(sz, sz + extra)]

    config_for_trial = {
        str(trial_id): config for trial_id, config in enumerate(configs)
    }
    return TuningJobState(
        hp_ranges=hp_ranges,
        config_for_trial=config_for_trial,
        trials_evaluations=trials_evaluations,
        failed_trials=failed_trials,
        pending_evaluations=pending_evaluations,
    )
