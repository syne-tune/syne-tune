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
from typing import Iterable, List, Optional, Iterator
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import logging
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    SurrogateModel,
    ScoringFunction,
    LocalOptimizer,
    SurrogateOutputModel,
    AcquisitionClassAndArgs,
    unwrap_acquisition_class_and_kwargs,
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)

logger = logging.getLogger(__name__)


class IndependentThompsonSampling(ScoringFunction):
    """
    Note: This is *not* Thompson sampling, but rather a variant called
    "independent Thompson sampling", where means and variances are drawn
    from the marginal rather than the joint distribution. This is cheap,
    but incorrect. In fact, the larger the number of candidates, the more
    likely the winning configuration is arising from pure chance.

    :param model: Surrogate model for statistics of predictive distribution
    :param random_state: PRN generator
    """

    def __init__(
        self, model: SurrogateModel, random_state: Optional[RandomState] = None
    ):
        self.model = model
        if random_state is None:
            random_state = RandomState(31415629)
        self.random_state = random_state

    def score(
        self,
        candidates: Iterable[Configuration],
        model: Optional[SurrogateModel] = None,
    ) -> List[float]:
        if model is None:
            model = self.model
        predictions_list = model.predict_candidates(candidates)
        scores = []
        # If the model supports fantasizing, posterior_means is a matrix. In
        # that case, samples are drawn for every column, then averaged (why
        # we need np.mean)
        for predictions in predictions_list:
            posterior_means = predictions["mean"]
            posterior_stds = predictions["std"]
            new_score = [
                np.mean(self.random_state.normal(m, s))
                for m, s in zip(posterior_means, posterior_stds)
            ]
            scores.append(new_score)
        return list(np.mean(np.array(scores), axis=0))


class LBFGSOptimizeAcquisition(LocalOptimizer):
    def __init__(
        self,
        hp_ranges: HyperparameterRanges,
        model: SurrogateOutputModel,
        acquisition_class: AcquisitionClassAndArgs,
        active_metric: str = None,
    ):
        super().__init__(hp_ranges, model, acquisition_class, active_metric)
        # Number criterion evaluations in last recent optimize call
        self.num_evaluations = None

    def optimize(
        self, candidate: Configuration, model: Optional[SurrogateOutputModel] = None
    ) -> Configuration:
        # Before local minimization, the model for this state_id should have been fitted.
        if model is None:
            model = self.model
        acquisition_class, acquisition_kwargs = unwrap_acquisition_class_and_kwargs(
            self.acquisition_class
        )
        acquisition_function = acquisition_class(
            model, self.active_metric, **acquisition_kwargs
        )

        x0 = self.hp_ranges.to_ndarray(candidate)
        bounds = self.hp_ranges.get_ndarray_bounds()
        n_evaluations = [0]  # wrapped in list to allow access from function

        # unwrap 2d arrays
        def f_df(x):
            n_evaluations[0] += 1
            return acquisition_function.compute_acq_with_gradient(x)

        res = fmin_l_bfgs_b(f_df, x0=x0, bounds=bounds, maxiter=1000)
        self.num_evaluations = n_evaluations[0]
        if res[2]["task"] == b"ABNORMAL_TERMINATION_IN_LNSRCH":
            # this condition was copied from the old GPyOpt code
            logger.warning(
                f"ABNORMAL_TERMINATION_IN_LNSRCH in lbfgs after {n_evaluations[0]} evaluations, "
                "returning original candidate"
            )
            return candidate  # returning original candidate
        else:
            # Clip to avoid situation where result is small epsilon out of bounds
            a_min, a_max = zip(*bounds)
            optimized_x = np.clip(res[0], a_min, a_max)
            # Make sure the above clipping does really just fix numerical
            # rounding issues in L-BFGS if any bigger change was made there is
            # a bug and we want to throw an exception
            assert np.linalg.norm(res[0] - optimized_x) < 1e-6, (
                res[0],
                optimized_x,
                bounds,
            )
            result = self.hp_ranges.from_ndarray(optimized_x.flatten())
            return result


class NoOptimization(LocalOptimizer):
    def optimize(
        self, candidate: Configuration, model: Optional[SurrogateModel] = None
    ) -> Configuration:
        return candidate


MAX_RETRIES_CANDIDATES_EN_BULK = 20
MAX_RETRIES_ON_DUPLICATES = 10000


class RandomStatefulCandidateGenerator(CandidateGenerator):
    """
    This generator maintains a random state, so if :meth:`generate_candidates`
    is called several times, different sequences are returned.

    :param hp_ranges: Feature generator for configurations
    :param random_state: PRN generator
    """

    def __init__(
        self, hp_ranges: HyperparameterRanges, random_state: np.random.RandomState
    ):
        self.hp_ranges = hp_ranges
        self.random_state = random_state

    def generate_candidates(self) -> Iterator[Configuration]:
        while True:
            yield self.hp_ranges.random_config(self.random_state)

    def generate_candidates_en_bulk(
        self, num_cands: int, exclusion_list=None
    ) -> List[Configuration]:
        if exclusion_list is None:
            return self.hp_ranges.random_configs(self.random_state, num_cands)
        else:
            assert isinstance(
                exclusion_list, ExclusionList
            ), "exclusion_list must be of type ExclusionList"
            configs = []
            num_done = 0
            for i in range(MAX_RETRIES_CANDIDATES_EN_BULK):
                # From iteration 1, we request more than what is still
                # needed, because the config space seems to not have
                # enough configs left
                num_requested = min(num_cands, (num_cands - num_done) * (i + 1))
                new_configs = [
                    config
                    for config in self.hp_ranges.random_configs(
                        self.random_state, num_requested
                    )
                    if not exclusion_list.contains(config)
                ]
                num_new = min(num_cands - num_done, len(new_configs))
                configs += new_configs[:num_new]
                num_done += num_new
                if num_done == num_cands:
                    break
            if num_done < num_cands:
                logger.warning(
                    f"Could only sample {num_done} candidates where "
                    f"{num_cands} were requested. len(exclusion_list) = "
                    f"{len(exclusion_list)}"
                )
            return configs


def generate_unique_candidates(
    candidates_generator: CandidateGenerator,
    num_candidates: int,
    exclusion_candidates: ExclusionList,
) -> List[Configuration]:
    exclusion_candidates = exclusion_candidates.copy()  # Copy
    result = []
    num_results = 0
    retries = 0
    just_added = True
    for i, cand in enumerate(candidates_generator.generate_candidates()):
        if just_added:
            if exclusion_candidates.config_space_exhausted():
                logger.warning(
                    "All entries of finite config space (size "
                    + f"{exclusion_candidates.configspace_size}) have been selected. Returning "
                    + f"{len(result)} configs instead of {num_candidates}"
                )
                break
            just_added = False
        if not exclusion_candidates.contains(cand):
            result.append(cand)
            num_results += 1
            exclusion_candidates.add(cand)
            retries = 0
            just_added = True
        else:
            # found a duplicate; retry
            retries += 1
        # End loop if enough candidates where generated, or after too many retries
        # (this latter can happen when most of them are duplicates, and must be done
        # to avoid infinite loops in the purely discrete case)
        if num_results == num_candidates or retries > MAX_RETRIES_ON_DUPLICATES:
            if retries > MAX_RETRIES_ON_DUPLICATES:
                logger.warning(
                    f"Reached limit of {MAX_RETRIES_ON_DUPLICATES} retries "
                    f"with i={i}. Returning {len(result)} candidates instead "
                    f"of the requested {num_candidates}"
                )
            break
    return result


class RandomFromSetCandidateGenerator(CandidateGenerator):
    """
    In this generator, candidates are sampled from a given set.

    :param base_set: Set of all configurations to sample from
    :param random_state: PRN generator
    :param ext_config: If given, each configuration is updated with this
        dictionary before being returned
    """

    def __init__(
        self,
        base_set: List[Configuration],
        random_state: np.random.RandomState,
        ext_config: Optional[Configuration] = None,
    ):
        self.random_state = random_state
        self.base_set = base_set
        self.num_base = len(base_set)
        self._ext_config = ext_config
        # Maintains index of positions of entries returned
        self.pos_returned = set()

    def generate_candidates(self) -> Iterator[Configuration]:
        while True:
            pos = self.random_state.randint(low=0, high=self.num_base)
            self.pos_returned.add(pos)
            config = self._extend_configs([self.base_set[pos]])[0]
            yield config

    def _extend_configs(self, configs: List[Configuration]) -> List[Configuration]:
        if self._ext_config is None:
            return configs
        else:
            return [dict(config, **self._ext_config) for config in configs]

    def generate_candidates_en_bulk(
        self, num_cands: int, exclusion_list=None
    ) -> List[Configuration]:
        if num_cands >= self.num_base:
            if exclusion_list is None:
                configs = self.base_set.copy()
                self.pos_returned = set(range(self.num_base))
            else:
                configs, new_pos = zip(
                    *[
                        (config, pos)
                        for pos, config in enumerate(self.base_set)
                        if not exclusion_list.contains(config)
                    ]
                )
                configs = list(configs)
                self.pos_returned = set(new_pos)
        else:
            if exclusion_list is None:
                randset = self.random_state.choice(
                    self.num_base, num_cands, replace=False
                )
                self.pos_returned.update(randset)
                configs = [self.base_set[pos] for pos in randset]
            else:
                randperm = self.random_state.permutation(self.num_base)
                configs = []
                new_pos = []
                len_configs = 0
                for pos in randperm:
                    if len_configs == num_cands:
                        break
                    config = self.base_set[pos]
                    if not exclusion_list.contains(config):
                        configs.append(config)
                        new_pos.append(pos)
                        len_configs += 1
                self.pos_returned.update(new_pos)
        return self._extend_configs(configs)


class DuplicateDetector:
    def contains(
        self, existing_candidates: ExclusionList, new_candidate: Configuration
    ) -> bool:
        raise NotImplementedError


class DuplicateDetectorNoDetection(DuplicateDetector):
    def contains(
        self, existing_candidates: ExclusionList, new_candidate: Configuration
    ) -> bool:
        return False  # no duplicate detection at all


class DuplicateDetectorIdentical(DuplicateDetector):
    def contains(
        self, existing_candidates: ExclusionList, new_candidate: Configuration
    ) -> bool:
        return existing_candidates.contains(new_candidate)
