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
import numpy as np
import pytest

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.utils import (
    _losses_for_rung,
)


class MadeUpPosteriorState:
    def __init__(self):
        self.num_fantasies = 1
        self._samples = None

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        num_data = test_features.shape[0]
        self._samples = random_state.randn(num_data, num_samples)
        return self._samples

    def get_losses(self, targets: np.ndarray):
        assert self._samples is not None
        num_data, num_samples = self._samples.shape
        assert targets.size == num_data
        targets = targets.reshape((-1,))
        result = np.zeros(num_samples)
        # Dumb on purpose!
        for j in range(num_data - 1):
            samp_j = self._samples[j]
            for k in range(j + 1, num_data):
                samp_k = self._samples[k]
                yj_lt_yk = targets[j] < targets[k]
                for ind in range(num_samples):
                    fj_lt_fk = samp_j[ind] < samp_k[ind]
                    result[ind] += int(yj_lt_yk != fj_lt_fk)
        result *= 2 / (num_data * (num_data - 1))
        return result


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "num_data, num_samples",
    [
        (10, 20),
        (5, 5),
        (2, 1),
        (10, 1),
        (100, 100),
    ],
)
def test_hypertune_ranking_losses(num_data, num_samples):
    seed = 31415927
    random_state = np.random.RandomState(seed)
    targets = random_state.randn(num_data, 1).reshape((-1,))
    features = random_state.randn(num_data, 2)  # Does not matter
    data = {"features": features, "targets": targets}
    poster_state = MadeUpPosteriorState()
    losses1 = _losses_for_rung(
        poster_state=poster_state,
        data_max_resource=data,
        num_samples=num_samples,
        random_state=random_state,
    )
    losses2 = poster_state.get_losses(targets)
    np.testing.assert_allclose(losses1, losses2, rtol=1e-5)
