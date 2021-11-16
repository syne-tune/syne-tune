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
import autograd.numpy as anp
from autograd import grad
from typing import Tuple, Dict, List, Optional
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import MeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.issm \
    import issm_likelihood_computations, posterior_computations, \
    predict_posterior_marginals, sample_posterior_marginals, \
    sample_posterior_joint, _inner_product
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import ISSModelParameters


# TODO:
# For simulation-based scoring, we need incremental updates with joint
# posterior sampling
class GaussProcISSMPosteriorState(object):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The model is the sum of a Gaussian process
    model for function values at r_max and independent Gaussian linear
    innovation state space models (ISSMs) of a particular power law decay
    form.

    Importantly, inference scales cubically only in the number of
    configurations, not in the number of observations.

    """
    def __init__(
            self, data: Dict, mean: MeanFunction, kernel: KernelFunction,
            iss_model: ISSModelParameters, noise_variance):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`. `iss_model` maintains the ISSM parameters.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param iss_model: ISS model
        :param noise_variance: Innovation and noise variance
        """
        self.mean = mean
        self.kernel = kernel
        self.iss_model = iss_model
        self.r_min = data['r_min']
        self.r_max = data['r_max']
        # Compute posterior state
        issm_params = iss_model.get_issm_params(data['configs'])
        issm_likelihood = issm_likelihood_computations(
            data['targets'], issm_params, self.r_min, self.r_max)
        self.poster_state = posterior_computations(
            data['features'], mean, kernel, issm_likelihood, noise_variance)
        self.num_data = issm_likelihood['num_data']

    def neg_log_likelihood(self):
        return self.poster_state['criterion']

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        We compute marginals over f(x, r_max), so at r = r_max. This does not
        need ISSM parameters.
        NOTE: The configurations corresponding to `test_features` must not
        contain any in the training set, otherwise predictive distributions
        computed here are wrong. This is not checked.

        :param test_features: Input points for test configs
        :return: posterior_means, posterior_variances
        """
        return predict_posterior_marginals(
            self.poster_state, self.mean, self.kernel, test_features)

    def sample_marginals(
            self, test_features: np.ndarray, num_samples: int=1,
            random_state: Optional[RandomState] = None) -> np.ndarray:
        """
        See comments of `predict`.

        :param test_features: Input points for test configs
        :param num_samples: Number of samples
        :return: Marginal samples, (num_test, num_samples)
        """
        if random_state is None:
            random_state = np.random
        return sample_posterior_marginals(
            self.poster_state, self.mean, self.kernel, test_features,
            random_state=random_state, num_samples=num_samples)

    def sample_curves(
            self, data: Dict, num_samples: int = 1,
            random_state: Optional[RandomState] = None) -> List[Dict]:
        """
        Given data from one or more configurations (as returned by
        `issm.prepare_data`), for each config, sample a curve from the
        joint posterior (predictive) distribution over latent variables.
        The curve for each config in `data` may be partly observed, but
        must not be fully observed. Samples for the different configs are
        independent. None of the configs in `data` must appear in the dataset
        used to compute the posterior state.

        The result is a list of Dict, one for each config. If for a config,
        targets in `data` are given for resource values range(r_min, r_obs),
        the joint samples returned are [f_r], r in range(r_obs-1, r_max+1),
        and [y_r], r in range(r_obs, r_max+1). If targets in `data` is
        empty, both [f_r] and [y_r] have r in range(r_min, r_max+1).

        :param data: Data for configs to predict at
        :param num_samples: Number of samples to draw from each curve
        :return: See above
        """
        if random_state is None:
            random_state = np.random
        result = []
        for feature, targets, config in zip(
                data['features'], data['targets'], data['configs']):
            issm_params = self.iss_model.get_issm_params([config])
            result.append(sample_posterior_joint(
                self.poster_state, self.mean, self.kernel, feature,
                targets, issm_params, self.r_min, self.r_max,
                random_state=random_state, num_samples=num_samples))
        return result

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Implements SurrogateModel.backward_gradient, see comments there.
        This is for a single posterior state. If the SurrogateModel uses
        MCMC, have to call this for every sample.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See SurrogateModel.backward_gradient
        :return:
        """
        test_feature = np.reshape(input, (1, -1))
        assert 'mean' in head_gradients, \
            "Need head_gradients['mean'] for backward_gradient"
        has_std = ('std' in head_gradients)

        def diff_test_feature(test_feature_array):
            pred_mean, pred_variance = self.predict(test_feature_array)
            head_gradients_mean = anp.reshape(
                head_gradients['mean'], pred_mean.shape)
            # Added to mimic mxnet.autograd.backward
            pred_mean_sum = _inner_product(pred_mean, head_gradients_mean)
            if has_std:
                pred_std = anp.sqrt(pred_variance)
                head_gradients_std = anp.reshape(
                    head_gradients['std'], pred_std.shape)
                pred_std_sum = _inner_product(pred_std, head_gradients_std)
            else:
                pred_std_sum = 0.0
            return pred_mean_sum + pred_std_sum
        
        test_feature_gradient = grad(diff_test_feature)
        return np.reshape(test_feature_gradient(test_feature), input.shape)
