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
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import MeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.issm \
    import issm_likelihood_slow_computations, posterior_computations, \
    predict_posterior_marginals, sample_posterior_marginals, \
    sample_posterior_joint, _inner_product, \
    issm_likelihood_computations, issm_likelihood_precomputations
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import ISSModelParameters
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.freeze_thaw \
    import resource_kernel_likelihood_slow_computations, \
    ExponentialDecayBaseKernelFunction, logdet_cholfact_cov_resource, \
    resource_kernel_likelihood_computations, \
    resource_kernel_likelihood_precomputations

logger = logging.getLogger(__name__)

__all__ = ['GaussProcISSMPosteriorState']


class GaussProcAdditivePosteriorState(object):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The (additive) model is the sum of a Gaussian
    process model for function values at r_max and independent Gaussian models
    over r only.

    Importantly, inference scales cubically only in the number of
    configurations, not in the number of observations.

    """

    def __init__(
            self, data: Dict, mean: MeanFunction, kernel: KernelFunction,
            noise_variance, **kwargs):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`. `iss_model` maintains the ISSM parameters.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param noise_variance: Noise variance
        """
        self.mean = mean
        self.kernel = kernel
        self.r_min = data['r_min']
        self.r_max = data['r_max']
        # Compute posterior state
        self.poster_state = None
        self._compute_posterior_state(data, noise_variance, **kwargs)

    def _compute_posterior_state(self, data: Dict, noise_variance, **kwargs):
        raise NotImplementedError()

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
            self, test_features: np.ndarray, num_samples: int = 1,
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

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: Dict[str, np.ndarray],
            mean_data: float, std_data: float) -> np.ndarray:
        """
        Implements SurrogateModel.backward_gradient, see comments there.
        This is for a single posterior state. If the SurrogateModel uses
        MCMC, have to call this for every sample.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See SurrogateModel.backward_gradient
        :param mean_data: Mean used to normalize targets
        :param std_data: Stddev used to normalize targets
        :return:
        """
        test_feature = np.reshape(input, (1, -1))

        def diff_test_feature(test_feature_array):
            norm_mean, norm_variance = self.predict(test_feature_array)
            # De-normalize, and variance -> stddev
            pred_mean = norm_mean * std_data + mean_data
            pred_std = anp.sqrt(norm_variance) * std_data
            head_gradients_mean = anp.reshape(head_gradients['mean'], pred_mean.shape)
            head_gradients_std = anp.reshape(head_gradients['std'], pred_std.shape)
            # Added to mimic mxnet.autograd.backward
            pred_mean_sum = _inner_product(pred_mean, head_gradients_mean)
            pred_std_sum = _inner_product(pred_std, head_gradients_std)
            return pred_mean_sum + pred_std_sum

        test_feature_gradient = grad(diff_test_feature)
        return np.reshape(test_feature_gradient(test_feature), input.shape)


class GaussProcISSMPosteriorState(GaussProcAdditivePosteriorState):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The model is the sum of a Gaussian process
    model for function values at r_max and independent Gaussian linear
    innovation state space models (ISSMs) of a particular power law decay
    form.

    """
    def __init__(
            self, data: Dict, mean: MeanFunction, kernel: KernelFunction,
            iss_model: ISSModelParameters, noise_variance, **kwargs):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`. `iss_model` maintains the ISSM parameters.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param iss_model: ISS model
        :param noise_variance: Innovation and noise variance
        """
        super().__init__(data, mean, kernel, noise_variance=noise_variance,
                         iss_model=iss_model, **kwargs)

    @staticmethod
    def _has_precomputations(data: Dict) -> bool:
        return all(k in data for k in ('ydims', 'num_configs', 'deltay', 'logr'))

    # TODO: Once unit tests confirm the equivalence of the two ways, remove
    # the slow one here and require precomputations to be part of `data`
    def _compute_posterior_state(self, data: Dict, noise_variance, **kwargs):
        profiler = kwargs.get('profiler')
        # Compute posterior state
        self.iss_model = kwargs['iss_model']
        issm_params = self.iss_model.get_issm_params(data['configs'])
        if self._has_precomputations(data):
            issm_likelihood = issm_likelihood_computations(
                precomputed=data, issm_params=issm_params, r_min=self.r_min,
                r_max=self.r_max, profiler=profiler)
        else:
            issm_likelihood = issm_likelihood_slow_computations(
                targets=data['targets'], issm_params=issm_params,
                r_min=self.r_min, r_max=self.r_max, profiler=profiler)
        if profiler is not None:
            profiler.start('poster_comp')
        self.poster_state = posterior_computations(
            data['features'], self.mean, self.kernel, issm_likelihood,
            noise_variance)
        if profiler is not None:
            profiler.stop('poster_comp')

    @staticmethod
    def data_precomputations(data: Dict):
        logger.info("Enhancing data dictionary by precomputed variables")
        data.update(issm_likelihood_precomputations(
            targets=data['targets'], r_min=data['r_min']))

    # NOTE: This works for ISSM only, not generic!
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
        :param random_state: PRNG state to be used for sampling
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


class GaussProcExpDecayPosteriorState(GaussProcAdditivePosteriorState):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The model is the sum of a Gaussian process
    model for function values at r_max and independent Gaussian processes over
    r, using an exponential decay covariance function.

    This is essentially the model from the Freeze Thaw paper (see also
    :class:`ExponentialDecayResourcesKernelFunction`).

    """
    def __init__(
            self, data: Dict, mean: MeanFunction, kernel: KernelFunction,
            res_kernel: ExponentialDecayBaseKernelFunction, noise_variance,
            **kwargs):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`. `iss_model` maintains the ISSM parameters.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param res_kernel: Kernel function k_r(r, r'), of exponential decay
            type
        :param noise_variance: Innovation and noise variance
        """
        super().__init__(data, mean, kernel, noise_variance=noise_variance,
                         res_kernel=res_kernel, **kwargs)

    @staticmethod
    def _has_precomputations(data: Dict) -> bool:
        return all(k in data for k in ('ydims', 'num_configs', 'yflat'))

    # TODO: Once unit tests confirm the equivalence of the two ways, remove
    # the slow one here and require precomputations to be part of `data`
    def _compute_posterior_state(self, data: Dict, noise_variance, **kwargs):
        profiler = kwargs.get('profiler')
        # Compute posterior state
        self.res_kernel = kwargs['res_kernel']
        if profiler is not None:
            profiler.start('likelihood')
        if self._has_precomputations(data):
            issm_likelihood = resource_kernel_likelihood_computations(
                precomputed=data, res_kernel=self.res_kernel,
                noise_variance=noise_variance)
        else:
            issm_likelihood = resource_kernel_likelihood_slow_computations(
                targets=data['targets'], res_kernel=self.res_kernel,
                noise_variance=noise_variance)
        if profiler is not None:
            profiler.stop('likelihood')
            profiler.start('poster_comp')
        self.poster_state = posterior_computations(
            data['features'], self.mean, self.kernel, issm_likelihood,
            noise_variance)
        if profiler is not None:
            profiler.stop('poster_comp')
        # Add missing term to criterion value
        part3 = logdet_cholfact_cov_resource(issm_likelihood)
        self.poster_state['criterion'] += part3

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Different to ISSM, g_r is not fixed to 0 at r = r_max.

        :param test_features: Input points for test configs
        :return: posterior_means, posterior_variances
        """
        post_means, post_variances = predict_posterior_marginals(
            self.poster_state, self.mean, self.kernel, test_features)
        feature = np.array([self.res_kernel.r_max]).reshape((1, 1))
        g_mean = self.res_kernel.mean_function(feature).item()
        g_variance = self.res_kernel.diagonal(feature).item()
        return post_means + g_mean, post_variances + g_variance

    @staticmethod
    def data_precomputations(data: Dict):
        logger.info("Enhancing data dictionary by precomputed variables")
        data.update(resource_kernel_likelihood_precomputations(
            targets=data['targets']))
