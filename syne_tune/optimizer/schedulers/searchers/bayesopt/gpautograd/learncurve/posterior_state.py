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
    issm_likelihood_computations, issm_likelihood_precomputations, \
    _rowvec, update_posterior_state, update_posterior_pvec, _flatvec
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import ISSModelParameters
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.freeze_thaw \
    import resource_kernel_likelihood_slow_computations, \
    ExponentialDecayBaseKernelFunction, logdet_cholfact_cov_resource, \
    resource_kernel_likelihood_computations, \
    resource_kernel_likelihood_precomputations
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration

logger = logging.getLogger(__name__)

__all__ = ['GaussProcAdditivePosteriorState',
           'IncrementalUpdateGPAdditivePosteriorState',
           'GaussProcISSMPosteriorState',
           'GaussProcExpDecayPosteriorState']


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
            self, data: Optional[Dict], mean: MeanFunction,
            kernel: KernelFunction, noise_variance, with_r2_r4=False,
            **kwargs):
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
        self.noise_variance = noise_variance
        self.poster_state = None
        if data is not None:
            self.r_min = data['r_min']
            self.r_max = data['r_max']
            # Compute posterior state
            self._compute_posterior_state(
                data, noise_variance, with_r2_r4=with_r2_r4, **kwargs)
        else:
            # Copy constructor, used by `IncrementalUpdateGPISSMPosteriorState`
            # subclass
            self.poster_state = kwargs['poster_state']
            self.r_min = kwargs['r_min']
            self.r_max = kwargs['r_max']

    def _compute_posterior_state(
            self, data: Dict, noise_variance, with_r2_r4, **kwargs):
        raise NotImplementedError()

    def neg_log_likelihood(self):
        assert 'criterion' in self.poster_state, \
            "neg_log_likelihood not defined for fantasizing posterior state"
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
        :param random_state: PRNG
        :return: Marginal samples, (num_test, num_samples)
        """
        if random_state is None:
            random_state = np.random
        return sample_posterior_marginals(
            self.poster_state, self.mean, self.kernel, test_features,
            random_state=random_state, num_samples=num_samples)

    # TODO:
    # Needs `sample_curves` here!

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


class IncrementalUpdateGPAdditivePosteriorState(GaussProcAdditivePosteriorState):
    """
    Extension of :class:`GaussProcAdditivePosteriorState` which allows for
    incremental updating (single config added to the dataset).
    This is required for simulation-based scoring, and for support of
    fantasizing.

    """
    def __init__(
            self, data: Optional[Dict], mean: MeanFunction,
            kernel: KernelFunction, noise_variance, **kwargs):
        # For incremental updating, the posterior state also has to contain
        # r_2 and r_4
        super().__init__(
            data, mean, kernel, noise_variance, with_r2_r4=True, **kwargs)

    def _prepare_update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> Tuple[float, float, float]:
        """
        Helper method for `update`. Returns new entries for d, s, r2 vectors.

        :param config: See `update`
        :param feature: See `update`
        :param targets: See `update`
        :return: (d_new, s_new, r2_new)
        """
        raise NotImplementedError()

    def _update_internal(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> Dict:
        """
        Update posterior state, given a single new datapoint. `config`,
        `feature`, `targets` are like one entry of `data`.
        The method returns a new object with the updated state

        :param config: See above
        :param feature: See above
        :param targets: See above
        :return: Arguments to create new posterior state
        """
        # Update posterior state
        feature = _rowvec(feature, _np=np)
        d_new, s_new, r2_new = self._prepare_update(config, feature, targets)
        new_poster_state = update_posterior_state(
            self.poster_state, self.kernel, feature, d_new, s_new, r2_new)
        # Return args to create new object by way of "copy constructor"
        return dict(
            data=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            poster_state=new_poster_state,
            r_min=self.r_min, r_max=self.r_max)

    def update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> 'IncrementalUpdateGPAdditivePosteriorState':
        raise NotImplementedError()

    def update_pvec(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> np.ndarray:
        """
        Part of `update`: Only update prediction vector p. This cannot be used
        to update p for several new datapoints.

        :param config:
        :param feature:
        :param targets:
        :return: New p vector
        """
        feature = _rowvec(feature, _np=np)
        d_new, s_new, r2_new = self._prepare_update(config, feature, targets)
        return update_posterior_pvec(
            self.poster_state, self.kernel, feature, d_new, s_new, r2_new)

    def _sample_posterior_joint_for_config(
            self, poster_state, config: Configuration, feature: np.ndarray,
            targets: np.ndarray, random_state: RandomState) -> np.ndarray:
        raise NotImplementedError()

    def sample_and_update_for_pending(
            self, data_pending: Dict, sample_all_nonobserved: bool = False,
            random_state: Optional[RandomState] = None) \
            -> (List[np.ndarray], 'IncrementalUpdateGPAdditivePosteriorState'):
        """
        This function is needed for sampling fantasy targets, and also to
        support simulation-based scoring.

        `issm.prepare_data_with_pending` creates two data dicts `data_nopending`,
        `data_pending`, the first for configs with observed data, but no
        pending evals, the second for configs with pending evals.
        You create the state with `data_nopending`, then call this method with
        `data_pending`.

        This method is iterating over configs (or trials) in `data_pending`.
        For each config, it draws a joint sample from some non-observed
        targets, then updates the state conditioned on observed and sampled
        targets (by calling `update`). If `sample_all_nonobserved` is False,
        the number of targets sampled is the entry in
        `data_pending['num_pending']`. Otherwise, targets are sampled for all
        non-observed positions.

        The method returns the list of sampled target vectors, and the state
        at the end (like `update` does as well).

        :param data_pending: See above
        :param sample_all_nonobserved: See above
        :param random_state: PRNG
        :return: pending_targets, final_state
        """
        if random_state is None:
            random_state = np.random
        curr_poster_state = self.poster_state
        targets_lst = []
        final_state = self
        for config, feature, targets, num_pending in zip(
                data_pending['configs'], data_pending['features'],
                data_pending['targets'], data_pending['num_pending']):
            # Draw joint sample
            fantasies = _flatvec(self._sample_posterior_joint_for_config(
                curr_poster_state, config, feature, targets, random_state),
                _np=np)
            if not sample_all_nonobserved:
                fantasies = fantasies[:num_pending]
            targets_lst.append(fantasies)
            # Update state
            full_targets = np.concatenate([np.array(targets), fantasies])
            final_state = self.update(config, feature, full_targets)
            curr_poster_state = final_state.poster_state
        return targets_lst, final_state


class GaussProcISSMPosteriorState(IncrementalUpdateGPAdditivePosteriorState):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The model is the sum of a Gaussian process
    model for function values at r_max and independent Gaussian linear
    innovation state space models (ISSMs) of a particular power law decay
    form.

    """
    def __init__(
            self, data: Optional[Dict], mean: MeanFunction,
            kernel: KernelFunction, iss_model: ISSModelParameters,
            noise_variance, **kwargs):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`. `iss_model` maintains the ISSM parameters.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param iss_model: ISS model
        :param noise_variance: Innovation and noise variance
        """
        self.iss_model = iss_model
        super().__init__(
            data, mean, kernel, noise_variance=noise_variance, **kwargs)

    @staticmethod
    def _has_precomputations(data: Dict) -> bool:
        return all(k in data for k in ('ydims', 'num_configs', 'deltay', 'logr'))

    def _compute_posterior_state(self, data: Dict, noise_variance, **kwargs):
        profiler = kwargs.get('profiler')
        # Compute posterior state
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

    def _prepare_update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> Tuple[float, float, float]:
        issm_params = self.iss_model.get_issm_params([config])
        issm_likelihood = issm_likelihood_slow_computations(
            [targets], issm_params, self.r_min, self.r_max)
        d_new = issm_likelihood['d'].item()
        vtv = issm_likelihood['vtv'].item()
        wtv = issm_likelihood['wtv'].item()
        s_sq = vtv / self.noise_variance
        s_new = np.sqrt(s_sq)
        muhat = _flatvec(self.mean(feature)).item() - issm_likelihood['c'].item()
        r2_new = wtv / self.noise_variance - s_sq * muhat
        return d_new, s_new, r2_new

    def update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> 'IncrementalUpdateGPAdditivePosteriorState':
        create_kwargs = self._update_internal(config, feature, targets)
        return GaussProcISSMPosteriorState(
            **create_kwargs, iss_model=self.iss_model)

    def _sample_posterior_joint_for_config(
            self, poster_state, config: Configuration, feature: np.ndarray,
            targets: np.ndarray, random_state: RandomState) -> np.ndarray:
        issm_params = self.iss_model.get_issm_params([config])
        result = sample_posterior_joint(
            poster_state=poster_state, mean=self.mean,
            kernel=self.kernel, feature=feature, targets=targets,
            issm_params=issm_params, r_min=self.r_min, r_max=self.r_max,
            random_state=random_state)
        return result['y']


class GaussProcExpDecayPosteriorState(IncrementalUpdateGPAdditivePosteriorState):
    """
    Represent posterior state for joint Gaussian model of learning curves over
    a number of configurations. The model is the sum of a Gaussian process
    model for function values at r_max and independent Gaussian processes over
    r, using an exponential decay covariance function.

    This is essentially the model from the Freeze Thaw paper (see also
    :class:`ExponentialDecayResourcesKernelFunction`).

    """
    def __init__(
            self, data: Optional[Dict], mean: MeanFunction,
            kernel: KernelFunction,
            res_kernel: ExponentialDecayBaseKernelFunction, noise_variance,
            **kwargs):
        """
        `data` contains input points and targets, as obtained from
        `issm.prepare_data`.

        :param data: Input points and targets
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param res_kernel: Kernel function k_r(r, r'), of exponential decay
            type
        :param noise_variance: Innovation and noise variance
        """
        self.res_kernel = res_kernel
        super().__init__(
            data, mean, kernel, noise_variance=noise_variance, **kwargs)

    @staticmethod
    def _has_precomputations(data: Dict) -> bool:
        return all(k in data for k in ('ydims', 'num_configs', 'yflat'))

    def _compute_posterior_state(self, data: Dict, noise_variance, **kwargs):
        profiler = kwargs.get('profiler')
        # Compute posterior state
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

    def _prepare_update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> Tuple[float, float, float]:
        issm_likelihood = resource_kernel_likelihood_slow_computations(
            [targets], self.res_kernel, self.noise_variance)
        vtv = issm_likelihood['vtv'].item()
        wtv = issm_likelihood['wtv'].item()
        s_sq = vtv / self.noise_variance
        s_new = np.sqrt(s_sq)
        muhat = _flatvec(self.mean(feature)).item()
        r2_new = wtv / self.noise_variance - s_sq * muhat
        return 0.0, s_new, r2_new

    def update(
            self, config: Configuration, feature: np.ndarray,
            targets: np.ndarray) -> 'IncrementalUpdateGPAdditivePosteriorState':
        create_kwargs = self._update_internal(config, feature, targets)
        return GaussProcExpDecayPosteriorState(
            **create_kwargs, res_kernel=self.res_kernel)

    # TODO!
    #def _sample_posterior_joint_for_config(
    #        self, poster_state, config: Configuration, feature: np.ndarray,
    #        targets: np.ndarray, random_state: RandomState) -> np.ndarray:
