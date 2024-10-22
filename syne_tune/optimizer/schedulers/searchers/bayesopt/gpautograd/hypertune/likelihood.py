from typing import Dict, Tuple, Any, Optional

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood import (
    IndependentGPPerResourceMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.posterior_state import (
    HyperTuneIndependentGPPosteriorState,
    HyperTuneJointGPPosteriorState,
    assert_ensemble_distribution,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    ScalarTargetTransform,
)


class HyperTuneIndependentGPMarginalLikelihood(
    IndependentGPPerResourceMarginalLikelihood
):
    """
    Variant of :class:`IndependentGPPerResourceMarginalLikelihood`, which has the
    same internal model and marginal likelihood function, but whose posterior
    state is of :class:`HyperTuneIndependentGPPosteriorState`, which uses an
    ensemble predictive distribution, whose weighting distribution has to be
    passed here at construction.
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        resource_attr_range: Tuple[int, int],
        ensemble_distribution: Dict[int, float],
        target_transform: Optional[ScalarTargetTransform] = None,
        separate_noise_variances: bool = False,
        initial_noise_variance=None,
        initial_covariance_scale=None,
        encoding_type=None,
        **kwargs,
    ):
        super().__init__(
            kernel=kernel,
            mean=mean,
            resource_attr_range=resource_attr_range,
            target_transform=target_transform,
            separate_noise_variances=separate_noise_variances,
            initial_noise_variance=initial_noise_variance,
            initial_covariance_scale=initial_covariance_scale,
            encoding_type=encoding_type,
            **kwargs,
        )
        self._ensemble_distribution = None
        self.set_ensemble_distribution(ensemble_distribution)

    @property
    def ensemble_distribution(self) -> Dict[int, float]:
        return self._ensemble_distribution

    def set_ensemble_distribution(self, distribution: Dict[int, float]):
        assert_ensemble_distribution(distribution, set(self.mean.keys()))
        self._ensemble_distribution = distribution.copy()

    def get_posterior_state(self, data: Dict[str, Any]) -> PosteriorState:
        GaussianProcessMarginalLikelihood.assert_data_entries(data)
        targets = self.target_transform(data["targets"])
        return HyperTuneIndependentGPPosteriorState(
            features=data["features"],
            targets=targets,
            kernel=self.kernel,
            mean=self.mean,
            covariance_scale=self._covariance_scale(),
            noise_variance=self._noise_variance(),
            resource_attr_range=self.resource_attr_range,
            ensemble_distribution=self._ensemble_distribution,
        )


class HyperTuneJointGPMarginalLikelihood(GaussianProcessMarginalLikelihood):
    """
    Variant of :class:`GaussianProcessMarginalLikelihood`, which has the
    same internal model and marginal likelihood function, but whose posterior
    state is of :class:`HyperTuneJointGPPosteriorState`, which uses an
    ensemble predictive distribution, whose weighting distribution has to be
    passed here at construction.
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: MeanFunction,
        resource_attr_range: Tuple[int, int],
        ensemble_distribution: Dict[int, float],
        target_transform: Optional[ScalarTargetTransform] = None,
        initial_noise_variance=None,
        encoding_type=None,
        **kwargs,
    ):
        super().__init__(
            kernel=kernel,
            mean=mean,
            target_transform=target_transform,
            initial_noise_variance=initial_noise_variance,
            encoding_type=encoding_type,
            **kwargs,
        )
        self._resource_attr_range = resource_attr_range
        self._ensemble_distribution = None
        self.set_ensemble_distribution(ensemble_distribution)

    @property
    def ensemble_distribution(self) -> Dict[int, float]:
        return self._ensemble_distribution

    def set_ensemble_distribution(self, distribution: Dict[int, float]):
        self._ensemble_distribution = distribution.copy()

    def get_posterior_state(self, data: Dict[str, Any]) -> PosteriorState:
        GaussianProcessMarginalLikelihood.assert_data_entries(data)
        targets = self.target_transform(data["targets"])
        return HyperTuneJointGPPosteriorState(
            features=data["features"],
            targets=targets,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self._noise_variance(),
            resource_attr_range=self._resource_attr_range,
            ensemble_distribution=self._ensemble_distribution,
        )
