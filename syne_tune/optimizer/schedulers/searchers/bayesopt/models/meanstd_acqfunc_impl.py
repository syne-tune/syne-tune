from typing import Dict, Optional, Set, List, Tuple
import logging
import itertools

import numpy as np
from scipy.stats import norm
from scipy.special import erfc

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc import (
    MeanStdAcquisitionFunction,
    HeadWithGradient,
    SamplePredictionsPerOutput,
    CurrentBestProvider,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BasePredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    OutputPredictor,
    Predictor,
)

logger = logging.getLogger(__name__)


MIN_COST = 1e-12  # For numerical stability when dividing EI / cost

MIN_STD_CONSTRAINT = (
    1e-12  # For numerical stability when computing the constraint probability in CEI
)


def _extract_active_and_secondary_metric(predictor_output_names, active_metric):
    """
    Returns the active metric and the secondary metric (such as the cost or constraint metric) from predictor_output_names.
    """
    assert len(predictor_output_names) == 2, (
        f"The predictor should consist of exactly 2 outputs, "
        f"while the current outputs are {predictor_output_names}"
    )
    assert active_metric in predictor_output_names, (
        f"{active_metric} is not a valid metric. "
        f"The metric name must match one of the following metrics "
        f"in the predictor output: {predictor_output_names}"
    )
    if predictor_output_names[0] == active_metric:
        secondary_metric = predictor_output_names[1]
    else:
        secondary_metric = predictor_output_names[0]
    logger.debug(
        f"There are two metrics in the output: {predictor_output_names}. "
        f"The metric to optimize was set to '{active_metric}'. "
        f"The secondary metric is assumed to be '{secondary_metric}'"
    )
    return active_metric, secondary_metric


def _postprocess_gradient(grad: np.ndarray, nf: int) -> np.ndarray:
    if nf > 1:
        assert grad.size == nf  # Sanity check
        return grad / nf
    else:
        return np.mean(grad, keepdims=True)


class EIAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus expected improvement acquisition function
    (minus because the convention is to always minimize acquisition functions)
    """

    def __init__(
        self,
        predictor: Predictor,
        active_metric: str = None,
        jitter: float = 0.01,
        debug_collect_stats: bool = False,
    ):
        assert isinstance(predictor, Predictor)
        super().__init__(predictor, active_metric)
        self.jitter = jitter
        if debug_collect_stats:
            self._debug_data = {"absdiff": [], "std": [], "u": []}
        else:
            self._debug_data = None

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_head(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> np.ndarray:
        assert current_best is not None
        means, stds = self._extract_mean_and_std(output_to_predictions)
        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        self._debug_append_data(means, stds, current_best, u)
        return np.mean((-stds) * (u * Phi + phi), axis=1)

    def _compute_head_and_gradient(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> HeadWithGradient:
        assert current_best is not None
        mean, std = self._extract_mean_and_std(output_to_predictions)
        nf_mean = mean.size
        assert current_best.size == nf_mean

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        self._debug_append_data(mean, std, current_best, u)
        f_acqu = std * (u * Phi + phi)
        dh_dmean = _postprocess_gradient(Phi, nf=nf_mean)
        dh_dstd = _postprocess_gradient(-phi, nf=1)
        return HeadWithGradient(
            hval=-np.mean(f_acqu),
            gradient={self.active_metric: dict(mean=dh_dmean, std=dh_dstd)},
        )

    def _debug_append_data(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_best: np.ndarray,
        u: np.ndarray,
    ):
        if self._debug_data is not None:
            new_data = [
                ("absdiff", np.abs(current_best - means)),
                ("std", stds),
                ("u", u),
            ]
            for k, v in new_data:
                self._debug_data[k].extend(list(v.flatten()))

    def debug_stats_message(self) -> str:
        if self._debug_data is None or len(self._debug_data["absdiff"]) == 0:
            return ""
        msg_parts = [
            f"{k:7s}: min={np.min(v):.2e}, median={np.median(v):.2e}, "
            f"max={np.max(v):.2e}, num={len(v)}"
            for k, v in self._debug_data.items()
        ]
        return "\n".join(msg_parts)


class LCBAcquisitionFunction(MeanStdAcquisitionFunction):
    r"""
    Lower confidence bound (LCB) acquisition function:

    .. math::

       h(\mu, \sigma) = \mu - \kappa * \sigma
    """

    def __init__(self, predictor: Predictor, kappa: float, active_metric: str = None):
        super().__init__(predictor, active_metric)
        assert isinstance(predictor, Predictor)
        assert kappa > 0, "kappa must be positive"
        self.kappa = kappa

    def _head_needs_current_best(self) -> bool:
        return False

    def _compute_head(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> np.ndarray:
        means, stds = self._extract_mean_and_std(output_to_predictions)
        return np.mean(means - stds * self.kappa, axis=1)

    def _compute_head_and_gradient(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> HeadWithGradient:
        mean, std = self._extract_mean_and_std(output_to_predictions)
        nf_mean = mean.size

        dh_dmean = np.ones_like(mean) / nf_mean
        dh_dstd = (-self.kappa) * np.ones_like(std)
        return HeadWithGradient(
            hval=np.mean(mean - std * self.kappa),
            gradient={self.active_metric: dict(mean=dh_dmean, std=dh_dstd)},
        )


class EIpuAcquisitionFunction(MeanStdAcquisitionFunction):
    r"""
    Minus cost-aware expected improvement acquisition function.

    This is defined as

    .. math::

       \mathrm{EIpu}(x) = \frac{\mathrm{EI(x)}}{\mathrm{power}(\mathrm{cost}(x), \mathrm{exponent_cost})},

    where :math:`\mathrm{EI}(x)` is expected improvement, :math:`\mathrm{cost}(x)`
    is the predictive mean of a cost model, and ``exponent_cost`` is an exponent in
    :math:`(0, 1]`.

    ``exponent_cost`` scales the influence of the cost term on the acquisition
    function. See also:

        | Lee etal.
        | Cost-aware Bayesian Optimization
        | https://arxiv.org/abs/2003.10870

    Note: two metrics are expected in the model output: the main objective and the cost.
    The main objective needs to be indicated as ``active_metric`` when initializing
    :class:`EIpuAcquisitionFunction`.
    The cost is automatically assumed to be the other metric.

    :param predictor: Predictors for main objective and cost
    :param active_metric: Name of main objective
    :param exponent_cost: Exponent for cost in denominator. Defaults to 1
    :param jitter: Jitter factor, must be positive. Defaults to 0.01
    """

    def __init__(
        self,
        predictor: OutputPredictor,
        active_metric: Optional[str] = None,
        exponent_cost: float = 1.0,
        jitter: float = 0.01,
    ):
        super().__init__(predictor, active_metric)
        assert (
            0 < exponent_cost <= 1
        ), f"exponent_cost = {exponent_cost} must lie in (0, 1]"
        self.jitter = jitter
        self.exponent_cost = exponent_cost
        self.active_metric, self.cost_metric = _extract_active_and_secondary_metric(
            self.predictor_output_names, active_metric
        )

    def _head_needs_current_best(self) -> bool:
        return True

    def _output_to_keys_predict(self) -> Dict[str, Set[str]]:
        """
        The cost model may be deterministic, as the acquisition function
        only needs the mean.
        """
        return {
            self.predictor_output_names[0]: {"mean", "std"},
            self.predictor_output_names[1]: {"mean"},
        }

    def _compute_head(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Returns minus the cost-aware expected improvement.
        """
        assert current_best is not None
        means, stds = self._extract_mean_and_std(output_to_predictions)
        pred_costs = self._extract_positive_cost(output_to_predictions)

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        f_acqu = stds * (u * Phi + phi) * np.power(pred_costs, -self.exponent_cost)
        return -np.mean(f_acqu, axis=1)

    def _compute_head_and_gradient(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> HeadWithGradient:
        """
        Returns minus cost-aware expected improvement and, for each output
        predictor, the gradients with respect to the mean and standard deviation of
        that predictor.
        """
        assert current_best is not None
        mean, std = self._extract_mean_and_std(output_to_predictions)
        pred_cost = self._extract_positive_cost(output_to_predictions)
        nf_active = mean.size
        nf_cost = pred_cost.size

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        inv_cost_power = np.power(pred_cost, -self.exponent_cost)
        f_acqu = std * (u * Phi + phi) * inv_cost_power

        dh_dmean_active = _postprocess_gradient(Phi * inv_cost_power, nf=nf_active)
        dh_dstd_active = _postprocess_gradient(-phi * inv_cost_power, nf=1)
        # Flip the sign twice: once because of the derivative of 1 / x, and
        # once because the head is actually - f_ei
        dh_dmean_cost = _postprocess_gradient(
            self.exponent_cost * f_acqu / pred_cost, nf=nf_cost
        )

        gradient = {
            self.active_metric: dict(mean=dh_dmean_active, std=dh_dstd_active),
            self.cost_metric: dict(mean=dh_dmean_cost),
        }
        return HeadWithGradient(hval=-np.mean(f_acqu), gradient=gradient)

    def _extract_positive_cost(self, output_to_predictions):
        pred_cost = output_to_predictions[self.cost_metric]["mean"]
        if np.any(pred_cost) < 0.0:
            logger.warning(
                f"The predictor for {self.cost_metric} predicted some negative cost. "
                f"Capping the minimum cost at {MIN_COST}."
            )
        pred_cost = np.maximum(
            pred_cost, MIN_COST
        )  # ensure that the predicted cost/run-time is positive
        return pred_cost


class ConstraintCurrentBestProvider(CurrentBestProvider):
    """
    Here, ``current_best`` depends on two predictors, for active and constraint metric.
    """

    def __init__(self, current_best_list: List[np.ndarray], num_samples_active: int):
        list_size = len(current_best_list)
        assert list_size > 0 and list_size % num_samples_active == 0
        self._active_and_constraint_current_best = [
            v.reshape((1, -1)) for v in current_best_list
        ]
        self._num_samples_active = num_samples_active

    def __call__(self, positions: Tuple[int, ...]) -> Optional[np.ndarray]:
        flat_pos = positions[1] * self._num_samples_active + positions[0]
        return self._active_and_constraint_current_best[flat_pos]


class CEIAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus legacy_constrained expected improvement acquisition function.
    (minus because the convention is to always minimize the acquisition function)

    This is defined as ``CEI(x) = EI(x) * P(c(x) <= 0)``, where ``EI`` is the
    standard expected improvement with respect to the current *feasible best*,
    and ``P(c(x) <= 0)`` is the probability that the hyperparameter configuration
    ``x`` satisfies the constraint modeled by ``c(x)``.

    If there are no feasible hyperparameters yet, the current feasible best is
    undefined. Thus, ``CEI`` is reduced to the ``P(c(x) <= 0)`` term until a feasible
    configuration is found.

    Two metrics are expected in the model output: the main objective and the
    constraint metric. The main objective needs to be indicated as ``active_metric``
    when initializing :class:`CEIAcquisitionFunction`.
    The constraint is automatically assumed to be the other metric.

    References on ``CEI``:

        | Gardner et al.
        | Bayesian Optimization with Inequality Constraints
        | ICML 2014
    and

        | Gelbart et al.
        | Bayesian Optimization with Unknown Constraints
        | UAI 2014.

    :param predictor: Predictors for main objective and cost
    :param active_metric: Name of main objective
    :param jitter: Jitter factor, must be positive. Defaults to 0.01
    """

    def __init__(
        self,
        predictor: OutputPredictor,
        active_metric: Optional[str] = None,
        jitter: float = 0.01,
    ):
        super().__init__(predictor, active_metric)
        self.jitter = jitter
        self._feasible_best_list = None
        (
            self.active_metric,
            self.constraint_metric,
        ) = _extract_active_and_secondary_metric(
            self.predictor_output_names, active_metric
        )

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_head(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Returns minus the legacy_constrained expected improvement (- CEI).
        """
        assert current_best is not None
        means, stds = self._extract_mean_and_std(output_to_predictions)
        means_constr, stds_constr = self._extract_mean_and_std(
            output_to_predictions, metric=self.constraint_metric
        )

        # Compute the probability of satisfying the constraint P(c(x) <= 0)
        constr_probs = norm.cdf(-means_constr / (stds_constr + MIN_STD_CONSTRAINT))
        # If for some fantasies there are not feasible candidates, there is also no current_best (i.e., a nan).
        # The acquisition function is replaced by only the P(c(x) <= 0) term when no feasible best exist.
        feas_idx = ~np.isnan(current_best).reshape((1, -1))

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        f_ei = stds * (u * Phi + phi)
        # CEI(x) = EI(x) * P(c(x) <= 0) if feasible best exists, CEI(x) = P(c(x) <= 0) otherwise
        f_acqu = np.where(feas_idx, f_ei * constr_probs, constr_probs)
        return -np.mean(f_acqu, axis=1)

    def _compute_head_and_gradient(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> HeadWithGradient:
        """
        Returns minus cost-aware expected improvement (- CEI) and, for each output predictor, the gradients
        with respect to the mean and standard deviation of that predictor.
        """
        assert current_best is not None
        mean, std = self._extract_mean_and_std(output_to_predictions)
        mean_constr, std_constr = self._extract_mean_and_std(
            output_to_predictions, metric=self.constraint_metric
        )
        nf_mean = mean.size
        nf_constr = mean_constr.size

        # Compute the probability of satisfying the constraint P(c(x) <= 0)
        std_constr = std_constr + MIN_STD_CONSTRAINT
        z = -mean_constr / std_constr
        constr_prob = norm.cdf(z)
        # Useful variables for computing the head gradients
        mean_over_squared_std_constr = mean_constr / std_constr**2
        inverse_std_constr = 1.0 / std_constr
        phi_constr = norm.pdf(z)

        # If for some fantasies there are not feasible candidates, there is also no current_best (i.e., a nan).
        # The acquisition function is replaced by only the P(c(x) <= 0) term when no feasible best exist.
        feas_idx = ~np.isnan(current_best)
        phi, Phi, u = get_quantiles(
            self.jitter, current_best, mean, std
        )  # phi, Phi is PDF and CDF of Gaussian
        f_ei = std * (u * Phi + phi)
        f_acqu = np.where(
            feas_idx, f_ei * constr_prob, constr_prob
        )  # CEI(x) = EI(x) * P(c(x) <= 0) if feasible best
        # exists, CEI(x) = P(c(x) <= 0) otherwise
        dh_dmean_constraint_feas = f_ei * inverse_std_constr * phi_constr
        dh_dstd_constraint_feas = -f_ei * mean_over_squared_std_constr * phi_constr
        dh_dmean_active_feas = Phi * constr_prob
        dh_dstd_active_feas = -phi * constr_prob
        dh_dmean_constraint_infeas = inverse_std_constr * phi_constr
        dh_dstd_constraint_infeas = -mean_over_squared_std_constr * phi_constr
        dh_dmean_active_infeas = np.zeros_like(phi_constr)
        dh_dstd_active_infeas = np.zeros_like(phi_constr)
        dh_dmean_active = _postprocess_gradient(
            np.where(feas_idx, dh_dmean_active_feas, dh_dmean_active_infeas), nf=nf_mean
        )
        dh_dstd_active = _postprocess_gradient(
            np.where(feas_idx, dh_dstd_active_feas, dh_dstd_active_infeas), nf=1
        )
        dh_dmean_constraint = _postprocess_gradient(
            np.where(feas_idx, dh_dmean_constraint_feas, dh_dmean_constraint_infeas),
            nf=nf_constr,
        )
        dh_dstd_constraint = _postprocess_gradient(
            np.where(feas_idx, dh_dstd_constraint_feas, dh_dstd_constraint_infeas), nf=1
        )
        gradient = {
            self.active_metric: dict(mean=dh_dmean_active, std=dh_dstd_active),
            self.constraint_metric: dict(
                mean=dh_dmean_constraint, std=dh_dstd_constraint
            ),
        }
        return HeadWithGradient(hval=-np.mean(f_acqu), gradient=gradient)

    def _get_current_bests_internal(
        self, predictor: OutputPredictor
    ) -> CurrentBestProvider:
        active_predictor = predictor[self.active_metric]
        assert isinstance(active_predictor, BasePredictor)
        all_means_active = active_predictor.predict_mean_current_candidates()
        num_samples_active = len(all_means_active)
        constraint_predictor = predictor[self.constraint_metric]
        assert isinstance(constraint_predictor, BasePredictor)
        all_means_constraint = constraint_predictor.predict_mean_current_candidates()
        common_shape = all_means_active[0].shape
        assert all(
            x.shape == common_shape for x in all_means_constraint
        ), "Shape mismatch between predictors for predict_mean_current_candidates"
        current_best_list = []
        for means_constraint, means_active in itertools.product(
            all_means_constraint, all_means_active
        ):
            # Remove all infeasible candidates (i.e., where means_constraint
            # is >= 0)
            means_active[means_constraint >= 0] = np.nan
            # Compute the current *feasible* best (separately for every fantasy)
            min_across_observations = np.nanmin(means_active, axis=0)
            current_best_list.append(min_across_observations)
        return ConstraintCurrentBestProvider(current_best_list, num_samples_active)


def get_quantiles(acquisition_par, fmin, m, s):
    """
    Quantiles of the Gaussian distribution, useful to determine the acquisition
    function values.

    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    """
    if isinstance(s, np.ndarray):
        s[s < 1e-10] = 1e-10
    elif s < 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par) / s

    phi = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    # vectorized version of erfc to not depend on scipy
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return phi, Phi, u
