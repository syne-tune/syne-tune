from functools import partial
from typing import Optional

from scipy import stats

import numpy as np


class GaussianTransform:
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated empirical CDF.
    :param y: shape (n, dim)
    :param random_state: If specified, randomize the rank when consecutive values exists between extreme values.
     If none use lowest rank of duplicated values.
    """

    def __init__(self, y: np.array, random_state: Optional[np.random.RandomState] = None):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.sorted = y.copy()
        self.sorted.sort(axis=0)
        self.random_state = random_state

    @staticmethod
    def z_transform(series, values_sorted, random_state: Optional[np.random.RandomState] = None):
        """
        :param series: shape (n, dim)
        :param values_sorted: series sorted on the first axis
        :param random_state: if not None, ranks are drawn uniformly for values with consecutive ranges
        :return: data with same shape as input series where distribution is normalized on all dimensions
        """
        # Cutoff ranks since `Phi^{-1}` is infinite at `0` and `1` with winsorized constants.
        def winsorized_delta(n):
            return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))
        delta = winsorized_delta(len(series))

        def quantile(values_sorted, values_to_insert, delta):
            low = np.searchsorted(values_sorted, values_to_insert, side='left')
            if random_state is not None:
                # in case where multiple occurences of the same value exists in sorted array
                # we return a random index in the valid range
                high = np.searchsorted(values_sorted, values_to_insert, side='right')
                res = random_state.randint(low, np.maximum(high, low + 1))
            else:
                res = low
            return np.clip(res / len(values_sorted), a_min=delta, a_max=1 - delta)

        quantiles = quantile(
            values_sorted,
            series,
            delta
        )

        quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)

        return stats.norm.ppf(quantiles)

    def transform(self, y: np.array):
        """
        :param y: shape (n, dim)
        :return: shape (n, dim), distributed along a normal
        """
        assert y.shape[1] == self.dim
        # compute truncated quantile, apply gaussian inv cdf
        return np.stack([
            self.z_transform(y[:, i], self.sorted[:, i], self.random_state)
            for i in range(self.dim)
        ]).T


class StandardTransform:

    def __init__(self, y: np.array):
        """
        Transformation that removes mean and divide by standard error.
        :param y:
        """
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)

    def transform(self, y: np.array):
        z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
        return z


def from_string(name: str, random_state: Optional[np.random.RandomState] = None):
    assert name in ["standard", "gaussian"]
    mapping = {
        "standard": StandardTransform,
        "gaussian": partial(GaussianTransform, random_state=random_state),
    }
    return mapping[name]
