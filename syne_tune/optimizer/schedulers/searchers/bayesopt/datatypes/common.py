from typing import Union, Dict, Optional, List
from dataclasses import dataclass
import numpy as np


INTERNAL_METRIC_NAME = "target"

INTERNAL_CONSTRAINT_NAME = "constraint"

INTERNAL_COST_NAME = "cost"


def dictionarize_objective(x):
    return {INTERNAL_METRIC_NAME: x}


MetricValues = Union[float, Dict[str, float]]


@dataclass
class TrialEvaluations:
    """
    For each fixed k, :code:`metrics[k]` is either a single value or a dict. The
    latter is used, for example, for multi-fidelity schedulers, where
    :code:`metrics[k][str(r)]` is the value at resource level ``r``.
    """

    trial_id: str
    metrics: Dict[str, MetricValues]

    def num_cases(
        self, metric_name: str = INTERNAL_METRIC_NAME, resource: Optional[int] = None
    ) -> int:
        """
        Counts the number of observations for metric ``metric_name``.

        :param metric_name: Defaults to :const:`INTERNAL_METRIC_NAME`
        :param resource: In the multi-fidelity case, we only count observations
            at this resource level
        :return: Number of observations
        """
        metric_vals = self.metrics.get(metric_name)
        if metric_vals is None:
            return 0
        elif isinstance(metric_vals, dict):
            if resource is None:
                return len(metric_vals)
            else:
                return 1 if str(resource) in metric_vals else 0
        else:
            return 1

    def _map_value_for_matching(
        self, value: MetricValues
    ) -> (Optional[List[str]], np.ndarray):
        if isinstance(value, dict):
            keys = list(sorted(value.keys()))
            vals = np.array(value[k] for k in keys)
        else:
            keys = None
            vals = np.array([value])
        return keys, vals

    def __eq__(self, other) -> bool:
        if not isinstance(other, TrialEvaluations):
            return False
        if self.trial_id != other.trial_id:
            return False
        if set(self.metrics.keys()) != set(other.metrics.keys()):
            return False
        for name, value in self.metrics.items():
            keys, vals = self._map_value_for_matching(value)
            keys_other, vals_other = self._map_value_for_matching(other.metrics[name])
            if keys != keys_other or (not np.allclose(vals, vals_other)):
                return False
        return True


class PendingEvaluation:
    """
    Maintains information for pending candidates (i.e. candidates which have
    been queried for labeling, but target feedback has not yet been obtained.

    The minimum information is the candidate which has been queried.
    """

    def __init__(self, trial_id: str, resource: Optional[int] = None):
        self._trial_id = trial_id
        self._resource = resource

    @property
    def trial_id(self) -> str:
        return self._trial_id

    @property
    def resource(self) -> Optional[int]:
        return self._resource


class FantasizedPendingEvaluation(PendingEvaluation):
    """
    Here, latent target values are integrated out by Monte Carlo samples,
    also called "fantasies".
    """

    def __init__(
        self,
        trial_id: str,
        fantasies: Dict[str, np.ndarray],
        resource: Optional[int] = None,
    ):
        super().__init__(trial_id, resource)
        fantasy_sizes = [fantasy_values.size for fantasy_values in fantasies.values()]
        assert all(
            fantasy_size > 0 for fantasy_size in fantasy_sizes
        ), "fantasies must be non-empty"
        assert len(set(fantasy_sizes)) == 1, "fantasies must all have the same length"
        self._fantasies = fantasies.copy()

    @property
    def fantasies(self):
        return self._fantasies
