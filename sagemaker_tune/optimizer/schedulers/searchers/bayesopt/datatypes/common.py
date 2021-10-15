from typing import Union, NamedTuple, Dict
import numpy as np


INTERNAL_METRIC_NAME = 'active_metric'

INTERNAL_CONSTRAINT_NAME = 'constraint_metric'

INTERNAL_COST_NAME = 'cost_metric'

def dictionarize_objective(x):
    return {INTERNAL_METRIC_NAME: x}


Hyperparameter = Union[str, int, float]

Configuration = Dict[str, Hyperparameter]


MetricValues = Union[float, Dict[str, float]]

class CandidateEvaluation(NamedTuple):
    """
    For each fixed k, `metrics[k]` is either a single value or a dict. The
    latter is used, for example, for multi-fidelity schedulers, where
    `metrics[k][str(r)]` is the value at resource level r.

    """
    candidate: Configuration
    metrics: Dict[str, MetricValues]

    def num_cases(self, metric_name: str = INTERNAL_METRIC_NAME) -> int:
        metric_vals = self.metrics.get(metric_name)
        if metric_vals is None:
            return 0
        elif isinstance(metric_vals, dict):
            return len(metric_vals)
        else:
            return 1


class PendingEvaluation(object):
    """
    Maintains information for pending candidates (i.e. candidates which have
    been queried for labeling, but target feedback has not yet been obtained.

    The minimum information is the candidate which has been queried.
    """
    def __init__(self, candidate: Configuration):
        super(PendingEvaluation, self).__init__()
        self._candidate = candidate

    @property
    def candidate(self):
        return self._candidate


class FantasizedPendingEvaluation(PendingEvaluation):
    """
    Here, latent target values are integrated out by Monte Carlo samples,
    also called "fantasies".

    """
    def __init__(self, candidate: Configuration, fantasies: Dict[str, np.ndarray]):
        super(FantasizedPendingEvaluation, self).__init__(candidate)
        fantasy_sizes = [
            fantasy_values.size for fantasy_values in fantasies.values()]
        assert all(fantasy_size > 0 for fantasy_size in fantasy_sizes), \
            "fantasies must be non-empty"
        assert len(set(fantasy_sizes)) == 1, \
            "fantasies must all have the same length"
        self._fantasies = fantasies.copy()

    @property
    def fantasies(self):
        return self._fantasies


