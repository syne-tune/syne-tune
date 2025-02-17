import logging
from typing import Optional, Union, List, Dict, Any

import numpy as np
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.optimizer.legacy_scheduler import LegacyTrialScheduler
from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
    NonDominatedPriority,
)

logger = logging.getLogger(__name__)


class LegacyMOASHA(LegacyTrialScheduler):
    """
    Implements MultiObjective Asynchronous Successive HAlving with different
    multiobjective sort options. References:

        | A multi-objective perspective on jointly tuning hardware and hyperparameters
        | David Salinas, Valerio Perrone, Cedric Archambeau and Olivier Cruchant
        | NAS workshop, ICLR2021.

    and

        | Multi-objective multi-fidelity hyperparameter optimization with application to fairness
        | Robin Schmucker, Michele Donini, Valerio Perrone, Cédric Archambeau

    :param config_space: Configuration space
    :param metrics: List of metric names MOASHA optimizes over
    :param mode: One of :code:`{"min", "max"}` or a list of these values (same
        size as ``metrics``). Determines whether objectives are minimized or
        maximized. Defaults to "min"
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param multiobjective_priority: The multiobjective priority that is used
        to sort multiobjective candidates. We support several choices such
        as non-dominated sort or linear scalarization, default is
        non-dominated sort.
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param grace_period: Only stop trials at least this old in time.
        The units are the same as the attribute named by ``time_attr``.
        Defaults to 1
    :param reduction_factor: Used to set halving rate and amount. This
        is simply a unit-less scalar. Defaults to 3
    :param brackets: Number of brackets. Each bracket has a different
        ``grace_period`` and number of rung levels. Defaults to 1
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metrics: List[str],
        mode: Optional[Union[str, List[str]]] = None,
        time_attr: str = "training_iteration",
        multiobjective_priority: Optional[MOPriority] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 3,
        brackets: int = 1,
    ):
        super(LegacyMOASHA, self).__init__(config_space=config_space)
        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "reduction factor not valid!"
        assert brackets > 0, "brackets must be positive!"
        if mode:
            if isinstance(mode, List):
                assert len(mode) == len(metrics), "one mode should be given per metric"
                assert all(
                    m in ["min", "max"] for m in mode
                ), "all modes should be 'min' or 'max'."
            else:
                assert mode in ["min", "max"], "``mode`` must be 'min' or 'max'."
        else:
            mode = "min"

        if multiobjective_priority is None:
            self._multiobjective_priority = NonDominatedPriority()
        else:
            self._multiobjective_priority = multiobjective_priority

        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self._brackets = [
            _Bracket(
                grace_period, max_t, reduction_factor, s, self._multiobjective_priority
            )
            for s in range(brackets)
        ]
        self._num_stopped = 0
        self._metrics = metrics
        self._mode = mode
        if isinstance(self._mode, List):
            self._metric_op = {
                metric: 1 if mode == "min" else -1
                for metric, mode in zip(metrics, self._mode)
            }
        else:
            if self._mode == "min":
                self._metric_op = dict(zip(self._metrics, [1.0] * len(self._metrics)))
            elif self._mode == "max":
                self._metric_op = dict(zip(self._metrics, [-1.0] * len(self._metrics)))
        self._time_attr = time_attr

    def metric_names(self) -> List[str]:
        return self._metrics

    def metric_mode(self) -> str:
        return self._mode

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        config = {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }
        return TrialSuggestion.start_suggestion(config)

    def on_trial_add(self, trial: Trial):
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        print(f"adding trial {trial.trial_id}")
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        self._check_metrics_are_present(result)
        if result[self._time_attr] >= self._max_t:
            action = SchedulerDecision.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            metrics = self._metric_dict(result)
            action = bracket.on_result(
                trial_id=trial.trial_id,
                cur_iter=result[self._time_attr],
                metrics=metrics,
            )
        if action == SchedulerDecision.STOP:
            self._num_stopped += 1
        return action

    def _metric_dict(self, reported_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            metric: reported_results[metric] * self._metric_op[metric]
            for metric in self._metrics
        }

    def _check_metrics_are_present(self, result: Dict[str, Any]):
        for key in [self._time_attr] + self._metrics:
            if key not in result:
                assert key in result, f"{key} not found in reported result {result}"

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        self._check_metrics_are_present(result)
        bracket = self._trial_info[trial.trial_id]
        bracket.on_result(
            trial_id=trial.trial_id,
            cur_iter=result[self._time_attr],
            metrics=self._metric_dict(result),
        )
        del self._trial_info[trial.trial_id]

    def on_trial_remove(self, trial: Trial):
        del self._trial_info[trial.trial_id]

    def is_multiobjective_scheduler(self) -> bool:
        """
        Return True if a scheduler is multi-objective.
        """
        return True


class _Bracket:
    """Bookkeeping system to track recorded values.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    """

    def __init__(
        self,
        min_t: int,
        max_t: int,
        reduction_factor: float,
        s: int,
        mo_priority: MOPriority = NonDominatedPriority(),
    ):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [
            (min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))
        ]
        self.priority = mo_priority

    def on_result(self, trial_id: int, cur_iter: int, metrics: Optional[dict]) -> str:
        action = SchedulerDecision.CONTINUE
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or trial_id in recorded:
                continue
            else:
                if not recorded:
                    # if no result was previously recorded, we saw the first result and we continue
                    action = SchedulerDecision.CONTINUE
                else:
                    # get the list of metrics seen for the rung, compute multiobjective priority and decide to continue
                    # if priority is in the top ones according to a rank induced by the ``reduction_factor``.
                    metric_recorded = np.array(
                        [list(x.values()) for x in recorded.values()]
                        + [list(metrics.values())]
                    )
                    priorities = self.priority(metric_recorded)

                    # self._plot(milestone, metric_recorded, priorities)

                    # We sort priorities at every call, assuming the cost of sort would be negligible
                    # in case this becomes slow, we could just maintain a sorted list of priorities in cost
                    # of memory.
                    ranks = np.searchsorted(sorted(priorities), priorities) / len(
                        priorities
                    )
                    new_priority_rank = ranks[-1]
                    if new_priority_rank > 1 / self.rf:
                        action = SchedulerDecision.STOP
                recorded[trial_id] = metrics
                break
        return action

    # TODO: Merge with plotting tools in ``experiments``
    def _plot(self, milestone, metric_recorded, priorities):
        """
        Plots the multiobjective candidates and the rank given by the multiobjective priority.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if len(metric_recorded) < 5:
            return

        fig, ax = plt.subplots()

        ranks = np.searchsorted(sorted(priorities), priorities)

        ax.scatter(metric_recorded[:, 0], metric_recorded[:, 1])

        font_size = 14
        plt.rcParams.update({"font.size": font_size})

        for i, indice in enumerate(ranks):
            ax.annotate(
                str(ranks[i]),
                metric_recorded[i],
                textcoords="offset points",  # how to position the text
                xytext=(-10, -10),  # distance from text to points (x,y)
            )

        # plt.legend()
        plt.tight_layout()
        plt.savefig(f"non-dominated-sorting-{milestone}.pdf")
        plt.show()
