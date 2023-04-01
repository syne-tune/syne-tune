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
import logging
from typing import List, Tuple, Dict, Any, Optional
from sortedcontainers import SortedList

logger = logging.getLogger(__name__)


class RungEntry:
    """
    Represents entry in a rung. This class is extended by rung level systems
    which need to maintain more information per entry.

    :param trial_id: ID of trial
    :param metric_val: Metric value
    """

    def __init__(self, trial_id: str, metric_val: float):
        self.trial_id = trial_id
        self.metric_val = metric_val


class Rung:
    """
    :param level: Rung level :math:`r_j`
    :param prom_quant: promotion quantile :math:`q_j`
    :param data: Data of all previous jobs reaching the level. This list is
        kept sorted w.r.t. ``metric_val``, so that best values come first
    """

    def __init__(
        self,
        level: int,
        prom_quant: float,
        mode: str,
        data: Optional[List[RungEntry]] = None,
    ):
        self.level = level
        assert 0 < prom_quant < 1
        self.prom_quant = prom_quant
        assert mode in ["min", "max"]
        self._is_min = mode == "min"
        # ``SortedList`` supports insertion in :math:`O(log n)`
        if data is None:
            data = []
        sign = 1 if self._is_min else -1
        self.data = SortedList(iterable=data, key=lambda x: sign * x.metric_val)
        # We need to test whether ``trial_id`` is in the rung, but not at which
        # position it is
        self._trial_ids = {entry.trial_id for entry in data}

    def add(self, entry: RungEntry):
        self.data.add(entry)
        self._trial_ids.add(entry.trial_id)

    def pop(self, pos: int) -> RungEntry:
        entry = self.data.pop(pos)
        self._trial_ids.remove(entry.trial_id)
        return entry

    def __contains__(self, trial_id: str):
        return trial_id in self._trial_ids

    def __len__(self) -> int:
        return len(self.data)

    def quantile(self) -> Optional[float]:
        """
        Returns same value as ``numpy.quantile(metric_vals, q)``, where
        ``metric_vals`` are the metric values in ``data``, and
        ``q = prom_quant`` if ``mode == "min"``, ``q = ``1 - prom_quant``
        otherwise. If ``len(data) < 2``, we return ``None``.

        See `here <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>`__.
        The default for ``numpy.quantile`` is ``method="linear"``.

        :return: See above
        """
        len_data = len(self.data)
        if len_data < 2:
            return None
        # In ``numpy.quantile``, values are sorted in increasing order. This is
        # the case for ``self.data`` for "min" mode, otherwise ``self.data`` is
        # sorted in decreasing order, so we need to use it in reverse
        q = self.prom_quant if self._is_min else 1 - self.prom_quant
        virt_index = (len_data - 1) * q + 1
        index = int(virt_index)  # i in ``numpy.quantile`` docs
        assert 1 <= index < len_data  # Sanity check
        frac_part = virt_index - index  # g in ``numpy.quantile`` docs
        if self._is_min:
            left_pos = index - 1
            g = frac_part
        else:
            left_pos = len_data - index - 1
            g = 1 - frac_part
        values = [x.metric_val for x in self.data.islice(left_pos, left_pos + 2)]
        return g * values[1] + (1 - g) * values[0]


PausedTrialsResult = List[Tuple[str, int, float, int]]


class RungSystem:
    """
    Terminology: Trials emit results at certain resource levels (e.g., epoch
    numbers). Some resource levels are rung levels, this is where scheduling
    decisions (stop, continue or pause, resume) are taken. For a running trial,
    the next rung level (or ``max_t``) it will reach is called its next
    milestone.

    Note that ``rung_levels``, ``promote_quantiles`` can be empty. All
    entries of ``rung_levels`` are smaller than ``max_t``.

    :param rung_levels: List of rung levels (positive int, increasing)
    :param promote_quantiles: List of promotion quantiles at each rung level
    :param metric: Name of metric to optimize
    :param mode: "min" or "max"
    :param resource_attr: Name of resource attribute
    :param max_t: Largest resource level
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
    ):
        self.num_rungs = len(rung_levels)
        assert len(promote_quantiles) == self.num_rungs
        assert self.num_rungs == 0 or rung_levels[-1] < max_t
        self._metric = metric
        self._mode = mode
        self._resource_attr = resource_attr
        self._max_t = max_t
        self._rungs = [
            Rung(level=x, prom_quant=y, mode=mode)
            for x, y in reversed(list(zip(rung_levels, promote_quantiles)))
        ]

    def on_task_schedule(self, new_trial_id: str) -> Dict[str, Any]:
        """Called when new task is to be scheduled.

        For a promotion-based rung system, check whether any trial can be
        promoted. If so, return dict with keys "trial_id", "resume_from"
        (rung level where trial is paused), "milestone" (next rung level
        the trial will reach, or None).

        If no trial can be promoted, or if the rung system is not
        promotion-based, the returned dictionary must not contain the
        "trial_id" key. It is nevertheless passed back via ``extra_kwargs`` in
        :meth:`~syne_tune.optimizer.schedulers.hyperband.HyperbandBracketManager.on_task_schedule`.
        The default is to return an empty dictionary, but some special subclasses
        can use this to return information in case a trial is not promoted.

        If no trial can be promoted, or if the rung system is not
        promotion-based, the returned dictionary must not contain the
        "trial_id" key. It is nevertheless passed back via ``extra_kwargs`` in
        :meth:`~syne_tune.optimizer.schedulers.hyperband.HyperbandBracketManager.on_task_schedule`.
        The default is to return an empty dictionary, but some special subclasses
        can use this to return information in case a trial is not promoted.

        :param new_trial_id: ID for new trial as passed to :meth:`_suggest`.
            Only needed by specific subclasses
        :return: See above
        """
        raise NotImplementedError

    def on_task_add(self, trial_id: str, skip_rungs: int, **kwargs):
        """Called when new task is started.

        :param trial_id: ID of trial to be started
        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :param kwargs: Additional arguments
        """
        pass

    def on_task_report(
        self, trial_id: str, result: Dict[str, Any], skip_rungs: int
    ) -> Dict[str, Any]:
        """Called when a trial reports metric results.

        Returns dict with keys "milestone_reached" (trial reaches its milestone),
        "task_continues" (trial should continue; otherwise it is stopped or
        paused), "next_milestone" (next milestone it will reach, or None).
        For certain subclasses, there may be additional entries.

        :param trial_id: ID of trial which reported results
        :param result: Reported metrics
        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :return: See above
        """
        raise NotImplementedError

    def on_task_remove(self, trial_id: str):
        """Called when task is removed.

        :param trial_id: ID of trial which is to be removed
        """
        pass

    def get_first_milestone(self, skip_rungs: int) -> int:
        """
        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :return: First milestone to be considered
        """
        return (
            self._rungs[-(skip_rungs + 1)].level
            if skip_rungs < self.num_rungs
            else self._max_t
        )

    def _milestone_rungs(self, skip_rungs: int) -> List[Rung]:
        if skip_rungs > 0:
            return self._rungs[:(-skip_rungs)]
        else:
            return self._rungs

    def get_milestones(self, skip_rungs: int) -> List[int]:
        """
        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :return: All milestones to be considered, in decreasing order; does
            not include ``max_t``
        """
        milestone_rungs = self._milestone_rungs(skip_rungs)
        return [x.level for x in milestone_rungs]

    def snapshot_rungs(self, skip_rungs: int) -> List[Tuple[int, List[RungEntry]]]:
        """
        A snapshot is a list of rung levels with entries ``(level, data)``,
        ordered from top to bottom (largest rung first).

        :param skip_rungs: This number of the smallest rung levels are not
            considered milestones for this task
        :return: Snapshot (see above)
        """
        milestone_rungs = self._milestone_rungs(skip_rungs)
        return [(x.level, list(x.data)) for x in milestone_rungs]

    @staticmethod
    def does_pause_resume() -> bool:
        """
        :return: Is this variant doing pause and resume scheduling, in the
            sense that trials can be paused and resumed later?
        """
        raise NotImplementedError

    def support_early_checkpoint_removal(self) -> bool:
        """
        :return: Do we support early checkpoint removal via
            :meth:`paused_trials`?
        """
        return False

    def paused_trials(self, resource: Optional[int] = None) -> PausedTrialsResult:
        """
        Only for pause and resume schedulers (:meth:`does_pause_resume` returns
        ``True``), where trials can be paused at certain rung levels only.
        If ``resource`` is not given, returns list of all paused trials
        ``(trial_id, rank, metric_val, level)``, where ``level`` is
        the rung level, and ``rank`` is the rank of the trial in the rung
        (0 for the best metric value). If ``resource`` is given, only the
        paused trials in the rung of this level are returned. If ``resource``
        is not a rung level, the returned list is empty.

        :param resource: If given, paused trials of only this rung level are
            returned. Otherwise, all paused trials are returned
        :return: See above
        """
        return []

    def information_for_rungs(self) -> List[Tuple[int, int, float]]:
        """
        :return: List of ``(resource, num_entries, prom_quant)``, where
            ``resource`` is a rung level, ``num_entries`` the number of entries
            in the rung, and ``prom_quant`` the promotion quantile
        """
        return [(rung.level, len(rung), rung.prom_quant) for rung in self._rungs]

    def _rung_pos_for_level(self, level: int) -> Optional[int]:
        try:
            rung_pos = next(i for i, v in enumerate(self._rungs) if v.level == level)
        except StopIteration:
            rung_pos = None
        return rung_pos


class StoppingRungSystem(RungSystem):
    r"""
    The decision on whether a trial :math:`\mathbf{x}` continues or is stopped
    at a rung level :math:`r`, is taken in :meth:`on_task_report`. To this end,
    the metric value :math:`f(\mathbf{x}, r)` is inserted into :math:`r.data`.
    Then:

    .. math::

       \mathrm{continues}(\mathbf{x}, r)\; \Leftrightarrow\;
       f(\mathbf{x}, r) \le \mathrm{np.quantile}(r.data, r.prom\_quant)

    in case ``mode == "min"``. See also :meth:`_task_continues`.
    """

    def _task_continues(
        self,
        trial_id: str,
        metric_val: float,
        rung: Rung,
    ) -> bool:
        r"""
        :param trial_id: ID of trial
        :param metric_val: :math:`f(\mathbf{x}, r)` for trial
            :math:`\mathbf{x}` at rung :math:`r`
        :param rung: Rung where new entry has just been inserted
        :return: Continue trial? Stop otherwise
        """
        cutoff = rung.quantile()
        if cutoff is None:
            return True
        return metric_val <= cutoff if self._mode == "min" else metric_val >= cutoff

    def on_task_schedule(self, new_trial_id: str) -> Dict[str, Any]:
        return dict()

    def on_task_report(
        self, trial_id: str, result: Dict[str, Any], skip_rungs: int
    ) -> Dict[str, Any]:
        resource = result[self._resource_attr]
        metric_val = result[self._metric]
        if resource == self._max_t:
            task_continues = False
            milestone_reached = True
            next_milestone = None
        else:
            task_continues = True
            milestone_reached = False
            next_milestone = self._max_t
            for rung in self._milestone_rungs(skip_rungs):
                milestone = rung.level
                if not (resource < milestone or trial_id in rung):
                    # Note: It is important for model-based searchers that
                    # milestones are reached exactly, not jumped over. In
                    # particular, if a future milestone is reported via
                    # register_pending, its reward value has to be passed
                    # later on via update.
                    if resource > milestone:
                        logger.warning(
                            f"resource = {resource} > {milestone} = milestone. "
                            "Make sure to report time attributes covering all milestones.\n"
                            f"Continueing, but milestone {milestone} has been skipped."
                        )
                    else:
                        milestone_reached = True
                        # Enter new metric value before checking condition
                        rung.add(RungEntry(trial_id=trial_id, metric_val=metric_val))
                        task_continues = self._task_continues(
                            trial_id=trial_id,
                            metric_val=metric_val,
                            rung=rung,
                        )
                    break
                next_milestone = milestone
        return {
            "task_continues": task_continues,
            "milestone_reached": milestone_reached,
            "next_milestone": next_milestone,
        }

    @staticmethod
    def does_pause_resume() -> bool:
        return False
