import numpy as np

from syne_tune.optimizer.scheduler import TrialScheduler


class BracketDistribution:
    """
    Configures asynchronous multi-fidelity schedulers such as
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` with
    distribution over brackets. This distribution can be fixed up front, or
    change adaptively during the course of an experiment. It has an effect
    only if the scheduler is run with more than one bracket.
    """

    def __call__(self) -> np.ndarray:
        """
        :return: Distribution over brackets
        """
        raise NotImplementedError

    def configure(self, scheduler: TrialScheduler):
        """
        This method is called in by the scheduler just after
        ``self.searcher.configure_scheduler``. The searcher must be accessible
        via ``self.searcher``.
        The :meth:`__call__` method cannot be used before this method has been
        called.
        """
        raise NotImplementedError


class DefaultHyperbandBracketDistribution(BracketDistribution):
    """
    Implements default bracket distribution, where probability for each bracket
    is proportional to the number of slots in each bracket in synchronous
    Hyperband.
    """

    def __init__(self):
        self.num_brackets = None
        self.rung_levels = None
        self._distribution = None

    def configure(self, scheduler: TrialScheduler):
        from syne_tune.optimizer.schedulers import LegacyHyperbandScheduler

        assert isinstance(
            scheduler, LegacyHyperbandScheduler
        ), "Scheduler must be HyperbandScheduler"
        self.num_brackets = scheduler.terminator.num_brackets
        self.rung_levels = scheduler.rung_levels + [scheduler.max_t]
        self._set_distribution()

    def __call__(self) -> np.ndarray:
        assert self._distribution is not None, "Call 'configure' first"
        return self._distribution

    def _set_distribution(self):
        if self.num_brackets > 1:
            smax_plus1 = len(self.rung_levels)
            assert self.num_brackets <= smax_plus1  # Sanity check
            self._distribution = np.array(
                [
                    smax_plus1 / ((smax_plus1 - s) * self.rung_levels[s])
                    for s in range(self.num_brackets)
                ]
            )
            self._distribution /= self._distribution.sum()
        else:
            self._distribution = np.ones(1)
