from abc import ABC, abstractmethod
import time
from datetime import datetime


class TimeKeeper(ABC):
    """
    To be used by tuner, backend, and scheduler to measure time differences
    and wait for a specified amount of time. By centralizing this
    functionality here, we can support simulating experiments much faster than
    real time if the training evaluation function corresponds to a tabulated
    benchmark.

    """
    @abstractmethod
    def start_of_time(self):
        """
        Called at the start of the experiment. Can be called multiple times
        if several experiments are run in sequence.
        """
        pass

    @abstractmethod
    def time(self) -> float:
        """
        :return: Time elapsed since the start of the experiment
        """
        pass

    @abstractmethod
    def time_stamp(self) -> datetime:
        """
        :return: Timestamp (datetime) corresponding to `time()`
        """
        pass

    @abstractmethod
    def advance(self, step: float):
        """
        Advance time by `step`. For real time, this means we sleep for
        `step`.
        """
        pass


class RealTimeKeeper(TimeKeeper):
    def __init__(self):
        self._start_time = None

    def start_of_time(self):
        # This can be called multiple times, if multiple experiments are
        # run in sequence
        self._start_time = time.time()

    def _assert_has_started(self):
        assert self._start_time is not None, \
            "RealTimeKeeper needs to be started, by calling start_of_time"

    def time(self) -> float:
        self._assert_has_started()
        return time.time() - self._start_time

    def time_stamp(self) -> datetime:
        self._assert_has_started()
        return datetime.now()

    def advance(self, step: float):
        self._assert_has_started()
        assert step >= 0
        time.sleep(step)
