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
import time
from datetime import datetime, timedelta

from syne_tune.backend.time_keeper import TimeKeeper


class SimulatedTimeKeeper(TimeKeeper):
    """
    Here, time is simulated. It needs to be advanced explicitly.

    In addition, `mark_exit` and `real_time_since_last_recent_exit` are used to
    measure real time spent outside the back-end (i.e., in the tuner loop and
    scheduler). Namely, every method of :class:`SimulatorBackend` calls
    `mark_exit` before leaving, and `real_time_since_last_recent_exit` at the
    start, advancing the time counter accordingly.

    """

    def __init__(self):
        self._current_time = None
        self._start_time_stamp = None
        self._last_recent_exit = None

    @property
    def start_time_stamp(self) -> datetime:
        """
        :return: Time stamp (datetime) of (last recent) call of
        `start_of_time`
        """
        self._assert_has_started()
        return self._start_time_stamp

    def start_of_time(self):
        # This can be called multiple times, if multiple experiments are
        # run in sequence
        self._current_time = 0
        self._start_time_stamp = datetime.now()
        self.mark_exit()

    def _assert_has_started(self):
        assert (
            self._current_time is not None
        ), "RealTimeKeeper needs to be started, by calling start_of_time"

    def time(self) -> float:
        self._assert_has_started()
        return self._current_time

    def time_stamp(self) -> datetime:
        self._assert_has_started()
        return self._start_time_stamp + timedelta(seconds=self._current_time)

    def advance(self, step: float):
        self._assert_has_started()
        assert step >= 0
        self._current_time += step

    def advance_to(self, to_time: float):
        self._assert_has_started()
        self._current_time = max(to_time, self._current_time)

    def mark_exit(self):
        self._last_recent_exit = time.time()

    def real_time_since_last_recent_exit(self) -> float:
        self._assert_has_started()
        return time.time() - self._last_recent_exit
