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
from datetime import datetime


class TimeKeeper:
    """
    To be used by tuner, backend, and scheduler to measure time differences
    and wait for a specified amount of time. By centralizing this
    functionality here, we can support simulating experiments much faster than
    real time if the training evaluation function corresponds to a tabulated
    benchmark.

    """

    def start_of_time(self):
        """
        Called at the start of the experiment. Can be called multiple times
        if several experiments are run in sequence.
        """
        raise NotImplementedError

    def time(self) -> float:
        """
        :return: Time elapsed since the start of the experiment
        """
        raise NotImplementedError

    def time_stamp(self) -> datetime:
        """
        :return: Timestamp (datetime) corresponding to `time()`
        """
        raise NotImplementedError

    def advance(self, step: float):
        """
        Advance time by `step`. For real time, this means we sleep for
        `step`.
        """
        raise NotImplementedError


class RealTimeKeeper(TimeKeeper):
    def __init__(self):
        self._start_time = None

    def start_of_time(self):
        # This can be called multiple times, if multiple experiments are
        # run in sequence
        self._start_time = time.time()

    def _assert_has_started(self):
        assert (
            self._start_time is not None
        ), "RealTimeKeeper needs to be started, by calling start_of_time"

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
