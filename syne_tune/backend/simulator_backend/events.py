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
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq


@dataclass
class Event:
    """
    Base class for events dealt with in the simulator.

    """

    trial_id: int


@dataclass
class StartEvent(Event):
    """
    Start training evaluation function for `trial_id`. In fact, the function
    is run completely, and `OnTrialResultEvent` events and one `CompleteEvent`
    are generated.

    """


@dataclass
class CompleteEvent(Event):
    """
    Job for trial `trial_id` completes with status `status`. This is registered
    at the back-end.

    """

    status: str


@dataclass
class StopEvent(Event):
    """
    Job for trial `trial_id` is stopped. This leads to all later events for
    `trial_id` to be deleted, and a new `CompleteEvent`.

    """


@dataclass
class OnTrialResultEvent(Event):
    """
    Result reported by some worker arrives at the back-end and is registered
    there.

    """

    result: dict


EventHeapType = List[Tuple[float, int, Event]]


class SimulatorState:
    """
    Maintains the state of the simulator, in particular the event heap.

    `event_heap` is the priority queue for events, the key being `(time, cnt)`,
    where `time` is the event time, and `cnt` is a non-negative int used to
    break ties. When an event is added, the `cnt` value is taken from
    `events_added`. This means that ties are broken first_in_first_out.

    """

    def __init__(
        self, event_heap: Optional[EventHeapType] = None, events_added: int = 0
    ):
        if event_heap is None:
            event_heap = []
        self.event_heap = event_heap
        self.events_added = events_added

    def push(self, event: Event, event_time: float):
        """
        Push new event onto heap

        :param event:
        :param event_time:
        """
        heapq.heappush(self.event_heap, (event_time, self.events_added, event))
        self.events_added += 1

    def remove_events(self, trial_id: int):
        """
        Remove all events with trial_id equal to `trial_id`.

        :param trial_id:
        """
        self.event_heap = [
            elem for elem in self.event_heap if elem[2].trial_id != trial_id
        ]
        heapq.heapify(self.event_heap)

    def next_until(self, time_until: float) -> Optional[Tuple[float, Event]]:
        """
        Returns (and pops) event on top of heap, if event time is <=
        `time_until`. Otherwise, returns None.

        :param time_until:
        :return:
        """
        result = None
        if self.event_heap:
            top_time, _, top_event = self.event_heap[0]
            if top_time <= time_until:
                heapq.heappop(self.event_heap)
                result = (top_time, top_event)
        return result
