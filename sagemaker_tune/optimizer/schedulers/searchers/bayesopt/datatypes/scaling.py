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
from abc import ABC, abstractmethod
import numpy as np


class Scaling(ABC):
    @abstractmethod
    def to_internal(self, value: float) -> float:
        pass

    @abstractmethod
    def from_internal(self, value: float) -> float:
        pass

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def __eq__(self, other):
        # For usage in tests. Make sure to edit if parameters are added.
        return self.__class__ == other.__class__


class LinearScaling(Scaling):
    def to_internal(self, value: float) -> float:
        return value

    def from_internal(self, value: float) -> float:
        return value


class LogScaling(Scaling):
    def to_internal(self, value: float) -> float:
        assert value > 0, "Value must be strictly positive to be log-scaled."
        return np.log(value)

    def from_internal(self, value: float) -> float:
        return np.exp(value)


class ReverseLogScaling(Scaling):
    def to_internal(self, value: float) -> float:
        assert 0 <= value < 1, \
            "Value must be between 0 (inclusive) and 1 (exclusive) to be reverse-log-scaled."
        return -np.log(1.0 - value)

    def from_internal(self, value: float) -> float:
        return 1.0 - np.exp(-value)
