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
from typing import List


class TabulatedBenchmark(object):
    """
    Base class for tabulated benchmarks, which can be used with
    @class:`SimulatorBackend`.

    """
    def __call__(self, config: dict) -> List[dict]:
        """
        For a configuration `config`, returns all results in the order in which
        they would be reported.
        Each result needs to contain an entry for elapsed time since start of
        evaluation, whose key is `elapsed_time_attr` in the simulator backend.

        :param config: Configuration to be evaluated
        :return: Sequence of results
        """
        raise NotImplementedError()
