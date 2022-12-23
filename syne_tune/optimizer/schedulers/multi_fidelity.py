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


class MultiFidelitySchedulerMixin:
    """
    Declares properties which are required for multi-fidelity schedulers.
    """

    @property
    def resource_attr(self) -> str:
        """
        :return: Name of resource attribute in reported results
        """
        raise NotImplementedError

    @property
    def max_resource_level(self) -> int:
        """
        :return: Maximum resource level
        """
        raise NotImplementedError

    @property
    def rung_levels(self) -> List[int]:
        """
        :return: Rung levels (positive int; increasing), may or may not
            include ``max_resource_level``
        """
        raise NotImplementedError

    @property
    def searcher_data(self) -> str:
        """
        :return: Relevant only if a model-based searcher is used.
            Example: For NN tuning and ``resource_attr == "epoch"``, we receive
            a result for each epoch, but not all epoch values are also rung
            levels. ``searcher_data`` determines which of these results are
            passed to the searcher. As a rule, the more data the searcher
            receives, the better its fit, but also the more expensive
            :meth:`get_config` may become. Choices:

            * "rungs": Only results at rung levels. Cheapest
            * "all": All results. Most expensive
            * "rungs_and_last": Results at rung levels plus last recent one.
              Not available for all multi-fidelity schedulers
        """
        raise NotImplementedError

    @property
    def num_brackets(self) -> int:
        """
        :return: Number of brackets (i.e., rung level systems). If the scheduler
            does not use brackets, it has to return 1
        """
        raise NotImplementedError
