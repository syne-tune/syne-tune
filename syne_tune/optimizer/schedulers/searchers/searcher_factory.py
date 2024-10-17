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
from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.kde import KernelDensityEstimator
from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import RandomSearcher


searcher_cls_dict = {
    "random_search": RandomSearcher,
    "bore": Bore,
    "kde": KernelDensityEstimator,
    "bayesopt": GPFIFOSearcher,
}


def searcher_factory(searcher: str, **searcher_kwargs):
    assert searcher in searcher_cls_dict
    return searcher_cls_dict[searcher](**searcher_kwargs)
