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

__all__ = []

import logging

try:
    from syne_tune.optimizer.schedulers.searchers.constrained.constrained_gp_fifo_searcher import (  # noqa: F401
        ConstrainedGPFIFOSearcher,
    )

    __all__.append("ConstrainedGPFIFOSearcher")
except ImportError as e:
    logging.debug(e)
