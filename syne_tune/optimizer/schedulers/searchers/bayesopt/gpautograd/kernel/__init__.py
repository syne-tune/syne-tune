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
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.base import (  # noqa: F401
    KernelFunction,
    Matern52,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.exponential_decay import (  # noqa: F401
    ExponentialDecayResourcesKernelFunction,
    ExponentialDecayResourcesMeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.fabolas import (  # noqa: F401
    FabolasKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.product_kernel import (  # noqa: F401
    ProductKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.freeze_thaw import (  # noqa: F401
    FreezeThawKernelFunction,
    FreezeThawMeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.cross_validation import (  # noqa: F401
    CrossValidationMeanFunction,
    CrossValidationKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel.range_kernel import (  # noqa: F401
    RangeKernelFunction,
)

__all__ = [
    "KernelFunction",
    "Matern52",
    "ExponentialDecayResourcesKernelFunction",
    "ExponentialDecayResourcesMeanFunction",
    "FabolasKernelFunction",
    "ProductKernelFunction",
    "FreezeThawKernelFunction",
    "FreezeThawMeanFunction",
    "CrossValidationMeanFunction",
    "CrossValidationKernelFunction",
    "RangeKernelFunction",
]
