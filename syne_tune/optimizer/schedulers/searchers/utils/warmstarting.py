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
from typing import Optional

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
    ConfigurationFilter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    Matern52,
    ProductKernelFunction,
    KernelFunction,
    RangeKernelFunction,
)


def create_hp_ranges_for_warmstarting(**kwargs) -> HyperparameterRanges:
    """
    See :class:`GPFIFOSearcher` for details on transfer_learning_task_attr',
    'transfer_learning_active_task', 'transfer_learning_active_config_space'
    as optional fields in `kwargs`. If given, they determine
    `active_config_space` and `prefix_keys` of `hp_ranges` created here,
    and they also places constraints on 'config_space'.

    This function is not only called in `gp_searcher_factory` to create
    `hp_ranges` for a new :class:`GPFIFOSearcher` object. It is also needed to
    create the `TuningJobState` containing the data to be used in warmstarting.

    """
    task_attr = kwargs.get("transfer_learning_task_attr")
    config_space = kwargs["config_space"]
    prefix_keys = None
    active_config_space = None
    if task_attr is not None:
        from syne_tune.config_space import Categorical

        active_task = kwargs.get("transfer_learning_active_task")
        assert (
            active_task is not None
        ), "transfer_learning_active_task is needed if transfer_learning_task_attr is given"
        hp_range = config_space.get(task_attr)
        assert isinstance(
            hp_range, Categorical
        ), f"config_space[{task_attr}] must be a categorical parameter"
        assert active_task in hp_range.categories, (
            f"'{active_task}' must be value in config_space[{task_attr}] "
            + f"(values: {hp_range.categories})"
        )
        prefix_keys = [task_attr]
        active_config_space = kwargs.get("transfer_learning_active_config_space")
        if active_config_space is None:
            active_config_space = config_space
        # The parameter `task_attr` in `active_config_space` must be restricted
        # to `active_task` as a single value
        task_param = Categorical(categories=[active_task])
        active_config_space = dict(active_config_space, **{task_attr: task_param})
    return make_hyperparameter_ranges(
        config_space, active_config_space=active_config_space, prefix_keys=prefix_keys
    )


def create_filter_observed_data_for_warmstarting(
    **kwargs,
) -> Optional[ConfigurationFilter]:
    """
    See :class:`GPFIFOSearcher` for details on transfer_learning_task_attr',
    'transfer_learning_active_task' as optional fields in `kwargs`.

    """
    task_attr = kwargs.get("transfer_learning_task_attr")
    if task_attr is not None:
        active_task = kwargs.get("transfer_learning_active_task")
        assert (
            active_task is not None
        ), "transfer_learning_active_task is needed if transfer_learning_task_attr is given"

        def filter_observed_data(config: Configuration) -> bool:
            return config[task_attr] == active_task

        return filter_observed_data
    else:
        return None


def create_base_gp_kernel_for_warmstarting(
    hp_ranges: HyperparameterRanges, **kwargs
) -> KernelFunction:
    """
    In the transfer learning case, the base kernel is a product of
    two `Matern52` kernels, the first non-ARD over the categorical
    parameter determining the task, the second ARD over the remaining
    parameters.

    """
    task_attr = kwargs.get("transfer_learning_task_attr")
    assert task_attr is not None
    # Note: This attribute is the first in `hp_ranges`, see
    # `create_hp_ranges_for_warmstarting`
    assert hp_ranges.internal_keys[0] == task_attr  # Sanity check
    _, categ_dim = hp_ranges.encoded_ranges[task_attr]
    full_dim = hp_ranges.ndarray_size
    model = kwargs.get("transfer_learning_model", "matern52_product")
    kernel2 = Matern52(full_dim - categ_dim, ARD=True)
    if model == "matern52_product":
        # Kernel is a product of Matern with single length scale on task_id
        # attribute, and Matern ARD kernel on the rest
        kernel1 = Matern52(categ_dim, ARD=False)
        return ProductKernelFunction(kernel1, kernel2)
    else:
        assert (
            model == "matern52_same"
        ), f"transfer_learning_model = {model} not supported"
        # Kernel is Matern ARD on rest, ignoring the task_id attribute
        return RangeKernelFunction(dimension=full_dim, kernel=kernel2, start=categ_dim)
