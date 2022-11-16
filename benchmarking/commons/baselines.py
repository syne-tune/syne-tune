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
from typing import Dict, Optional, Callable

from syne_tune.config_space import ordinal, Categorical
from syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import import (
    CONFIGURATION_SPACE,
)
from syne_tune.optimizer.scheduler import TrialScheduler


@dataclass
class MethodArguments:
    """
    Arguments for creating HPO method (scheduler). We collect the union of
    optional arguments for all use cases here.

    :param config_space: Configuration space (typically taken from benchmark
        definition)
    :param metric: Name of metric to optimize
    :param mode: Whether `metric` is minimized ("min") or maximized ("max")
    :param random_seed: Different for different repetitions
    :param resource_attr: Name of resource attribute
    :param max_resource_attr: Name of `max_resource_value` in `config_space`.
        One of `max_resource_attr`, `max_t` is mandatory
    :param max_t: Value for `max_resource_value`. One of `max_resource_attr`,
        `max_t` is mandatory
    :param transfer_learning_evaluations: Support for transfer learning. Only
        for simulator back-end experiments right now
    :param use_surrogates: For simulator back-end experiments, defaults to False
    :param num_brackets: Parameter for Hyperband schedulers, optional
    :param verbose: If True, fine-grained log information about the tuning is
        printed. Defaults to False
    :param num_samples: Parameter for Hyper-Tune schedulers, defaults to 50
    :param fcnet_ordinal: Only for simulator back-end and `fcnet` blackbox.
        This blackbox is tabulated with finite domains, one of which has
        irregular spacing. If `fcnet_ordinal="none"`, this is left as
        categorical, otherwise we use ordinal encoding with
        `kind=fcnet_ordinal`.
    :param scheduler_kwargs: If given, overwrites defaults of scheduler
        arguments
    """

    config_space: dict
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    max_resource_attr: Optional[str] = None
    max_t: Optional[int] = None
    transfer_learning_evaluations: Optional[Dict] = None
    use_surrogates: bool = False
    num_brackets: Optional[int] = None
    verbose: Optional[bool] = False
    num_samples: int = 50
    fcnet_ordinal: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None


MethodDefinitions = Dict[str, Callable[[MethodArguments], TrialScheduler]]


def search_options(args: MethodArguments) -> dict:
    return {"debug_log": args.verbose}


def convert_categorical_to_ordinal(config_space: dict) -> dict:
    """
    :param config_space: Configuration space
    :return: New configuration space where all categorical domains are
        replaced by ordinal ones (with `kind="equal"`)
    """
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in config_space.items()
    }


def _is_fcnet(config_space: dict) -> bool:
    fcnet_keys = set(CONFIGURATION_SPACE.keys())
    return fcnet_keys.issubset(set(config_space.keys()))


def convert_categorical_to_ordinal_numeric(
    config_space: dict,
    kind: Optional[str],
    do_convert: Optional[Callable[[dict], bool]] = None,
) -> dict:
    """
    Converts categorical domains to ordinal ones, of type `kind`. This is not
    done if `kind="none"`, or if `do_convert(config_space) == False`.

    :param config_space: Configuration space
    :param kind: Type of ordinal, or `"none"`
    :param do_convert: See above. The default is testing for the config space
        of the `fcnet` blackbox
    :return: New configuration space
    """
    if do_convert is None:
        do_convert = _is_fcnet
    if kind is None or kind == "none" or not do_convert(config_space):
        return config_space
    else:
        return {
            name: (
                ordinal(domain.categories, kind=kind)
                if isinstance(domain, Categorical)
                else domain
            )
            for name, domain in config_space.items()
        }
