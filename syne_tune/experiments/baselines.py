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
from typing import Dict, Optional, Callable, Any

from syne_tune.blackbox_repository.conversion_scripts.scripts.fcnet_import import (
    CONFIGURATION_SPACE,
)
from syne_tune.config_space import ordinal, Categorical, Domain, Ordinal
from syne_tune.optimizer.scheduler import TrialScheduler


@dataclass
class MethodArguments:
    """
    Arguments for creating HPO method (scheduler). We collect the union of
    optional arguments for all use cases here.

    :param config_space: Configuration space (typically taken from benchmark
        definition)
    :param metric: Name of metric to optimize
    :param mode: Whether ``metric`` is minimized ("min") or maximized ("max")
    :param random_seed: Different for different repetitions
    :param resource_attr: Name of resource attribute
    :param max_resource_attr: Name of ``max_resource_value`` in ``config_space``.
        One of ``max_resource_attr``, ``max_t`` is mandatory
    :param scheduler_kwargs: If given, overwrites defaults of scheduler
        arguments
    :param transfer_learning_evaluations: Support for transfer learning. Only
        for simulator backend experiments right now
    :param use_surrogates: For simulator backend experiments, defaults to
        ``False``
    :param fcnet_ordinal: Only for simulator backend and ``fcnet`` blackbox.
        This blackbox is tabulated with finite domains, one of which has
        irregular spacing. If ``fcnet_ordinal="none"``, this is left as
        categorical, otherwise we use ordinal encoding with
        ``kind=fcnet_ordinal``.
    :param num_gpus_per_trial: Only for local backend and GPU training. Number
        of GPUs assigned to a trial.
        This is passed here, because it needs to be written into the
        configuration space for some benchmarks. Defaults to 1
    """

    config_space: Dict[str, Any]
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    max_resource_attr: Optional[str] = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    transfer_learning_evaluations: Optional[Dict[str, Any]] = None
    use_surrogates: bool = False
    fcnet_ordinal: Optional[str] = None
    num_gpus_per_trial: int = 1


MethodDefinitions = Dict[str, Callable[[MethodArguments], TrialScheduler]]


def default_arguments(
    args: MethodArguments,
    extra_args: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict() if args.scheduler_kwargs is None else args.scheduler_kwargs.copy()
    result.update(
        dict(
            mode=args.mode,
            metric=args.metric,
            max_resource_attr=args.max_resource_attr,
            random_seed=args.random_seed,
        )
    )
    result.update(extra_args)
    return result


def convert_categorical_to_ordinal(config_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param config_space: Configuration space
    :return: New configuration space where all categorical domains are
        replaced by ordinal ones (with ``kind="equal"``)
    """
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in config_space.items()
    }


def _is_fcnet(config_space: Dict[str, Any]) -> bool:
    fcnet_keys = set(CONFIGURATION_SPACE.keys())
    return fcnet_keys.issubset(set(config_space.keys()))


def _to_be_converted(domain: Domain) -> bool:
    return (
        isinstance(domain, Categorical)
        and not isinstance(domain, Ordinal)
        and domain.value_type != str
    )


def convert_categorical_to_ordinal_numeric(
    config_space: Dict[str, Any],
    kind: Optional[str],
    do_convert: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """
    Converts categorical domains to ordinal ones, of type ``kind``. This is not
    done if ``kind="none"``, or if ``do_convert(config_space) == False``.

    :param config_space: Configuration space
    :param kind: Type of ordinal, or ``"none"``
    :param do_convert: See above. The default is testing for the config space
        of the ``fcnet`` blackbox
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
                if _to_be_converted(domain)
                else domain
            )
            for name, domain in config_space.items()
        }
