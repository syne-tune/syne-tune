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
from typing import Dict, Any

from syne_tune.experiments.baselines import (
    default_arguments,
    MethodArguments,
    convert_categorical_to_ordinal_numeric,
)
from syne_tune.optimizer.baselines import (
    RandomSearch as _RandomSearch,
    GridSearch as _GridSearch,
    BayesianOptimization as _BayesianOptimization,
    KDE as _KDE,
    BORE as _BORE,
    BoTorch as _BoTorch,
    REA as _REA,
    ConstrainedBayesianOptimization as _ConstrainedBayesianOptimization,
    ASHA as _ASHA,
    MOBSTER as _MOBSTER,
    HyperTune as _HyperTune,
    BOHB as _BOHB,
    DyHPO as _DyHPO,
    ASHABORE as _ASHABORE,
    SyncHyperband as _SyncHyperband,
    SyncBOHB as _SyncBOHB,
    DEHB as _DEHB,
    SyncMOBSTER as _SyncMOBSTER,
    MOREA as _MOREA,
    NSGA2 as _NSGA2,
    MORandomScalarizationBayesOpt as _MORandomScalarizationBayesOpt,
    MOLinearScalarizationBayesOpt as _MOLinearScalarizationBayesOpt,
)
from syne_tune.util import recursive_merge


def _baseline_kwargs(
    method_arguments: MethodArguments,
    kwargs: Dict[str, Any],
    is_multifid: bool = False,
) -> Dict[str, Any]:
    config_space = convert_categorical_to_ordinal_numeric(
        method_arguments.config_space, kind=method_arguments.fcnet_ordinal
    )
    da_input = dict(config_space=config_space)
    if is_multifid:
        da_input["resource_attr"] = method_arguments.resource_attr
    result = recursive_merge(
        default_arguments(method_arguments, da_input),
        kwargs,
        stop_keys=["config_space"],
    )
    return result


def RandomSearch(method_arguments: MethodArguments, **kwargs):
    return _RandomSearch(**_baseline_kwargs(method_arguments, kwargs))


def GridSearch(method_arguments: MethodArguments, **kwargs):
    return _GridSearch(**_baseline_kwargs(method_arguments, kwargs))


def BayesianOptimization(method_arguments: MethodArguments, **kwargs):
    return _BayesianOptimization(**_baseline_kwargs(method_arguments, kwargs))


def KDE(method_arguments: MethodArguments, **kwargs):
    return _KDE(**_baseline_kwargs(method_arguments, kwargs))


def BORE(method_arguments: MethodArguments, **kwargs):
    return _BORE(**_baseline_kwargs(method_arguments, kwargs))


def BoTorch(method_arguments: MethodArguments, **kwargs):
    return _BoTorch(**_baseline_kwargs(method_arguments, kwargs))


def REA(method_arguments: MethodArguments, **kwargs):
    return _REA(**_baseline_kwargs(method_arguments, kwargs))


def ConstrainedBayesianOptimization(method_arguments: MethodArguments, **kwargs):
    return _ConstrainedBayesianOptimization(
        **_baseline_kwargs(method_arguments, kwargs)
    )


def ASHA(method_arguments: MethodArguments, **kwargs):
    return _ASHA(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def MOBSTER(method_arguments: MethodArguments, **kwargs):
    return _MOBSTER(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def HyperTune(method_arguments: MethodArguments, **kwargs):
    return _HyperTune(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def BOHB(method_arguments: MethodArguments, **kwargs):
    return _BOHB(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def DyHPO(method_arguments: MethodArguments, **kwargs):
    return _DyHPO(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def ASHABORE(method_arguments: MethodArguments, **kwargs):
    return _ASHABORE(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def SyncHyperband(method_arguments: MethodArguments, **kwargs):
    return _SyncHyperband(
        **_baseline_kwargs(method_arguments, kwargs, is_multifid=True)
    )


def SyncBOHB(method_arguments: MethodArguments, **kwargs):
    return _SyncBOHB(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def DEHB(method_arguments: MethodArguments, **kwargs):
    return _DEHB(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def SyncMOBSTER(method_arguments: MethodArguments, **kwargs):
    return _SyncMOBSTER(**_baseline_kwargs(method_arguments, kwargs, is_multifid=True))


def MOREA(method_arguments: MethodArguments, **kwargs):
    return _MOREA(**_baseline_kwargs(method_arguments, kwargs))


def LSOBO(method_arguments: MethodArguments, **kwargs):
    return _MOLinearScalarizationBayesOpt(**_baseline_kwargs(method_arguments, kwargs))


def NSGA2(method_arguments: MethodArguments, **kwargs):
    return _NSGA2(**_baseline_kwargs(method_arguments, kwargs))


def MORandomScalarizationBayesOpt(method_arguments: MethodArguments, **kwargs):
    return _MORandomScalarizationBayesOpt(**_baseline_kwargs(method_arguments, kwargs))
