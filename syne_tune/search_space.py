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

# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.

# ========================================================
# DEPRECATED:
# `search_space` is deprecated, use `config_space` instead
# ========================================================

import logging
from typing import Dict, List
import argparse

from syne_tune.config_space import uniform as _uniform, \
    loguniform as _loguniform, choice as _choice, \
    randint as _randint, lograndint as _lograndint, \
    finrange as _finrange, logfinrange as _logfinrange, \
    add_to_argparse as _add_to_argparse

logger = logging.getLogger(__name__)


def _deprecated_warning(name: str):
    logger.warning(
        f"\n*****\n***** syne_tune.search_space.{name} is deprecated, use "
        f"syne_tune.config_space.{name} instead!\n"
        "***** syne_tune.search_space will be removed in the next release\n*****")


def uniform(lower: float, upper: float):
    _deprecated_warning('uniform')
    return _uniform(lower, upper)


def loguniform(lower: float, upper: float):
    _deprecated_warning('loguniform')
    return _loguniform(lower, upper)


def choice(categories: List):
    _deprecated_warning('choice')
    return _choice(categories)


def randint(lower: int, upper: int):
    _deprecated_warning('randint')
    return _randint(lower, upper)


def lograndint(lower: int, upper: int):
    _deprecated_warning('lograndint')
    return _lograndint(lower, upper)


def finrange(lower: float, upper: float, size: int, cast_int: bool = False):
    _deprecated_warning('finrange')
    return _finrange(lower, upper, size, cast_int=cast_int)


def logfinrange(lower: float, upper: float, size: int, cast_int: bool = False):
    _deprecated_warning('logfinrange')
    return _logfinrange(lower, upper, size, cast_int=cast_int)


def add_to_argparse(parser: argparse.ArgumentParser, config_space: Dict):
    _deprecated_warning('add_to_argparse')
    return _add_to_argparse(parser, config_space)
