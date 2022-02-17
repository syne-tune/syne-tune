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
from typing import Dict

from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband import \
    SynchronousHyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system \
    import SynchronousHyperbandRungSystem
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults, Integer, Float, filter_by_key


_ARGUMENT_KEYS = {
    'grace_period', 'max_resource_level', 'reduction_factor',
    'brackets'}

_DEFAULT_OPTIONS = {
    'grace_period': 1,
    'reduction_factor': 3,
    'brackets': 1,
}

_CONSTRAINTS = {
    'grace_period': Integer(1, None),
    'max_resource_level': Integer(1, None),
    'reduction_factor': Float(2, None),
    'brackets': Integer(1, None),
}


class SynchronousGeometricHyperbandScheduler(SynchronousHyperbandScheduler):
    """
    Special case of :class:`SynchronousHyperbandScheduler` with rung system
    defined by geometric sequences (see
    `SynchronousHyperbandRungSystem.geometric`). This is the most frequently
    used case.

    Additional parameters
    ---------------------
    grace_period : int
        Smallest (resource) rung level. Must be positive int.
    reduction_factor : float
        Approximate ratio of successive rung levels. Must be >= 2.
    max_resource_level : int (optional)
        Largest rung level, corresponds to `max_t` in
        :class:`FIFOScheduler`. Must be positive int larger than
        `grace_period`. If this is not given, it is inferred like in
        :class:`FIFOScheduler`.
    brackets : int
        Number of brackets to be used. The value 1 corresponds to successive
        halving. Is capped to the largest number of supported brackets.

    """
    def __init__(self, config_space: Dict, **kwargs):
        TrialScheduler.__init__(self, config_space)
        # Additional parameters to determine rung systems
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS,
            dict_name='scheduler_options')
        self.grace_period = kwargs['grace_period']
        self.reduction_factor = kwargs['reduction_factor']
        num_brackets = kwargs['brackets']
        max_resource_level = self._infer_max_resource_level(
            kwargs.get('max_resource_level'), kwargs.get('max_resource_attr'))
        assert max_resource_level is not None, \
            "The maximum resource level must be specified, either as " +\
            "explicit argument 'max_resource_level', or as entry in " +\
            "'config_space', with name 'max_resource_attr'"
        self.max_resource_level = max_resource_level
        bracket_rungs = SynchronousHyperbandRungSystem.geometric(
            min_resource=self.grace_period,
            max_resource=max_resource_level,
            reduction_factor=self.reduction_factor,
            num_brackets=num_brackets)
        self._create_internal(
            bracket_rungs, **filter_by_key(kwargs, _ARGUMENT_KEYS))
