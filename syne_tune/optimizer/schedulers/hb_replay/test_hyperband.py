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
import numbers
from typing import Optional

from syne_tune.optimizer.schedulers.hyperband import (
    HyperbandScheduler,
    HyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.hyperband_stopping import (
    RungSystem,
    RungEntry,
)
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges import (
    HyperparameterRanges
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import (
    make_hyperparameter_ranges
)


def state_representation_hyperband(trial_scheduler: HyperbandScheduler):
    """
     Converts inner state of `HyperbandScheduler` into a representation which
     can be compared for equality. The mapping is not invertible.
     Useful for testing, in the context of `replay_scheduling_events`.

     Note: Depends on internals of `HyperbandScheduler` and other classes used
     there. Use for testing only.

     Note: The representation does not contain information which is irrelevant
     for comparisons, such as time stamps.

     """
    hp_ranges = make_hyperparameter_ranges(trial_scheduler.config_space)
    return {
        "active_trials": {
            k: _convert_to_representation(
                v, hp_ranges, parent_key="active_trials"
            )
            for k, v in trial_scheduler._active_trials.items()
        },
        "terminator": _convert_to_representation(
            trial_scheduler.terminator, hp_ranges
        ),
        "cost_offset": _convert_to_representation(
            trial_scheduler._cost_offset, hp_ranges
        ),
    }


def _convert_to_representation(
        x,
        hp_ranges: HyperparameterRanges,
        parent_key: Optional[str] = None):
    """
    :param x:
    :param hp_ranges: Used to map config's to match strings
    :param parent_key: Used to identify certain objects for special treatment
    :return: Representation of x
    """
    convert_to_repr = lambda a: _convert_to_representation(a, hp_ranges)
    if isinstance(x, dict):
        is_active_trials = (parent_key == 'active_trials')
        skip_keys = {"config", "time_stamp"} if is_active_trials else set()
        result = {
            str(k): convert_to_repr(v) for k, v in x.items()
            if k not in skip_keys
        }
        if is_active_trials:
            result["config"] = hp_ranges.config_to_match_string(x["config"])
    elif isinstance(x, list) or isinstance(x, tuple):
        result = [convert_to_repr(elem) for elem in x]
    elif isinstance(x, HyperbandBracketManager):
        supported_types = {"stopping", "promotion"}
        if x._scheduler_type not in supported_types:
            raise NotImplementedError(
                f"Only implement for type in {supported_types}")
        result = {
            "task_info": convert_to_repr(x._task_info),
            "rung_systems": convert_to_repr(x._rung_systems),
        }
    elif isinstance(x, RungSystem):
        result = {"rungs": convert_to_repr(x._rungs)}
        if isinstance(x, PromotionRungSystem):
            result["running"] = convert_to_repr(x._running)
    elif isinstance(x, RungEntry):
        result = convert_to_repr(x.__dict__)
    elif x is None:
        result = "None"
    elif isinstance(x, bool):
        result = "True" if x else "False"
    elif isinstance(x, numbers.Real):
        result = f"{x:.3e}"
    else:
        result = x
    return result
