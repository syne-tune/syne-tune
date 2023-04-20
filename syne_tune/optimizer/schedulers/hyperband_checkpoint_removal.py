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
from typing import Optional, Dict, Any, Callable

from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCallback,
    HyperbandRemoveCheckpointsBaselineCallback,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuning_status import TuningStatus
from syne_tune.tuner_callback import TunerCallback


def create_callback_for_checkpoint_removal(
    callback_kwargs: Dict[str, Any],
    stop_criterion: Callable[[TuningStatus], bool],
) -> Optional[TunerCallback]:
    if isinstance(stop_criterion, StoppingCriterion):
        # Obtain ``max_wallclock_time`` from stopping criterion
        max_wallclock_time = stop_criterion.max_wallclock_time
        if max_wallclock_time is not None:
            callback_kwargs = dict(
                callback_kwargs,
                max_wallclock_time=max_wallclock_time,
            )
    if "baseline" in callback_kwargs:
        callback = HyperbandRemoveCheckpointsBaselineCallback(**callback_kwargs)
    else:
        callback = HyperbandRemoveCheckpointsCallback(**callback_kwargs)
    return callback
