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
import datetime

import pytest

from syne_tune.remote.scheduling import backoff


@backoff(errorname="AttributeError", ntimes_resource_wait=10, length2sleep=0.01)
def errorfunction(starttime: datetime.datetime):
    if starttime is None:
        raise ValueError

    # This will fail if the function tries to finish less than 0.1s after being called
    time_since_start = datetime.datetime.now() - starttime
    if time_since_start < datetime.timedelta(microseconds=int(1e5)):
        raise AttributeError

    return True


def test_backoff_completes():
    errorfunction(datetime.datetime.now())


def test_backoff_failure():
    with pytest.raises(ValueError) as e_info:
        errorfunction(None)
