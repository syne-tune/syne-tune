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
import functools
import logging
import time

logger = logging.getLogger(__name__)


def backoff(errorname: str, ntimes_resource_wait: int = 100, length2sleep: float = 600):
    """
    Decorator that back offs for a fixed about of s after a given error is detected
    """

    def errorcatch(some_function):
        @functools.wraps(some_function)
        def wrapper(*args, **kwargs):
            for idx in range(ntimes_resource_wait):
                try:
                    return some_function(*args, **kwargs)
                except Exception as e:
                    if not e.__class__.__name__ == errorname:
                        raise (e)

                logger.info(
                    f"{errorname} detected when calling <{some_function.__name__}>, waiting {length2sleep / 60} minutes before retring"
                )
                time.sleep(length2sleep)
                continue

        return wrapper

    return errorcatch
