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

__all__ = ['dict_get']


def dict_get(params: dict, key: str, default):
    """
    Returns `params[key]` if this exists and is not None, and `default` otherwise.
    Note that this is not the same as `params.get(key, default)`. Namely, if `params[key]`
    is equal to None, this would return None, but this method returns `default`.

    This function is particularly helpful when dealing with a dict returned by
    :class:`argparse.ArgumentParser`. Whenever `key` is added as argument to the parser,
    but a value is not provided, this leads to `params[key] = None`.

    """
    v = params.get(key)
    return default if v is None else v
