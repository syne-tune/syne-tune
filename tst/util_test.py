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
import tempfile

from syne_tune.backend import LocalBackend


def temporary_local_backend(entry_point: str):
    """
    :param entry_point:
    :return: a backend whose files are deleted after finishing to avoid side-effects. This is used in unit-tests.
    """
    with tempfile.TemporaryDirectory() as local_path:
        backend = LocalBackend(entry_point=entry_point)
        backend.set_path(results_root=local_path)
        return backend
