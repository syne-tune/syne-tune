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

__all__ = []

import logging

from syne_tune.try_import import try_import_backend_message

try:
    from syne_tune.backend.local_backend import LocalBackend  # noqa: F401

    __all__.append("LocalBackend")
except ImportError:
    logging.info(try_import_backend_message("LocalBackend"))

try:
    from syne_tune.backend.python_backend.python_backend import (  # noqa: F401
        PythonBackend,
    )

    __all__.append("PythonBackend")
except ImportError:
    logging.info(try_import_backend_message("PythonBackend"))

try:
    from syne_tune.backend.sagemaker_backend.sagemaker_backend import (  # noqa: F401
        SageMakerBackend,
    )

    __all__.append("SageMakerBackend")
except ImportError:
    logging.info(try_import_backend_message("SageMakerBackend"))
