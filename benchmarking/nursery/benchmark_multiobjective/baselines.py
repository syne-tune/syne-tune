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
from typing import Dict, Any, Optional, List, Union

from benchmarking.commons.default_baselines import (
    RandomSearch,
    MOREABench,
    LSOBOBench,
)


class Methods:
    RS = "RS"
    MOREA = "MOREA"
    LSOBO = "LSOBO"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(method_arguments),
    Methods.MOREA: lambda method_arguments: MOREABench(method_arguments),
    Methods.LSOBO: lambda method_arguments: LSOBOBench(method_arguments),
}
