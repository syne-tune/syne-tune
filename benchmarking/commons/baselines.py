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
from dataclasses import dataclass
from typing import Dict, Optional

from syne_tune.config_space import ordinal, Categorical


@dataclass
class MethodArguments:
    config_space: dict
    metric: str
    mode: str
    random_seed: int
    resource_attr: str
    max_t: Optional[int] = None
    max_resource_attr: Optional[str] = None
    transfer_learning_evaluations: Optional[Dict] = None
    use_surrogates: bool = False
    num_brackets: Optional[int] = None
    verbose: Optional[bool] = False
    num_samples: int = 50


def search_options(args: MethodArguments) -> dict:
    return {"debug_log": args.verbose}


def convert_categorical_to_ordinal(args: MethodArguments) -> dict:
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in args.config_space.items()
    }
