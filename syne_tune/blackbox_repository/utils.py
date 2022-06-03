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
from typing import List, Optional, Tuple

from syne_tune.blackbox_repository.blackbox import Blackbox


def metrics_for_configuration(
    blackbox: Blackbox,
    config: dict,
    resource_attr: str,
    fidelity_range: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Returns all results for configuration `config` at fidelities in range
    `fidelity_range`.

    :param blackbox: Blackbox
    :param config: Configuration
    :param resource_attr: Name of resource attribute
    :param fidelity_range: Range [min_f, max_f], only fidelities in this range
        (both ends inclusive) are returned. Default is no filtering
    :param seed: Seed for queries to blackbox. Drawn at random if not
        given
    :return: List of result dicts

    """
    all_fidelities = blackbox.fidelity_values
    assert all_fidelities is not None, "Blackbox must come with fidelities"
    res = []
    if fidelity_range is None:
        fidelity_range = (min(all_fidelities), max(all_fidelities))
    else:
        assert (
            len(fidelity_range) == 2 and fidelity_range[0] <= fidelity_range[1]
        ), f"fidelity_range = {fidelity_range} must be tuple (min, max), min <= max"
    objective_values = blackbox._objective_function(config, seed=seed)
    for fidelity, value in enumerate(all_fidelities):
        if value >= fidelity_range[0] and value <= fidelity_range[1]:
            res_dict = dict(zip(blackbox.objectives_names, objective_values[fidelity]))
            res_dict[resource_attr] = value
            res.append(res_dict)
    return res
