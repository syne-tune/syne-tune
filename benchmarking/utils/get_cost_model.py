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
from typing import Tuple

__all__ = ['get_cost_model_for_batch_size']


def get_cost_model_for_batch_size(
        params: dict, batch_size_key: str, batch_size_range: Tuple[int, int]):
    """
    Returns cost model depending on the batch size only.

    :param params: Command line arguments
    :param batch_size_key: Name of batch size entry in config
    :param batch_size_range: (lower, upper) for batch size, both sides are
        inclusive
    :return: Cost model (or None if dependencies cannot be imported)

    """
    try:
        cost_model_type = params.get('cost_model_type')
        if cost_model_type is None:
            cost_model_type = 'quadratic_spline'
        if cost_model_type == 'biasonly':
            from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.linear_cost_model \
                import BiasOnlyLinearCostModel

            cost_model = BiasOnlyLinearCostModel()
        else:
            from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.sklearn_cost_model \
                import UnivariateSplineCostModel

            def scalar_attribute(config_dct):
                return float(config_dct[batch_size_key])

            assert cost_model_type in {'quadratic_spline', 'cubic_spline'}, \
                f"cost_model_type = '{cost_model_type}' is not supported"
            cost_model = UnivariateSplineCostModel(
                scalar_attribute=scalar_attribute,
                input_range=batch_size_range,
                spline_degree=(2 if cost_model_type[0] == 'q' else 3))
        return cost_model
    except Exception:
        return None
