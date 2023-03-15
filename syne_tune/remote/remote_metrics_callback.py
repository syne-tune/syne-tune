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
from typing import Dict, Any, Optional
import logging

from sagemaker.estimator import EstimatorBase

from syne_tune import Reporter
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import Domain
from syne_tune.constants import MAX_METRICS_SUPPORTED_BY_SAGEMAKER
from syne_tune.tuner_callback import TunerCallback
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    add_metric_definitions_to_sagemaker_estimator,
)

logger = logging.getLogger(__name__)


BEST_METRIC_VALUE = "best_metric_value"

BEST_TRIAL_ID = "best_trial_id"

BEST_RESOURCE_VALUE = "best_resource_value"

BEST_HP_PREFIX = "best_hp_"  # Followed by hyperparameter name


class RemoteTuningMetricsCallback(TunerCallback):
    """
    Reports metrics related to the experiment run by :class:`~syne_tune.Tuner`.
    With remote tuning, if these metrics are registered with the SageMaker
    estimator running the experiment, they are visualized in the SageMaker
    console. Metrics reported are:

    * :const:`BEST_METRIC_VALUE`: Best value of ``metric`` reported to tuner so
      far
    * :const:`BEST_TRIAL_ID`: ID of trial for which the best metric value was
      reported so far
    * :const:`BEST_RESOURCE_VALUE`: Resource value for which the best metric
      value was reported so far. Only if ``resource_attr`` is given
    * If ``config_space`` is given, then for each hyperparameter ``name`` in
      there (entry with domain), we add a metric :code:`BEST_HP_PREFIX + name`.
      However, at most :const:`~syne_tune.constants.MAX_METRICS_SUPPORTED_BY_SAGEMAKER`
      are supported
    """

    def __init__(
        self,
        metric: str,
        mode: str,
        config_space: Optional[Dict[str, Any]] = None,
        resource_attr: Optional[str] = None,
    ):
        self._metric = metric
        supported_mode = ("min", "max")
        assert mode in supported_mode, f"mode must be in {supported_mode}"
        self._metric_sign = 1 if mode == "min" else -1
        self._resource_attr = resource_attr
        self._reporter = None
        self._best_metric_value = None
        self._report_config = config_space is not None
        self.metric_names = self.get_metric_names(config_space, resource_attr)

    @staticmethod
    def get_metric_names(
        config_space: Optional[Dict[str, Any]],
        resource_attr: Optional[str] = None,
    ):
        metric_names = [BEST_METRIC_VALUE, BEST_TRIAL_ID]
        if resource_attr is not None:
            metric_names.append(BEST_RESOURCE_VALUE)
        if config_space is not None:
            for name, domain in config_space.items():
                if isinstance(domain, Domain):
                    metric_names.append(BEST_HP_PREFIX + name)
        if len(metric_names) > MAX_METRICS_SUPPORTED_BY_SAGEMAKER:
            metric_names = metric_names[:MAX_METRICS_SUPPORTED_BY_SAGEMAKER]
        return metric_names

    def register_metrics_with_estimator(self, estimator: EstimatorBase):
        """
        Registers metrics reported here at SageMaker estimator ``estimator``. This
        should be the one which runs the remote experiment.

        Note: The total number of metric definitions must not exceed
        :const:`~syne_tune.constants.MAX_METRICS_SUPPORTED_BY_SAGEMAKER`. Otherwise,
        only the initial part of ``metric_names`` is registered.

        :param estimator: SageMaker estimator to run the experiment
        """
        add_metric_definitions_to_sagemaker_estimator(estimator, self.metric_names)

    def on_tuning_start(self, tuner):
        self._reporter = Reporter()
        self._best_metric_value = None

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        metric_value = result[self._metric]
        if (
            self._best_metric_value is None
            or self._metric_sign * (metric_value - self._best_metric_value) < 0
        ):
            self._best_metric_value = metric_value
            report_dict = {
                BEST_METRIC_VALUE: metric_value,
                BEST_TRIAL_ID: trial.trial_id,
            }
            if self._resource_attr is not None:
                report_dict[BEST_RESOURCE_VALUE] = result[self._resource_attr]
            if self._report_config:
                for mname in self.metric_names:
                    if mname.startswith(BEST_HP_PREFIX):
                        report_dict[mname] = trial.config[mname[len(BEST_HP_PREFIX) :]]
            self._reporter(**report_dict)
