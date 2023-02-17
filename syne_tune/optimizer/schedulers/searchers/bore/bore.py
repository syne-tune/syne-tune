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
from typing import Optional, List, Dict, Any
import time
import xgboost
import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from syne_tune.optimizer.schedulers.searchers import (
    StochasticAndFilterDuplicatesSearcher,
)
from syne_tune.optimizer.schedulers.searchers.bore.de import (
    DifferentialevolutionOptimizer,
)

logger = logging.getLogger(__name__)


class Bore(StochasticAndFilterDuplicatesSearcher):
    """
    Implements "Bayesian optimization by Density Ratio Estimation" as described
    in the following paper:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning
        | https://arxiv.org/abs/2102.09009

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`:

    :param mode: Can be "min" (default) or "max".
    :param gamma: Defines the percentile, i.e how many percent of configurations
        are used to model :math:`l(x)`. Defaults to 0.25
    :param calibrate: If set to true, we calibrate the predictions of the
        classifier via CV. Defaults to False
    :param classifier: The binary classifier to model the acquisition
        function. Choices: :code:`{"mlp", "gp", "xgboost", "rf", "logreg"}`.
        Defaults to "xgboost"
    :param acq_optimizer: The optimization method to maximize the acquisition
        function. Choices: :code:`{"de", "rs", "rs_with_replacement"}`. Defaults
        to "rs"
    :param feval_acq: Maximum allowed function evaluations of the acquisition
        function. Defaults to 500
    :param random_prob: probability for returning a random configurations
        (epsilon greedy). Defaults to 0
    :param init_random: :meth:`get_config` returns randomly drawn configurations
        until at least ``init_random`` observations have been recorded in
        :meth:`update`. After that, the BORE algorithm is used. Defaults to 6
    :param classifier_kwargs: Parameters for classifier. Optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        allow_duplicates: Optional[bool] = None,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[str] = None,
        gamma: Optional[float] = None,
        calibrate: Optional[bool] = None,
        classifier: Optional[str] = None,
        acq_optimizer: Optional[str] = None,
        feval_acq: Optional[int] = None,
        random_prob: Optional[float] = None,
        init_random: Optional[int] = None,
        classifier_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            restrict_configurations=restrict_configurations,
            **kwargs,
        )
        if mode is None:
            mode = "min"
        if gamma is None:
            gamma = 0.25
        if calibrate is None:
            calibrate = False
        if classifier is None:
            classifier = "xgboost"
        if acq_optimizer is None:
            acq_optimizer = "rs"
        if feval_acq is None:
            feval_acq = 500
        if random_prob is None:
            random_prob = 0.0
        if init_random is None:
            init_random = 6

        self.calibrate = calibrate
        self.gamma = gamma
        self.classifier = classifier
        assert acq_optimizer in {"rs", "de", "rs_with_replacement"}
        assert (
            acq_optimizer != "de" or restrict_configurations is None
        ), "acq_optimizer == 'de' does not support restrict_configurations"
        self.acq_optimizer = acq_optimizer
        self.feval_acq = feval_acq
        self.init_random = init_random
        self.random_prob = random_prob
        self.mode = mode

        if classifier_kwargs is None:
            classifier_kwargs = dict()
        if self.classifier == "xgboost":
            self.model = xgboost.XGBClassifier(use_label_encoder=False)
        elif self.classifier == "logreg":
            self.model = LogisticRegression()
        elif self.classifier == "rf":
            self.model = RandomForestClassifier()
        elif self.classifier == "gp":
            from syne_tune.optimizer.schedulers.searchers.bore.gp_classififer import (
                GPModel,
            )

            self.model = GPModel(**classifier_kwargs)
        elif self.classifier == "mlp":
            from syne_tune.optimizer.schedulers.searchers.bore.mlp_classififer import (
                MLP,
            )

            self.model = MLP(n_inputs=self._hp_ranges.ndarray_size, **classifier_kwargs)

        self.inputs = []
        self.targets = []

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def _loss(self, x):
        if len(x.shape) < 2:
            y = -self.model.predict_proba(x[None, :])
        else:
            y = -self.model.predict_proba(x)
        if self.classifier in ["gp", "mlp"]:
            return y[:, 0]
        else:
            return y[:, 1]  # return probability of class 1

    def _get_config(self, **kwargs):
        start_time = time.time()
        config = self._next_initial_config()
        if config is None:
            if (
                len(self.inputs) < self.init_random
                or self.random_state.rand() < self.random_prob
            ):
                config = self._get_random_config()
            else:
                # train model
                if not self._train_model(self.inputs, self.targets):
                    config = self._get_random_config()
                elif self.acq_optimizer == "de":

                    def wrapper(x):
                        l = self._loss(x)
                        return l[:, None]

                    bounds = np.array(self._hp_ranges.get_ndarray_bounds())
                    lower = bounds[:, 0]
                    upper = bounds[:, 1]

                    de = DifferentialevolutionOptimizer(
                        wrapper, lower, upper, self.feval_acq
                    )
                    best, traj = de.run()
                    config = self._hp_ranges.from_ndarray(best)

                else:
                    # sample random configurations with or without replacement
                    values = []
                    X = []
                    if self.acq_optimizer == "rs":
                        new_excl_list = self._excl_list.copy()
                    else:
                        assert self.acq_optimizer == "rs_with_replacement"
                        new_excl_list = None
                    for _ in range(self.feval_acq):
                        xi = self._get_random_config(new_excl_list)
                        if xi is None:
                            break
                        X.append(xi)
                        values.append(self._loss(self._hp_ranges.to_ndarray(xi))[0])
                        if new_excl_list is not None:
                            new_excl_list.add(xi)
                    if len(values) > 0:
                        if len(values) < self.feval_acq:
                            logging.warning(
                                f"Only {len(values)} instead of {self.feval_acq} configurations "
                                f"sampled to optimize the acquisition function"
                            )
                        ind = np.array(values).argmin()
                        config = X[ind]
                    else:
                        # This can happen if the configuration space is almost exhausted
                        logger.warning(
                            "Could not find configuration by optimizing the acquisition function. Drawing at random instead."
                        )
                        config = self._get_random_config()

        if config is not None:
            opt_time = time.time() - start_time
            logging.debug(
                f"[Select new candidate: "
                f"config={config}] "
                f"optimization time : {opt_time}"
            )

        return config

    def _train_model(self, train_data: np.ndarray, train_targets: np.ndarray) -> bool:
        """
        :param train_data: Training input feature matrix X
        :param train_targets: Training targets y
        :return: Was training successful?
        """

        start_time = time.time()

        X = np.array(train_data)

        if self.mode == "min":
            y = np.array(train_targets)
        else:
            y = -np.array(train_targets)

        tau = np.quantile(y, q=self.gamma)
        z = np.less(y, tau)

        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.model, cv=2, method=self.calibration
            )
            self.model.fit(X, np.array(z, dtype=np.int64))
        else:
            self.model.fit(X, np.array(z, dtype=np.int64))

        z_hat = self.model.predict(X)
        accuracy = np.mean(z_hat == z)

        train_time = time.time() - start_time
        logging.debug(
            f"[Model fit: "
            f"accuracy={accuracy:.3f}] "
            f"dataset size: {X.shape[0]}, "
            f"train time : {train_time}"
        )
        return True

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        self.inputs.append(self._hp_ranges.to_ndarray(config))
        self.targets.append(result[self._metric])

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError
