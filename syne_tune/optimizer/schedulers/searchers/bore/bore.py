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
import time
import xgboost
import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from typing import Dict

from syne_tune.optimizer.schedulers.searchers.searcher import SearcherWithRandomSeed
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bore.de import (
    DifferentialevolutionOptimizer,
)

logger = logging.getLogger(__name__)


class Bore(SearcherWithRandomSeed):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate=None,
        mode: str = "max",
        gamma: float = 0.25,
        calibrate: bool = False,
        classifier: str = "xgboost",
        acq_optimizer: str = "rs",
        feval_acq: int = 500,
        random_prob: float = 0.0,
        init_random: int = 6,
        classifier_kwargs: dict = None,
        **kwargs,
    ):

        """
        Implements "Bayesian optimization by Density Ratio Estimation" as described in the following paper:

        BORE: Bayesian Optimization by Density-Ratio Estimation,
        Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        Proceedings of the 38th International Conference on Machine Learning


        Note: Bore only works in the non-parallel non-multi-fideltiy setting. Make sure that you use it with the
        FIFO scheduler and set num_workers to 1 in the backend.

        :param config_space: Configuration space. Constant parameters are filtered out
        :param metric: Name of metric reported by evaluation function.
        :param points_to_evaluate:
        :param gamma: Defines the percentile, i.e how many percent of configuration are used to model l(x).
        :param calibrate: If set to true, we calibrate the predictions of the classifier via CV
        :param classifier: The binary classifier to model the acquisition function.
            Choices: {'mlp', 'gp', 'xgboost', 'rf}
        :param random_seed: seed for the random number generator
        :param acq_optimizer: The optimization method to maximize the acquisition function. Choices: {'de', 'rs'}
        :param feval_acq: Maximum allowed function evaluations of the acquisition function.
        :param random_prob: probability for returning a random configurations (epsilon greedy)
        :param init_random: Number of initial random configurations before we start with the optimization.
        :param classifier_kwargs: Dict that contains all hyperparameters for the classifier
        """

        super().__init__(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            **kwargs,
        )

        self.calibrate = calibrate
        self.gamma = gamma
        self.classifier = classifier
        assert acq_optimizer in {"rs", "de", "rs_with_replacement"}
        self.acq_optimizer = acq_optimizer
        self.feval_acq = feval_acq
        self.init_random = init_random
        self.random_prob = random_prob
        self.mode = mode

        self._hp_ranges = make_hyperparameter_ranges(config_space)

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
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        assert isinstance(
            scheduler, FIFOScheduler
        ), "This searcher requires FIFOScheduler scheduler"

        super().configure_scheduler(scheduler)

    def loss(self, x):
        if len(x.shape) < 2:
            y = -self.model.predict_proba(x[None, :])
        else:
            y = -self.model.predict_proba(x)
        if self.classifier in ["gp", "mlp"]:
            return y[:, 0]
        else:
            return y[:, 1]  # return probability of class 1

    def get_config(self, **kwargs):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new
        configuration.

        Note: Query `_next_initial_config` for initial configs to return first.

        Args:
        kwargs:
            Extra information may be passed from scheduler to searcher
        returns: config
            must return a valid configuration
        """

        start_time = time.time()

        if len(self.inputs) < self.init_random or np.random.rand() < self.random_prob:
            config = self._hp_ranges.random_config(self.random_state)

        else:
            # train model
            self.train_model(self.inputs, self.targets)

            if self.model is None:
                config = self._hp_ranges.random_config(self.random_state)

            else:

                if self.acq_optimizer == "de":

                    def wrapper(x):
                        l = self.loss(x)
                        return l[:, None]

                    bounds = np.array(self._hp_ranges.get_ndarray_bounds())
                    lower = bounds[:, 0]
                    upper = bounds[:, 1]

                    de = DifferentialevolutionOptimizer(
                        wrapper, lower, upper, self.feval_acq
                    )
                    best, traj = de.run()
                    config = self._hp_ranges.from_ndarray(best)

                elif self.acq_optimizer == "rs_with_replacement":
                    values = []
                    X = []
                    for i in range(self.feval_acq):
                        xi = self._hp_ranges.random_config(self.random_state)
                        X.append(xi)
                        values.append(self.loss(self._hp_ranges.to_ndarray(xi))[0])

                    ind = np.array(values).argmin()
                    config = X[ind]
                else:

                    # sample random configurations without replacement
                    values = []
                    X = []
                    counter = 0
                    while len(values) < self.feval_acq and counter < 10:
                        xi = self._hp_ranges.random_config(self.random_state)
                        if xi not in X:
                            X.append(xi)
                            values.append(self.loss(self._hp_ranges.to_ndarray(xi))[0])
                            counter = 0
                        else:
                            logging.warning(
                                "Re-sampled the same configuration. Retry..."
                            )
                            counter += 1  # we stop sampling if after 10 retires we are not able to find a new config
                    if len(values) < self.feval_acq:
                        logging.warning(
                            f"Only {len(values)} instead of {self.feval_acq} configurations "
                            f"sampled to optimize the acquisition function"
                        )
                    ind = np.array(values).argmin()
                    config = X[ind]

        opt_time = time.time() - start_time
        logging.debug(
            f"[Select new candidate: "
            f"config={config}] "
            f"optimization time : {opt_time}"
        )

        return config

    def train_model(self, train_data, train_targets):

        start_time = time.time()

        X = np.array(self.inputs)

        if self.mode == "min":
            y = np.array(self.targets)
        else:
            y = -np.array(self.targets)

        tau = np.quantile(y, q=self.gamma)
        z = np.less(y, tau)

        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.model, cv=2, method=self.calibration
            )
            self.model.fit(X, np.array(z, dtype=np.int))
        else:
            self.model.fit(X, np.array(z, dtype=np.int))

        z_hat = self.model.predict(X)
        accuracy = np.mean(z_hat == z)

        train_time = time.time() - start_time
        logging.debug(
            f"[Model fit: "
            f"accuracy={accuracy:.3f}] "
            f"dataset size: {X.shape[0]}, "
            f"train time : {train_time}"
        )

    def _update(self, trial_id: str, config: Dict, result: Dict):
        """Update surrogate model with result

        :param config: new configuration
        :param result: observed results from the train function
        """

        self.inputs.append(self._hp_ranges.to_ndarray(config))
        self.targets.append(result[self._metric])

    def clone_from_state(self, state):
        pass
