from typing import Optional, List, Dict, Any
import time
import xgboost
import logging
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from syne_tune.optimizer.schedulers.searchers.bore.mlp_classififer import MLP
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)

from syne_tune.optimizer.schedulers.searchers.bore.de import (
    DifferentialevolutionOptimizer,
)

logger = logging.getLogger(__name__)


class Bore(SingleObjectiveBaseSearcher):
    """
    Implements "Bayesian optimization by Density Ratio Estimation" as described
    in the following paper:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning
        | https://arxiv.org/abs/2102.09009

    :param config_space: Configuration space for the evaluation function.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
    :param random_seed: Seed for initializing random number generators.
    :param gamma: Defines the percentile, i.e how many percent of configurations
        are used to model :math:`l(x)`. Defaults to 0.25
    :param calibrate: If set to true, we calibrate the predictions of the
        classifier via CV. Defaults to False
    :param classifier: The binary classifier to model the acquisition
        function. Choices: :code:`{"mlp", "xgboost", "rf", "logreg"}`.
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
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
        gamma: Optional[float] = 0.25,
        calibrate: Optional[bool] = False,
        classifier: Optional[str] = "xgboost",
        acq_optimizer: Optional[str] = "rs",
        feval_acq: Optional[int] = 500,
        random_prob: Optional[float] = 0.0,
        init_random: Optional[int] = 6,
        classifier_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

        self.calibrate = calibrate
        self.gamma = gamma
        self.classifier = classifier
        assert acq_optimizer in {"rs", "de", "rs_with_replacement"}
        self.acq_optimizer = acq_optimizer
        self.feval_acq = feval_acq
        self.init_random = init_random
        self.random_prob = random_prob
        self.random_state = np.random.RandomState(self.random_seed)

        self._hp_ranges = make_hyperparameter_ranges(config_space)

        if classifier_kwargs is None:
            classifier_kwargs = dict()

        if self.classifier == "xgboost":
            self.model = xgboost.XGBClassifier(random_state=self.random_state)
        elif self.classifier == "logreg":
            self.model = LogisticRegression(
                random_state=self.random_state, **classifier_kwargs
            )
        elif self.classifier == "rf":
            self.model = RandomForestClassifier(
                random_state=self.random_state, **classifier_kwargs
            )
        elif self.classifier == "mlp":
            self.model = MLP(
                n_inputs=self._hp_ranges.ndarray_size,
                random_state=self.random_state,
                **classifier_kwargs,
            )

        self.inputs = []
        self.targets = []

    def _loss(self, x):
        if len(x.shape) < 2:
            y = -self.model.predict_proba(x[None, :])
        else:
            y = -self.model.predict_proba(x)
        if self.classifier in ["gp", "mlp"]:
            return y[:, 0]
        else:
            return y[:, 1]  # return probability of class 1

    def _get_random_config(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }

    def suggest(self, **kwargs):
        start_time = time.time()
        config = self._next_points_to_evaluate()

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

                elif self.acq_optimizer == "rs_with_replacement":
                    # sample random configurations with replacement
                    candidates = [
                        self._get_random_config() for _ in range(self.feval_acq)
                    ]
                    values = [
                        self._loss(self._hp_ranges.to_ndarray(candidate))[0]
                        for candidate in candidates
                    ]
                    ind = np.array(values).argmin()
                    config = candidates[ind]

                else:
                    # sample random configurations without replacement
                    values = []
                    candidates = []
                    counter = 0
                    while len(candidates) < self.feval_acq:
                        xi = self._get_random_config()
                        counter += 1
                        if counter > 10000:
                            logging.error(
                                f"Tried 10000 times to sample a new configuration "
                                f"without replacement with no success."
                                f"We will stop now! Current candidate set contains {len(candidates)} "
                                f"configurations. Try reduce the total number of samples feval_acq."
                            )
                            break
                        if xi in candidates:
                            continue
                        counter = 0
                        candidates.append(xi)
                        values.append(self._loss(self._hp_ranges.to_ndarray(xi))[0])
                    ind = np.array(values).argmin()
                    config = candidates[ind]

        if config is not None:
            opt_time = time.time() - start_time
            logging.debug(
                f"[Select new candidate: "
                f"config={config}] "
                f"optimization time : {opt_time}"
            )

        return config

    def _train_model(self, train_data: list, train_targets: list) -> bool:
        """
        :param train_data: Training input feature matrix X
        :param train_targets: Training targets y
        :return: Was training successful?
        """

        start_time = time.time()

        X = np.array(train_data)
        y = np.array(train_targets)

        tau = np.quantile(y, q=self.gamma)
        z = np.less(y, tau)

        if self.calibrate:
            self.model = CalibratedClassifierCV(
                self.model,
                cv=2,
            )
            self.model.fit(X, np.array(z, dtype=np.int64))
        else:
            self.model.fit(X, np.array(z, dtype=np.int64))

        z_hat = self.model.predict(X)
        if len(z_hat.shape) == 2:
            z_hat = z_hat[:, 0]
        accuracy = np.mean(z_hat == z)

        train_time = time.time() - start_time
        logging.debug(
            f"[Model fit: "
            f"accuracy={accuracy:.3f}] "
            f"dataset size: {X.shape[0]}, "
            f"train time : {train_time}"
        )
        return True

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        self.inputs.append(self._hp_ranges.to_ndarray(config))
        self.targets.append(metric)
