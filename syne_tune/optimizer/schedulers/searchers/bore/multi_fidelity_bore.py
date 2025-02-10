import logging

from typing import Optional, List, Dict, Any
from collections import OrderedDict

from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.optimizer.schedulers.searchers.multi_fidelity_searcher import MultiFidelityBaseSearcher

logger = logging.getLogger(__name__)


class MultiFidelityBore(MultiFidelityBaseSearcher):
    """
    Adapts BORE (Tiao et al.) for the multi-fidelity Hyperband setting following
    BOHB (Falkner et al.). Once we collected enough data points on the smallest
    resource level, we fit a probabilistic classifier and sample from it until we have
    a sufficient amount of data points for the next higher resource level. We then
    refit the classifier on the data of this resource level. These steps are
    iterated until we reach the highest resource level. References:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning

    and

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.bore.Bore`:

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
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

        self.gamma = gamma
        self.calibrate = calibrate
        self.classifier = classifier
        self.acq_optimizer = acq_optimizer
        self.feval_acq = feval_acq
        self.random_prob = random_prob
        self.init_random = init_random
        self.classifier_kwargs = classifier_kwargs

        self.models = OrderedDict()
        self.models[0] = self.initialize_model()

    def initialize_model(self):
        return Bore(
            config_space=self.config_space,
            gamma = self.gamma,
        calibrate = self.calibrate,
        classifier = self.classifier,
        acq_optimizer = self.acq_optimizer,
        feval_acq = self.feval_acq,
        random_prob = self.random_prob,
        init_random = self.init_random,
        classifier_kwargs = self.classifier_kwargs,

            random_seed=self.random_seed
        )

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_points_to_evaluate` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """
        suggestion = self._next_points_to_evaluate()
        if suggestion is None:
            highest_observed_resource = next(reversed(self.models))
            return self.models[highest_observed_resource].suggest()

    def on_trial_result(
            self,
            trial_id: int,
            config: Dict[str, Any],
            metric: float,
            resource_level: int,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(trial_id=trial_id, config=config, metric=metric)

    def on_trial_complete(
            self,
            trial_id: int,
            config: Dict[str, Any],
            metric: float,
            resource_level: int,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """

        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(trial_id=trial_id, config=config, metric=metric)
