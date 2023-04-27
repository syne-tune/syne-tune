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
import logging

from syne_tune.optimizer.schedulers.searchers.model_based_searcher import (
    BayesianOptimizationSearcher,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    gp_fifo_searcher_factory,
    gp_fifo_searcher_defaults,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)

logger = logging.getLogger(__name__)


class GPFIFOSearcher(BayesianOptimizationSearcher):
    """Gaussian process Bayesian optimization for FIFO scheduler

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. It provides
    Bayesian optimization, based on a Gaussian process surrogate model.

    It is *not* recommended creating :class:`GPFIFOSearcher` searcher objects
    directly, but rather to create
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` objects with
    ``searcher="bayesopt"``, and passing arguments here in ``search_options``.
    This will use the appropriate functions from
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`` to
    create components in a consistent way.

    Most of the implementation is generic in
    :class:`~syne_tune.optimizer.schedulers.searchers.model_based_searcher.BayesianOptimizationSearcher`.

    Note: If metric values are to be maximized (``mode-"max"`` in scheduler),
    the searcher uses ``map_reward`` to map metric values to internal
    criterion values, and *minimizes* the latter. The default choice is
    to multiply values by -1.

    Pending configurations (for which evaluation tasks are currently running)
    are dealt with by fantasizing (i.e., target values are drawn from the
    current posterior, and acquisition functions are averaged over this
    sample, see ``num_fantasy_samples``).

    The GP surrogate model uses a Matern 5/2 covariance function with automatic
    relevance determination (ARD) of input attributes, and a constant mean
    function. The acquisition function is expected improvement (EI). All
    hyperparameters of the surrogate model are estimated by empirical Bayes
    (maximizing the marginal likelihood). In general, this hyperparameter
    fitting is the most expensive part of a :meth:`get_config` call.

    Note that the full logic of construction based on arguments is given in
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory``. In
    particular, see
    :func:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory.gp_fifo_searcher_defaults`
    for default values.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param clone_from_state: Internal argument, do not use
    :type clone_from_state: bool
    :param resource_attr: Name of resource attribute in reports. This is
        optional here, but required for multi-fidelity searchers.
        If ``resource_attr`` and ``cost_attr`` are given, cost values are read from
        each report and stored in the state. This allows cost models to be fit
        on more data.
    :type resource_attr: str, optional
    :param cost_attr: Name of cost attribute in data obtained from reporter
        (e.g., elapsed training time). Needed only by cost-aware searchers.
        Depending on whether ``resource_attr`` is given, cost values are read
        from each report or only at the end.
    :type cost_attr: str, optional
    :param num_init_random: Number of initial :meth:`get_config` calls for which
        randomly sampled configs are returned. Afterwards, the model-based
        searcher is used. Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS`
    :type num_init_random: int, optional
    :param num_init_candidates: Number of initial candidates sampled at
        random in order to seed the model-based search in ``get_config``.
        Defaults to :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_NUM_INITIAL_CANDIDATES`
    :type num_init_candidates: int, optional
    :param num_fantasy_samples: Number of samples drawn for fantasizing
        (latent target values for pending evaluations), defaults to 20
    :type num_fantasy_samples: int, optional
    :param no_fantasizing: If True, fantasizing is not done and pending
        evaluations are ignored. This may lead to loss of diversity in
        decisions. Defaults to ``False``
    :type no_fantasizing: bool, optional
    :param input_warping: If ``True``, we use a warping transform, so the kernel
        function becomes :math:`k(w(x), w(x'))`, where :math:`w(x)` is a warping
        transform parameterized by two non-negative numbers per component, which
        are learned as hyperparameters. See also
        :class:`~syne_tune.optimizer.schedulers.searcher.bayesopt.gpautograd.warping.Warping`.
        Coordinates which belong to categorical hyperparameters, are not warped.
        Defaults to ``False``.
    :type input_warping: bool, optional
    :param boxcox_transform: If ``True``, target values are transformed before
        being fitted with a Gaussian marginal likelihood. This is using the Box-Cox
        transform with a parameter :math:`\lambda`, which is learned alongside
        other parameters of the surrogate model. The transform is :math:`\log y`
        for :math:`\lambda = 0`, and :math:`y - 1` for :math:`\lambda = 1`. This
        option requires the targets to be positive. Defaults to ``False``.
    :type boxcox_transform: bool, optional
    :param initial_scoring: Scoring function to rank initial candidates
        (local optimization of EI is started from top scorer):

        * "thompson_indep": Independent Thompson sampling; randomized score,
          which can increase exploration
        * "acq_func": score is the same (EI) acquisition function which is
          used for local optimization afterwards

        Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_INITIAL_SCORING`
    :type initial_scoring: str, optional
    :param skip_local_optimization: If ``True``, the local gradient-based
        optimization of the acquisition function is skipped, and the
        top-ranked initial candidate (after initial scoring) is returned
        instead. In this case, ``initial_scoring="acq_func"`` makes most
        sense, otherwise the acquisition function will not be used.
        Defaults to False
    :type skip_local_optimization: bool, optional
    :param opt_nstarts: Parameter for surrogate model fitting. Number of
        random restarts. Defaults to 2
    :type opt_nstarts: int, optional
    :param opt_maxiter: Parameter for surrogate model fitting. Maximum
        number of iterations per restart. Defaults to 50
    :type opt_maxiter: int, optional
    :param opt_warmstart: Parameter for surrogate model fitting. If ``True``,
        each fitting is started from the previous optimum. Not recommended
        in general. Defaults to ``False``
    :type opt_warmstart: bool, optional
    :param opt_verbose: Parameter for surrogate model fitting. If ``True``,
        lots of output. Defaults to ``False``
    :type opt_verbose: bool, optional
    :param max_size_data_for_model: If this is set, we limit the number of
        observations the surrogate model is fitted on this value. If there are
        more observations, they are down sampled, see
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state.SubsampleSingleFidelityStateConverter`
        for details. This down sampling is repeated every time the model is
        fit. The ``opt_skip_*`` predicates are evaluated before the state is
        downsampled. Pass ``None`` not to apply such a threshold. The default is
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_MAX_SIZE_DATA_FOR_MODEL`.
    :type max_size_data_for_model: int, optional
    :param max_size_top_fraction: Only used if ``max_size_data_for_model`` is
        set. This fraction of the down sampled set is filled with the top entries
        in the full set, the remaining ones are sampled at random from the full
        set, see
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state.SubsampleSingleFidelityStateConverter`
        for details. Defaults to 0.25.
    :type max_size_top_fraction: float, optional
    :param opt_skip_init_length: Parameter for surrogate model fitting,
        skip predicate. Fitting is never skipped as long as number of
        observations below this threshold. Defaults to 150
    :type opt_skip_init_length: int, optional
    :param opt_skip_period: Parameter for surrogate model fitting, skip
        predicate. If ``>1``, and number of observations above
        ``opt_skip_init_length``, fitting is done only K-th call, and skipped
        otherwise. Defaults to 1 (no skipping)
    :type opt_skip_period: int, optional
    :param allow_duplicates: If ``True``, :meth:`get_config` may return the same
        configuration more than once. Defaults to ``False``
    :type allow_duplicates: bool, optional
    :param restrict_configurations: If given, the searcher only suggests
        configurations from this list. This needs
        ``skip_local_optimization == True``. If ``allow_duplicates == False``,
         entries are popped off this list once suggested.
    :type restrict_configurations: List[dict], optional
    :param map_reward: In the scheduler, the metric may be minimized or
        maximized, but internally, Bayesian optimization is minimizing
        the criterion. ``map_reward`` converts from metric to internal
        criterion:

        * "minus_x": ``criterion = -metric``
        * "<a>_minus_x": ``criterion = <a> - metric``. For example "1_minus_x"
          maps accuracy to zero-one error

        From a technical standpoint, it does not matter what is chosen here,
        because criterion is only used internally. Also note that criterion
        data is always normalized to mean 0, variance 1 before fitted with a
        Gaussian process. Defaults to "1_minus_x"
    :type map_reward: str or :class:`MapReward`, optional
    :param transfer_learning_task_attr: Used to support transfer HPO, where
        the state contains observed data from several tasks, one of which
        is the active one. To this end, ``config_space`` must contain a
        categorical parameter of name ``transfer_learning_task_attr``, whose
        range are all task IDs. Also, ``transfer_learning_active_task`` must
        denote the active task, and ``transfer_learning_active_config_space``
        is used as ``active_config_space`` argument in
        :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`.
        This allows us to use a narrower search space for the active task than
        for the union of all tasks (``config_space`` must be that), which is
        needed if some configurations of non-active tasks lie outside of the
        ranges in ``active_config_space``. One of the implications is that
        :meth:`filter_observed_data` is selecting configs of the active task,
        so that incumbents or exclusion lists are restricted to data from the
        active task.
    :type transfer_learning_task_attr: str, optional
    :param transfer_learning_active_task: See ``transfer_learning_task_attr``.
    :type transfer_learning_active_task: str, optional
    :param transfer_learning_active_config_space:
        See ``transfer_learning_task_attr``. If not given, ``config_space`` is the
        search space for the active task as well. This active config space need
        not contain the ``transfer_learning_task_attr`` parameter. In fact, this
        parameter is set to a categorical with ``transfer_learning_active_task``
        as single value, so that new configs are chosen for the active task
        only.
    :type transfer_learning_active_config_space: Dict[str, Any], optional
    :param transfer_learning_model: See ``transfer_learning_task_attr``.
        Specifies the surrogate model to be used for transfer learning:

        * "matern52_product": Kernel is product of Matern 5/2 (not ARD) on
          ``transfer_learning_task_attr`` and Matern 5/2 (ARD) on the rest.
          Assumes that data from same task are more closely related than
          data from different tasks
        * "matern52_same": Kernel is Matern 5/2 (ARD) on the rest of the
          variables, ``transfer_learning_task_attr`` is ignored. Assumes
          that data from all tasks can be merged together

        Defaults to "matern52_product"
    :type transfer_learning_model: str, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        clone_from_state: bool = False,
        **kwargs,
    ):
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            random_seed_generator=kwargs.get("random_seed_generator"),
            random_seed=kwargs.get("random_seed"),
        )
        if not clone_from_state:
            kwargs["config_space"] = config_space
            kwargs["metric"] = metric
            kwargs_int = self._create_kwargs_int(kwargs)
        else:
            # Internal constructor, bypassing the factory
            # Note: Members which are part of the mutable state, will be
            # overwritten in ``_restore_from_state``
            kwargs_int = kwargs.copy()
        self._call_create_internal(kwargs_int)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_fifo_searcher_defaults(kwargs), dict_name="search_options"
        )
        kwargs_int = gp_fifo_searcher_factory(**_kwargs)
        # Extra arguments not parsed in factory
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _call_create_internal(self, kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        self._create_internal(**kwargs_int)

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        skip_optimization = state["skip_optimization"]
        model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = GPFIFOSearcher(
            **self._new_searcher_kwargs_for_clone(),
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
