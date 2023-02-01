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
from typing import Dict, Any

from syne_tune.optimizer.schedulers.scheduler_searcher import TrialSchedulerWithSearcher
from syne_tune.optimizer.schedulers.synchronous.hyperband import (
    SynchronousHyperbandScheduler,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import (
    SynchronousHyperbandRungSystem,
)
from syne_tune.optimizer.schedulers.synchronous.dehb import (
    DifferentialEvolutionHyperbandScheduler,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Integer,
    Float,
    filter_by_key,
)


_ARGUMENT_KEYS = {"grace_period", "reduction_factor", "brackets"}

_DEFAULT_OPTIONS = {
    "grace_period": 1,
    "reduction_factor": 3,
}

_CONSTRAINTS = {
    "grace_period": Integer(1, None),
    "reduction_factor": Float(2, None),
    "brackets": Integer(1, None),
}


class SynchronousGeometricHyperbandScheduler(SynchronousHyperbandScheduler):
    """
    Special case of :class:`SynchronousHyperbandScheduler` with rung system
    defined by geometric sequences (see
    :meth:`SynchronousHyperbandRungSystem.geometric`). This is the most
    frequently used case.

    :param config_space: Configuration space for trial evaluation function
    :param metric: Name of metric to optimize, key in result's obtained via
        :meth:`on_trial_result`
    :type metric: str
    :param grace_period: Smallest (resource) rung level. Must be positive int.
        Defaults to 1
    :type grace_period: int, optional
    :param reduction_factor: Approximate ratio of successive rung levels. Must
        be >= 2. Defaults to 3
    :type reduction_factor: float, optional
    :param brackets: Number of brackets to be used. The default is to use the
        maximum number of brackets per iteration. Pass 1 for successive halving.
    :type brackets: int, optional
    :param searcher: Selects searcher. Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`.
        Defaults to "random"
    :type searcher: str, optional
    :param search_options: Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`.
    :type search_options: Dict[str, Any], optional
    :param mode: Mode to use for the metric given, can be "min" (default) or
        "max"
    :type mode: str, optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If None (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the scheduler
        or searcher are seeded using
        :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param max_resource_level: Largest rung level, corresponds to ``max_t`` in
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Must be positive
        int larger than ``grace_period``. If this is not given, it is inferred
        like in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. In
        particular, it is not needed if ``max_resource_attr`` is given.
    :type max_resource_level: int, optional
    :param max_resource_attr: Key name in config for fixed attribute
        containing the maximum resource. If given, trials need not be
        stopped, which can run more efficiently.
    :type max_resource_attr: str, optional
    :param resource_attr: Name of resource attribute in results obtained via
        ``:meth:`on_trial_result`. The type of resource must be int. Default to
        "epoch"
    :type resource_attr: str, optional
    :param searcher_data: Relevant only if a model-based searcher is used.
        Example: For NN tuning and ``resource_attr == "epoch"``, we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:

        * "rungs" (default): Only results at rung levels. Cheapest
        * "all": All results. Most expensive

        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to "all".
    :type searcher_data: str, optional
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        TrialSchedulerWithSearcher.__init__(self, config_space, **kwargs)
        # Additional parameters to determine rung systems
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        self.grace_period = kwargs["grace_period"]
        self.reduction_factor = kwargs["reduction_factor"]
        max_resource_level = self._infer_max_resource_level(
            kwargs.get("max_resource_level"), kwargs.get("max_resource_attr")
        )
        assert max_resource_level is not None, (
            "The maximum resource level must be specified, either as "
            + "explicit argument 'max_resource_level', or as entry in "
            + "'config_space', with name 'max_resource_attr'"
        )
        bracket_rungs = SynchronousHyperbandRungSystem.geometric(
            min_resource=self.grace_period,
            max_resource=max_resource_level,
            reduction_factor=self.reduction_factor,
            num_brackets=kwargs.get("brackets"),
        )
        self._create_internal(bracket_rungs, **filter_by_key(kwargs, _ARGUMENT_KEYS))


class GeometricDifferentialEvolutionHyperbandScheduler(
    DifferentialEvolutionHyperbandScheduler
):
    """
    Special case of :class:`DifferentialEvolutionHyperbandScheduler` with
    rung system defined by geometric sequences. This is the most frequently
    used case.

    :param config_space: Configuration space for trial evaluation function
    :param grace_period: Smallest (resource) rung level. Must be positive int.
        Defaults to 1
    :type grace_period: int, optional
    :param reduction_factor: Approximate ratio of successive rung levels. Must
        be >= 2. Defaults to 3
    :type reduction_factor: float, optional
    :param brackets: Number of brackets to be used. The default is to use the
        maximum number of brackets per iteration. Pass 1 for successive halving.
    :type brackets: int, optional
    :param metric: Name of metric to optimize, key in result's obtained via
        :meth:`on_trial_result`
    :type metric: str
    :param searcher: Selects searcher. Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`..
        If ``searcher == "random_encoded"`` (default), the encoded configs are
        sampled directly, each entry independently from U([0, 1]).
        This distribution has higher entropy than for "random" if
        there are discrete hyperparameters in ``config_space``. Note that
        ``points_to_evaluate`` is still used in this case.
    :type searcher: str, optional
    :param search_options: Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`.
    :type search_options: Dict[str, Any], optional
    :param mode: Mode to use for the metric given, can be "min" (default) or
        "max"
    :type mode: str, optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If None (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the scheduler
        or searcher are seeded using
        :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param max_resource_level: Largest rung level, corresponds to ``max_t`` in
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Must be positive
        int larger than ``grace_period``. If this is not given, it is inferred
        like in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. In
        particular, it is not needed if ``max_resource_attr`` is given.
    :type max_resource_level: int, optional
    :param max_resource_attr: Key name in config for fixed attribute
        containing the maximum resource. If given, trials need not be
        stopped, which can run more efficiently.
    :type max_resource_attr: str, optional
    :param resource_attr: Name of resource attribute in results obtained via
        :meth:`on_trial_result`. The type of resource must be int. Default to
        "epoch"
    :type resource_attr: str, optional
    :param mutation_factor: In :math:`(0, 1]`. Factor :math:`F` used in the rand/1
        mutation operation of DE. Default to 0.5
    :type mutation_factor: float, optional
    :param crossover_probability: In :math:`(0, 1)`. Probability :math:`p` used
        in crossover operation (child entries are chosen with probability
        :math:`p`). Defaults to 0.5
    :type crossover_probability: float, optional
    :param support_pause_resume: If ``True``, :meth:`_suggest` supports pause and
        resume in the first bracket (this is the default). If the objective
        supports checkpointing, this is made use of. Defaults to ``True``.
        Note: The resumed trial still gets assigned a new ``trial_id``, but it
        starts from the earlier checkpoint.
    :type support_pause_resume: bool, optional
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        TrialSchedulerWithSearcher.__init__(self, config_space, **kwargs)
        # Additional parameters to determine rung systems
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        self.grace_period = kwargs["grace_period"]
        self.reduction_factor = kwargs["reduction_factor"]
        num_brackets = kwargs.get("brackets")
        max_resource_level = self._infer_max_resource_level(
            kwargs.get("max_resource_level"), kwargs.get("max_resource_attr")
        )
        assert max_resource_level is not None, (
            "The maximum resource level must be specified, either as "
            + "explicit argument 'max_resource_level', or as entry in "
            + "'config_space', with name 'max_resource_attr':\n"
            + f"max_resource_attr = {kwargs.get('max_resource_attr')}\n"
            + f"config_space = {config_space}"
        )
        bracket_rungs = SynchronousHyperbandRungSystem.geometric(
            min_resource=self.grace_period,
            max_resource=max_resource_level,
            reduction_factor=self.reduction_factor,
            num_brackets=num_brackets,
        )
        num_brackets = len(bracket_rungs)
        rungs_first_bracket = bracket_rungs[0]
        self._create_internal(
            rungs_first_bracket=rungs_first_bracket,
            num_brackets_per_iteration=num_brackets,
            **filter_by_key(kwargs, _ARGUMENT_KEYS),
        )
