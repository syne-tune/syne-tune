import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Status
from syne_tune.blackbox_repository import add_surrogate, load_blackbox
from syne_tune.blackbox_repository.blackbox import Blackbox
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.utils import metrics_for_configuration
from syne_tune.config_space import (
    Domain,
    config_space_from_json_dict,
    config_space_to_json_dict,
)

logger = logging.getLogger(__name__)


class _BlackboxSimulatorBackend(SimulatorBackend):
    """
    Shared parent of :class:`BlackboxRepositoryBackend` and
    :class:`UserBlackboxBackend`, see comments of
    :class:`BlackboxRepositoryBackend`.
    """

    def __init__(
        self,
        elapsed_time_attr: str,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        support_checkpointing: bool = True,
        **simulatorbackend_kwargs,
    ):
        super().__init__(
            entry_point=str(Path(__file__)),  # Dummy value
            elapsed_time_attr=elapsed_time_attr,
            **simulatorbackend_kwargs,
        )
        self._max_resource_attr = max_resource_attr
        self.simulatorbackend_kwargs = simulatorbackend_kwargs
        self._seed = seed
        self._support_checkpointing = support_checkpointing
        self._seed_for_trial = dict()
        self._resource_paused_for_trial = dict()

    @property
    def blackbox(self) -> Blackbox:
        raise NotImplementedError

    @property
    def resource_attr(self):
        return self.blackbox.fidelity_name()

    def _pause_trial(self, trial_id: int, result: Optional[dict]):
        """
        From ``result``, we obtain the resource level at which the trial is
        paused by the scheduler. This is required in order to properly
        resume the trial in ``_run_job_and_collect_results``.
        """
        super()._pause_trial(trial_id, result)
        resource_attr = self.resource_attr
        if result is not None and resource_attr in result:
            resource = int(result[resource_attr])
            self._resource_paused_for_trial[trial_id] = resource

    def _filter_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_space = self.blackbox.configuration_space
        return {k: v for k, v in config.items() if k in config_space}

    def config_objectives(self, config: Dict[str, Any], seed: int) -> List[dict]:
        mattr = self._max_resource_attr
        if mattr is not None and mattr in config:
            max_resource = int(config[mattr])
            fidelity_range = (min(self.blackbox.fidelity_values), max_resource)
        else:
            fidelity_range = None  # All fidelity values
        # ``config`` may contain keys not in ``blackbox.configuration_space`` (for
        # example, ``self._max_resource_attr``). These are filtered out before
        # passing the configuration
        return metrics_for_configuration(
            blackbox=self.blackbox,
            config=self._filter_config(config),
            resource_attr=self.resource_attr,
            fidelity_range=fidelity_range,
            seed=seed,
        )

    def _run_job_and_collect_results(
        self, trial_id: int, config: Optional[dict] = None
    ) -> (str, List[dict]):
        assert (
            trial_id in self._trial_dict
        ), f"Trial with trial_id = {trial_id} not registered with backend"
        if config is None:
            config = self._trial_dict[trial_id].config

        # Seed for query to blackbox. It is important to use the same
        # seed for all queries for the same ``trial_id``
        seed = None
        if self._seed is not None:
            seed = self._seed
        elif isinstance(self.blackbox, BlackboxTabular):
            seed = self._seed_for_trial.get(trial_id)
            if seed is None:
                seed = np.random.randint(0, self.blackbox.num_seeds)
                self._seed_for_trial[trial_id] = seed

        # Fetch all results for this trial from the table
        all_results = self.config_objectives(config, seed=seed)

        status = Status.completed
        resource_paused = self._resource_paused_for_trial.get(trial_id)
        if resource_paused is not None and self._support_checkpointing:
            # If checkpointing is supported and trial has been paused, we
            # can ignore results up until the paused level. Also, the
            # elapsed_time field in later results needs to be corrected
            # to not count the time for skipped results
            resource_attr = self.resource_attr
            elapsed_time_offset = 0
            results = []
            for result in all_results:
                resource = int(result[resource_attr])
                if resource > resource_paused:
                    results.append(result)
                elif resource == resource_paused:
                    elapsed_time_offset = result[self.elapsed_time_attr]
            for result in results:
                result[self.elapsed_time_attr] -= elapsed_time_offset
        else:
            # Use all results from the start
            results = all_results

        # Makes sure that time is monotonically increasing which may not be the
        # case due to numerical errors or due to the use of a surrogate
        et_attr = self.elapsed_time_attr
        results[0][et_attr] = max(results[0][et_attr], 0.01)
        for i in range(1, len(results)):
            results[i][et_attr] = max(
                results[i][et_attr],
                results[i - 1][et_attr] + 0.01,
            )

        return status, results


def make_surrogate(
    surrogate: Optional[str] = None, surrogate_kwargs: Optional[dict] = None
):
    """Creates surrogate model (scikit-learn estimater)

    :param surrogate: A model that is fitted to predict objectives given any
        configuration. Possible examples: "KNeighborsRegressor", MLPRegressor",
        "XGBRegressor", which would enable using the corresponding scikit-learn
        estimator.
        The model is fit on top of pipeline that applies basic feature-processing
        to convert hyperparameters rows in X to vectors. The ``configuration_space``
        hyperparameters types are used to deduce the types of columns in X (for
        instance, categorical hyperparameters are one-hot encoded).
    :param surrogate_kwargs: Arguments for the scikit-learn estimator, for
        instance :code:`{"n_neighbors": 1}` can be used if
        ``surrogate="KNeighborsRegressor"`` is chosen.
    :return: Scikit-learn estimator representing surrogate model
    """
    if surrogate is None:
        return None
    else:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor
        import xgboost

        surrogate_dict = {
            "KNeighborsRegressor": KNeighborsRegressor,
            "MLPRegressor": MLPRegressor,
            "XGBRegressor": xgboost.XGBRegressor,
            "RandomForestRegressor": RandomForestRegressor,
        }
        assert surrogate in surrogate_dict, (
            f"surrogate passed {surrogate} is not supported, "
            f"only {list(surrogate_dict.keys())} are available"
        )
        if surrogate_kwargs is None:
            surrogate_kwargs = dict()
        return surrogate_dict[surrogate](**surrogate_kwargs)


class BlackboxRepositoryBackend(_BlackboxSimulatorBackend):
    """
    Allows to simulate a blackbox from blackbox-repository, selected by
    ``blackbox_name``. See ``examples/launch_simulated_benchmark.py`` for an
    example on how to use. If you want to add a new dataset, see the *Adding a
    new dataset* section of ``syne_tune/blackbox_repository/README.md``.

    In each result reported to the simulator backend, the value for key
    ``elapsed_time_attr`` must be the time since the start of the
    evaluation. For example, if resource (or fidelity) equates to epochs
    trained, this would be the time from start of training until the end
    of the epoch. If the blackbox contains this information in a column,
    ``elapsed_time_attr`` should be its key.

    If this backend is used with pause-and-resume multi-fidelity
    scheduling, it needs to track at which resource level each trial is
    paused. Namely, once a trial is resumed, all results for resources
    smaller or equal to that level are ignored, which simulates the
    situation that training is resumed from a checkpoint. This feature
    relies on ``result`` to be passed to :meth:`pause_trial`. If this is not
    done, the backend cannot know from which resource level to resume
    a trial, so it starts the trial from scratch (which is equivalent to
    no checkpointing). The same happens if ``support_checkpointing`` is
    False.

    .. note::
       If the blackbox maintains cumulative time (elapsed_time), this is
       different from what
       :class:`~syne_tune.backend.simulator_backend.SimulatorBackend` requires
       for ``elapsed_time_attr``, if a pause-and-resume scheduler is used. Namely,
       the backend requires the time since the start of the last recent
       resume. This conversion is done here internally in
       :meth:`_run_job_and_collect_results`, which is called for each resume.
       This means that the field ``elapsed_time_attr`` is not what is received
       from the blackbox table, but instead what the backend needs.

    ``max_resource_attr`` plays the same role as in
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    If given, it is the key in a configuration ``config`` for the maximum
    resource. This is used by schedulers which limit each evaluation by
    setting this argument (e.g., promotion-based Hyperband).

    If ``seed`` is given, entries of the blackbox are queried for this
    seed. Otherwise, a seed is drawn at random for every trial, but the
    same seed is used for all :meth:`_run_job_and_collect_results` calls for
    the same trial. This is important for pause and resume scheduling.

    :param blackbox_name: Name of a blackbox, must have been registered in
        blackbox repository.
    :param elapsed_time_attr: Name of the column containing cumulative time
    :param max_resource_attr: See above
    :param seed: If given, this seed is used for all trial evaluations.
        Otherwise, seed is sampled at random for each trial. Only relevant
        for blackboxes with multiple seeds
    :param support_checkpointing: If ``False``, the simulation does not do
        checkpointing, so resumed trials are started from scratch. Defaults
        to ``True``
    :param dataset: Selects different versions of the blackbox (typically, the
        same ML model has been trained on different datasets)
    :param surrogate: Optionally, a model that is fitted to predict objectives
        given any configuration.
        Examples: "KNeighborsRegressor", "MLPRegressor", "XGBRegressor",
        which would enable using the corresponding scikit-learn estimator, see
        also :func:`make_surrogate`.
        The model is fit on top of pipeline that applies basic feature-processing
        to convert hyperparameter rows in X to vectors. The ``configuration_space``
        hyperparameter types are used to deduce the types of columns in X (for
        instance, categorical hyperparameters are one-hot encoded).
    :param surrogate_kwargs: Arguments for the scikit-learn estimator, for
        instance :code:`{"n_neighbors": 1}` can be used if
        ``surrogate="KNeighborsRegressor"`` is chosen.
        If ``blackbox_name`` is a YAHPO blackbox, then ``surrogate_kwargs`` is passed
        as ``yahpo_kwargs`` to
        :func:`~syne_tune.blackbox_repository.load_blackbox`. In this case,
        ``surrogate`` is ignored (YAHPO always uses surrogates).
    :param config_space_surrogate: If ``surrogate`` is given, this is the
        configuration space for the surrogate blackbox. If not given, the
        space of the original blackbox is used. However, its numerical parameters
        have finite domains (categorical or ordinal), which is usually not what
        we want for a surrogate.
    :param simulatorbackend_kwargs: Additional arguments to parent
        :class:`~syne_tune.backend.simulator_backend.SimulatorBackend`
    """

    def __init__(
        self,
        blackbox_name: str,
        elapsed_time_attr: str,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        support_checkpointing: bool = True,
        dataset: Optional[str] = None,
        surrogate: Optional[str] = None,
        surrogate_kwargs: Optional[dict] = None,
        add_surrogate_kwargs: Optional[dict] = None,
        config_space_surrogate: Optional[dict] = None,
        **simulatorbackend_kwargs,
    ):
        assert (
            config_space_surrogate is None or surrogate is not None
        ), "config_space_surrogate only together with surrogate"
        super().__init__(
            elapsed_time_attr=elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            seed=seed,
            support_checkpointing=support_checkpointing,
            **simulatorbackend_kwargs,
        )
        self.blackbox_name = blackbox_name
        self.dataset = dataset
        self._blackbox = None
        if surrogate is not None:
            # makes sure the surrogate can be constructed
            make_surrogate(surrogate=surrogate, surrogate_kwargs=surrogate_kwargs)
        self._surrogate = surrogate
        self._surrogate_kwargs = (
            surrogate_kwargs if surrogate_kwargs is not None else dict()
        )
        self._add_surrogate_kwargs = (
            add_surrogate_kwargs if add_surrogate_kwargs is not None else dict()
        )
        if config_space_surrogate is not None:
            self._config_space_surrogate = {
                k: v for k, v in config_space_surrogate.items() if isinstance(v, Domain)
            }
        else:
            self._config_space_surrogate = None

    @property
    def blackbox(self) -> Blackbox:
        if self._blackbox is None:
            # Pass ``self._surrogate_kwargs`` as ``yahpo_kwargs``. This is used if
            # ``self.blackbox_name`` is a YAHPO blackbox, and is ignored otherwise
            self._blackbox = load_blackbox(
                self.blackbox_name,
                yahpo_kwargs=self._surrogate_kwargs,
            )
            if self.dataset is None:
                assert not isinstance(self._blackbox, dict), (
                    f"blackbox_name = '{self.blackbox_name}' maps to a dict, "
                    + "dataset argument must be given"
                )
            else:
                self._blackbox = self._blackbox[self.dataset]
            if self._surrogate is not None:
                surrogate = make_surrogate(
                    surrogate=self._surrogate, surrogate_kwargs=self._surrogate_kwargs
                )
                self._blackbox = add_surrogate(
                    blackbox=self._blackbox,
                    surrogate=surrogate,
                    configuration_space=self._config_space_surrogate,
                    **self._add_surrogate_kwargs,
                )

        return self._blackbox

    def __getstate__(self):
        # we serialize only required metadata information since the blackbox data is contained in the repository and
        # its raw data does not need to be saved.
        state = {
            "elapsed_time_attr": self.elapsed_time_attr,
            "max_resource_attr": self._max_resource_attr,
            "seed": self._seed,
            "simulatorbackend_kwargs": self.simulatorbackend_kwargs,
            "support_checkpointing": self._support_checkpointing,
            "seed_for_trial": self._seed_for_trial,
            "blackbox_name": self.blackbox_name,
            "dataset": self.dataset,
            "surrogate": self._surrogate,
            "surrogate_kwargs": self._surrogate_kwargs,
        }
        if self._config_space_surrogate is not None:
            state["config_space_surrogate"] = config_space_to_json_dict(
                self._config_space_surrogate
            )
        return state

    def __setstate__(self, state):
        super().__init__(
            elapsed_time_attr=state["elapsed_time_attr"],
            max_resource_attr=state["max_resource_attr"],
            seed=state["seed"],
            support_checkpointing=state["support_checkpointing"],
            **state["simulatorbackend_kwargs"],
        )
        self._seed_for_trial = state["seed_for_trial"]
        self.blackbox_name = state["blackbox_name"]
        self.dataset = state["dataset"]
        self._surrogate = state["surrogate"]
        self._surrogate_kwargs = state["surrogate_kwargs"]
        self._blackbox = None
        if "config_space_surrogate" in state:
            self._config_space_surrogate = config_space_from_json_dict(
                state["config_space_surrogate"]
            )
        else:
            self._config_space_surrogate = None


class UserBlackboxBackend(_BlackboxSimulatorBackend):
    """
    Version of :class:`_BlackboxSimulatorBackend`, where the blackbox is
    given as explicit :class:`Blackbox` object.
    See ``examples/launch_simulated_benchmark.py`` for an example on how to use.

    Additional arguments on top of parent :class:`_BlackboxSimulatorBackend`:

    :param blackbox: Blackbox to be used for simulation
    """

    def __init__(
        self,
        blackbox: Blackbox,
        elapsed_time_attr: str,
        max_resource_attr: Optional[str] = None,
        seed: Optional[int] = None,
        support_checkpointing: bool = True,
        **simulatorbackend_kwargs,
    ):
        super().__init__(
            elapsed_time_attr=elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            seed=seed,
            support_checkpointing=support_checkpointing,
            **simulatorbackend_kwargs,
        )
        self._blackbox = blackbox

    @property
    def blackbox(self) -> Blackbox:
        return self._blackbox
