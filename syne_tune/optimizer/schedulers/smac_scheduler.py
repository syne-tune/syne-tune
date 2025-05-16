import logging
from typing import Optional, List, Dict, Any

from ConfigSpace import Configuration
from smac import Scenario, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue, TrialInfo

from syne_tune.backend.trial_status import Trial
from syne_tune.constants import ST_WORKER_TIME
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    TrialSuggestion,
)

logging.getLogger().setLevel(logging.INFO)

import ConfigSpace as CS

from syne_tune.config_space import Domain, Integer, is_log_space, Float


def to_smac_configspace(
    config_space: Dict[str, Domain], random_seed: int
) -> CS.ConfigurationSpace:
    cs = CS.ConfigurationSpace(seed=random_seed)
    for hp_name, hp in config_space.items():
        assert hasattr(hp, "sample"), "does not support constants."
        log = is_log_space(hp)
        if isinstance(hp.get_sampler(), Float._Uniform):
            cs.add_hyperparameter(CS.Float(hp_name, bounds=(hp.lower, hp.upper)))
        elif isinstance(hp.get_sampler(), Float._LogUniform):
            cs.add_hyperparameter(
                CS.Float(hp_name, bounds=(hp.lower, hp.upper), log=log)
            )
        elif isinstance(hp.get_sampler(), Integer._Uniform):
            cs.add_hyperparameter(
                CS.Integer(hp_name, bounds=(hp.lower, hp.upper), log=log)
            )
        elif isinstance(hp.get_sampler(), Integer._LogUniform):
            cs.add_hyperparameter(
                CS.Integer(hp_name, bounds=(hp.lower, hp.upper), log=log)
            )
        elif isinstance(hp.get_sampler(), Integer._LogUniform):
            cs.add_hyperparameter(
                CS.Integer(hp_name, bounds=(hp.lower, hp.upper), log=log)
            )
        elif hasattr(hp, "values"):
            # Convert logfinrange and finrange
            cs.add_hyperparameter(CS.Categorical(hp_name, items=hp.values))
        elif hasattr(hp, "categories"):
            cs.add_hyperparameter(CS.Categorical(hp_name, items=hp.categories))
        else:
            raise ValueError(
                f"Conversion to SMAC configspace for Hyperparameter {hp_name} {hp} not supported, "
                f"add a conversion to this function to support it."
            )
    return cs


class SMACScheduler(TrialScheduler):
    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        points_to_evaluate=None,
        random_seed: Optional[int] = None,
    ):
        """
        Wrapper to SMAC3. Requires SMAC3 to be installed, see https://github.com/automl/SMAC3 for instructions.
        :param config_space:
        :param metric: metric to be optimized, should be present in reported results dictionary
        :param do_minimize: True if we minimize the objective function
        :param points_to_evaluate: list of points to consider before calling the optimizer
        :param random_seed: to fix the behavior of smac
        """
        super(SMACScheduler, self).__init__(random_seed=random_seed)

        # compute part of the config space that are constants to add those when calling the blackbox
        self.config_space_constants = {
            k: v for k, v in config_space.items() if not hasattr(v, "sample")
        }
        config_space_non_constants = {
            k: v for k, v in config_space.items() if hasattr(v, "sample")
        }
        self.smac_configspace = to_smac_configspace(
            config_space_non_constants, random_seed=random_seed
        )
        self.metric = metric
        self.do_minimize = do_minimize
        self.points_to_evaluate = points_to_evaluate if points_to_evaluate else []

        scenario = Scenario(
            self.smac_configspace,
            deterministic=False,
            n_trials=1000,
            seed=random_seed if random_seed else -1,
        )

        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )

        self.smac = HyperparameterOptimizationFacade(
            scenario,
            lambda seed: -1,  # pass a dummy function since we are just using the ask & tell interface
            intensifier=intensifier,
            overwrite=True,
        )
        self.trial_info = {}

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        info = self.trial_info[trial.trial_id]
        cost = result[self.metric]
        if not self.do_minimize:
            cost *= -1
        self.smac.tell(
            info,
            TrialValue(
                cost=cost,
                time=result.get(ST_WORKER_TIME, 0),
            ),
        )

    def suggest(self) -> Optional[TrialSuggestion]:
        if self.points_to_evaluate:
            config = self.points_to_evaluate.pop()
            info = TrialInfo(
                config=Configuration(
                    configuration_space=self.smac_configspace, values=config
                ),
                seed=0,
            )
        else:
            info = self.smac.ask()
            config = dict(info.config)
            config.update(self.config_space_constants)

        trial_id = len(self.trial_info)
        self.trial_info[trial_id] = info
        return TrialSuggestion.start_suggestion(config)

    def __getstate__(self):
        # Avoid serialization issues with swig
        return None

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        metadata["metric_names"] = self.metric_names()
        metadata["metric_mode"] = self.metric_mode()
        return metadata

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"
