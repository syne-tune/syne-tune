import logging
from typing import Dict, Callable, Optional
import pandas as pd

import syne_tune.search_space as sp
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningScheduler, TransferLearningTaskEvaluations
from syne_tune.search_space import Categorical


class BoundingBox(TransferLearningScheduler):
    def __init__(
            self,
            scheduler_fun: Callable[[Dict, str, str], TrialScheduler],
            config_space: Dict,
            metric: str,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            mode: Optional[str] = None,
            num_hyperparameters_per_task: int = 1,
    ):
        """
        Simple baseline that computes a bounding-box of the best candidate found in previous tasks to restrict the
         search space to only good candidates. The bounding-box is obtained by restricting to the min-max of best
         numerical hyperparameters and restricting to the set of best candidates on categorical parameters.

        Reference: Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning.
        Valerio Perrone, Huibin Shen, Matthias Seeger, Cédric Archambeau, Rodolphe Jenatton. Neurips 2019.

        :param scheduler_fun: function that takes a configuration space (Dict), a mode (str) and a metric (str)
        and returns a scheduler. This is required since the final configuration space is known only after computing
        a bounding-box. For instance,
        `scheduler_fun=lambda new_config_space, mode, metric: RandomSearch(new_config_space, metric, mode)`
        will consider a random-search on the config-space is restricted to the bounding of best evaluations of previous
        tasks.
        :param config_space: initial search-space to consider, will be updated to the bounding of best evaluations of
        previous tasks
        :param metric: objective name to optimize, must be present in transfer learning evaluations.
        :param transfer_learning_evaluations: dictionary from task name to offline evaluations.
        :param mode: mode to be considered, default to min.
        :param num_hyperparameters_per_task: number of best hyperparameter to take per task when computing the bounding
        box, default to 1.
        """
        self._check_consistency(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=[metric],
        )
        self.mode = mode if mode is not None else "min"

        new_config_space = self.compute_box(
            config_space=config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            mode=self.mode,
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            metric=metric
        )
        self.config_space = new_config_space
        print(f"hyperparameter ranges of best previous configurations {new_config_space}")
        print(f"({sp.search_space_size(new_config_space)} options)")
        self.scheduler = scheduler_fun(new_config_space, mode, metric)
        super(BoundingBox, self).__init__(
            config_space=new_config_space,
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric_names=[metric],
        )

    @staticmethod
    def compute_box(
            config_space: Dict,
            transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
            mode: str,
            num_hyperparameters_per_task: int,
            metric: str
    ) -> Dict:
        # find the best hyperparameters on all tasks
        best_hps = []
        for task, evaluation in transfer_learning_evaluations.items():
            # average over seed and take last fidelity
            avg_objective_last_fidelity = evaluation.objective_values(objective_name=metric).mean(axis=1)[:, -1]
            best_hp_task_indices = avg_objective_last_fidelity.argsort()
            if mode == 'max':
                best_hp_task_indices = best_hp_task_indices[::-1]
            best_hps.append(evaluation.hyperparameters.loc[best_hp_task_indices[:num_hyperparameters_per_task]])
        hp_df = pd.concat(best_hps)

        # compute bounding-box on all hyperparameters that are numerical or categorical
        new_config_space = {}
        for i, (name, domain) in enumerate(config_space.items()):
            if hasattr(domain, "sample"):
                if isinstance(domain, Categorical):
                    hp_values = list(sorted(hp_df.loc[:, name].unique()))
                    new_config_space[name] = sp.choice(hp_values)
                elif hasattr(domain, "lower") and hasattr(domain, "upper"):
                    # domain is numerical, set new lower and upper ranges with bounding-box values
                    new_domain_dict = sp.to_dict(domain)
                    new_domain_dict['domain_kwargs']['lower'] = hp_df.loc[:, name].min()
                    new_domain_dict['domain_kwargs']['upper'] = hp_df.loc[:, name].max()
                    new_domain = sp.from_dict(new_domain_dict)
                    new_config_space[name] = new_domain
                else:
                    # no known way to compute bounding over non numerical domains such as functional
                    new_config_space[name] = domain
            else:
                new_config_space[name] = domain
        logging.info(f"new configuration space obtained after computing bounding-box: {new_config_space}")

        return new_config_space

    def suggest(self, *args, **kwargs):
        return self.scheduler.suggest(*args, **kwargs)

    def on_trial_add(self, *args, **kwargs):
        self.scheduler.on_trial_add(*args, **kwargs)

    def on_trial_complete(self, *args, **kwargs):
        self.scheduler.on_trial_complete(*args, **kwargs)

    def on_trial_remove(self, *args, **kwargs):
        self.scheduler.on_trial_remove(*args, **kwargs)

    def on_trial_error(self, *args, **kwargs):
        self.scheduler.on_trial_error(*args, **kwargs)

    def on_trial_result(self, *args, **kwargs) -> str:
        return self.scheduler.on_trial_result(*args, **kwargs)