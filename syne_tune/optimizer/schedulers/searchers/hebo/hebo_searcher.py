from typing import Any
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
import syne_tune.config_space as sp

# optuna distributions for conversion
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

import optunahub
import optuna

logger = logging.getLogger(__name__)


# extract attributes from Syne Tune domains
def _get_attr(obj: Any, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _syne_tune_domain_to_optuna_dist(name: str, dom: Any):
    """
    Convert a Syne Tune domain (sp.*) or literal constant into an Optuna distribution.
    Returns optuna_dist or None for constants
    """
    # literal constant
    if isinstance(dom, (int, float, bool, str)):
        return None, {"constant": dom}

    # Syne Tune Categorical
    if isinstance(dom, sp.Categorical):
        choices = _get_attr(dom, "categories", "choices", default=None)
        if choices is None:
            # fallback: try repr
            raise RuntimeError(
                f"Categorical domain {name} has no categories attribute."
            )
        return CategoricalDistribution(choices), {
            "type": "categorical",
            "choices": choices,
        }

    # FiniteRange (discrete set, maybe cast_int)
    if isinstance(dom, sp.FiniteRange):
        lower = _get_attr(dom, "lower", "lower_bound", default=None)
        upper = _get_attr(dom, "upper", "upper_bound", default=None)
        size = _get_attr(dom, "size", default=None)
        cast_int = _get_attr(dom, "cast_int", default=False)
        step = _get_attr(dom, "step", default=None)
        if size is None:
            if step is not None:
                size = int(np.round((upper - lower) / step)) + 1
            else:
                raise RuntimeError(f"Cannot infer size/step for FiniteRange '{name}'")
        # If cast_int -> integer discrete values
        if cast_int:
            # compute step that maps to int grid
            computed_step = (upper - lower) / (size - 1) if size > 1 else 1
            # create IntDistribution with step if integer-valued
            return IntDistribution(
                int(lower),
                int(upper),
                step=int(computed_step) if computed_step.is_integer() else 1,
            ), {
                "type": "int_finite",
                "low": lower,
                "high": upper,
                "size": size,
                "step": computed_step,
                "cast_int": True,
            }
        else:
            # finite floats -> treat as FloatDistribution with step encoded (Optuna FloatDistribution supports step)
            computed_step = (upper - lower) / (size - 1) if size > 1 else None
            return FloatDistribution(
                float(lower),
                float(upper),
                step=float(computed_step) if computed_step is not None else None,
            ), {
                "type": "float_finite",
                "low": lower,
                "high": upper,
                "size": size,
                "step": computed_step,
                "cast_int": False,
            }

    # Integer domain
    if isinstance(dom, sp.Integer):
        low = _get_attr(dom, "lower", "lb", default=None)
        high = _get_attr(dom, "upper", "ub", default=None)
        if low is None or high is None:
            raise RuntimeError(f"Integer domain {name} missing bounds.")
        return IntDistribution(int(low), int(high)), {
            "type": "int",
            "low": int(low),
            "high": int(high),
        }

    # Float domain
    if isinstance(dom, sp.Float):
        low = _get_attr(dom, "lower", "lb", default=None)
        high = _get_attr(dom, "upper", "ub", default=None)
        log_flag = _get_attr(dom, "log", "log_scale", default=False)
        if low is None or high is None:
            raise RuntimeError(f"Float domain {name} missing bounds.")
        return FloatDistribution(float(low), float(high), log=bool(log_flag)), {
            "type": "float",
            "low": low,
            "high": high,
            "log": bool(log_flag),
        }

    raise NotImplementedError(f"Unsupported domain type for key '{name}': {type(dom)}")


def _convert_syne_to_optuna_space(
    syne_space: dict[str, Any]
) -> dict[str, BaseDistribution]:
    """
    Convert entire Syne Tune config space mapping -> optuna distributions dict.
    """
    optuna_space: dict[str, BaseDistribution] = {}

    for name, dom in syne_space.items():
        optuna_dist, info = _syne_tune_domain_to_optuna_dist(name, dom)
        if optuna_dist is not None:
            optuna_space[name] = optuna_dist
    return optuna_space


class HEBOSearcher(SingleObjectiveBaseSearcher):
    """
    Syne Tune searcher that converts Syne Tune config-space -> Optuna distributions,
    instantiates optunahub's HEBOSampler, and uses the ask/tell interface to query the Sampler.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        do_minimize: bool = True,
        random_seed: int | None = None,
    ):
        optuna_space = _convert_syne_to_optuna_space(config_space)
        self._optuna_space = optuna_space
        self.trials = []

        super().__init__(
            config_space=config_space, points_to_evaluate=None, random_seed=random_seed
        )

        self._do_minimize = do_minimize

        # Instantiate optunahub HEBOSampler
        HEBOSampler = optunahub.load_module("samplers/hebo").HEBOSampler

        self._study = optuna.create_study(
            sampler=HEBOSampler(seed=random_seed),
            direction="minimize" if do_minimize else "maximize",
        )

    def suggest(self, **kwargs):
        trial = self._study.ask(self._optuna_space)
        self.trials.append(trial)
        return trial.params

    def on_trial_complete(self, trial_id, config, metric):
        # Report back the objective
        self._study.tell(config, metric, trial=self.trials[-1])
