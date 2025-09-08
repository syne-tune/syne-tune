from typing import Any, TYPE_CHECKING
import logging

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)

import syne_tune.config_space as sp


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution

logger = logging.getLogger(__name__)


def _syne_tune_domain_to_optuna_dist(name: str, dom: Any) -> tuple[Any, dict]:
    """
    Convert a Syne Tune domain (sp.*) or literal constant into an Optuna distribution.
    Returns (optuna_dist or None for constants, metadata dict)
    """
    # literal constant
    if isinstance(dom, (int, float, bool, str)):
        return None, {"constant": dom}

    # Categorical
    if isinstance(dom, sp.Categorical):
        choices = dom.categories
        if choices is None:
            raise RuntimeError(
                f"Categorical domain {name} has no categories attribute."
            )
        return CategoricalDistribution(choices), {
            "type": "categorical",
            "choices": choices,
        }

    # FiniteRange
    if isinstance(dom, sp.FiniteRange):
        lower = dom.lower
        upper = dom.upper
        size = dom.size
        cast_int = dom.cast_int if hasattr(dom, "cast_int") else False
        log_scale = dom.log_scale if hasattr(dom, "log_scale") else False

        if size is None:
            raise RuntimeError(f"Cannot infer size for FiniteRange '{name}'")

        # compute linear step
        computed_step = (upper - lower) / (size - 1) if size > 1 else 0.0

        if cast_int:
            # IntDistribution
            step_for_int = (
                int(computed_step)
                if float(computed_step).is_integer() and computed_step != 0
                else 1
            )
            return IntDistribution(
                int(lower),
                int(upper),
                step=step_for_int,
            ), {
                "type": "int_finite",
                "low": lower,
                "high": upper,
                "size": size,
                "step": computed_step,
                "cast_int": True,
                "log_scale": bool(log_scale),
            }
        else:
            # Finite floats -> use FloatDistribution. If the space is log-spaced,
            # set log=True so Optuna treats it as log-scale.
            return FloatDistribution(
                float(lower),
                float(upper),
                step=float(computed_step) if size > 1 else None,
                log=bool(log_scale),
            ), {
                "type": "float_finite",
                "low": lower,
                "high": upper,
                "size": size,
                "step": computed_step,
                "cast_int": False,
                "log_scale": bool(log_scale),
            }

    # Integer
    if isinstance(dom, sp.Integer):
        low = dom.lower
        high = dom.upper
        if low is None or high is None:
            raise RuntimeError(f"Integer domain {name} missing bounds.")
        return IntDistribution(int(low), int(high)), {
            "type": "int",
            "low": int(low),
            "high": int(high),
            "log": bool(sp.is_log_space(dom)),
        }

    # Float
    if isinstance(dom, sp.Float):
        low = dom.lower
        high = dom.upper
        if low is None or high is None:
            raise RuntimeError(f"Float domain {name} missing bounds.")
        log_flag = bool(sp.is_log_space(dom))
        return FloatDistribution(float(low), float(high), log=log_flag), {
            "type": "float",
            "low": low,
            "high": high,
            "log": bool(log_flag),
        }

    raise NotImplementedError(f"Unsupported domain type for key '{name}': {type(dom)}")


def _convert_syne_to_optuna_space(
    syne_space: dict[str, Any]
) -> dict[str, "BaseDistribution"]:
    """
    Convert entire Syne Tune config space mapping -> optuna distributions dict.
    """
    optuna_space: dict[str, "BaseDistribution"] = {}

    for name, dom in syne_space.items():
        optuna_dist, info = _syne_tune_domain_to_optuna_dist(name, dom)
        if optuna_dist is not None:
            optuna_space[name] = optuna_dist
    return optuna_space


class HEBOSearcher(SingleObjectiveBaseSearcher):
    """
    Syne Tune searcher that converts Syne Tune config-space -> Optuna distributions,
    instantiates Optunahub's HEBOSampler, and uses the ask/tell interface to query the Sampler.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        do_minimize: bool = True,
        random_seed: int | None = None,
    ):
        try:
            import optuna  # type: ignore
            import optunahub  # type: ignore
            # import distribution classes into module scope so
            # _syne_tune_domain_to_optuna_dist can reference them
            from optuna.distributions import (
                BaseDistribution,
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
            )
        except Exception as exc:
            raise RuntimeError(
                "HEBOSearcher requires the 'optuna' and 'optunahub' packages. "
                "Install them with: pip install optuna optunahub"
            ) from exc

        globals().update(
            {
                "optuna": optuna,
                "optunahub": optunahub,
                "BaseDistribution": BaseDistribution,
                "CategoricalDistribution": CategoricalDistribution,
                "FloatDistribution": FloatDistribution,
                "IntDistribution": IntDistribution,
            }
        )

        optuna_space = _convert_syne_to_optuna_space(config_space)
        self._optuna_space = optuna_space
        self._trial_map: dict[int, Any] = {}

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
        self._trial_map[trial._trial_id] = trial
        return trial.params

    def on_trial_complete(self, trial_id, config, metric):
        self._study.tell(trial=self._trial_map[trial_id])
