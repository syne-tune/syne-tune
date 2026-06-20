import json
import random

from dataclasses import dataclass, field

import numpy as np
from syne_tune.config_space import (
    Categorical,
    Float,
    Integer,
    Domain,
    config_space_from_json_dict,
    FiniteRange,
    is_log_space,
)
from syne_tune.experiments import ExperimentResult


def quantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Quantize a value x to be in [0, q-1] based on the range [x_min, x_max].
    q is the number of quantization levels.
    """
    if x_min == x_max:
        return 0
    if log_scale:
        x = np.log(x + 1e-10)
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
    x_norm = (x - x_min) / (x_max - x_min)
    return int(round(x_norm * (q - 1)))


def dequantize(x, x_min, x_max, q=1000, log_scale=False):
    """
    Dequantize a value x from [0, q-1] to the range [x_min, x_max].
    q is the number of quantization levels.
    """
    if log_scale:
        x_min = np.log(x_min + 1e-10)
        x_max = np.log(x_max + 1e-10)
        return np.exp(x / (q - 1) * (x_max - x_min) + x_min)
    return x / (q - 1) * (x_max - x_min) + x_min


def encode(x, hp: Domain, q: int = 1000, hp_name: str = ""):
    """
    Encode a value x based on the type of hyperparameter hp.
    """
    if isinstance(hp, Categorical):
        # TODO: handle this in a more principled way
        if hp_name == "proc.skew_threshold" and np.isnan(x):
            x = "None"
        if hp_name == "proc.skew_threshold" and isinstance(x, float):
            x = str(x)
        if hp_name == "num_layers" and isinstance(x, np.int64):
            x = str(x)
        if hp_name == "max_features":
            x = str(x)
        return f"<{hp.categories.index(x)}>"
    elif isinstance(hp, (Float, Integer, FiniteRange)):
        return quantize(x, hp.lower, hp.upper, q, log_scale=is_log_space(hp))
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")


@dataclass
class Trial:
    config: dict
    metric: int


@dataclass
class History:
    name: str
    algorithm: str
    config_space: dict
    num_numeric_tokens: int = 1000
    metric_names: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    remove_names: bool = False

    def add_trial(self, config, result):

        trial = Trial(config, result)
        self.trials.append(trial)

    def get_prompt(self, shuffle=False):
        string = ""
        if not self.remove_names:
            string += f"benchmark:{self.name},"
        string += f"algorithm:{self.algorithm},"
        hypers = list(self.config_space.items())
        if shuffle:
            random.shuffle(hypers)
        # sort hyperparameters: continuous first, categorical last
        continues_hypers = []
        categorical_hypers = []
        for hp_name, hp in hypers:
            if isinstance(hp, Categorical):
                categorical_hypers.append((hp_name, hp))
            else:
                continues_hypers.append((hp_name, hp))
        hypers = continues_hypers + categorical_hypers
        string += f"search-space:"
        for hp_name, hp in hypers:
            string += "{"
            if not self.remove_names:
                string += f"name:{hp_name},"

            if isinstance(hp, Categorical):

                string += f"type:CAT,"
                string += (
                    f"categories:{[i for i in range(len(hp.categories))]}".replace(
                        " ", ""
                    )
                )
            elif isinstance(hp, Float):
                string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
                string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            elif isinstance(hp, Integer):
                string += f"type:INT,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
                string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            elif isinstance(hp, FiniteRange):
                if hp.cast_int:
                    string += f"type:INT,"
                else:
                    string += f"type:UNI,"
                string += f"min_value:{hp.lower},"
                string += f"max_value:{hp.upper},"
                string += f"log_scale" if is_log_space(hp) else f"linear_scale"
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
            string += "}"

        string += ",history:"

        if len(self.trials) > 0:
            y_min = min(trial.metric for trial in self.trials)
            y_max = max(trial.metric for trial in self.trials)
            if y_min == y_max:
                y_max += 1  # Avoid division by zero in quantization
            for trial in self.trials:
                for i, (hp_name, hp) in enumerate(hypers):
                    if not isinstance(hp, Domain):
                        continue
                    if i > 0:
                        string += ","

                    hp_encoded = encode(
                        trial.config[hp_name],
                        hp,
                        hp_name=hp_name,
                        q=self.num_numeric_tokens,
                    )
                    string += str(hp_encoded)
                string += f"*"

                string += (
                    f"{quantize(trial.metric, y_min, y_max, q=self.num_numeric_tokens)}"
                )
                string += f"|"
        return string

    @classmethod
    def from_syne_tune_experiment(
        cls,
        experiment: ExperimentResult,
        num_numeric_tokens: int = 1000,
        remove_names: bool = False,
        max_num_trials: int = None,
    ):
        """
        Create a History object from a Syne Tune ExperimentResult.
        """
        metadata = experiment.metadata
        config_space = config_space_from_json_dict(json.loads(metadata["config_space"]))
        metric_name = metadata["metric_names"][0]
        results = experiment.results
        mode = metadata["metric_mode"] if "metric_mode" in metadata else "min"
        benchmark_name = (
            metadata["benchmark"] if "benchmark" in metadata else metadata["entrypoint"]
        )
        algorithm_name = (
            metadata["algorithm"]
            if "algorithm" in metadata
            else metadata["scheduler_name"]
        )
        hist = cls(
            config_space=config_space,
            name=benchmark_name,
            algorithm=algorithm_name,
            metric_names=metric_name,
            num_numeric_tokens=num_numeric_tokens,
            remove_names=remove_names,
        )

        for i, (trial_id, trial) in enumerate(results.groupby("trial_id")):
            row = trial.iloc[-1]
            config = {k: row[f"config_{k}"] for k in config_space.keys()}
            result = row[metric_name]
            if mode == "max":
                result = -result
            hist.add_trial(config, result)
            if max_num_trials is not None and i >= max_num_trials - 1:
                break

        return hist
