import os
import logging
import numpy as np
import torch

from typing import Any
from pathlib import Path

from syne_tune.config_space import Integer, Float, FiniteRange, is_log_space
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.fmbo.history import History, dequantize

logger = logging.getLogger(__name__)


class ConfigGrammar:
    """
    Generates a regex pattern to constrain LLM output to valid configurations.

    The output format is: {cont_values},{cat_values}*{output}|

    Example with 2 continuous and 1 categorical hyperparameter:
        "500,400,<0>*123|"

    Structure:
        - Continuous values: token IDs 0 to num_numeric_tokens-1, decoded to their string representation
        - Categorical values: tokens <0>, <1>, ..., <num_categorical_tokens-1>
        - All hyperparameter values are comma-separated
        - '*' separates hyperparameters from the predicted output
        - '|' marks the end of the sequence

    The regex is built using actual token strings from the tokenizer vocabulary,
    ensuring the model only generates valid token sequences.
    """

    def __init__(
        self,
        tokenizer,
        config_space,
        n_continuous: int,
        n_categorical: int,
        hp_cat_names: list[str],
        num_numeric_tokens: int = 1000,
        num_categorical_tokens: int = 15,
    ):
        self.tokenizer = tokenizer
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.hp_cat_names = hp_cat_names
        self.config_space = config_space
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens

    def _get_continuous_tokens(self) -> list[str]:
        return [str(i) for i in range(self.num_numeric_tokens)]

    def _get_categorical_tokens(self) -> list[str]:
        return [f"<{i}>" for i in range(self.num_categorical_tokens)]

    def _get_separator_tokens(self) -> dict[str, str]:
        token_to_id = self.tokenizer.convert_tokens_to_ids
        return {
            "comma": self.tokenizer.convert_ids_to_tokens(token_to_id(",")),
            "star": self.tokenizer.convert_ids_to_tokens(token_to_id("*")),
            "pipe": self.tokenizer.convert_ids_to_tokens(token_to_id("|")),
        }

    def _escape_regex(self, s: str) -> str:
        import re

        return re.escape(s)

    def _build_continuous_pattern(self) -> str:
        tokens = self._get_continuous_tokens()
        escaped = [self._escape_regex(t) for t in tokens]
        return "(" + "|".join(escaped) + ")"

    def _build_categorical_pattern(self, max_categories: int = None) -> str:
        if max_categories is None:
            tokens = self._get_categorical_tokens()
        else:
            tokens = [f"<{i}>" for i in range(max_categories)]
        escaped = [self._escape_regex(t) for t in tokens]
        return "(" + "|".join(escaped) + ")"

    def build_regex(self) -> str:
        # TODO important note, right now we constrain the model to predict a token among the 1000 options
        #  it would be more efficient to check if values are in a range as values are continuous
        cont_pattern = self._build_continuous_pattern()
        separators = self._get_separator_tokens()

        comma = self._escape_regex(separators["comma"])
        star = self._escape_regex(separators["star"])
        pipe = self._escape_regex(separators["pipe"])

        patterns = []

        for _ in range(self.n_continuous):
            patterns.append(cont_pattern)

        for hp_cat in self.hp_cat_names:
            n_categories = len(self.config_space[hp_cat].categories)
            patterns.append(self._build_categorical_pattern(n_categories))

        if patterns:
            hp_pattern = comma.join(patterns)
            regex = hp_pattern + star + cont_pattern + pipe
        else:
            regex = star + cont_pattern + pipe

        return regex


def resolve_checkpoint(checkpoint_dir: str | Path) -> Path:
    """
    Resolve a checkpoint to a local path.

    If ``checkpoint_dir`` is an existing local directory, return it as-is.
    Otherwise treat it as a HuggingFace repo ID (e.g. ``"synetune/qwen3_30M_..."``
    or just ``"qwen3_30M_..."`` which is resolved to ``synetune/<name>``) and
    download it to the local HuggingFace cache.
    """
    path = Path(checkpoint_dir)
    if path.exists():
        return path
    repo_id = str(checkpoint_dir)
    if "/" not in repo_id:
        repo_id = f"synetune/{repo_id}"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id))


def detect_hf_checkpoint(path):
    """Returns True if the checkpoint is a HuggingFace model, False if litgpt."""
    path = os.fspath(path)
    files = set(os.listdir(path))

    lit_markers = {"lit_model.pth", "hyperparameters.yaml", "model_config.yaml"}
    if lit_markers & files:
        logging.debug("found litgpt model")
        return False

    hf_model_files = [f for f in files if f.endswith(".safetensors")]
    if "config.json" in files and hf_model_files:
        logging.debug("found hf model")
        return True


class FMBOSearcher(SingleObjectiveBaseSearcher):
    """
    Searcher using a pretrained FMBO LLM to suggest hyperparameter configurations.

    Supports litgpt checkpoints and HuggingFace checkpoints (including vllm inference).

    :param config_space: Configuration space
    :param points_to_evaluate: List of configurations to be evaluated initially (in that
        order). Each config in the list can be partially specified, or even be an empty
        dict. For each hyperparameter not specified, the default value is determined
        using a midpoint heuristic. If ``None`` (default), this is mapped to
        ``[dict()]``, a single default config determined by the midpoint heuristic. If
        ``[]`` (empty list), no initial configurations are specified.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        checkpoint_dir: str
        | Path = "synetune/qwen3_80M_token_2B_lr_5e-3_bsz_16_seed_0",
        task_info: dict | None = None,
        points_to_evaluate: list[dict[str, Any]] | None = None,
        random_seed: int | None = None,
        num_numeric_tokens: int = 1000,
        num_categorical_tokens: int = 15,
        remove_names: bool = True,
        n_sample_configurations: int = 1,
        use_vllm: bool = True,
        tokenizer_dir: str | Path = "synetune/bbo-pile-tokenizer",
    ):
        """
        :param checkpoint_dir: Local path to a model checkpoint, or a HuggingFace repo
            ID (e.g. ``"synetune/qwen3_30M_token_400M_lr_1e-2_bsz_8_seed_0"`` or just
            the short name without the ``synetune/`` prefix). When a repo ID is given
            the checkpoint is downloaded automatically via ``huggingface_hub``.
        :param config_space: Configuration space
        :param task_info: Dict with keys 'name', 'algorithm', 'metric_names'
        :param points_to_evaluate: Initial configurations to evaluate
        :param random_seed: Random seed
        :param num_numeric_tokens: Number of quantization levels for continuous values
        :param num_categorical_tokens: Maximum number of categories
        :param n_sample_configurations: Number of configurations to sample; picks the
            one with best predicted performance
        :param use_vllm: Use vllm for inference (requires HF checkpoint)
        :param tokenizer_dir: Local path to a tokenizer, or a HuggingFace repo ID.
            Defaults to ``"synetune/bbo-pile-tokenizer"`` which is downloaded
            automatically via ``huggingface_hub`` if not present locally.
        """
        super().__init__(config_space, points_to_evaluate, random_seed)
        checkpoint_dir = resolve_checkpoint(checkpoint_dir)
        tokenizer_dir = resolve_checkpoint(tokenizer_dir)
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        self.use_hf_checkpoint = detect_hf_checkpoint(checkpoint_dir)
        self.use_vllm = use_vllm
        if self.use_vllm:
            assert (
                self.use_hf_checkpoint
            ), "Can only use vllm with a HF checkpoint, convert the litgpt checkpoint first."
        if self.use_vllm:
            from vllm import LLM
            from vllm.config.structured_outputs import StructuredOutputsConfig
            from transformers import AutoTokenizer

            self.model = LLM(
                model=str(checkpoint_dir),
                enforce_eager=True,
                structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.use_hf_checkpoint:
            from transformers import AutoTokenizer, Qwen3ForCausalLM

            self.model = Qwen3ForCausalLM.from_pretrained(checkpoint_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            from litgpt.tokenizer import Tokenizer
            from litgpt.model import GPT
            from litgpt.config import Config

            config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
            self.model = GPT(config).cuda()
            self.tokenizer = Tokenizer(str(tokenizer_dir))
            state_dict = torch.load(
                str(checkpoint_dir / "lit_model.pth"),
                weights_only=True,
                map_location=torch.device("cpu")
                if not torch.cuda.is_available()
                else torch.device("cuda"),
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)

        self.random_state = np.random.RandomState(random_seed)
        self.num_numeric_tokens = num_numeric_tokens
        self.num_categorical_tokens = num_categorical_tokens
        self.n_sample_configurations = n_sample_configurations
        self.history = []

        if task_info is None:
            self.task_info = {
                "name": "tst",
                "algorithm": "BORE",
                "metric_names": "error",
            }
        else:
            self.task_info = task_info

        self.study = History(
            config_space=config_space,
            name=self.task_info["name"],
            algorithm=self.task_info["algorithm"],
            metric_names=[self.task_info["metric_names"]],
            num_numeric_tokens=self.num_numeric_tokens,
            remove_names=remove_names,
        )

        self.hp_cont_names = [
            hp_name
            for hp_name, hp in config_space.items()
            if isinstance(hp, (Float, Integer, FiniteRange))
        ]
        self.hp_cat_names = [
            hp_name
            for hp_name, hp in config_space.items()
            if not isinstance(hp, (Float, Integer, FiniteRange))
        ]
        self.config_space = {
            k: self.config_space[k] for k in self.hp_cont_names + self.hp_cat_names
        }

    def suggest(self, **kwargs) -> dict[str, Any] | None:
        config = self._next_points_to_evaluate()
        if config is not None:
            return config
        else:
            configs, ys = self._sample_n_configs()
            if len(configs) == 0:
                logging.warning("Sampling failed, return a random configuration!")
                return {k: v.sample() for k, v in self.config_space.items()}
            print(f"valid config: {len(configs)}/{self.n_sample_configurations}")
            return configs[np.argmin(ys)]

    def _sample_n_configs(self):
        configs = []
        ys = []
        prompt = self.study.get_prompt()
        completions = self._generate_n_suggestions(prompt=prompt)

        for completion in completions:
            try:
                config, y = self._decode_config(completion)

                for k, v in self.config_space.items():
                    if not hasattr(v, "sample"):
                        config[k] = v
                configs.append(config)
                ys.append(y)

            except ValueError as e:
                logging.warning(
                    f"Could not sample because of error: {str(e)}, skipping sampled configuration."
                )

        return configs, ys

    def _generate_n_suggestions(self, prompt: str) -> list[list[int]]:
        """Generate a string like `500,400,<0>*123|`"""
        if self.use_hf_checkpoint:
            if self.use_vllm:
                # 500,400,<0>*123| => 2N+2 tokens (should count | as well in vllm)
                max_new_tokens = (
                    len(self.hp_cont_names) + len(self.hp_cat_names)
                ) * 2 + 2

                grammar = ConfigGrammar(
                    tokenizer=self.tokenizer,
                    config_space=self.config_space,
                    n_continuous=len(self.hp_cont_names),
                    n_categorical=len(self.hp_cat_names),
                    hp_cat_names=self.hp_cat_names,
                    num_numeric_tokens=self.num_numeric_tokens,
                    num_categorical_tokens=self.num_categorical_tokens,
                )
                regex_pattern = grammar.build_regex()

                from vllm import SamplingParams
                from vllm.sampling_params import StructuredOutputsParams

                sampling_params = SamplingParams(
                    max_tokens=max_new_tokens,
                    n=self.n_sample_configurations,
                    structured_outputs=StructuredOutputsParams(regex=regex_pattern),
                )
                outputs = self.model.generate([prompt], sampling_params)
                tokens_configs = [
                    list(output.token_ids) for output in outputs[0].outputs
                ]
            else:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(
                        self.model.device
                    )
                    prompt_length = inputs["input_ids"].shape[1]

                    max_new_tokens = (
                        len(self.hp_cont_names) + len(self.hp_cat_names)
                    ) * 2 + 1
                    eos_token_id = self.tokenizer.convert_tokens_to_ids("|")

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=self.n_sample_configurations,
                        do_sample=True,
                        eos_token_id=eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    tokens_configs = [
                        output[prompt_length:].tolist() for output in outputs
                    ]
        else:
            from litgpt.generate.base import generate

            with torch.no_grad():
                prompt_tokens = self.tokenizer.encode(prompt)[
                    -self.model.max_seq_length :
                ]
                self.model.set_kv_cache(batch_size=1)

                max_returned_tokens = (
                    len(prompt_tokens)
                    + (len(self.hp_cont_names) + len(self.hp_cat_names)) * 2
                    + 1
                )

                tokens_configs = [
                    generate(
                        model=self.model,
                        prompt=prompt_tokens,
                        max_returned_tokens=max_returned_tokens,
                        include_prompt=False,
                        eos_id=self.tokenizer.token_to_id("|"),
                    ).tolist()
                    for _ in range(self.n_sample_configurations)
                ]
        return tokens_configs

    def _decode_config(self, tokens_config: list[int]) -> tuple[dict[str, Any], float]:

        token_to_id = (
            self.tokenizer.convert_tokens_to_ids
            if self.use_hf_checkpoint
            else self.tokenizer.token_to_id
        )

        star_index = tokens_config.index(token_to_id("*"))

        if star_index >= len(tokens_config) - 1:
            raise ValueError(f"Star index {star_index} is out of bounds.")
        tokens_hps = tokens_config[:star_index]
        token_output = tokens_config[star_index + 1]

        config = {}
        hp_value_tokens = [x for x in tokens_hps if x != token_to_id(",")]

        if len(hp_value_tokens) != len(self.hp_cont_names) + len(self.hp_cat_names):
            logging.warning("wrong length")

        for i, (hp_name, hp_token) in enumerate(
            zip(self.hp_cont_names + self.hp_cat_names, hp_value_tokens)
        ):
            is_continuous_hp = i < len(self.hp_cont_names)
            if is_continuous_hp:
                if self.use_hf_checkpoint:
                    int_token = int(self.tokenizer.convert_ids_to_tokens(hp_token))
                else:
                    int_token = int(self.tokenizer.id_to_token(hp_token))
                config[hp_name] = dequantize(
                    x=int_token,
                    x_min=self.config_space[hp_name].lower,
                    x_max=self.config_space[hp_name].upper,
                    q=self.num_numeric_tokens,
                    log_scale=is_log_space(self.config_space[hp_name]),
                )
            else:
                tokens_per_category = {
                    token_to_id(f"<{i}>"): cat
                    for i, cat in enumerate(self.config_space[hp_name].categories)
                }

                if hp_token not in tokens_per_category:
                    logging.warning(
                        f"Could not read category {hp_name}, got token {hp_token}."
                    )
                    config[hp_name] = self.config_space[hp_name].sample(
                        random_state=self.random_state
                    )
                else:
                    config[hp_name] = tokens_per_category[hp_token]

        for hp_name in self.hp_cat_names:
            if hp_name not in config:
                logging.warning(f"Did not sample category {hp_name}, sampling randomly")
                config[hp_name] = self.config_space[hp_name].sample(
                    random_state=self.random_state
                )

        # Return token_output as predicted metric (monotonic with actual value, sufficient for argmin)
        return config, token_output

    def on_trial_complete(
        self,
        trial_id: int,
        config: dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        if isinstance(metric, list):
            self.study.add_trial(config, metric[0])
        else:
            self.study.add_trial(config, metric)

    def on_trial_error(self, trial_id: int):
        return
