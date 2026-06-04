import json
from pathlib import Path

import numpy as np
import pandas as pd

from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    default_metric,
    metric_elapsed_time,
    time_attr,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path
from syne_tune.config_space import choice, loguniform, randint, uniform
from syne_tune.util import catchtime

# ── Metric name constants (all with metric_ prefix) ───────────────────────────

METRIC_ELAPSED_TIME = "metric_elapsed_time"
METRIC_RECON_LOSS = "metric_valid_recon_loss"  # multi-fidelity (per epoch)
METRIC_DOWNSTREAM = "metric_avg_ml_task_performance"

# SCHC-specific downstream metrics (final epoch only)
METRIC_AUTHOR_CELL_TYPE = "metric_author_cell_type"
METRIC_AGE_GROUP = "metric_age_group"
METRIC_SEX_SCHC = "metric_sex_schc"

# TCGA-specific downstream metrics (final epoch only)
METRIC_CANCER_TYPE = "metric_cancer_type"
METRIC_SUBTYPE = "metric_subtype"
METRIC_ONCOTREE_CODE = "metric_oncotree_code"
METRIC_SEX_TCGA = "metric_sex_tcga"  # same string as SCHC — shared column name
METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE = "metric_ajcc_pathologic_tumor_stage"
METRIC_GRADE = "metric_grade"
METRIC_PATH_N_STAGE = "metric_path_n_stage"
METRIC_DSS_STATUS = "metric_dss_status"
METRIC_OS_STATUS = "metric_os_status"

TIME_ATTR = "epoch"

# ── Per-dataset objective lists ───────────────────────────────────────────────
# Order matters: index 0 is RECON_LOSS (multi-fidelity), the rest are final-only.

TCGA_OBJECTIVES = [
    METRIC_ELAPSED_TIME,
    METRIC_RECON_LOSS,
    METRIC_SEX_TCGA,
    METRIC_DSS_STATUS,
    METRIC_OS_STATUS,
    METRIC_PATH_N_STAGE,
    METRIC_GRADE,
    METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE,
    METRIC_CANCER_TYPE,
    METRIC_SUBTYPE,
    METRIC_ONCOTREE_CODE,
    METRIC_DOWNSTREAM,
]

SCHC_OBJECTIVES = [
    METRIC_ELAPSED_TIME,
    METRIC_RECON_LOSS,
    METRIC_AUTHOR_CELL_TYPE,
    METRIC_AGE_GROUP,
    METRIC_SEX_SCHC,
    METRIC_DOWNSTREAM,
]

# Map the lower-cased dataset prefix (first token of task_name before "_") to
# the corresponding objective list.
DATASET_OBJECTIVES: dict[str, list[str]] = {
    "tcga": TCGA_OBJECTIVES,
    "schc": SCHC_OBJECTIVES,
}

ONTOLOGY_ARCHITECTURES = {"ontix"}

# ── Hyperparameter search spaces ──────────────────────────────────────────────

_SHARED_HPS = {
    "k_filter": choice([128, 256, 512, 1024, 2048, 4096]),
    "n_layers": choice([2, 3, 4]),
    "enc_factor": choice([1, 2, 3, 4]),
    "batch_size": choice([32, 64, 128, 256]),
    "learning_rate": loguniform(1e-5, 1e-1),
    "drop_p": uniform(0, 0.9),
    "weight_decay": loguniform(1e-5, 1e-1),
}

ARCHITECTURE_CONFIG_SPACES = {
    "vanillix": {**_SHARED_HPS, "latent_dim": choice([2, 4, 8, 16, 32, 64])},
    "varix": {
        **_SHARED_HPS,
        "beta": loguniform(0.001, 10),
        "latent_dim": choice([2, 4, 8, 16, 32, 64]),
    },
    "ontix": {**_SHARED_HPS, "beta": loguniform(0.0001, 1)},
    "disentanglix": {
        **_SHARED_HPS,
        "latent_dim": choice([2, 4, 8, 16, 32, 64]),
        "beta_mi": loguniform(0.001, 10.0),
        "beta_tc": loguniform(0.1, 10000),
        "beta_dimKL": loguniform(0.001, 10.0),
    },
}

# CHANGE TO YOUR LOCAL PATH
RESULTS_ROOT = Path("./bbomix_results")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dataset_prefix(task_name: str) -> str:
    """Return the lower-cased dataset identifier from a task name like 'tcga_...'."""
    return task_name.split("_")[0].lower()


def _objectives_for_task(task_name: str) -> list[str]:
    """Return the objective list for a given task, falling back to TCGA."""
    prefix = _dataset_prefix(task_name)
    if prefix not in DATASET_OBJECTIVES:
        print(
            f"  [WARN] Unknown dataset prefix '{prefix}' in task '{task_name}'; "
            f"falling back to TCGA objectives."
        )
    return DATASET_OBJECTIVES.get(prefix, TCGA_OBJECTIVES)


def _seed_parent_map(modalities_dir: Path, architecture: str) -> dict[str, Path]:
    """
    Return {ontology_suffix: seed_parent_dir} for one modalities directory.

    For ontology architectures subdirs are either seed dirs (no ontology level)
    or ontology dirs (one level deeper). For all others maps "" → modalities_dir.
    """
    if architecture not in ONTOLOGY_ARCHITECTURES:
        return {"": modalities_dir}

    subdirs = [d for d in sorted(modalities_dir.iterdir()) if d.is_dir()]
    if not subdirs or any(d.name.startswith("seed_") for d in subdirs):
        return {"": modalities_dir}

    return {d.name: d for d in subdirs}


def load_json_results(results_root: Path) -> dict[tuple, list[dict]]:
    """
    Walk results_root/<arch>/<dataset>/<modalities>/[<ontology>/]seed_<n>/<run>.json
    and return a mapping (architecture, dataset, modalities) → list[record].
    """
    records: dict[tuple, list[dict]] = {}
    n_files = 0

    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        architecture = arch_dir.name.lower()

        for dataset_dir in sorted(arch_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            for modalities_dir in sorted(dataset_dir.iterdir()):
                if not modalities_dir.is_dir():
                    continue

                for ontology_suffix, seed_parent in _seed_parent_map(
                    modalities_dir, architecture
                ).items():
                    task_modalities = (
                        f"{modalities_dir.name}_{ontology_suffix}"
                        if ontology_suffix
                        else modalities_dir.name
                    )
                    key = (architecture, dataset_dir.name, task_modalities)
                    records.setdefault(key, [])

                    for seed_dir in sorted(seed_parent.iterdir()):
                        if not seed_dir.is_dir() or not seed_dir.name.startswith(
                            "seed_"
                        ):
                            print(f"  [SKIP] unexpected directory: {seed_dir}")
                            continue
                        if not seed_dir.name[len("seed_") :].isdigit():
                            print(f"  [SKIP] non-integer seed directory: {seed_dir}")
                            continue

                        for json_file in sorted(seed_dir.glob("*.json")):
                            with open(json_file) as fh:
                                records[key].append(json.load(fh))
                            n_files += 1

    print(
        f"  Found {n_files} JSON files across "
        f"{len(records)} (architecture, dataset, modalities) combos"
    )
    if empty := [k for k, v in records.items() if not v]:
        print(f"  WARNING: 0 files found for {len(empty)} combo(s):")
        for k in empty:
            print(f"    {k}")

    return records


def _extract_hps(record: dict, hp_keys: list[str]) -> dict:
    """Extract hyperparameter values from a record's HYPERPARAMETERS block."""
    return {k: record["HYPERPARAMETERS"][k] for k in hp_keys}


def _hp_key(hp_row: dict) -> tuple:
    return tuple(sorted(hp_row.items()))


def _build_arch_index(
    all_runs: list[dict], hp_keys: list[str]
) -> tuple[pd.DataFrame, dict[tuple, int], list[int], int]:
    max_epochs = 300

    seed_configs: dict[int, set[tuple]] = {}
    hp_rows: dict[tuple, dict] = {}

    for record in all_runs:
        hp_row = _extract_hps(record, hp_keys)
        key = _hp_key(hp_row)
        seed_configs.setdefault(record["SEED"], set()).add(key)
        if key not in hp_rows:
            hp_rows[key] = hp_row

    all_seeds = sorted(seed_configs)
    shared_configs = set.intersection(*(seed_configs[s] for s in all_seeds))

    n_union = len(set.union(*(seed_configs[s] for s in all_seeds)))
    n_shared = len(shared_configs)
    print(
        f"Global HP intersection: keeping {n_shared} / {n_union} configs present in all seeds."
    )

    hp_order = [k for k in hp_rows if k in shared_configs]
    union_hp_df = pd.DataFrame([hp_rows[k] for k in hp_order]).reset_index(drop=True)

    return union_hp_df, {k: i for i, k in enumerate(hp_order)}, all_seeds, max_epochs


# ── Mapping from metric_ name → JSON record key ───────────────────────────────
# Only final-epoch metrics need an explicit mapping; RECON_LOSS is handled
# specially because it is read from the per-epoch loss_dict.

_METRIC_TO_RECORD_KEY: dict[str, str] = {
    # shared
    METRIC_ELAPSED_TIME: "RUNTIME_SECONDS",
    METRIC_DOWNSTREAM: "AVG_ML_TASK_PERFORMANCE",
    # SCHC
    METRIC_AUTHOR_CELL_TYPE: "author_cell_type",
    METRIC_AGE_GROUP: "age_group",
    METRIC_SEX_SCHC: "sex",
    # TCGA
    METRIC_CANCER_TYPE: "CANCER_TYPE",
    METRIC_SUBTYPE: "SUBTYPE",
    METRIC_ONCOTREE_CODE: "ONCOTREE_CODE",
    METRIC_SEX_TCGA: "SEX",
    METRIC_AJCC_PATHOLOGIC_TUMOR_STAGE: "AJCC_PATHOLOGIC_TUMOR_STAGE",
    METRIC_GRADE: "GRADE",
    METRIC_PATH_N_STAGE: "PATH_N_STAGE",
    METRIC_DSS_STATUS: "DSS_STATUS",
    METRIC_OS_STATUS: "OS_STATUS",
}


def _fill_objectives(
    records: list[dict],
    hp_keys: list[str],
    hp_index: dict[tuple, int],
    seed_to_idx: dict[int, int],
    shape: tuple[int, int, int],
    objectives: list[str],
) -> np.ndarray:
    """
    Build objectives_evaluations array of shape (num_hps, num_seeds, max_epochs, len(objectives)).

    Filling strategy
    ----------------
    - METRIC_RECON_LOSS   : filled at *every* epoch from loss_per_epoch (multi-fidelity).
    - All other objectives: filled only at the *final* epoch of each run.
    """
    obj_array = np.full((*shape, len(objectives)), np.nan, dtype=np.float64)
    obj_idx = {name: i for i, name in enumerate(objectives)}
    final_only_metrics = [m for m in objectives if m != METRIC_RECON_LOSS]

    for record in records:
        hp_row = _extract_hps(record, hp_keys)
        key = _hp_key(hp_row)
        if key not in hp_index:
            continue

        hp_idx = hp_index[key]
        s_idx = seed_to_idx[record["SEED"]]
        loss_dict: dict[str, float] = record["loss_per_epoch"]
        final_e = max(int(k) for k in loss_dict)  # 0-indexed final (e.g. 299)

        if METRIC_RECON_LOSS in obj_idx:
            for epoch_str, recon_loss in loss_dict.items():
                e = int(epoch_str)  # 0..299
                obj_array[hp_idx, s_idx, e, obj_idx[METRIC_RECON_LOSS]] = recon_loss

        per_task = record.get("PER_TASK_PERFORMANCE") or {}  # safe fallback

        for metric_name in final_only_metrics:
            if metric_name not in obj_idx:
                continue
            record_key = _METRIC_TO_RECORD_KEY.get(metric_name)
            if record_key is None:
                print(
                    f"  [WARN] No record key mapping for metric '{metric_name}'; skipping."
                )
                continue
            if record_key in ("AVG_ML_TASK_PERFORMANCE", "RUNTIME_SECONDS"):
                value = record.get(record_key)
            else:
                value = per_task.get(record_key)
            if value is None:
                continue
            obj_array[hp_idx, s_idx, final_e, obj_idx[metric_name]] = value

    return obj_array


# ── Main conversion entry point ───────────────────────────────────────────────


def generate_bbomix_from_json(results_root: Path = RESULTS_ROOT) -> None:
    with catchtime("loading JSON results"):
        all_records = load_json_results(results_root)

    if not all_records:
        raise FileNotFoundError(
            f"No JSON run files found under '{results_root}'. "
            "Expected layout: <arch>/<dataset>/<modalities>/[<ontology>/]seed_<n>/<run>.json"
        )

    # Group by architecture: arch → {task_name: [records]}
    arch_tasks: dict[str, dict[str, list[dict]]] = {}
    for (arch, dataset, modalities), records in all_records.items():
        arch_tasks.setdefault(arch, {})[f"{dataset}_{modalities}"] = records

    for architecture, task_records in arch_tasks.items():
        config_space = ARCHITECTURE_CONFIG_SPACES.get(architecture)
        if config_space is None:
            print(f"[SKIP] No config space for '{architecture}'.")
            continue

        hp_keys = list(config_space.keys())
        all_runs = [r for recs in task_records.values() for r in recs]
        print(
            f"\nArchitecture: {architecture}  ({len(all_runs)} runs, {len(task_records)} tasks)"
        )

        # Build ONE global HP index shared across all tasks in this architecture.
        # The HP space is dataset-agnostic, so we can safely build it from all runs.
        global_hp_df, global_hp_index, all_seeds, max_epochs = _build_arch_index(
            all_runs, hp_keys
        )
        seed_to_idx = {s: i for i, s in enumerate(all_seeds)}
        num_hps = len(global_hp_df)
        num_seeds = len(all_seeds)
        fidelity_space = {TIME_ATTR: randint(lower=1, upper=300)}

        print(f"Seeds: {all_seeds}")
        print(f"Global HP configs: {num_hps}")

        # Group tasks by dataset prefix so each serialized blackbox contains
        # only tasks that share the same objective list (syne-tune constraint).
        dataset_bb_dicts: dict[str, dict[str, BlackboxTabular]] = {}

        for task_name, records in task_records.items():
            dataset_prefix = _dataset_prefix(task_name)
            objectives = _objectives_for_task(task_name)
            print(
                f"  Converting {len(records):5d} runs  →  task={task_name}  "
                f"dataset={dataset_prefix}  ({len(objectives)} objectives)"
            )

            with catchtime(f"    filling {task_name}"):
                obj_array = _fill_objectives(
                    records,
                    hp_keys,
                    global_hp_index,
                    seed_to_idx,
                    shape=(num_hps, num_seeds, max_epochs),
                    objectives=objectives,
                )

            dataset_bb_dicts.setdefault(dataset_prefix, {})[
                task_name
            ] = BlackboxTabular(
                hyperparameters=global_hp_df,
                configuration_space=config_space,
                fidelity_space=fidelity_space,
                objectives_evaluations=obj_array,
                objectives_names=objectives,
            )

        # Serialize one blackbox per (architecture, dataset) pair.
        for dataset_prefix, bb_dict in dataset_bb_dicts.items():
            blackbox_name = f"bbomix_{architecture}_{dataset_prefix}"
            with catchtime(f"  serializing {blackbox_name}"):
                serialize(
                    bb_dict=bb_dict,
                    path=repository_path / blackbox_name,
                    metadata={
                        metric_elapsed_time: METRIC_ELAPSED_TIME,
                        default_metric: METRIC_DOWNSTREAM,
                        time_attr: TIME_ATTR,
                    },
                )
            print(f"  Saved {blackbox_name}  tasks: {sorted(bb_dict)}")


# ── Recipe classes ────────────────────────────────────────────────────────────


class BBOmixJsonRecipe(BlackboxRecipe):
    def __init__(self, architecture: str, dataset: str):
        super().__init__(
            name=f"bbomix_{architecture}_{dataset}",
            cite_reference=(
                "BBOmix: A Tabular Benchmark for Hyperparameter Optimization "
                "of Unsupervised Biological Representation Learning, "
                "Luca Thale-Bombien, Jan Ewald, Ralf Koenig, Aaron Klein. Arxiv, 2026"
            ),
        )
        self.architecture = architecture
        self.dataset = dataset

    def _generate_on_disk(self) -> None:
        if not (repository_path / self.name).exists():
            generate_bbomix_from_json()


def _make_recipe(class_name: str, architecture: str, dataset: str):
    return type(
        class_name,
        (BBOmixJsonRecipe,),
        {
            "__init__": lambda self, _a=architecture, _d=dataset: BBOmixJsonRecipe.__init__(
                self, _a, _d
            )
        },
    )


BBOmixVanillixSchcJsonRecipe = _make_recipe(
    "BBOmixVanillixSchcJsonRecipe", "vanillix", "schc"
)
BBOmixVanillixTcgaJsonRecipe = _make_recipe(
    "BBOmixVanillixTcgaJsonRecipe", "vanillix", "tcga"
)
BBOmixVarixTcgaJsonRecipe = _make_recipe("BBOmixVarixTcgaJsonRecipe", "varix", "tcga")
BBOmixVarixSchcJsonRecipe = _make_recipe("BBOmixVarixSchcJsonRecipe", "varix", "schc")
BBOmixOntixTcgaJsonRecipe = _make_recipe("BBOmixOntixTcgaJsonRecipe", "ontix", "tcga")
BBOmixOntixSchcJsonRecipe = _make_recipe("BBOmixOntixSchcJsonRecipe", "ontix", "schc")
BBOmixDisentanglixTcgaJsonRecipe = _make_recipe(
    "BBOmixDisentanglixTcgaJsonRecipe", "disentanglix", "tcga"
)
BBOmixDisentanglixSchcJsonRecipe = _make_recipe(
    "BBOmixDisentanglixSchcJsonRecipe", "disentanglix", "schc"
)


if __name__ == "__main__":
    recipes = [
        BBOmixVanillixSchcJsonRecipe,
        BBOmixVanillixTcgaJsonRecipe,
        BBOmixVarixTcgaJsonRecipe,
        BBOmixVarixSchcJsonRecipe,
        BBOmixOntixTcgaJsonRecipe,
        BBOmixOntixSchcJsonRecipe,
        BBOmixDisentanglixTcgaJsonRecipe,
        BBOmixDisentanglixSchcJsonRecipe,
    ]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=True)
