#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()

SCRIPT173_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script173_successful_corrective_six_way_architecture_selection_core_sha256.txt"
)

SELECTED_CONFIGURATION_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_aggregate_v2/"
      "selected_corrective_architecture_registry.tsv"
)

SCRIPT175_PATH = (
    PROJECT / "scripts/175_generate_and_freeze_final_transfer_splits.py"
)
EXPECTED_SCRIPT175_SHA256 = (
    "985f37447f45d65760273577d733239394316017959e1a13d48fc0f75076b3a5"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1"
)
TABLE_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/preregistration_v1"
)

PROTOCOL_OUTPUT = OUTPUT_ROOT / "final_transfer_protocol_v1.tsv"
SOURCE_REGIME_OUTPUT = OUTPUT_ROOT / "final_transfer_source_regime_registry_v1.tsv"
TARGET_PANEL_OUTPUT = OUTPUT_ROOT / "final_transfer_target_panel_registry_v1.tsv"
CONFIGURATION_OUTPUT = OUTPUT_ROOT / "frozen_outer_configuration_registry_v1.tsv"
EXPERIMENT_CELL_OUTPUT = OUTPUT_ROOT / "final_transfer_experiment_cell_registry_v1.tsv"
INPUT_MANIFEST_OUTPUT = OUTPUT_ROOT / "script174_input_manifest.tsv"
PLAN_SUMMARY_OUTPUT = TABLE_ROOT / "final_transfer_plan_summary_v1.tsv"
OUTPUT_MANIFEST = OUTPUT_ROOT / "script174_outputs_sha256.txt"
FREEZE_OUTPUT = (
    OUTPUT_ROOT
    / "script174_successful_final_transfer_preregistration_core_sha256.txt"
)

RANDOM_SEED = 20260814
MODEL_SEEDS = (20260815, 20260816, 20260817)
TARGET_PROTOCOLS = (
    "random_pair",
    "genome_disjoint",
    "drug_held_out",
)
SUPPORT_BUDGETS = (0, 1, 5, 10)

KP_PANEL = (
    "amikacin",
    "aztreonam",
    "cefepime",
    "cefmetazole",
    "cefotaxime",
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "cefuroxime",
    "ciprofloxacin",
    "imipenem",
    "levofloxacin",
    "meropenem",
    "minocycline",
    "tetracycline",
    "tigecycline",
    "tobramycin",
)
EC_PANEL = tuple(sorted(set(KP_PANEL).union({"ampicillin", "chloramphenicol"})))
SE_PANEL = (
    "ampicillin",
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "chloramphenicol",
    "ciprofloxacin",
    "meropenem",
    "tetracycline",
)

TARGET_PANELS = {
    "kp": KP_PANEL,
    "ec": EC_PANEL,
    "se": SE_PANEL,
}

COMMON_SIX = (
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "ciprofloxacin",
    "meropenem",
    "tetracycline",
)

SOURCE_REGIMES = (
    ("se", "kp_to_se", "kp", "single_source"),
    ("se", "ec_to_se", "ec", "single_source"),
    ("se", "kp_plus_ec_to_se", "kp|ec", "equal_species_multisource"),
    ("ec", "kp_to_ec", "kp", "single_source"),
    ("ec", "se_to_ec", "se", "single_source"),
    ("ec", "kp_plus_se_to_ec", "kp|se", "equal_species_multisource"),
    ("kp", "se_to_kp", "se", "single_source"),
    ("kp", "ec_to_kp", "ec", "single_source"),
    ("kp", "se_plus_ec_to_kp", "se|ec", "equal_species_multisource"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else PROJECT / path


def verify_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)

    verified: list[Path] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed SHA line {line_number}: {path}")
        expected, value = parts
        candidate = project_path(value)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != expected:
            raise RuntimeError(f"SHA mismatch: {candidate}")
        verified.append(candidate)

    if not verified:
        raise RuntimeError(f"Empty SHA manifest: {path}")
    return verified


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def write_manifest(paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for candidate in sorted(
            {path.resolve() for path in paths},
            key=lambda value: value.as_posix(),
        ):
            try:
                display = candidate.relative_to(PROJECT)
            except ValueError:
                display = candidate
            handle.write(f"{sha256_file(candidate)}  {display}\n")


def main() -> None:
    required = [
        SCRIPT173_FREEZE,
        SELECTED_CONFIGURATION_PATH,
        SCRIPT175_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified_architecture = verify_manifest(SCRIPT173_FREEZE)

    observed_script175_sha = sha256_file(SCRIPT175_PATH)
    if observed_script175_sha != EXPECTED_SCRIPT175_SHA256:
        raise RuntimeError(
            "Script 175 SHA mismatch: "
            f"{observed_script175_sha}"
        )

    selected = read_tsv(SELECTED_CONFIGURATION_PATH)
    required_columns = {
        "outer_target_code",
        "genome_representation",
        "corrective_genome_variant",
        "shared_hp_id",
        "low_rank_interaction_rank",
        "drug_representation",
        "drug_view_fusion_method",
        "drug_view_low_rank",
        "cross_modal_architecture",
    }
    missing = sorted(required_columns - set(selected.columns))
    if missing:
        raise RuntimeError(
            f"Selected-configuration registry missing columns: {missing}"
        )

    selected = selected.sort_values("outer_target_code").reset_index(drop=True)
    if set(selected["outer_target_code"]) != {"kp", "ec", "se"}:
        raise RuntimeError("Expected exactly one configuration per outer target.")
    if selected["outer_target_code"].duplicated().any():
        raise RuntimeError("Duplicate outer-target configuration.")

    target_panel_rows: list[dict[str, object]] = []
    for target_code, panel in TARGET_PANELS.items():
        target_panel_rows.append(
            {
                "target_species_code": target_code,
                "target_drug_count": len(panel),
                "target_antibiotics": "|".join(sorted(panel)),
                "common_six_antibiotics": "|".join(COMMON_SIX),
                "final_evaluation_policy": (
                    "evaluate full target panel and additionally report "
                    "source-seen, source-unseen and common-six subsets"
                ),
            }
        )
    target_panel_registry = pd.DataFrame(target_panel_rows).sort_values(
        "target_species_code"
    ).reset_index(drop=True)

    source_rows: list[dict[str, object]] = []
    selected_lookup = selected.set_index("outer_target_code").to_dict(
        orient="index"
    )
    for target, regime_id, source_codes, sampling_policy in SOURCE_REGIMES:
        config = selected_lookup[target]
        source_list = source_codes.split("|")
        source_union = sorted(
            set().union(*(set(TARGET_PANELS[source]) for source in source_list))
        )
        source_rows.append(
            {
                "outer_target_code": target,
                "source_regime_id": regime_id,
                "source_species_codes": source_codes,
                "source_species_count": len(source_list),
                "source_sampling_policy": sampling_policy,
                "source_training_panel_policy": (
                    "complete eligible source panel for every source species"
                ),
                "source_union_drug_count": len(source_union),
                "source_union_antibiotics": "|".join(source_union),
                "target_drug_count": len(TARGET_PANELS[target]),
                "target_antibiotics": "|".join(sorted(TARGET_PANELS[target])),
                "genome_representation": config["genome_representation"],
                "corrective_genome_variant": config[
                    "corrective_genome_variant"
                ],
                "shared_hp_id": config["shared_hp_id"],
                "low_rank_interaction_rank": config[
                    "low_rank_interaction_rank"
                ],
                "drug_representation": config["drug_representation"],
                "drug_view_fusion_method": config[
                    "drug_view_fusion_method"
                ],
                "drug_view_low_rank": config["drug_view_low_rank"],
                "cross_modal_architecture": config[
                    "cross_modal_architecture"
                ],
                "outer_target_query_labels_used_for_configuration": "NO",
            }
        )
    source_registry = pd.DataFrame(source_rows).sort_values(
        ["outer_target_code", "source_regime_id"]
    ).reset_index(drop=True)

    cell_rows: list[dict[str, object]] = []
    for regime in source_registry.to_dict(orient="records"):
        for protocol in TARGET_PROTOCOLS:
            for budget in SUPPORT_BUDGETS:
                cell_rows.append(
                    {
                        "outer_target_code": regime["outer_target_code"],
                        "source_regime_id": regime["source_regime_id"],
                        "target_protocol": protocol,
                        "target_support_budget_percent": budget,
                        "transfer_mode": (
                            "zero_shot" if budget == 0 else "few_shot"
                        ),
                        "query_policy": (
                            "identical frozen query cohort for zero-shot and "
                            "all few-shot budgets"
                        ),
                        "configuration_status": "FROZEN",
                    }
                )
    experiment_cells = pd.DataFrame(cell_rows).sort_values(
        [
            "outer_target_code",
            "source_regime_id",
            "target_protocol",
            "target_support_budget_percent",
        ]
    ).reset_index(drop=True)

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_id",
                "value": "final_cross_species_zero_and_few_shot_transfer_v1",
            },
            {
                "item": "outer_configuration_policy",
                "value": (
                    "one frozen target-specific configuration selected without "
                    "using the held-out target species"
                ),
            },
            {
                "item": "source_regimes",
                "value": "9: two single-source and one equal-sampled multi-source per target",
            },
            {
                "item": "source_training_panel",
                "value": "complete eligible panel of each source species",
            },
            {
                "item": "target_evaluation_panel",
                "value": (
                    "complete eligible target panel; source-seen, source-unseen "
                    "and common-six subsets reported separately"
                ),
            },
            {
                "item": "target_protocols",
                "value": "random_pair|genome_disjoint|drug_held_out",
            },
            {
                "item": "random_pair_interpretation",
                "value": "pair-level interpolation; not unseen-isolate prediction",
            },
            {
                "item": "genome_disjoint_interpretation",
                "value": "unseen target isolate; not claimed as unseen lineage",
            },
            {
                "item": "drug_held_out_interpretation",
                "value": (
                    "target antibiotic receives no target MIC label; distinguish "
                    "source-seen from unseen in source MIC supervision"
                ),
            },
            {
                "item": "target_folds",
                "value": "5 random-pair folds and 5 genome-disjoint folds per species",
            },
            {
                "item": "drug_folds",
                "value": "one leave-one-target-antibiotic-out fold per eligible target drug",
            },
            {
                "item": "target_support_budgets_percent",
                "value": "0|1|5|10",
            },
            {
                "item": "nested_target_support",
                "value": "S_1_subset_S_5_subset_S_10",
            },
            {
                "item": "support_reuse",
                "value": (
                    "same support observations reused across source regimes, "
                    "model seeds and target-only controls"
                ),
            },
            {
                "item": "split_seed",
                "value": str(RANDOM_SEED),
            },
            {
                "item": "model_seeds",
                "value": "|".join(str(seed) for seed in MODEL_SEEDS),
            },
            {
                "item": "primary_metric",
                "value": "per-antibiotic macro RMSE",
            },
            {
                "item": "secondary_metrics",
                "value": (
                    "macro MAE|one-tier accuracy|R2|Pearson|Spearman|worst-drug RMSE"
                ),
            },
            {
                "item": "fold_seed_summary",
                "value": (
                    "average model seeds within each target fold, then report "
                    "mean and sample SD across folds"
                ),
            },
            {
                "item": "zero_shot_checkpoint_reuse",
                "value": (
                    "one source checkpoint per source regime and model seed; "
                    "evaluate on every frozen target query subset"
                ),
            },
            {
                "item": "adaptation_policy_status",
                "value": (
                    "must be frozen before few-shot; selected without the outer "
                    "target or defined a priori"
                ),
            },
            {
                "item": "target_query_use_in_tuning",
                "value": "PROHIBITED",
            },
            {
                "item": "models_trained_by_script174",
                "value": "NO",
            },
        ]
    )

    plan_summary = (
        experiment_cells.groupby(
            ["outer_target_code", "target_protocol"],
            as_index=False,
        )
        .agg(
            source_regimes=("source_regime_id", "nunique"),
            support_budgets=("target_support_budget_percent", "nunique"),
            broad_experiment_cells=("source_regime_id", "size"),
        )
        .sort_values(["outer_target_code", "target_protocol"])
        .reset_index(drop=True)
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    write_tsv(protocol, PROTOCOL_OUTPUT)
    write_tsv(source_registry, SOURCE_REGIME_OUTPUT)
    write_tsv(target_panel_registry, TARGET_PANEL_OUTPUT)
    write_tsv(selected, CONFIGURATION_OUTPUT)
    write_tsv(experiment_cells, EXPERIMENT_CELL_OUTPUT)
    write_tsv(plan_summary, PLAN_SUMMARY_OUTPUT)

    input_paths = [
        Path(__file__).resolve(),
        *required,
    ]
    input_manifest = pd.DataFrame(
        [
            {
                "file_path": str(path.relative_to(PROJECT)),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(
                {candidate.resolve() for candidate in input_paths},
                key=lambda value: value.as_posix(),
            )
        ]
    )
    write_tsv(input_manifest, INPUT_MANIFEST_OUTPUT)

    output_paths = [
        PROTOCOL_OUTPUT,
        SOURCE_REGIME_OUTPUT,
        TARGET_PANEL_OUTPUT,
        CONFIGURATION_OUTPUT,
        EXPERIMENT_CELL_OUTPUT,
        INPUT_MANIFEST_OUTPUT,
        PLAN_SUMMARY_OUTPUT,
    ]
    write_manifest(output_paths, OUTPUT_MANIFEST)
    verify_manifest(OUTPUT_MANIFEST)

    freeze_paths = [
        Path(__file__).resolve(),
        SCRIPT175_PATH,
        SCRIPT173_FREEZE,
        *verified_architecture,
        OUTPUT_MANIFEST,
        *output_paths,
    ]
    write_manifest(freeze_paths, FREEZE_OUTPUT)
    verify_manifest(FREEZE_OUTPUT)

    print("===== SCRIPT 174 FINAL TRANSFER PREREGISTRATION =====")
    print(
        source_registry[
            [
                "outer_target_code",
                "source_regime_id",
                "source_species_codes",
                "source_union_drug_count",
                "target_drug_count",
                "corrective_genome_variant",
                "drug_representation",
                "drug_view_fusion_method",
                "cross_modal_architecture",
            ]
        ].to_string(index=False)
    )
    print()
    print("Frozen outer configurations:", len(selected))
    print("Source regimes:", len(source_registry))
    print("Target protocols:", len(TARGET_PROTOCOLS))
    print("Support budgets: 0|1|5|10")
    print("Broad experiment cells:", len(experiment_cells))
    print("Split seed:", RANDOM_SEED)
    print("Model seeds:", "|".join(str(seed) for seed in MODEL_SEEDS))
    print("Script 175 frozen SHA:", EXPECTED_SCRIPT175_SHA256)
    print("Models trained: NO")
    print()
    print("STATUS: SCRIPT 174 FINAL TRANSFER EXPERIMENTS PREREGISTERED")


if __name__ == "__main__":
    main()
