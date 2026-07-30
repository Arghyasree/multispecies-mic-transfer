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
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT167_AGGREGATE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_runs_v2/"
      "aggregate_outputs_sha256.txt"
)

SCRIPT166_RUN_PLAN = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_v2/"
      "corrective_drug_representation_run_plan_v2.tsv"
)

SCRIPT166_CONFIGURATION_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_v2/"
      "corrective_drug_representation_configuration_registry_v2.tsv"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_v1"
)

RUN_PLAN_PATH = (
    OUTPUT_ROOT
    / "corrective_drug_view_fusion_run_plan_v1.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    OUTPUT_ROOT
    / "corrective_drug_view_fusion_configuration_registry_v1.tsv"
)

PROTOCOL_PATH = (
    OUTPUT_ROOT
    / "corrective_drug_view_fusion_protocol_v1.tsv"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script170_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script170_successful_corrective_drug_view_fusion_preregistration_core_sha256.txt"
)

PLAN_SUMMARY_PATH = (
    TABLE_ROOT
    / "corrective_drug_view_fusion_plan_summary_v1.tsv"
)

MULTIVIEW_REPRESENTATIONS = (
    "ChemBERTa_mean_plus_Morgan",
    "ChemBERTa_mean_plus_Morgan_plus_RDKit",
)

NEW_FUSION_METHODS = (
    "separate_encoder_projected",
    "separate_encoder_low_rank",
)

CONFIRMATION_SEEDS = (
    20260811,
    20260812,
    20260813,
)

# Fresh, fixed capacity for this corrective study.
# It is not imported as a prior winner and is not tuned on the same results.
DRUG_VIEW_LOW_RANK = 16

EXPECTED_OUTERS = {"ec", "kp", "se"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(str(value).strip())
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
            raise RuntimeError(
                f"Malformed SHA line {line_number}: {path}"
            )

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


def write_manifest(paths: Iterable[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(
        {candidate.resolve() for candidate in paths},
        key=lambda candidate: candidate.as_posix(),
    )

    with path.open("w", encoding="utf-8") as handle:
        for candidate in unique:
            try:
                display = candidate.relative_to(PROJECT)
            except ValueError:
                display = candidate
            handle.write(f"{sha256_file(candidate)}  {display}\n")


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


def slug(value: object) -> str:
    return (
        str(value)
        .strip()
        .replace("+", "_plus_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def main() -> None:
    required = [
        SCRIPT167_AGGREGATE_MANIFEST,
        SCRIPT166_RUN_PLAN,
        SCRIPT166_CONFIGURATION_REGISTRY,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified167 = verify_manifest(
        SCRIPT167_AGGREGATE_MANIFEST
    )

    old_plan = read_tsv(
        SCRIPT166_RUN_PLAN
    )

    old_registry = read_tsv(
        SCRIPT166_CONFIGURATION_REGISTRY
    )

    required_columns = {
        "outer_target_code",
        "drug_representation",
        "source_species_code",
        "evaluation_species_code",
        "seed",
        "run_id",
        "configuration_id",
        "genome_representation",
        "corrective_genome_variant",
        "shared_hp_id",
        "cross_modal_architecture",
    }

    missing = sorted(
        required_columns.difference(old_plan.columns)
    )
    if missing:
        raise RuntimeError(
            f"Script 166 run plan missing columns: {missing}"
        )

    filtered = old_plan.loc[
        old_plan["drug_representation"].isin(
            MULTIVIEW_REPRESENTATIONS
        )
        & pd.to_numeric(
            old_plan["seed"],
            errors="raise",
        ).isin(CONFIRMATION_SEEDS)
    ].copy()

    expected_old_rows = (
        len(EXPECTED_OUTERS)
        * len(MULTIVIEW_REPRESENTATIONS)
        * 2
        * len(CONFIRMATION_SEEDS)
    )

    if len(filtered) != expected_old_rows:
        raise RuntimeError(
            "Expected exactly "
            f"{expected_old_rows} frozen multiview rows from Script 166; "
            f"observed {len(filtered)}."
        )

    if set(filtered["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            "Unexpected outer-target set in Script 166 multiview rows."
        )

    run_records: list[dict[str, object]] = []
    config_records: list[dict[str, object]] = []

    grouping = filtered.groupby(
        [
            "outer_target_code",
            "drug_representation",
        ],
        sort=True,
        dropna=False,
    )

    for (outer, representation), group in grouping:
        direction_seed_rows = group.sort_values(
            [
                "source_species_code",
                "evaluation_species_code",
                "seed",
            ]
        ).reset_index(drop=True)

        if len(direction_seed_rows) != 6:
            raise RuntimeError(
                f"Expected 6 rows for outer={outer}, "
                f"representation={representation}; "
                f"observed {len(direction_seed_rows)}."
            )

        reference = direction_seed_rows.iloc[0].to_dict()

        for fusion_method in NEW_FUSION_METHODS:
            configuration_id = (
                f"outer_{outer}__corrective_drug_view_fusion__"
                f"{slug(representation)}__{fusion_method}"
            )

            config = dict(reference)
            for column in [
                "run_id",
                "seed",
                "source_species_code",
                "evaluation_species_code",
            ]:
                config.pop(column, None)

            config.update(
                {
                    "configuration_id": configuration_id,
                    "outer_target_code": outer,
                    "drug_representation": representation,
                    "drug_view_fusion_method": fusion_method,
                    "drug_view_low_rank": (
                        DRUG_VIEW_LOW_RANK
                        if fusion_method
                        == "separate_encoder_low_rank"
                        else 0
                    ),
                    "cross_modal_architecture": (
                        "projected_concatenation_MLP"
                    ),
                    "selection_eligible": "YES",
                    "corrective_analysis_stage": (
                        "corrective_drug_view_fusion_screen_v1"
                    ),
                    "outer_target_labels_used": "NO",
                }
            )

            config_records.append(config)

            for old_row in direction_seed_rows.to_dict(
                orient="records"
            ):
                seed = int(float(old_row["seed"]))
                source = str(
                    old_row["source_species_code"]
                )
                evaluation = str(
                    old_row["evaluation_species_code"]
                )

                run = dict(old_row)
                run.update(
                    {
                        "configuration_id": configuration_id,
                        "run_id": (
                            f"{configuration_id}__"
                            f"{slug(source)}_to_{slug(evaluation)}__"
                            f"seed_{seed}"
                        ),
                        "drug_view_fusion_method": fusion_method,
                        "drug_view_low_rank": (
                            DRUG_VIEW_LOW_RANK
                            if fusion_method
                            == "separate_encoder_low_rank"
                            else 0
                        ),
                        "cross_modal_architecture": (
                            "projected_concatenation_MLP"
                        ),
                        "selection_eligible": "YES",
                        "corrective_analysis_stage": (
                            "corrective_drug_view_fusion_screen_v1"
                        ),
                        "outer_target_labels_used": "NO",
                    }
                )

                run_records.append(run)

    configuration_registry = (
        pd.DataFrame(config_records)
        .drop_duplicates("configuration_id")
        .sort_values(
            [
                "outer_target_code",
                "drug_representation",
                "drug_view_fusion_method",
            ]
        )
        .reset_index(drop=True)
    )

    run_plan = (
        pd.DataFrame(run_records)
        .sort_values(
            [
                "outer_target_code",
                "drug_representation",
                "drug_view_fusion_method",
                "source_species_code",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    if len(configuration_registry) != 12:
        raise RuntimeError(
            f"Expected 12 configurations; "
            f"observed {len(configuration_registry)}."
        )

    if len(run_plan) != 72:
        raise RuntimeError(
            f"Expected 72 new training fits; observed {len(run_plan)}."
        )

    if configuration_registry[
        "configuration_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate drug-view-fusion configuration IDs."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate drug-view-fusion run IDs."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_role",
                "value": (
                    "corrective extension of the drug-representation "
                    "screen to compare internal multiview fusion"
                ),
            },
            {
                "item": "existing_results_reused",
                "value": (
                    "all Script 167 single-view and raw single-encoder "
                    "concatenation results under the same three seeds"
                ),
            },
            {
                "item": "new_multiview_sets",
                "value": "|".join(MULTIVIEW_REPRESENTATIONS),
            },
            {
                "item": "new_fusion_methods",
                "value": "|".join(NEW_FUSION_METHODS),
            },
            {
                "item": "fixed_drug_view_low_rank",
                "value": DRUG_VIEW_LOW_RANK,
            },
            {
                "item": "rank_policy",
                "value": (
                    "fresh fixed rank chosen before new runs; "
                    "not imported as a prior winner and not tuned"
                ),
            },
            {
                "item": "fixed_cross_modal_architecture",
                "value": "projected_concatenation_MLP",
            },
            {
                "item": "primary_metric",
                "value": (
                    "bidirectional-average per-antibiotic macro RMSE"
                ),
            },
            {
                "item": "secondary_metric",
                "value": (
                    "seedwise worst-direction macro RMSE, mean and sample SD"
                ),
            },
            {
                "item": "selection_pool_after_completion",
                "value": (
                    "single views, raw multiview concatenation, separate-"
                    "encoder projected fusion, and separate-encoder "
                    "low-rank fusion"
                ),
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "no held-out outer-target MIC label is used"
                ),
            },
            {
                "item": "models_trained_by_script170",
                "value": "NO",
            },
        ]
    )

    plan_summary = (
        run_plan.groupby(
            [
                "outer_target_code",
                "drug_representation",
                "drug_view_fusion_method",
            ],
            as_index=False,
        )
        .agg(
            directions=(
                "source_species_code",
                "nunique",
            ),
            seeds=(
                "seed",
                "nunique",
            ),
            planned_runs=(
                "run_id",
                "size",
            ),
        )
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    write_tsv(
        configuration_registry,
        CONFIGURATION_REGISTRY_PATH,
    )
    write_tsv(
        run_plan,
        RUN_PLAN_PATH,
    )
    write_tsv(
        protocol,
        PROTOCOL_PATH,
    )
    write_tsv(
        plan_summary,
        PLAN_SUMMARY_PATH,
    )

    input_manifest_path = (
        OUTPUT_ROOT
        / "script170_input_manifest.tsv"
    )

    input_paths = [
        Path(__file__).resolve(),
        *required,
    ]

    write_tsv(
        pd.DataFrame(
            [
                {
                    "file_path": str(
                        path.relative_to(PROJECT)
                    ),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(
                    set(input_paths),
                    key=lambda value: value.as_posix(),
                )
            ]
        ),
        input_manifest_path,
    )

    output_paths = [
        CONFIGURATION_REGISTRY_PATH,
        RUN_PLAN_PATH,
        PROTOCOL_PATH,
        PLAN_SUMMARY_PATH,
        input_manifest_path,
    ]

    write_manifest(
        output_paths,
        OUTPUT_MANIFEST,
    )
    verify_manifest(
        OUTPUT_MANIFEST
    )

    freeze_paths = [
        Path(__file__).resolve(),
        OUTPUT_MANIFEST,
        *output_paths,
        SCRIPT167_AGGREGATE_MANIFEST,
        *verified167,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_MANIFEST,
    )
    verify_manifest(
        FREEZE_MANIFEST
    )

    print(
        "===== SCRIPT 170 CORRECTIVE DRUG-VIEW FUSION PREREGISTRATION ====="
    )
    print(
        plan_summary.to_string(index=False)
    )
    print()
    print(
        "Multiview sets:",
        len(MULTIVIEW_REPRESENTATIONS),
    )
    print(
        "New fusion methods:",
        len(NEW_FUSION_METHODS),
    )
    print(
        "Configurations:",
        len(configuration_registry),
    )
    print(
        "New training fits:",
        len(run_plan),
    )
    print(
        "Fixed low-rank:",
        DRUG_VIEW_LOW_RANK,
    )
    print(
        "Seeds:",
        "|".join(str(seed) for seed in CONFIRMATION_SEEDS),
    )
    print(
        "Outer-target MIC labels used: NO"
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 170 CORRECTIVE DRUG-VIEW "
        "FUSION SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
