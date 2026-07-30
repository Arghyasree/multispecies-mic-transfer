#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT155_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script155_successful_drug_selection_core_sha256.txt"
)

SCRIPT156_METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "architecture_screen_runs_v1"
)

SCRIPT156_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "architecture_screen_aggregate_v1"
)

SCRIPT156_AGGREGATE_MANIFEST = (
    SCRIPT156_METADATA_ROOT
    / "aggregate_outputs_sha256.txt"
)

SCRIPT156_CONFIG_SUMMARY = (
    SCRIPT156_AGGREGATE_ROOT
    / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

SCRIPT156_DIRECTION_SUMMARY = (
    SCRIPT156_AGGREGATE_ROOT
    / "direction_three_seed_mean_sd.tsv"
)

SCRIPT154_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "drug_representation_screen_aggregate_v1"
)

SCRIPT154_CONFIG_SUMMARY = (
    SCRIPT154_AGGREGATE_ROOT
    / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

SCRIPT154_DIRECTION_SUMMARY = (
    SCRIPT154_AGGREGATE_ROOT
    / "direction_three_seed_mean_sd.tsv"
)

SCRIPT152_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "genome_representation_screen_aggregate_v1"
)

SCRIPT152_CONFIG_SUMMARY = (
    SCRIPT152_AGGREGATE_ROOT
    / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

SCRIPT152_DIRECTION_SUMMARY = (
    SCRIPT152_AGGREGATE_ROOT
    / "direction_three_seed_mean_sd.tsv"
)

SELECTED_DRUG_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "architecture_screen_v1/"
      "nested_loso_selected_drug_representation_registry_v1.tsv"
)

SELECTED_GENOME_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "drug_representation_screen_v1/"
      "nested_loso_selected_genome_representation_registry_v1.tsv"
)

SELECTED_KMER_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_selected_kmer_registry_v1.tsv"
)

ARCHITECTURE_SELECTION_ALIAS_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "architecture_screen_v1/"
      "nested_loso_selected_architecture_registry_v1.tsv"
)

FUSED_MATRIX_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_selected_kmer_plus_common_amr_matrix_registry_v1.tsv"
)

COMMON_AMR_MATRIX_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "common_cross_species_amr_matrix_v1/"
      "nested_loso_common_cross_species_amr_matrix_registry_v1.tsv"
)

FULL_KMER_RUN_PLAN = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "full_kmer_grid_v1/"
      "nested_loso_full_kmer_run_plan_v1.tsv"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "multiview_sensitivity_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "multiview_sensitivity_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script157_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script157_successful_architecture_selection_core_sha256.txt"
)

EXPECTED_OUTERS = {"ec", "kp", "se"}
EXPECTED_SCRIPT156_RUNS = 72
PRACTICAL_RMSE_TIE_THRESHOLD = 0.002

ARCHITECTURE_NAME_TO_ID = {
    "projected_concatenation_MLP": "cross_modal_projected_concat",
    "dual_tower_interaction": "dual_tower_explicit_interaction",
    "cross_modal_GMU": "cross_modal_gmu",
    "low_rank_bilinear": "cross_modal_low_rank_bilinear",
    "drug_to_genome_FiLM": "drug_to_genome_film",
}

SENSITIVITY_VARIANTS = (
    "selected_kmer_plus_common_AMR_concat",
    "selected_kmer_plus_common_AMR_low_rank_fusion",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
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
            raise RuntimeError(
                f"Malformed SHA line {line_number}: {path}"
            )

        expected, value = parts
        candidate = project_path(value)

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        observed = sha256_file(candidate)

        if observed != expected:
            raise RuntimeError(
                f"SHA mismatch: {candidate}"
            )

        verified.append(candidate)

    if not verified:
        raise RuntimeError(
            f"Empty SHA manifest: {path}"
        )

    return verified


def write_manifest(
    paths: Iterable[Path],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    unique = sorted(
        set(paths),
        key=lambda candidate: candidate.as_posix(),
    )

    with path.open("w", encoding="utf-8") as handle:
        for candidate in unique:
            handle.write(
                f"{sha256_file(candidate)}  "
                f"{candidate.relative_to(PROJECT)}\n"
            )


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


def write_tsv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def verify_script156_runs() -> list[Path]:
    flags = sorted(
        SCRIPT156_METADATA_ROOT.glob("*/RUN_COMPLETE")
    )

    if len(flags) != EXPECTED_SCRIPT156_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_SCRIPT156_RUNS} Script 156 "
            f"completion markers; observed {len(flags)}."
        )

    verified: list[Path] = []

    for flag in flags:
        if flag.read_text(
            encoding="utf-8"
        ).strip() != "0":
            raise RuntimeError(
                f"Nonzero RUN_COMPLETE: {flag}"
            )

        manifest = flag.parent / "outputs_sha256.txt"
        verified.extend(
            verify_manifest(manifest)
        )
        verified.extend(
            [flag, manifest]
        )

    return verified


def numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in result.columns:
        if (
            column.endswith("_mean")
            or column.endswith("_sd")
            or column in {
                "seed_count",
                "parameter_count",
                "provisional_rank",
            }
        ):
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def selected_lookup(
    frame: pd.DataFrame,
    candidate_column: str,
) -> dict[str, str]:
    if set(frame["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            f"Unexpected outer targets in {candidate_column} registry."
        )

    return dict(
        zip(
            frame["outer_target_code"].astype(str),
            frame[candidate_column].astype(str),
        )
    )


def projected_baseline_rows(
    selected_drug: pd.DataFrame,
    selected_genome: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_lookup = selected_lookup(
        selected_drug,
        "drug_representation",
    )
    genome_lookup = selected_lookup(
        selected_genome,
        "candidate_id",
    )

    config154 = numeric_columns(
        read_tsv(SCRIPT154_CONFIG_SUMMARY)
    )
    direction154 = numeric_columns(
        read_tsv(SCRIPT154_DIRECTION_SUMMARY)
    )
    config152 = numeric_columns(
        read_tsv(SCRIPT152_CONFIG_SUMMARY)
    )
    direction152 = numeric_columns(
        read_tsv(SCRIPT152_DIRECTION_SUMMARY)
    )

    config_rows: list[pd.DataFrame] = []
    direction_rows: list[pd.DataFrame] = []

    for outer in sorted(EXPECTED_OUTERS):
        drug = drug_lookup[outer]
        genome = genome_lookup[outer]

        if drug == "Morgan":
            config_source = config152
            direction_source = direction152
        else:
            config_source = config154
            direction_source = direction154

        config_mask = (
            config_source["outer_target_code"].eq(outer)
            & config_source["drug_representation"].eq(drug)
            & config_source["cross_modal_architecture"].eq(
                "projected_concatenation_MLP"
            )
        )

        if "candidate_id" in config_source.columns:
            config_mask &= config_source[
                "candidate_id"
            ].eq(genome)
        elif "genome_representation" in config_source.columns:
            acceptable = {
                genome,
                "selected_genome_representation",
            }
            config_mask &= config_source[
                "genome_representation"
            ].isin(acceptable)

        config_match = config_source.loc[
            config_mask
        ].copy()

        if len(config_match) != 1:
            raise RuntimeError(
                f"Expected one projected baseline for outer {outer}; "
                f"observed {len(config_match)}."
            )

        direction_mask = (
            direction_source["outer_target_code"].eq(outer)
            & direction_source["drug_representation"].eq(drug)
            & direction_source["cross_modal_architecture"].eq(
                "projected_concatenation_MLP"
            )
        )

        if "candidate_id" in direction_source.columns:
            direction_mask &= direction_source[
                "candidate_id"
            ].eq(genome)
        elif "genome_representation" in direction_source.columns:
            acceptable = {
                genome,
                "selected_genome_representation",
            }
            direction_mask &= direction_source[
                "genome_representation"
            ].isin(acceptable)

        direction_match = direction_source.loc[
            direction_mask
        ].copy()

        if len(direction_match) != 2:
            raise RuntimeError(
                f"Expected two projected directions for outer {outer}; "
                f"observed {len(direction_match)}."
            )

        config_match["selected_genome_candidate"] = genome
        config_match["metric_source_stage"] = (
            "script152_reused_projected"
            if drug == "Morgan"
            else "script154_reused_projected"
        )

        direction_match[
            "selected_genome_candidate"
        ] = genome

        config_rows.append(config_match)
        direction_rows.append(direction_match)

    return (
        pd.concat(config_rows, ignore_index=True),
        pd.concat(direction_rows, ignore_index=True),
    )


def architecture_candidates(
    selected_drug: pd.DataFrame,
    selected_genome: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config156 = numeric_columns(
        read_tsv(SCRIPT156_CONFIG_SUMMARY)
    )
    direction156 = numeric_columns(
        read_tsv(SCRIPT156_DIRECTION_SUMMARY)
    )

    if len(config156) != 12:
        raise RuntimeError(
            f"Expected 12 Script 156 configurations; "
            f"observed {len(config156)}."
        )

    projected_config, projected_direction = (
        projected_baseline_rows(
            selected_drug,
            selected_genome,
        )
    )

    selected_genome_lookup = selected_lookup(
        selected_genome,
        "candidate_id",
    )

    config156["selected_genome_candidate"] = (
        config156["outer_target_code"].map(
            selected_genome_lookup
        )
    )
    config156["metric_source_stage"] = "script156"

    combined_config = pd.concat(
        [
            config156,
            projected_config,
        ],
        ignore_index=True,
        sort=False,
    )

    combined_direction = pd.concat(
        [
            direction156,
            projected_direction,
        ],
        ignore_index=True,
        sort=False,
    )

    required_architectures = set(
        ARCHITECTURE_NAME_TO_ID
    )

    for outer in sorted(EXPECTED_OUTERS):
        observed = set(
            combined_config.loc[
                combined_config[
                    "outer_target_code"
                ].eq(outer),
                "cross_modal_architecture",
            ].astype(str)
        )

        if observed != required_architectures:
            raise RuntimeError(
                f"Outer {outer}: architecture set mismatch. "
                f"Observed {sorted(observed)}."
            )

    direction_key = [
        "outer_target_code",
        "drug_representation",
        "cross_modal_architecture",
    ]

    worst_direction = (
        combined_direction.groupby(
            direction_key,
            dropna=False,
        )["macro_rmse_mean"]
        .max()
        .rename("worst_direction_macro_rmse_mean")
        .reset_index()
    )

    combined_config = combined_config.merge(
        worst_direction,
        on=direction_key,
        how="left",
        validate="one_to_one",
    )

    combined_config[
        "architecture_id"
    ] = combined_config[
        "cross_modal_architecture"
    ].map(ARCHITECTURE_NAME_TO_ID)

    if combined_config["architecture_id"].isna().any():
        raise RuntimeError(
            "Unmapped architecture name."
        )

    combined_config[
        "selection_metric"
    ] = (
        "three_seed_mean_bidirectional_macro_rmse"
    )
    combined_config[
        "practical_rmse_tie_threshold"
    ] = PRACTICAL_RMSE_TIE_THRESHOLD
    combined_config[
        "outer_target_labels_used"
    ] = "NO"

    selected_records: list[pd.Series] = []
    ranked_groups: list[pd.DataFrame] = []

    for outer, group in combined_config.groupby(
        "outer_target_code",
        sort=True,
    ):
        group = group.copy()

        best_rmse = float(
            group[
                "bidirectional_macro_rmse_mean"
            ].min()
        )

        group[
            "rmse_difference_from_best"
        ] = (
            group[
                "bidirectional_macro_rmse_mean"
            ]
            - best_rmse
        )

        group[
            "within_practical_rmse_tie"
        ] = group[
            "rmse_difference_from_best"
        ].le(PRACTICAL_RMSE_TIE_THRESHOLD)

        eligible = group.loc[
            group[
                "within_practical_rmse_tie"
            ]
        ].copy()

        if len(eligible) != 1:
            details = eligible[
                [
                    "cross_modal_architecture",
                    "bidirectional_macro_rmse_mean",
                    "bidirectional_macro_rmse_sd",
                    "worst_direction_macro_rmse_mean",
                ]
            ].to_dict(orient="records")

            raise RuntimeError(
                f"Outer {outer}: {len(eligible)} architectures are "
                f"within the preregistered 0.002 RMSE tie threshold. "
                "A within-species guardrail is required before selection. "
                f"Candidates: {details}"
            )

        selected_architecture = str(
            eligible.iloc[0][
                "cross_modal_architecture"
            ]
        )

        group[
            "selected_architecture"
        ] = group[
            "cross_modal_architecture"
        ].eq(selected_architecture)

        group = group.sort_values(
            [
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
                "parameter_count",
                "cross_modal_architecture",
            ]
        ).reset_index(drop=True)

        group[
            "selection_rank"
        ] = np.arange(
            1,
            len(group) + 1,
            dtype=int,
        )

        ranked_groups.append(group)

        selected_records.append(
            group.loc[
                group[
                    "selected_architecture"
                ]
            ].iloc[0]
        )

    ranking = pd.concat(
        ranked_groups,
        ignore_index=True,
    )

    selected = pd.DataFrame(
        selected_records
    ).reset_index(drop=True)

    if len(selected) != 3:
        raise RuntimeError(
            "Expected three selected architecture rows."
        )

    return ranking, selected


def prepare_sensitivity_plan(
    selected_architecture: pd.DataFrame,
    selected_kmer: pd.DataFrame,
    fused_registry: pd.DataFrame,
    common_amr_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_plan = read_tsv(FULL_KMER_RUN_PLAN)

    kmer_lookup = selected_lookup(
        selected_kmer,
        "genome_representation",
    )

    selected_drug_lookup = dict(
        zip(
            selected_architecture[
                "outer_target_code"
            ].astype(str),
            selected_architecture[
                "drug_representation"
            ].astype(str),
        )
    )

    selected_architecture_lookup = dict(
        zip(
            selected_architecture[
                "outer_target_code"
            ].astype(str),
            selected_architecture[
                "cross_modal_architecture"
            ].astype(str),
        )
    )

    fused_lookup = (
        fused_registry.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )

    amr_lookup = (
        common_amr_registry.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )

    rows: list[pd.DataFrame] = []
    registry_records: list[dict[str, object]] = []

    for outer in sorted(EXPECTED_OUTERS):
        architecture = (
            selected_architecture_lookup[outer]
        )
        drug = selected_drug_lookup[outer]
        selected_kmer_id = kmer_lookup[outer]

        template = full_plan.loc[
            full_plan[
                "outer_target_code"
            ].eq(outer)
            & full_plan[
                "genome_representation"
            ].eq("canonical_4mer")
            & full_plan[
                "drug_representation"
            ].eq("Morgan")
            & full_plan[
                "cross_modal_architecture"
            ].eq(architecture)
        ].copy()

        if len(template) != 6:
            raise RuntimeError(
                f"Outer {outer}: expected six run-template rows "
                f"for architecture {architecture}; "
                f"observed {len(template)}."
            )

        fused_row = fused_lookup[outer]
        amr_row = amr_lookup[outer]

        fused_matrix_path = str(
            fused_row["fused_matrix_path"]
        )

        kmer_dimension = int(
            float(
                fused_row[
                    "selected_kmer_dimension"
                ]
            )
        )

        amr_dimension = int(
            float(
                fused_row[
                    "common_amr_dimension"
                ]
            )
        )

        fused_dimension = int(
            float(
                fused_row[
                    "fused_dimension"
                ]
            )
        )

        common_amr_matrix_path = str(
            amr_row["matrix_path"]
        )

        for variant in SENSITIVITY_VARIANTS:
            variant_plan = template.copy()

            representation_id = (
                f"outer_{outer}__{variant}"
            )

            variant_plan[
                "genome_representation"
            ] = representation_id

            variant_plan[
                "drug_representation"
            ] = drug

            variant_plan[
                "configuration_id"
            ] = (
                f"outer_{outer}__{variant}__"
                f"{drug}__{architecture}"
            )

            variant_plan[
                "run_id"
            ] = (
                variant_plan[
                    "configuration_id"
                ].astype(str)
                + "__"
                + variant_plan[
                    "source_species_code"
                ].astype(str)
                + "_to_"
                + variant_plan[
                    "evaluation_species_code"
                ].astype(str)
                + "__seed_"
                + variant_plan[
                    "seed"
                ].astype(str)
            )

            variant_plan[
                "selected_kmer_representation"
            ] = selected_kmer_id
            variant_plan[
                "selected_kmer_dimension"
            ] = kmer_dimension
            variant_plan[
                "common_amr_dimension"
            ] = amr_dimension
            variant_plan[
                "fused_dimension"
            ] = fused_dimension
            variant_plan[
                "fused_matrix_path"
            ] = fused_matrix_path
            variant_plan[
                "common_amr_matrix_path"
            ] = common_amr_matrix_path
            variant_plan[
                "within_genome_fusion_variant"
            ] = variant

            rows.append(variant_plan)

            registry_records.append(
                {
                    "outer_target_code": outer,
                    "configuration_id": (
                        f"outer_{outer}__{variant}__"
                        f"{drug}__{architecture}"
                    ),
                    "genome_representation": (
                        representation_id
                    ),
                    "within_genome_fusion_variant": (
                        variant
                    ),
                    "selected_kmer_representation": (
                        selected_kmer_id
                    ),
                    "selected_kmer_dimension": (
                        kmer_dimension
                    ),
                    "common_amr_dimension": (
                        amr_dimension
                    ),
                    "fused_dimension": (
                        fused_dimension
                    ),
                    "fused_matrix_path": (
                        fused_matrix_path
                    ),
                    "common_amr_matrix_path": (
                        common_amr_matrix_path
                    ),
                    "drug_representation": drug,
                    "cross_modal_architecture": (
                        architecture
                    ),
                    "new_training_fits": 6,
                    "outer_target_labels_used": "NO",
                }
            )

    run_plan = pd.concat(
        rows,
        ignore_index=True,
    ).sort_values(
        [
            "outer_target_code",
            "within_genome_fusion_variant",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    if len(run_plan) != 36:
        raise RuntimeError(
            f"Expected 36 sensitivity runs; "
            f"observed {len(run_plan)}."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sensitivity run IDs."
        )

    configuration_registry = pd.DataFrame(
        registry_records
    ).sort_values(
        [
            "outer_target_code",
            "within_genome_fusion_variant",
        ]
    ).reset_index(drop=True)

    if len(configuration_registry) != 6:
        raise RuntimeError(
            "Expected six sensitivity configurations."
        )

    baseline_registry = selected_architecture[
        [
            "outer_target_code",
            "selected_genome_candidate",
            "drug_representation",
            "cross_modal_architecture",
            "architecture_id",
            "seed_count",
            "parameter_count",
            "bidirectional_macro_rmse_mean",
            "bidirectional_macro_rmse_sd",
        ]
    ].copy()

    baseline_registry[
        "baseline_genome_representation"
    ] = "common_cross_species_AMR"

    baseline_registry[
        "training_policy"
    ] = (
        "reuse selected architecture-screen result; no retraining"
    )

    baseline_registry[
        "outer_target_labels_used"
    ] = "NO"

    return (
        run_plan,
        configuration_registry,
        baseline_registry,
    )


def main() -> None:
    required = [
        SCRIPT155_FREEZE,
        SCRIPT156_AGGREGATE_MANIFEST,
        SCRIPT156_CONFIG_SUMMARY,
        SCRIPT156_DIRECTION_SUMMARY,
        SCRIPT154_CONFIG_SUMMARY,
        SCRIPT154_DIRECTION_SUMMARY,
        SCRIPT152_CONFIG_SUMMARY,
        SCRIPT152_DIRECTION_SUMMARY,
        SELECTED_DRUG_REGISTRY,
        SELECTED_GENOME_REGISTRY,
        SELECTED_KMER_REGISTRY,
        FUSED_MATRIX_REGISTRY,
        COMMON_AMR_MATRIX_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified155 = verify_manifest(
        SCRIPT155_FREEZE
    )
    verified156_runs = verify_script156_runs()
    verified156_aggregate = verify_manifest(
        SCRIPT156_AGGREGATE_MANIFEST
    )

    selected_drug = read_tsv(
        SELECTED_DRUG_REGISTRY
    )
    selected_genome = read_tsv(
        SELECTED_GENOME_REGISTRY
    )
    selected_kmer = read_tsv(
        SELECTED_KMER_REGISTRY
    )
    fused_registry = read_tsv(
        FUSED_MATRIX_REGISTRY
    )
    common_amr_registry = read_tsv(
        COMMON_AMR_MATRIX_REGISTRY
    )

    ranking, selected_architecture = (
        architecture_candidates(
            selected_drug,
            selected_genome,
        )
    )

    (
        run_plan,
        configuration_registry,
        baseline_registry,
    ) = prepare_sensitivity_plan(
        selected_architecture,
        selected_kmer,
        fused_registry,
        common_amr_registry,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking_path = (
        TABLE_ROOT
        / "nested_loso_architecture_complete_ranking_v1.tsv"
    )
    selected_path = (
        OUTPUT_ROOT
        / "nested_loso_selected_architecture_registry_v1.tsv"
    )
    run_plan_path = (
        OUTPUT_ROOT
        / "nested_loso_multiview_sensitivity_run_plan_v1.tsv"
    )
    configuration_path = (
        OUTPUT_ROOT
        / "nested_loso_multiview_sensitivity_configuration_registry_v1.tsv"
    )
    baseline_path = (
        OUTPUT_ROOT
        / "nested_loso_multiview_sensitivity_amr_baseline_registry_v1.tsv"
    )
    protocol_path = (
        OUTPUT_ROOT
        / "nested_loso_multiview_sensitivity_protocol_v1.tsv"
    )
    input_manifest_path = (
        OUTPUT_ROOT
        / "script157_input_manifest.tsv"
    )

    protocol = pd.DataFrame(
        [
            {
                "item": "objective",
                "value": (
                    "test whether selected-kmer plus AMR lost because "
                    "whole-genome k-mer information is not transferable "
                    "or because raw concatenation was an inadequate "
                    "within-genome fusion mechanism"
                ),
            },
            {
                "item": "architecture_selection_metric",
                "value": (
                    "three-seed mean bidirectional per-antibiotic "
                    "macro RMSE"
                ),
            },
            {
                "item": "architecture_practical_tie_threshold",
                "value": PRACTICAL_RMSE_TIE_THRESHOLD,
            },
            {
                "item": "architecture_tie_policy",
                "value": (
                    "halt and require the preregistered within-species "
                    "guardrail if more than one candidate lies within "
                    "0.002 RMSE"
                ),
            },
            {
                "item": "amr_baseline",
                "value": (
                    "reuse AMR-only result under the selected drug "
                    "representation and selected cross-modal architecture"
                ),
            },
            {
                "item": "new_variant_1",
                "value": (
                    "selected k-mer plus common AMR raw feature "
                    "concatenation passed through one genome encoder"
                ),
            },
            {
                "item": "new_variant_2",
                "value": (
                    "separate selected-kmer and AMR encoders with a "
                    "projected-concatenation base plus low-rank bilinear "
                    "within-genome residual"
                ),
            },
            {
                "item": "new_training_fits",
                "value": 36,
            },
            {
                "item": "seeds",
                "value": "20260726|20260727|20260728",
            },
            {
                "item": "target_label_policy",
                "value": (
                    "no held-out outer-target MIC label used"
                ),
            },
            {
                "item": "models_trained_by_script157",
                "value": "NO",
            },
        ]
    )

    write_tsv(
        ranking,
        ranking_path,
    )
    write_tsv(
        selected_architecture,
        selected_path,
    )
    write_tsv(
        selected_architecture,
        ARCHITECTURE_SELECTION_ALIAS_PATH,
    )
    write_tsv(
        run_plan,
        run_plan_path,
    )
    write_tsv(
        configuration_registry,
        configuration_path,
    )
    write_tsv(
        baseline_registry,
        baseline_path,
    )
    write_tsv(
        protocol,
        protocol_path,
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
                    "file_size_bytes": (
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(path),
                }
                for path in sorted(
                    set(input_paths),
                    key=lambda candidate:
                        candidate.as_posix(),
                )
            ]
        ),
        input_manifest_path,
    )

    output_paths = [
        ranking_path,
        selected_path,
        ARCHITECTURE_SELECTION_ALIAS_PATH,
        run_plan_path,
        configuration_path,
        baseline_path,
        protocol_path,
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
        SCRIPT155_FREEZE,
        SCRIPT156_AGGREGATE_MANIFEST,
        *verified156_aggregate,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_MANIFEST,
    )
    verify_manifest(
        FREEZE_MANIFEST
    )

    print(
        "===== SCRIPT 157 ARCHITECTURE SELECTION ====="
    )
    print(
        selected_architecture[
            [
                "outer_target_code",
                "selected_genome_candidate",
                "drug_representation",
                "cross_modal_architecture",
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
                "worst_direction_macro_rmse_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Verified Script 155 frozen files:",
        len(verified155),
    )
    print(
        "Verified Script 156 runs:",
        EXPECTED_SCRIPT156_RUNS,
    )
    print(
        "Verified Script 156 run files:",
        len(verified156_runs),
    )
    print(
        "Architecture candidates:",
        len(ranking),
    )
    print(
        "Selected architectures:",
        len(selected_architecture),
    )
    print(
        "Sensitivity configurations:",
        len(configuration_registry),
    )
    print(
        "Sensitivity new training fits:",
        len(run_plan),
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 157 ARCHITECTURES SELECTED "
        "AND MULTIVIEW SENSITIVITY PREPARED"
    )


if __name__ == "__main__":
    main()
