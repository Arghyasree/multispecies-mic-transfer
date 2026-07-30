#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT171_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script171_successful_corrective_drug_view_fusion_selection_core_sha256.txt"
)

DRUG_FUSION_AGGREGATE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_runs_v1/"
      "aggregate_outputs_sha256.txt"
)

SELECTED_DRUG_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_aggregate_v1/"
      "selected_final_corrective_drug_representation_registry_v1.tsv"
)

DRUG_RANKING_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_aggregate_v1/"
      "complete_corrective_drug_representation_and_fusion_ranking_v1.tsv"
)

FINAL_CONFIGURATION_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_v1/"
      "corrective_final_genome_configuration_registry_v1.tsv"
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
      "corrective_architecture_screen_v2"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_v2"
)

NEW_CONFIGURATION_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_architecture_new_configuration_registry_v2.tsv"
)

ALL_CANDIDATE_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_architecture_all_candidate_registry_v2.tsv"
)

RUN_PLAN_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_architecture_new_run_plan_v2.tsv"
)

SELECTED_INPUT_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_selected_genome_drug_registry_v2.tsv"
)

PROTOCOL_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_architecture_screen_protocol_v2.tsv"
)

INPUT_MANIFEST_OUTPUT = (
    OUTPUT_ROOT
    / "script172_input_manifest.tsv"
)

PLAN_SUMMARY_OUTPUT = (
    RESULT_ROOT
    / "corrective_architecture_screen_plan_summary_v2.tsv"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script172_outputs_sha256.txt"
)

FREEZE_OUTPUT = (
    PROJECT
    / "metadata/config_selection/"
      "script172_successful_corrective_architecture_preregistration_core_sha256.txt"
)

SCREEN_SEEDS = (
    20260811,
    20260812,
    20260813,
)

ALL_ARCHITECTURES = (
    "additive_linear",
    "projected_concatenation_MLP",
    "dual_tower_interaction",
    "cross_modal_GMU",
    "low_rank_bilinear",
    "drug_to_genome_FiLM",
)

NEW_ARCHITECTURES = (
    "additive_linear",
    "dual_tower_interaction",
    "cross_modal_GMU",
    "low_rank_bilinear",
    "drug_to_genome_FiLM",
)

TEMPLATE_ARCHITECTURES = (
    "projected_concatenation_MLP",
    "dual_tower_interaction",
    "cross_modal_GMU",
    "low_rank_bilinear",
    "drug_to_genome_FiLM",
)

EXPECTED_OUTERS = {
    "ec",
    "kp",
    "se",
}

CROSS_MODAL_BILINEAR_RANK = 32


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


def write_manifest(
    paths: Iterable[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for path in sorted(
            {candidate.resolve() for candidate in paths},
            key=lambda value: value.as_posix(),
        ):
            display = path.relative_to(PROJECT)
            handle.write(
                f"{sha256_file(path)}  {display}\n"
            )


def first_existing(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise RuntimeError(
        "None of the required columns exists: "
        + "|".join(candidates)
    )


def slug(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(value),
    ).strip("_")


def main() -> None:
    required = [
        Path(__file__).resolve(),
        SCRIPT171_FREEZE,
        DRUG_FUSION_AGGREGATE_MANIFEST,
        SELECTED_DRUG_PATH,
        DRUG_RANKING_PATH,
        FINAL_CONFIGURATION_PATH,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified171 = verify_manifest(
        SCRIPT171_FREEZE
    )
    verified_drug_fusion_aggregate = verify_manifest(
        DRUG_FUSION_AGGREGATE_MANIFEST
    )

    selected_drug = read_tsv(
        SELECTED_DRUG_PATH
    )
    drug_ranking = read_tsv(
        DRUG_RANKING_PATH
    )
    final_configurations = read_tsv(
        FINAL_CONFIGURATION_PATH
    )
    full_plan = read_tsv(
        FULL_KMER_RUN_PLAN
    )

    if len(selected_drug) != 3:
        raise RuntimeError(
            "Expected three selected drug representations."
        )

    if set(
        selected_drug["outer_target_code"]
    ) != EXPECTED_OUTERS:
        raise RuntimeError(
            "Selected drug registry has unexpected outer targets."
        )

    supported_selected_fusions = {
        "single_view",
        "raw_single_encoder_concatenation",
    }

    observed_selected_fusions = set(
        selected_drug[
            "drug_view_fusion_method"
        ].astype(str)
    )

    if not observed_selected_fusions.issubset(
        supported_selected_fusions
    ):
        raise RuntimeError(
            "The selected final drug configuration requires an unsupported "
            "within-drug fusion implementation: "
            + "|".join(
                sorted(
                    observed_selected_fusions
                    - supported_selected_fusions
                )
            )
        )

    selected_records: list[dict[str, Any]] = []

    for row in selected_drug.to_dict(
        orient="records"
    ):
        outer = str(
            row["outer_target_code"]
        )
        variant = str(
            row["corrective_genome_variant"]
        )

        matches = final_configurations.loc[
            final_configurations[
                "outer_target_code"
            ].astype(str).eq(outer)
            & final_configurations[
                "corrective_genome_variant"
            ].astype(str).eq(variant)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not resolve exactly one frozen genome configuration "
                f"for outer={outer}, variant={variant}; observed "
                f"{len(matches)}."
            )

        merged = matches.iloc[0].to_dict()
        merged.update(row)
        selected_records.append(merged)

    selected_inputs = pd.DataFrame(
        selected_records
    ).sort_values(
        "outer_target_code"
    ).reset_index(drop=True)

    outer_column = first_existing(
        full_plan,
        ["outer_target_code"],
    )
    source_column = first_existing(
        full_plan,
        ["source_species_code"],
    )
    evaluation_column = first_existing(
        full_plan,
        ["evaluation_species_code"],
    )
    seed_column = first_existing(
        full_plan,
        ["seed"],
    )
    genome_column = first_existing(
        full_plan,
        ["genome_representation"],
    )
    drug_column = first_existing(
        full_plan,
        ["drug_representation"],
    )
    architecture_column = first_existing(
        full_plan,
        ["cross_modal_architecture"],
    )

    template = (
        full_plan.loc[
            full_plan[
                genome_column
            ].astype(str).eq(
                "canonical_4mer"
            )
            & full_plan[
                architecture_column
            ].astype(str).isin(
                TEMPLATE_ARCHITECTURES
            )
        ]
        .sort_values(
            [
                outer_column,
                drug_column,
                architecture_column,
                source_column,
                evaluation_column,
                seed_column,
            ]
        )
        .drop_duplicates(
            [
                outer_column,
                drug_column,
                architecture_column,
                source_column,
                evaluation_column,
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    run_records: list[dict[str, Any]] = []
    new_configuration_records: list[
        dict[str, Any]
    ] = []
    all_candidate_records: list[
        dict[str, Any]
    ] = []

    selected_lookup = {
        str(row["outer_target_code"]): row
        for row in selected_inputs.to_dict(
            orient="records"
        )
    }

    for outer in sorted(
        EXPECTED_OUTERS
    ):
        selected = selected_lookup[outer]
        drug = str(
            selected[
                "drug_representation"
            ]
        )
        genome_representation = str(
            selected[
                "genome_representation"
            ]
        )
        genome_variant = str(
            selected[
                "corrective_genome_variant"
            ]
        )

        baseline_matches = drug_ranking.loc[
            drug_ranking[
                "outer_target_code"
            ].astype(str).eq(outer)
            & drug_ranking[
                "drug_representation"
            ].astype(str).eq(drug)
            & drug_ranking[
                "drug_view_fusion_method"
            ].astype(str).eq(
                str(
                    selected[
                        "drug_view_fusion_method"
                    ]
                )
            )
        ]

        if len(baseline_matches) != 1:
            raise RuntimeError(
                "Could not resolve exactly one projected-concatenation "
                f"baseline for outer={outer}, drug={drug}; observed "
                f"{len(baseline_matches)}."
            )

        baseline = baseline_matches.iloc[0].to_dict()
        baseline_record = {
            **selected,
            **baseline,
            "configuration_id": (
                f"outer_{outer}__corrective_architecture_screen__"
                f"{slug(genome_variant)}__{slug(drug)}__"
                f"{slug(selected['drug_view_fusion_method'])}__"
                "projected_concatenation_MLP"
            ),
            "cross_modal_architecture": (
                "projected_concatenation_MLP"
            ),
            "result_source": (
                "reused_script171_selected_drug_projected_baseline"
            ),
            "new_training_required": "NO",
            "screen_seeds": "|".join(
                str(seed)
                for seed in SCREEN_SEEDS
            ),
            "cross_modal_bilinear_rank": (
                CROSS_MODAL_BILINEAR_RANK
            ),
        }
        all_candidate_records.append(
            baseline_record
        )

        for architecture in NEW_ARCHITECTURES:
            template_architecture = (
                "projected_concatenation_MLP"
                if architecture == "additive_linear"
                else architecture
            )

            rows = template.loc[
                template[
                    outer_column
                ].astype(str).eq(outer)
                & template[
                    drug_column
                ].astype(str).eq(drug)
                & template[
                    architecture_column
                ].astype(str).eq(
                    template_architecture
                )
            ]

            if len(rows) != 2:
                raise RuntimeError(
                    "Expected exactly two development directions for "
                    f"outer={outer}, drug={drug}, "
                    f"architecture={architecture}; observed {len(rows)}."
                )

            configuration_id = (
                f"outer_{outer}__corrective_architecture_screen__"
                f"{slug(genome_variant)}__{slug(drug)}__"
                f"{slug(selected['drug_view_fusion_method'])}__"
                f"{slug(architecture)}"
            )

            config = rows.iloc[0].to_dict()

            for column in [
                "run_id",
                seed_column,
                source_column,
                evaluation_column,
            ]:
                config.pop(column, None)

            updates = {
                "configuration_id": configuration_id,
                "outer_target_code": outer,
                "genome_representation": (
                    genome_representation
                ),
                "selected_genome_candidate": (
                    genome_variant
                ),
                "corrective_genome_variant": (
                    genome_variant
                ),
                "genome_matrix_path": selected.get(
                    "genome_matrix_path",
                    "",
                ),
                "shared_hp_id": selected.get(
                    "shared_hp_id",
                    "",
                ),
                "low_rank_interaction_rank": selected.get(
                    "low_rank_interaction_rank",
                    "0",
                ),
                "selected_kmer_dimension": selected.get(
                    "selected_kmer_dimension",
                    "",
                ),
                "common_amr_dimension": selected.get(
                    "common_amr_dimension",
                    "",
                ),
                "latent_width": selected.get(
                    "latent_width",
                    "",
                ),
                "genome_hidden_multiplier": selected.get(
                    "genome_hidden_multiplier",
                    "",
                ),
                "drug_hidden_multiplier": selected.get(
                    "drug_hidden_multiplier",
                    "",
                ),
                "fusion_hidden_multiplier": selected.get(
                    "fusion_hidden_multiplier",
                    "",
                ),
                "dropout": selected.get(
                    "dropout",
                    "",
                ),
                "learning_rate": selected.get(
                    "learning_rate",
                    "",
                ),
                "weight_decay": selected.get(
                    "weight_decay",
                    "",
                ),
                "batch_size": selected.get(
                    "batch_size",
                    "",
                ),
                "maximum_epochs": selected.get(
                    "maximum_epochs",
                    "",
                ),
                "early_stopping_patience": selected.get(
                    "early_stopping_patience",
                    "",
                ),
                "minimum_rmse_improvement": selected.get(
                    "minimum_rmse_improvement",
                    "",
                ),
                "gradient_clip_norm": selected.get(
                    "gradient_clip_norm",
                    "",
                ),
                "drug_representation": drug,
                "drug_view_fusion_method": selected.get(
                    "drug_view_fusion_method",
                    "",
                ),
                "drug_view_low_rank": selected.get(
                    "drug_view_low_rank",
                    "0",
                ),
                "cross_modal_architecture": (
                    architecture
                ),
                "cross_modal_bilinear_rank": (
                    CROSS_MODAL_BILINEAR_RANK
                ),
                "result_source": (
                    "new_script173_training"
                ),
                "new_training_required": "YES",
                "screen_seeds": "|".join(
                    str(seed)
                    for seed in SCREEN_SEEDS
                ),
                "outer_target_labels_used": "NO",
            }

            config.update(updates)
            new_configuration_records.append(
                config
            )
            all_candidate_records.append(
                config
            )

            for base in rows.to_dict(
                orient="records"
            ):
                source = str(
                    base[source_column]
                )
                evaluation = str(
                    base[evaluation_column]
                )

                for seed in SCREEN_SEEDS:
                    run = dict(base)
                    run.update(updates)
                    run[seed_column] = seed
                    run["run_id"] = (
                        f"{configuration_id}__"
                        f"{slug(source)}_to_"
                        f"{slug(evaluation)}__"
                        f"seed_{seed}"
                    )
                    run_records.append(run)

    new_configurations = pd.DataFrame(
        new_configuration_records
    ).sort_values(
        [
            "outer_target_code",
            "cross_modal_architecture",
        ]
    ).reset_index(drop=True)

    all_candidates = pd.DataFrame(
        all_candidate_records
    ).sort_values(
        [
            "outer_target_code",
            "cross_modal_architecture",
        ]
    ).reset_index(drop=True)

    run_plan = pd.DataFrame(
        run_records
    ).sort_values(
        [
            "outer_target_code",
            "cross_modal_architecture",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    if len(new_configurations) != 15:
        raise RuntimeError(
            f"Expected 15 new architecture configurations; "
            f"observed {len(new_configurations)}."
        )

    if len(all_candidates) != 18:
        raise RuntimeError(
            f"Expected 18 total architecture candidates; "
            f"observed {len(all_candidates)}."
        )

    if len(run_plan) != 90:
        raise RuntimeError(
            f"Expected 90 new architecture fits; "
            f"observed {len(run_plan)}."
        )

    if new_configurations[
        "configuration_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate new architecture configuration IDs."
        )

    if run_plan[
        "run_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate new architecture run IDs."
        )

    run_plan["_direction_id"] = (
        run_plan[
            "source_species_code"
        ].astype(str)
        + "_to_"
        + run_plan[
            "evaluation_species_code"
        ].astype(str)
    )

    summary = (
        run_plan.groupby(
            [
                "outer_target_code",
                "cross_modal_architecture",
            ],
            as_index=False,
        )
        .agg(
            configurations=(
                "configuration_id",
                "nunique",
            ),
            directions=(
                "_direction_id",
                "nunique",
            ),
            planned_runs=(
                "run_id",
                "size",
            ),
        )
    )

    run_plan = run_plan.drop(
        columns=[
            "_direction_id"
        ]
    )

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_role",
                "value": (
                    "corrective six-way cross-modal architecture screen "
                    "after frozen genome, drug representation and within-drug fusion selection"
                ),
            },
            {
                "item": "architecture_candidates",
                "value": "|".join(
                    ALL_ARCHITECTURES
                ),
            },
            {
                "item": "additive_control_definition",
                "value": (
                    "pure no-interaction model: intercept plus independent "
                    "linear genome-latent and drug-latent contributions; "
                    "equivalent to concatenation followed by a single linear head"
                ),
            },
            {
                "item": "projected_baseline_policy",
                "value": (
                    "reuse the selected final-drug projected-concatenation "
                    "result from Script 167/171 because it uses the same frozen "
                    "genome, selected drug views, within-drug fusion method, "
                    "numerical bundle, directions and seeds"
                ),
            },
            {
                "item": "new_architectures",
                "value": "|".join(
                    NEW_ARCHITECTURES
                ),
            },
            {
                "item": "new_training_fits",
                "value": "90",
            },
            {
                "item": "screen_seeds",
                "value": "|".join(
                    str(seed)
                    for seed in SCREEN_SEEDS
                ),
            },
            {
                "item": "cross_modal_bilinear_rank",
                "value": str(
                    CROSS_MODAL_BILINEAR_RANK
                ),
            },
            {
                "item": "primary_metric",
                "value": (
                    "bidirectional-average per-antibiotic macro RMSE"
                ),
            },
            {
                "item": "primary_reporting",
                "value": (
                    "mean and sample SD with ddof=1 across three paired seeds"
                ),
            },
            {
                "item": "secondary_metric",
                "value": (
                    "seedwise worst-direction macro RMSE, mean and sample SD"
                ),
            },
            {
                "item": "selection_rule",
                "value": (
                    "minimum mean bidirectional macro RMSE per outer target; "
                    "exact ties resolved by lower worst-direction mean and "
                    "then lower parameter count"
                ),
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "held-out outer-target MIC labels not used"
                ),
            },
            {
                "item": "models_trained_by_script172",
                "value": "NO",
            },
        ]
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_tsv(
        new_configurations,
        NEW_CONFIGURATION_OUTPUT,
    )
    write_tsv(
        all_candidates,
        ALL_CANDIDATE_OUTPUT,
    )
    write_tsv(
        run_plan,
        RUN_PLAN_OUTPUT,
    )
    write_tsv(
        selected_inputs,
        SELECTED_INPUT_OUTPUT,
    )
    write_tsv(
        protocol,
        PROTOCOL_OUTPUT,
    )
    write_tsv(
        summary,
        PLAN_SUMMARY_OUTPUT,
    )

    input_manifest = pd.DataFrame(
        [
            {
                "path": str(
                    path.relative_to(PROJECT)
                ),
                "sha256": sha256_file(path),
            }
            for path in required
        ]
    )
    write_tsv(
        input_manifest,
        INPUT_MANIFEST_OUTPUT,
    )

    output_paths = [
        NEW_CONFIGURATION_OUTPUT,
        ALL_CANDIDATE_OUTPUT,
        RUN_PLAN_OUTPUT,
        SELECTED_INPUT_OUTPUT,
        PROTOCOL_OUTPUT,
        INPUT_MANIFEST_OUTPUT,
        PLAN_SUMMARY_OUTPUT,
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
        SCRIPT171_FREEZE,
        DRUG_FUSION_AGGREGATE_MANIFEST,
        *verified171,
        *verified_drug_fusion_aggregate,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_OUTPUT,
    )
    verify_manifest(
        FREEZE_OUTPUT
    )

    display = selected_inputs[
        [
            "outer_target_code",
            "corrective_genome_variant",
            "shared_hp_id",
            "low_rank_interaction_rank",
            "drug_representation",
        ]
    ]

    print(
        "===== SCRIPT 172 CORRECTIVE ARCHITECTURE SCREEN PREREGISTRATION ====="
    )
    print(
        display.to_string(
            index=False
        )
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Architecture candidates:",
        len(ALL_ARCHITECTURES),
    )
    print(
        "Projected baselines reused:",
        3,
    )
    print(
        "New configurations:",
        len(new_configurations),
    )
    print(
        "New training fits:",
        len(run_plan),
    )
    print(
        "Seeds:",
        "|".join(
            str(seed)
            for seed in SCREEN_SEEDS
        ),
    )
    print(
        "Outer-target MIC labels used: NO"
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 172 CORRECTIVE ARCHITECTURE "
        "SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
