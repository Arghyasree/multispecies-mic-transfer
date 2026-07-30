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

SCRIPT163_AGGREGATE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_low_rank_interaction_screen_runs_v1/"
      "aggregate_outputs_sha256.txt"
)

SELECTED_BUNDLE_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_low_rank_interaction_screen_v1/"
      "corrective_selected_shared_hyperparameter_bundle_registry_v1.tsv"
)

SELECTED_LOW_RANK_REGISTRY = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_low_rank_interaction_screen_aggregate_v1/"
      "selected_low_rank_interaction_registry.tsv"
)

STAGEA_RUN_PLAN = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_v1/"
      "corrective_stageA_run_plan_v1.tsv"
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
      "corrective_final_genome_confirmation_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script164_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script164_successful_corrective_final_confirmation_preregistration_core_sha256.txt"
)

EXPECTED_OUTERS = {"ec", "kp", "se"}

FINAL_VARIANTS = (
    "selected_kmer_only",
    "common_AMR_only",
    "raw_kmer_plus_AMR_single_encoder",
    "separate_encoder_projected_kmer_plus_AMR",
    "separate_encoder_low_rank_kmer_plus_AMR",
)

FINAL_SEEDS = (
    20260805,
    20260806,
    20260807,
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

    with path.open("w", encoding="utf-8") as handle:
        for candidate in sorted(
            set(paths),
            key=lambda value: value.as_posix(),
        ):
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


def main() -> None:
    required = [
        SCRIPT163_AGGREGATE_MANIFEST,
        SELECTED_BUNDLE_REGISTRY,
        SELECTED_LOW_RANK_REGISTRY,
        STAGEA_RUN_PLAN,
        FUSED_MATRIX_REGISTRY,
        COMMON_AMR_MATRIX_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified_stageB = verify_manifest(
        SCRIPT163_AGGREGATE_MANIFEST
    )

    selected_bundles = read_tsv(
        SELECTED_BUNDLE_REGISTRY
    ).sort_values(
        "outer_target_code"
    ).reset_index(drop=True)

    selected_ranks = read_tsv(
        SELECTED_LOW_RANK_REGISTRY
    ).sort_values(
        "outer_target_code"
    ).reset_index(drop=True)

    stageA_plan = read_tsv(
        STAGEA_RUN_PLAN
    )

    fused_registry = read_tsv(
        FUSED_MATRIX_REGISTRY
    ).set_index(
        "outer_target_code"
    )

    amr_registry = read_tsv(
        COMMON_AMR_MATRIX_REGISTRY
    ).set_index(
        "outer_target_code"
    )

    full_plan = read_tsv(
        FULL_KMER_RUN_PLAN
    )

    if set(selected_bundles["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            "Unexpected outer targets in selected bundle registry."
        )

    if set(selected_ranks["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            "Unexpected outer targets in selected rank registry."
        )

    selected_rank_lookup = (
        selected_ranks.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )

    run_frames: list[pd.DataFrame] = []
    configuration_records: list[dict[str, object]] = []

    for bundle in selected_bundles.to_dict(
        orient="records"
    ):
        outer = str(
            bundle["outer_target_code"]
        )
        shared_hp_id = str(
            bundle["shared_hp_id"]
        )

        rank_row = selected_rank_lookup[outer]

        selected_rank = int(
            float(
                rank_row[
                    "low_rank_interaction_rank"
                ]
            )
        )

        hp_rows = stageA_plan.loc[
            stageA_plan[
                "outer_target_code"
            ].eq(outer)
            & stageA_plan[
                "shared_hp_id"
            ].eq(shared_hp_id)
        ].copy()

        if len(hp_rows) != 16:
            raise RuntimeError(
                f"Outer {outer}: expected 16 rows for selected bundle "
                f"{shared_hp_id}; observed {len(hp_rows)}."
            )

        hp_reference = hp_rows.iloc[0]

        fused_row = fused_registry.loc[outer]
        amr_row = amr_registry.loc[outer]

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

        kmer_path = str(
            fused_row[
                "kmer_matrix_path"
            ]
        )
        amr_path = str(
            amr_row[
                "matrix_path"
            ]
        )
        fused_path = str(
            fused_row[
                "fused_matrix_path"
            ]
        )
        selected_kmer_id = str(
            fused_row[
                "selected_kmer_representation"
            ]
        )

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
            ].eq(
                "projected_concatenation_MLP"
            )
        ].copy()

        direction_template = (
            template.drop(
                columns=["seed"]
            )
            .drop_duplicates(
                subset=[
                    "outer_target_code",
                    "source_species_code",
                    "evaluation_species_code",
                ]
            )
        )

        if len(direction_template) != 2:
            raise RuntimeError(
                f"Outer {outer}: expected two development directions."
            )

        variant_specs = {
            "selected_kmer_only": {
                "matrix_path": kmer_path,
                "genome_dimension": kmer_dimension,
                "low_rank_interaction_rank": 0,
            },
            "common_AMR_only": {
                "matrix_path": amr_path,
                "genome_dimension": amr_dimension,
                "low_rank_interaction_rank": 0,
            },
            "raw_kmer_plus_AMR_single_encoder": {
                "matrix_path": fused_path,
                "genome_dimension": fused_dimension,
                "low_rank_interaction_rank": 0,
            },
            "separate_encoder_projected_kmer_plus_AMR": {
                "matrix_path": fused_path,
                "genome_dimension": fused_dimension,
                "low_rank_interaction_rank": 0,
            },
            "separate_encoder_low_rank_kmer_plus_AMR": {
                "matrix_path": fused_path,
                "genome_dimension": fused_dimension,
                "low_rank_interaction_rank": selected_rank,
            },
        }

        for variant in FINAL_VARIANTS:
            spec = variant_specs[variant]

            representation_id = (
                f"outer_{outer}__final_{variant}__{shared_hp_id}"
            )

            if (
                variant
                == "separate_encoder_low_rank_kmer_plus_AMR"
            ):
                representation_id += (
                    f"__rank_{selected_rank}"
                )

            configuration_id = (
                f"{representation_id}__Morgan__"
                "projected_concatenation_MLP"
            )

            for seed in FINAL_SEEDS:
                frame = direction_template.copy()
                frame["seed"] = seed
                frame["genome_representation"] = (
                    representation_id
                )
                frame["drug_representation"] = "Morgan"
                frame["cross_modal_architecture"] = (
                    "projected_concatenation_MLP"
                )
                frame["configuration_id"] = (
                    configuration_id
                )
                frame["run_id"] = (
                    configuration_id
                    + "__"
                    + frame[
                        "source_species_code"
                    ].astype(str)
                    + "_to_"
                    + frame[
                        "evaluation_species_code"
                    ].astype(str)
                    + "__seed_"
                    + str(seed)
                )
                frame["shared_hp_id"] = (
                    shared_hp_id
                )
                frame["corrective_genome_variant"] = (
                    variant
                )
                frame["low_rank_interaction_rank"] = (
                    spec[
                        "low_rank_interaction_rank"
                    ]
                )
                frame["selected_kmer_representation"] = (
                    selected_kmer_id
                )
                frame["selected_kmer_dimension"] = (
                    kmer_dimension
                )
                frame["common_amr_dimension"] = (
                    amr_dimension
                )
                frame["genome_dimension"] = (
                    spec["genome_dimension"]
                )
                frame["genome_matrix_path"] = (
                    spec["matrix_path"]
                )

                for field in [
                    "latent_width",
                    "genome_hidden_multiplier",
                    "drug_hidden_multiplier",
                    "fusion_hidden_multiplier",
                    "dropout",
                    "learning_rate",
                    "weight_decay",
                    "batch_size",
                    "maximum_epochs",
                    "early_stopping_patience",
                    "minimum_rmse_improvement",
                    "gradient_clip_norm",
                ]:
                    frame[field] = (
                        hp_reference[field]
                    )

                run_frames.append(frame)

            configuration_records.append(
                {
                    "outer_target_code": outer,
                    "configuration_id": configuration_id,
                    "genome_representation": representation_id,
                    "corrective_genome_variant": variant,
                    "shared_hp_id": shared_hp_id,
                    "low_rank_interaction_rank": (
                        spec[
                            "low_rank_interaction_rank"
                        ]
                    ),
                    "selected_kmer_representation": selected_kmer_id,
                    "selected_kmer_dimension": kmer_dimension,
                    "common_amr_dimension": amr_dimension,
                    "genome_dimension": (
                        spec["genome_dimension"]
                    ),
                    "genome_matrix_path": (
                        spec["matrix_path"]
                    ),
                    "drug_representation": "Morgan",
                    "cross_modal_architecture": (
                        "projected_concatenation_MLP"
                    ),
                    **{
                        field: hp_reference[field]
                        for field in [
                            "latent_width",
                            "genome_hidden_multiplier",
                            "drug_hidden_multiplier",
                            "fusion_hidden_multiplier",
                            "dropout",
                            "learning_rate",
                            "weight_decay",
                            "batch_size",
                            "maximum_epochs",
                            "early_stopping_patience",
                            "minimum_rmse_improvement",
                            "gradient_clip_norm",
                        ]
                    },
                    "directions": 2,
                    "confirmation_seeds": "|".join(
                        str(seed)
                        for seed in FINAL_SEEDS
                    ),
                    "planned_runs": 6,
                    "outer_target_labels_used": "NO",
                }
            )

    run_plan = pd.concat(
        run_frames,
        ignore_index=True,
    ).sort_values(
        [
            "outer_target_code",
            "corrective_genome_variant",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    if len(run_plan) != 90:
        raise RuntimeError(
            f"Expected 90 final-confirmation runs; observed {len(run_plan)}."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate final-confirmation run IDs."
        )

    configuration_registry = pd.DataFrame(
        configuration_records
    ).sort_values(
        [
            "outer_target_code",
            "corrective_genome_variant",
        ]
    ).reset_index(drop=True)

    if len(configuration_registry) != 15:
        raise RuntimeError(
            f"Expected 15 final configurations; "
            f"observed {len(configuration_registry)}."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_role",
                "value": (
                    "Stage C fresh-seed confirmation of the preregistered "
                    "corrective initial genome-representation analysis"
                ),
            },
            {
                "item": "confirmed_variants",
                "value": "|".join(FINAL_VARIANTS),
            },
            {
                "item": "confirmation_seeds",
                "value": "|".join(
                    str(seed)
                    for seed in FINAL_SEEDS
                ),
            },
            {
                "item": "fixed_shared_bundle",
                "value": (
                    "outer-target-specific Stage-A winner selected before "
                    "these confirmation seeds"
                ),
            },
            {
                "item": "fixed_low_rank",
                "value": (
                    "outer-target-specific Stage-B winner selected before "
                    "these confirmation seeds"
                ),
            },
            {
                "item": "fixed_drug_representation",
                "value": "Morgan",
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
                "item": "final_selection_rule",
                "value": (
                    "minimum mean bidirectional-average macro RMSE within "
                    "each outer target; worst-direction may resolve only an "
                    "exact numerical tie"
                ),
            },
            {
                "item": "sample_sd",
                "value": "ddof=1",
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "no held-out outer-target MIC label is used"
                ),
            },
            {
                "item": "models_trained_by_script164",
                "value": "NO",
            },
        ]
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration_path = (
        OUTPUT_ROOT
        / "corrective_final_genome_configuration_registry_v1.tsv"
    )
    run_plan_path = (
        OUTPUT_ROOT
        / "corrective_final_genome_run_plan_v1.tsv"
    )
    protocol_path = (
        OUTPUT_ROOT
        / "corrective_final_genome_confirmation_protocol_v1.tsv"
    )
    input_manifest_path = (
        OUTPUT_ROOT
        / "script164_input_manifest.tsv"
    )
    plan_summary_path = (
        TABLE_ROOT
        / "corrective_final_genome_confirmation_plan_summary_v1.tsv"
    )

    plan_summary = (
        configuration_registry.groupby(
            "outer_target_code",
            as_index=False,
        )
        .agg(
            genome_variants=(
                "corrective_genome_variant",
                "nunique",
            ),
            configurations=(
                "configuration_id",
                "nunique",
            ),
            planned_runs=(
                "planned_runs",
                "sum",
            ),
        )
    )

    write_tsv(
        configuration_registry,
        configuration_path,
    )
    write_tsv(
        run_plan,
        run_plan_path,
    )
    write_tsv(
        protocol,
        protocol_path,
    )
    write_tsv(
        plan_summary,
        plan_summary_path,
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
        configuration_path,
        run_plan_path,
        protocol_path,
        input_manifest_path,
        plan_summary_path,
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
        SCRIPT163_AGGREGATE_MANIFEST,
        *verified_stageB,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_MANIFEST,
    )
    verify_manifest(
        FREEZE_MANIFEST
    )

    selected_summary = (
        configuration_registry.loc[
            configuration_registry[
                "corrective_genome_variant"
            ].eq(
                "separate_encoder_low_rank_kmer_plus_AMR"
            ),
            [
                "outer_target_code",
                "shared_hp_id",
                "low_rank_interaction_rank",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "outer_target_code"
        )
    )

    print(
        "===== SCRIPT 164 CORRECTIVE FINAL GENOME CONFIRMATION PREREGISTRATION ====="
    )
    print(
        selected_summary.to_string(
            index=False
        )
    )
    print()
    print(
        plan_summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Genome variants:",
        len(FINAL_VARIANTS),
    )
    print(
        "Final configurations:",
        len(configuration_registry),
    )
    print(
        "Final new training fits:",
        len(run_plan),
    )
    print(
        "Confirmation seeds:",
        "|".join(
            str(seed)
            for seed in FINAL_SEEDS
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
        "STATUS: SCRIPT 164 CORRECTIVE FINAL "
        "GENOME CONFIRMATION PREREGISTERED"
    )


if __name__ == "__main__":
    main()
