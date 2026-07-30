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

SCRIPT157_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script157_successful_architecture_selection_core_sha256.txt"
)

SELECTED_ARCHITECTURE_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "multiview_sensitivity_v1/"
      "nested_loso_selected_architecture_registry_v1.tsv"
)

SELECTED_KMER_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_selected_kmer_registry_v1.tsv"
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
      "fresh_multiview_hyperparameter_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script158_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script158_successful_fresh_multiview_preregistration_core_sha256.txt"
)

EXPECTED_OUTERS = {"ec", "kp", "se"}
TUNING_SEEDS = (20260729, 20260730)

VARIANTS = (
    "fresh_common_AMR_only",
    "fresh_selected_kmer_plus_AMR_projected_concat",
)

# Freshly defined bundles for the current multispecies study.
SHARED_HYPERPARAMETER_BUNDLES = (
    {
        "shared_hp_id": "fresh_hp_01_compact_fast",
        "latent_width": 32,
        "genome_hidden_multiplier": 1,
        "drug_hidden_multiplier": 1,
        "fusion_hidden_multiplier": 1,
        "dropout": 0.05,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 256,
        "maximum_epochs": 120,
        "early_stopping_patience": 12,
        "minimum_rmse_improvement": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "drug_pairwise_rank": 4,
    },
    {
        "shared_hp_id": "fresh_hp_02_balanced",
        "latent_width": 48,
        "genome_hidden_multiplier": 2,
        "drug_hidden_multiplier": 1,
        "fusion_hidden_multiplier": 2,
        "dropout": 0.15,
        "learning_rate": 5.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 512,
        "maximum_epochs": 180,
        "early_stopping_patience": 18,
        "minimum_rmse_improvement": 5.0e-5,
        "gradient_clip_norm": 3.0,
        "drug_pairwise_rank": 8,
    },
    {
        "shared_hp_id": "fresh_hp_03_wider",
        "latent_width": 80,
        "genome_hidden_multiplier": 2,
        "drug_hidden_multiplier": 2,
        "fusion_hidden_multiplier": 2,
        "dropout": 0.25,
        "learning_rate": 2.0e-4,
        "weight_decay": 5.0e-5,
        "batch_size": 768,
        "maximum_epochs": 240,
        "early_stopping_patience": 24,
        "minimum_rmse_improvement": 1.0e-5,
        "gradient_clip_norm": 5.0,
        "drug_pairwise_rank": 16,
    },
    {
        "shared_hp_id": "fresh_hp_04_wide_regularised",
        "latent_width": 112,
        "genome_hidden_multiplier": 3,
        "drug_hidden_multiplier": 2,
        "fusion_hidden_multiplier": 3,
        "dropout": 0.35,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 1024,
        "maximum_epochs": 300,
        "early_stopping_patience": 30,
        "minimum_rmse_improvement": 1.0e-6,
        "gradient_clip_norm": 8.0,
        "drug_pairwise_rank": 32,
    },
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
            raise RuntimeError(f"SHA mismatch: {candidate}")

        verified.append(candidate)

    if not verified:
        raise RuntimeError(f"Empty SHA manifest: {path}")

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


def one_per_outer(
    frame: pd.DataFrame,
    value_column: str,
) -> dict[str, str]:
    if len(frame) != 3:
        raise RuntimeError(
            f"Expected three rows for {value_column}; observed {len(frame)}."
        )

    if set(frame["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            f"Unexpected outer targets for {value_column}."
        )

    return dict(
        zip(
            frame["outer_target_code"].astype(str),
            frame[value_column].astype(str),
        )
    )


def main() -> None:
    required = [
        SCRIPT157_FREEZE,
        SELECTED_ARCHITECTURE_REGISTRY,
        SELECTED_KMER_REGISTRY,
        FUSED_MATRIX_REGISTRY,
        COMMON_AMR_MATRIX_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified157 = verify_manifest(
        SCRIPT157_FREEZE
    )

    selected_architecture = read_tsv(
        SELECTED_ARCHITECTURE_REGISTRY
    )
    selected_kmer = read_tsv(
        SELECTED_KMER_REGISTRY
    )
    fused_registry = read_tsv(
        FUSED_MATRIX_REGISTRY
    )
    amr_registry = read_tsv(
        COMMON_AMR_MATRIX_REGISTRY
    )
    full_plan = read_tsv(
        FULL_KMER_RUN_PLAN
    )

    architecture_lookup = one_per_outer(
        selected_architecture,
        "cross_modal_architecture",
    )
    drug_lookup = one_per_outer(
        selected_architecture,
        "drug_representation",
    )
    kmer_lookup = one_per_outer(
        selected_kmer,
        "genome_representation",
    )

    fused_lookup = (
        fused_registry.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )
    amr_lookup = (
        amr_registry.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )

    bundle_registry = pd.DataFrame(
        SHARED_HYPERPARAMETER_BUNDLES
    )

    if len(bundle_registry) != 4:
        raise RuntimeError(
            "Expected four fresh shared hyperparameter bundles."
        )

    run_frames: list[pd.DataFrame] = []
    config_records: list[dict[str, object]] = []

    for outer in sorted(EXPECTED_OUTERS):
        architecture = architecture_lookup[outer]
        drug = drug_lookup[outer]
        selected_kmer_id = kmer_lookup[outer]

        template = full_plan.loc[
            full_plan["outer_target_code"].eq(outer)
            & full_plan["genome_representation"].eq(
                "canonical_4mer"
            )
            & full_plan["drug_representation"].eq("Morgan")
            & full_plan["cross_modal_architecture"].eq(
                architecture
            )
        ].copy()

        if len(template) != 6:
            raise RuntimeError(
                f"Outer {outer}: expected six direction/seed template rows "
                f"for architecture {architecture}; observed {len(template)}."
            )

        # Keep the two development directions but use two new tuning seeds.
        direction_template = (
            template[
                [
                    column
                    for column in template.columns
                    if column != "seed"
                ]
            ]
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

        fused_row = fused_lookup[outer]
        amr_row = amr_lookup[outer]

        fused_path = str(
            fused_row["fused_matrix_path"]
        )
        amr_path = str(
            amr_row["matrix_path"]
        )

        kmer_dimension = int(
            float(
                fused_row["selected_kmer_dimension"]
            )
        )
        amr_dimension = int(
            float(
                fused_row["common_amr_dimension"]
            )
        )
        fused_dimension = int(
            float(
                fused_row["fused_dimension"]
            )
        )

        for hp in SHARED_HYPERPARAMETER_BUNDLES:
            hp_id = str(hp["shared_hp_id"])

            for variant in VARIANTS:
                if variant == "fresh_common_AMR_only":
                    matrix_path = amr_path
                    genome_dimension = amr_dimension
                else:
                    matrix_path = fused_path
                    genome_dimension = fused_dimension

                representation_id = (
                    f"outer_{outer}__{variant}__{hp_id}"
                )

                configuration_id = (
                    f"outer_{outer}__{variant}__{hp_id}__"
                    f"{drug}__{architecture}"
                )

                for tuning_seed in TUNING_SEEDS:
                    frame = direction_template.copy()
                    frame["seed"] = tuning_seed
                    frame["genome_representation"] = (
                        representation_id
                    )
                    frame["drug_representation"] = drug
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
                        + str(tuning_seed)
                    )
                    frame["shared_hp_id"] = hp_id
                    frame["fresh_genome_variant"] = variant
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
                        genome_dimension
                    )
                    frame["genome_matrix_path"] = (
                        matrix_path
                    )

                    for key, value in hp.items():
                        frame[key] = value

                    run_frames.append(frame)

                config_records.append(
                    {
                        "outer_target_code": outer,
                        "configuration_id": configuration_id,
                        "genome_representation": representation_id,
                        "fresh_genome_variant": variant,
                        "shared_hp_id": hp_id,
                        "selected_kmer_representation": selected_kmer_id,
                        "selected_kmer_dimension": kmer_dimension,
                        "common_amr_dimension": amr_dimension,
                        "genome_dimension": genome_dimension,
                        "genome_matrix_path": matrix_path,
                        "drug_representation": drug,
                        "cross_modal_architecture": architecture,
                        **hp,
                        "directions": 2,
                        "tuning_seeds": "|".join(
                            str(seed)
                            for seed in TUNING_SEEDS
                        ),
                        "planned_runs": 4,
                        "outer_target_labels_used": "NO",
                    }
                )

    run_plan = pd.concat(
        run_frames,
        ignore_index=True,
    ).sort_values(
        [
            "outer_target_code",
            "shared_hp_id",
            "fresh_genome_variant",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    if len(run_plan) != 96:
        raise RuntimeError(
            f"Expected 96 fresh hyperparameter-screen runs; "
            f"observed {len(run_plan)}."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate fresh hyperparameter-screen run IDs."
        )

    configuration_registry = pd.DataFrame(
        config_records
    ).sort_values(
        [
            "outer_target_code",
            "shared_hp_id",
            "fresh_genome_variant",
        ]
    ).reset_index(drop=True)

    if len(configuration_registry) != 24:
        raise RuntimeError(
            f"Expected 24 fresh screen configurations; "
            f"observed {len(configuration_registry)}."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "purpose",
                "value": (
                    "select a fresh shared optimisation and capacity bundle "
                    "for the current multispecies study without inheriting "
                    "any frozen external pilot numeric hyperparameter bundle"
                ),
            },
            {
                "item": "balanced_selection_basis",
                "value": (
                    "for each shared bundle, average the development-species "
                    "bidirectional macro RMSE of AMR-only and separate-encoder "
                    "projected-concatenation multiview models"
                ),
            },
            {
                "item": "variants_in_shared_screen",
                "value": "|".join(VARIANTS),
            },
            {
                "item": "shared_bundle_count",
                "value": len(SHARED_HYPERPARAMETER_BUNDLES),
            },
            {
                "item": "tuning_seeds",
                "value": "|".join(
                    str(seed)
                    for seed in TUNING_SEEDS
                ),
            },
            {
                "item": "new_training_fits",
                "value": len(run_plan),
            },
            {
                "item": "fresh_values_selected",
                "value": (
                    "latent width|genome hidden multiplier|drug hidden "
                    "multiplier|fusion hidden multiplier|dropout|learning "
                    "rate|weight decay|batch size|maximum epochs|early "
                    "stopping patience|minimum RMSE improvement|gradient "
                    "clip norm|drug pairwise rank"
                ),
            },
            {
                "item": "fixed_algorithmic_choices",
                "value": (
                    "observation-level MSE|AdamW|source-only validation-fold "
                    "epoch selection|source-only scaling"
                ),
            },
            {
                "item": "primary_selection_metric",
                "value": (
                    "bidirectional-average per-antibiotic macro RMSE"
                ),
            },
            {
                "item": "secondary_robustness_metric",
                "value": (
                    "seedwise worst-direction macro RMSE: within each seed "
                    "take max of the two development transfer directions, "
                    "then report mean and sample SD across seeds"
                ),
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "held-out outer-target MIC labels are not used"
                ),
            },
            {
                "item": "models_trained_by_script158",
                "value": "NO",
            },
        ]
    )

    implementation_spec = pd.DataFrame(
        [
            {
                "component": "AMR-only genome encoder",
                "definition": (
                    "one view encoder with hidden width equal to "
                    "genome_hidden_multiplier × latent_width"
                ),
            },
            {
                "component": "projected multiview genome encoder",
                "definition": (
                    "separate selected-kmer and AMR encoders using the same "
                    "latent width; concatenate both latents and project through "
                    "a fusion MLP"
                ),
            },
            {
                "component": "drug encoder",
                "definition": (
                    "separate encoders per selected drug view; for multiple "
                    "drug views, projected-concatenation base plus pairwise "
                    "low-rank residual using the bundle-specific "
                    "drug_pairwise_rank"
                ),
            },
            {
                "component": "cross-modal architecture",
                "definition": (
                    "target-specific architecture already selected from "
                    "development species: FiLM for outer Ec, projected "
                    "concatenation for outer Kp, GMU for outer Se"
                ),
            },
            {
                "component": "output dimensions",
                "definition": (
                    "identical genome latent width across AMR-only and "
                    "projected multiview variants within the same bundle"
                ),
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

    bundle_path = (
        OUTPUT_ROOT
        / "fresh_shared_hyperparameter_bundle_registry_v1.tsv"
    )
    configuration_path = (
        OUTPUT_ROOT
        / "fresh_shared_hyperparameter_screen_configuration_registry_v1.tsv"
    )
    run_plan_path = (
        OUTPUT_ROOT
        / "fresh_shared_hyperparameter_screen_run_plan_v1.tsv"
    )
    protocol_path = (
        OUTPUT_ROOT
        / "fresh_shared_hyperparameter_screen_protocol_v1.tsv"
    )
    implementation_path = (
        OUTPUT_ROOT
        / "fresh_multiview_implementation_specification_v1.tsv"
    )
    input_manifest_path = (
        OUTPUT_ROOT
        / "script158_input_manifest.tsv"
    )
    summary_path = (
        TABLE_ROOT
        / "fresh_shared_hyperparameter_screen_summary_v1.tsv"
    )

    summary = (
        configuration_registry.groupby(
            [
                "outer_target_code",
                "fresh_genome_variant",
            ],
            as_index=False,
        )
        .agg(
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
        bundle_registry,
        bundle_path,
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
        implementation_spec,
        implementation_path,
    )
    write_tsv(
        summary,
        summary_path,
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
                    key=lambda value:
                        value.as_posix(),
                )
            ]
        ),
        input_manifest_path,
    )

    output_paths = [
        bundle_path,
        configuration_path,
        run_plan_path,
        protocol_path,
        implementation_path,
        input_manifest_path,
        summary_path,
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
        SCRIPT157_FREEZE,
        *verified157,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_MANIFEST,
    )
    verify_manifest(
        FREEZE_MANIFEST
    )

    print(
        "===== SCRIPT 158 FRESH MULTIVIEW HYPERPARAMETER PREREGISTRATION ====="
    )
    print(
        bundle_registry.to_string(
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
        "Shared bundles:",
        len(bundle_registry),
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
        "Tuning seeds:",
        "|".join(
            str(seed)
            for seed in TUNING_SEEDS
        ),
    )
    print(
        "External numerical bundle imported: NO"
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 158 FRESH MULTIVIEW "
        "HYPERPARAMETER SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
