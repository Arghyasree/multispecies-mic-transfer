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
      "corrective_initial_genome_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script160_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script160_successful_corrective_genome_screen_preregistration_core_sha256.txt"
)

EXPECTED_OUTERS = {"ec", "kp", "se"}
TUNING_SEEDS = (20260801, 20260802)

BASE_VARIANTS = (
    "selected_kmer_only",
    "common_AMR_only",
    "raw_kmer_plus_AMR_single_encoder",
    "separate_encoder_projected_kmer_plus_AMR",
)

# Fresh candidates for this corrective initial-screen branch.
FRESH_SHARED_BUNDLES = (
    {
        "shared_hp_id": "corrective_hp_01_compact",
        "latent_width": 32,
        "genome_hidden_multiplier": 1,
        "drug_hidden_multiplier": 1,
        "fusion_hidden_multiplier": 1,
        "dropout": 0.10,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 256,
        "maximum_epochs": 150,
        "early_stopping_patience": 15,
        "minimum_rmse_improvement": 1.0e-4,
        "gradient_clip_norm": 1.0,
    },
    {
        "shared_hp_id": "corrective_hp_02_balanced",
        "latent_width": 64,
        "genome_hidden_multiplier": 2,
        "drug_hidden_multiplier": 1,
        "fusion_hidden_multiplier": 2,
        "dropout": 0.20,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 512,
        "maximum_epochs": 200,
        "early_stopping_patience": 20,
        "minimum_rmse_improvement": 5.0e-5,
        "gradient_clip_norm": 3.0,
    },
    {
        "shared_hp_id": "corrective_hp_03_wide_regularised",
        "latent_width": 96,
        "genome_hidden_multiplier": 2,
        "drug_hidden_multiplier": 2,
        "fusion_hidden_multiplier": 2,
        "dropout": 0.30,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 768,
        "maximum_epochs": 250,
        "early_stopping_patience": 25,
        "minimum_rmse_improvement": 1.0e-5,
        "gradient_clip_norm": 5.0,
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


def one_per_outer(
    frame: pd.DataFrame,
    value_column: str,
) -> dict[str, str]:
    if len(frame) != 3:
        raise RuntimeError(
            f"Expected three rows for {value_column}; "
            f"observed {len(frame)}."
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
        FRESH_SHARED_BUNDLES
    )

    run_frames: list[pd.DataFrame] = []
    configuration_records: list[dict[str, object]] = []

    for outer in sorted(EXPECTED_OUTERS):
        selected_kmer_id = kmer_lookup[outer]
        fused_row = fused_lookup[outer]
        amr_row = amr_lookup[outer]

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

        kmer_path = str(
            fused_row["kmer_matrix_path"]
        )
        amr_path = str(
            amr_row["matrix_path"]
        )
        fused_path = str(
            fused_row["fused_matrix_path"]
        )

        template = full_plan.loc[
            full_plan["outer_target_code"].eq(outer)
            & full_plan["genome_representation"].eq(
                "canonical_4mer"
            )
            & full_plan["drug_representation"].eq("Morgan")
            & full_plan["cross_modal_architecture"].eq(
                "projected_concatenation_MLP"
            )
        ].copy()

        if len(template) != 6:
            raise RuntimeError(
                f"Outer {outer}: expected six template rows; "
                f"observed {len(template)}."
            )

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
            },
            "common_AMR_only": {
                "matrix_path": amr_path,
                "genome_dimension": amr_dimension,
            },
            "raw_kmer_plus_AMR_single_encoder": {
                "matrix_path": fused_path,
                "genome_dimension": fused_dimension,
            },
            "separate_encoder_projected_kmer_plus_AMR": {
                "matrix_path": fused_path,
                "genome_dimension": fused_dimension,
            },
        }

        for hp in FRESH_SHARED_BUNDLES:
            hp_id = str(
                hp["shared_hp_id"]
            )

            for variant in BASE_VARIANTS:
                variant_spec = variant_specs[variant]

                representation_id = (
                    f"outer_{outer}__{variant}__{hp_id}"
                )

                configuration_id = (
                    f"outer_{outer}__{variant}__{hp_id}__"
                    "Morgan__projected_concatenation_MLP"
                )

                for seed in TUNING_SEEDS:
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
                    frame["shared_hp_id"] = hp_id
                    frame["corrective_genome_variant"] = (
                        variant
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
                        variant_spec["genome_dimension"]
                    )
                    frame["genome_matrix_path"] = (
                        variant_spec["matrix_path"]
                    )

                    for key, value in hp.items():
                        frame[key] = value

                    run_frames.append(frame)

                configuration_records.append(
                    {
                        "outer_target_code": outer,
                        "configuration_id": configuration_id,
                        "genome_representation": representation_id,
                        "corrective_genome_variant": variant,
                        "shared_hp_id": hp_id,
                        "selected_kmer_representation": selected_kmer_id,
                        "selected_kmer_dimension": kmer_dimension,
                        "common_amr_dimension": amr_dimension,
                        "genome_dimension": (
                            variant_spec["genome_dimension"]
                        ),
                        "genome_matrix_path": (
                            variant_spec["matrix_path"]
                        ),
                        "drug_representation": "Morgan",
                        "cross_modal_architecture": (
                            "projected_concatenation_MLP"
                        ),
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
            "corrective_genome_variant",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    expected_runs = (
        len(EXPECTED_OUTERS)
        * len(FRESH_SHARED_BUNDLES)
        * len(BASE_VARIANTS)
        * 2
        * len(TUNING_SEEDS)
    )

    if len(run_plan) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} Stage-A runs; "
            f"observed {len(run_plan)}."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate corrective-screen run IDs."
        )

    configuration_registry = pd.DataFrame(
        configuration_records
    ).sort_values(
        [
            "outer_target_code",
            "shared_hp_id",
            "corrective_genome_variant",
        ]
    ).reset_index(drop=True)

    expected_configurations = (
        len(EXPECTED_OUTERS)
        * len(FRESH_SHARED_BUNDLES)
        * len(BASE_VARIANTS)
    )

    if len(configuration_registry) != expected_configurations:
        raise RuntimeError(
            f"Expected {expected_configurations} configurations; "
            f"observed {len(configuration_registry)}."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_role",
                "value": (
                    "preregistered corrective development-only reanalysis "
                    "of the initial genome-representation stage"
                ),
            },
            {
                "item": "reason",
                "value": (
                    "the original stage compared k-mer, common AMR and raw "
                    "k-mer-plus-AMR concatenation but did not include "
                    "separate-encoder projected fusion or low-rank "
                    "interaction fusion"
                ),
            },
            {
                "item": "stage_A_objective",
                "value": (
                    "select fresh shared numerical hyperparameters using "
                    "four non-interaction genome variants weighted equally"
                ),
            },
            {
                "item": "stage_A_variants",
                "value": "|".join(BASE_VARIANTS),
            },
            {
                "item": "stage_B_objective",
                "value": (
                    "after Stage A, screen fresh low-rank interaction ranks "
                    "using the selected shared bundle"
                ),
            },
            {
                "item": "stage_C_objective",
                "value": (
                    "confirm five genome variants with three new seeds: "
                    "k-mer, AMR, raw concatenation, separate-encoder projected "
                    "fusion and separate-encoder low-rank fusion"
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
                    "seedwise worst-direction macro RMSE: max of the two "
                    "development directions within each seed, then mean and "
                    "sample SD"
                ),
            },
            {
                "item": "shared_bundle_selection",
                "value": (
                    "within each outer target and seed, average the "
                    "bidirectional macro RMSE equally across the four Stage-A "
                    "genome variants; select the bundle with the lowest mean"
                ),
            },
            {
                "item": "fresh_numeric_values",
                "value": (
                    "all capacity, optimisation and stopping values are "
                    "selected from the newly preregistered bundles; no frozen "
                    "external pilot numerical hyperparameter bundle is imported"
                ),
            },
            {
                "item": "current_downstream_branch_status",
                "value": (
                    "provisional until this corrective branch resolves; if "
                    "the final genome winner changes, drug and architecture "
                    "selection must be rerun from the new genome winner"
                ),
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "no held-out outer-target MIC label is used"
                ),
            },
            {
                "item": "models_trained_by_script160",
                "value": "NO",
            },
        ]
    )

    implementation_spec = pd.DataFrame(
        [
            {
                "variant": "selected_kmer_only",
                "definition": (
                    "one selected-kmer encoder producing the common genome "
                    "latent dimension"
                ),
            },
            {
                "variant": "common_AMR_only",
                "definition": (
                    "one common-AMR encoder producing the same genome latent "
                    "dimension"
                ),
            },
            {
                "variant": "raw_kmer_plus_AMR_single_encoder",
                "definition": (
                    "raw selected-kmer and AMR features concatenated before "
                    "one genome encoder"
                ),
            },
            {
                "variant": "separate_encoder_projected_kmer_plus_AMR",
                "definition": (
                    "separate selected-kmer and AMR encoders followed by a "
                    "projected-concatenation fusion base"
                ),
            },
            {
                "variant": "separate_encoder_low_rank_kmer_plus_AMR",
                "definition": (
                    "Stage B/C only: exactly the same encoders and projected "
                    "base as the preceding variant plus a low-rank bilinear "
                    "interaction residual"
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
        / "corrective_shared_hyperparameter_bundle_registry_v1.tsv"
    )
    configuration_path = (
        OUTPUT_ROOT
        / "corrective_stageA_configuration_registry_v1.tsv"
    )
    run_plan_path = (
        OUTPUT_ROOT
        / "corrective_stageA_run_plan_v1.tsv"
    )
    protocol_path = (
        OUTPUT_ROOT
        / "corrective_initial_genome_screen_protocol_v1.tsv"
    )
    implementation_path = (
        OUTPUT_ROOT
        / "corrective_genome_variant_specification_v1.tsv"
    )
    input_manifest_path = (
        OUTPUT_ROOT
        / "script160_input_manifest.tsv"
    )
    summary_path = (
        TABLE_ROOT
        / "corrective_stageA_plan_summary_v1.tsv"
    )

    summary = (
        configuration_registry.groupby(
            [
                "outer_target_code",
                "corrective_genome_variant",
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

    write_tsv(bundle_registry, bundle_path)
    write_tsv(configuration_registry, configuration_path)
    write_tsv(run_plan, run_plan_path)
    write_tsv(protocol, protocol_path)
    write_tsv(implementation_spec, implementation_path)
    write_tsv(summary, summary_path)

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
        "===== SCRIPT 160 CORRECTIVE INITIAL GENOME SCREEN PREREGISTRATION ====="
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
        "Stage-A variants:",
        len(BASE_VARIANTS),
    )
    print(
        "Shared bundles:",
        len(FRESH_SHARED_BUNDLES),
    )
    print(
        "Stage-A configurations:",
        len(configuration_registry),
    )
    print(
        "Stage-A new training fits:",
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
        "Fixed drug representation: Morgan"
    )
    print(
        "Fixed cross-modal architecture: "
        "projected_concatenation_MLP"
    )
    print(
        "External numerical bundle imported: NO"
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 160 CORRECTIVE INITIAL "
        "GENOME SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
