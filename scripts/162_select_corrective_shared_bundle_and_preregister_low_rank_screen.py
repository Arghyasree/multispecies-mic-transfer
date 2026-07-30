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

SCRIPT161_AGGREGATE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_runs_v1/"
      "aggregate_outputs_sha256.txt"
)

STAGEA_BUNDLE_RANKING = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_aggregate_v1/"
      "corrective_shared_hyperparameter_bundle_ranking.tsv"
)

STAGEA_CONFIGURATION_RANKING = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_aggregate_v1/"
      "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

STAGEA_WORST_DIRECTION = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_aggregate_v1/"
      "configuration_seedwise_worst_direction_mean_sd.tsv"
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

FULL_KMER_RUN_PLAN = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "full_kmer_grid_v1/"
      "nested_loso_full_kmer_run_plan_v1.tsv"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_low_rank_interaction_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_low_rank_interaction_screen_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script162_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script162_successful_low_rank_screen_preregistration_core_sha256.txt"
)

EXPECTED_OUTERS = {"ec", "kp", "se"}

# Fresh rank candidates for the current multispecies corrective analysis.
LOW_RANK_CANDIDATES = (4, 8, 16, 32)

RANK_TUNING_SEEDS = (20260803, 20260804)


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
        SCRIPT161_AGGREGATE_MANIFEST,
        STAGEA_BUNDLE_RANKING,
        STAGEA_CONFIGURATION_RANKING,
        STAGEA_WORST_DIRECTION,
        STAGEA_RUN_PLAN,
        FUSED_MATRIX_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified_stageA = verify_manifest(
        SCRIPT161_AGGREGATE_MANIFEST
    )

    bundle_ranking = read_tsv(
        STAGEA_BUNDLE_RANKING
    )
    configuration_ranking = read_tsv(
        STAGEA_CONFIGURATION_RANKING
    )
    worst_direction = read_tsv(
        STAGEA_WORST_DIRECTION
    )
    stageA_plan = read_tsv(
        STAGEA_RUN_PLAN
    )
    fused_registry = read_tsv(
        FUSED_MATRIX_REGISTRY
    )
    full_plan = read_tsv(
        FULL_KMER_RUN_PLAN
    )

    if set(bundle_ranking["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError(
            "Unexpected outer targets in Stage-A bundle ranking."
        )

    rank_numeric = pd.to_numeric(
        bundle_ranking["selection_rank"],
        errors="raise",
    )

    selected_bundle = (
        bundle_ranking.loc[
            rank_numeric.eq(1)
        ]
        .copy()
        .sort_values(
            "outer_target_code"
        )
        .reset_index(drop=True)
    )

    if len(selected_bundle) != 3:
        raise RuntimeError(
            "Expected one selected Stage-A bundle per outer target."
        )

    if selected_bundle[
        "outer_target_code"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate selected Stage-A outer target."
        )

    fused_lookup = (
        fused_registry.set_index(
            "outer_target_code"
        ).to_dict(orient="index")
    )

    run_frames: list[pd.DataFrame] = []
    configuration_records: list[dict[str, object]] = []

    for selected in selected_bundle.to_dict(
        orient="records"
    ):
        outer = str(
            selected["outer_target_code"]
        )
        shared_hp_id = str(
            selected["shared_hp_id"]
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
                f"Outer {outer}: expected 16 Stage-A rows for "
                f"{shared_hp_id}; observed {len(hp_rows)}."
            )

        hp_reference = hp_rows.iloc[0]

        fused_row = fused_lookup[outer]

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

        for low_rank in LOW_RANK_CANDIDATES:
            representation_id = (
                f"outer_{outer}__separate_encoder_low_rank_"
                f"kmer_plus_AMR__rank_{low_rank}__{shared_hp_id}"
            )

            configuration_id = (
                f"outer_{outer}__separate_encoder_low_rank_"
                f"kmer_plus_AMR__rank_{low_rank}__{shared_hp_id}__"
                "Morgan__projected_concatenation_MLP"
            )

            for seed in RANK_TUNING_SEEDS:
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
                    "separate_encoder_low_rank_kmer_plus_AMR"
                )
                frame["low_rank_interaction_rank"] = (
                    low_rank
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
                    fused_dimension
                )
                frame["genome_matrix_path"] = (
                    fused_path
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
                    "corrective_genome_variant": (
                        "separate_encoder_low_rank_kmer_plus_AMR"
                    ),
                    "shared_hp_id": shared_hp_id,
                    "low_rank_interaction_rank": low_rank,
                    "selected_kmer_representation": selected_kmer_id,
                    "selected_kmer_dimension": kmer_dimension,
                    "common_amr_dimension": amr_dimension,
                    "genome_dimension": fused_dimension,
                    "genome_matrix_path": fused_path,
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
                    "rank_tuning_seeds": "|".join(
                        str(seed)
                        for seed in RANK_TUNING_SEEDS
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
            "low_rank_interaction_rank",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    if len(run_plan) != 48:
        raise RuntimeError(
            f"Expected 48 low-rank screen runs; observed {len(run_plan)}."
        )

    if run_plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate low-rank screen run IDs."
        )

    configuration_registry = pd.DataFrame(
        configuration_records
    ).sort_values(
        [
            "outer_target_code",
            "low_rank_interaction_rank",
        ]
    ).reset_index(drop=True)

    if len(configuration_registry) != 12:
        raise RuntimeError(
            f"Expected 12 low-rank configurations; "
            f"observed {len(configuration_registry)}."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "analysis_role",
                "value": (
                    "Stage B of the preregistered corrective initial "
                    "genome-representation analysis"
                ),
            },
            {
                "item": "selected_shared_bundle_policy",
                "value": (
                    "one fresh shared bundle per outer target, selected in "
                    "Stage A by minimum equal-weight balanced bidirectional "
                    "macro RMSE across four non-interaction genome variants"
                ),
            },
            {
                "item": "low_rank_candidates",
                "value": "|".join(
                    str(value)
                    for value in LOW_RANK_CANDIDATES
                ),
            },
            {
                "item": "rank_tuning_seeds",
                "value": "|".join(
                    str(seed)
                    for seed in RANK_TUNING_SEEDS
                ),
            },
            {
                "item": "controlled_comparison",
                "value": (
                    "same k-mer encoder, AMR encoder, projected-concatenation "
                    "base, drug representation, cross-modal architecture and "
                    "shared numerical bundle; only interaction rank varies"
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
                "item": "rank_selection_rule",
                "value": (
                    "minimum mean bidirectional-average macro RMSE within "
                    "each outer target; worst-direction is secondary reporting "
                    "and may be used only for an exact numerical tie"
                ),
            },
            {
                "item": "external_rank_imported",
                "value": "NO",
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "no held-out outer-target MIC label is used"
                ),
            },
            {
                "item": "models_trained_by_script162",
                "value": "NO",
            },
        ]
    )

    selected_stageA_results = (
        configuration_ranking.merge(
            selected_bundle[
                [
                    "outer_target_code",
                    "shared_hp_id",
                ]
            ],
            on=[
                "outer_target_code",
                "shared_hp_id",
            ],
            how="inner",
            validate="many_to_one",
        )
        .sort_values(
            [
                "outer_target_code",
                "bidirectional_macro_rmse_mean",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_bundle_path = (
        OUTPUT_ROOT
        / "corrective_selected_shared_hyperparameter_bundle_registry_v1.tsv"
    )
    candidate_path = (
        OUTPUT_ROOT
        / "corrective_low_rank_candidate_registry_v1.tsv"
    )
    configuration_path = (
        OUTPUT_ROOT
        / "corrective_low_rank_configuration_registry_v1.tsv"
    )
    run_plan_path = (
        OUTPUT_ROOT
        / "corrective_low_rank_run_plan_v1.tsv"
    )
    protocol_path = (
        OUTPUT_ROOT
        / "corrective_low_rank_screen_protocol_v1.tsv"
    )
    input_manifest_path = (
        OUTPUT_ROOT
        / "script162_input_manifest.tsv"
    )
    selected_stageA_path = (
        TABLE_ROOT
        / "corrective_selected_bundle_stageA_genome_results_v1.tsv"
    )
    plan_summary_path = (
        TABLE_ROOT
        / "corrective_low_rank_screen_plan_summary_v1.tsv"
    )

    candidate_registry = pd.DataFrame(
        {
            "low_rank_interaction_rank": (
                LOW_RANK_CANDIDATES
            ),
            "fresh_candidate_for_current_study": "YES",
            "imported_as_external_winner": "NO",
        }
    )

    plan_summary = (
        configuration_registry.groupby(
            "outer_target_code",
            as_index=False,
        )
        .agg(
            candidate_ranks=(
                "low_rank_interaction_rank",
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
        selected_bundle,
        selected_bundle_path,
    )
    write_tsv(
        candidate_registry,
        candidate_path,
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
        selected_stageA_results,
        selected_stageA_path,
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
        selected_bundle_path,
        candidate_path,
        configuration_path,
        run_plan_path,
        protocol_path,
        input_manifest_path,
        selected_stageA_path,
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
        SCRIPT161_AGGREGATE_MANIFEST,
        *verified_stageA,
    ]

    write_manifest(
        freeze_paths,
        FREEZE_MANIFEST,
    )
    verify_manifest(
        FREEZE_MANIFEST
    )

    print(
        "===== SCRIPT 162 STAGE-A SELECTION AND LOW-RANK SCREEN PREREGISTRATION ====="
    )
    print(
        selected_bundle[
            [
                "outer_target_code",
                "shared_hp_id",
                "balanced_bidirectional_macro_rmse_mean",
                "balanced_bidirectional_macro_rmse_sd",
                "latent_width",
                "dropout",
                "learning_rate",
                "batch_size",
            ]
        ].to_string(
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
        "Selected bundles:",
        len(selected_bundle),
    )
    print(
        "Candidate ranks:",
        "|".join(
            str(value)
            for value in LOW_RANK_CANDIDATES
        ),
    )
    print(
        "Low-rank configurations:",
        len(configuration_registry),
    )
    print(
        "Low-rank new training fits:",
        len(run_plan),
    )
    print(
        "Rank tuning seeds:",
        "|".join(
            str(seed)
            for seed in RANK_TUNING_SEEDS
        ),
    )
    print(
        "Interaction rank selected within the present protocol: YES"
    )
    print(
        "Models trained: NO"
    )
    print()
    print(
        "STATUS: SCRIPT 162 LOW-RANK "
        "INTERACTION SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
