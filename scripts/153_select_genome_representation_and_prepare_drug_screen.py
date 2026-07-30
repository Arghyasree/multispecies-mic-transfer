#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT151_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script151_successful_selected_kmer_and_matrix_core_sha256.txt"
)

SCREEN152_METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_runs_v1"
)

SCREEN152_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "genome_representation_screen_aggregate_v1"
)

SCREEN152_AGGREGATE_MANIFEST = (
    SCREEN152_METADATA_ROOT
    / "aggregate_outputs_sha256.txt"
)

SCREEN152_RANKING = (
    SCREEN152_AGGREGATE_ROOT
    / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

SELECTED_KMER_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_selected_kmer_registry_v1.tsv"
)

FUSED_REGISTRY = (
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
      "drug_representation_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "drug_representation_screen_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script153_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script153_successful_genome_selection_core_sha256.txt"
)

EXPECTED_COMPLETED_RUNS = 36
EXPECTED_OUTERS = {"ec", "kp", "se"}

NON_MORGAN_DRUG_REPRESENTATIONS = {
    "identity_seen_drug_control",
    "RDKit",
    "ChemBERTa_mean",
    "ChemBERTa_first_token_ablation",
    "ChemBERTa_mean_plus_Morgan",
    "ChemBERTa_mean_plus_Morgan_plus_RDKit",
}


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
    paths: list[Path],
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


def verify_screen152_runs() -> list[Path]:
    flags = sorted(
        SCREEN152_METADATA_ROOT.glob("*/RUN_COMPLETE")
    )

    if len(flags) != EXPECTED_COMPLETED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETED_RUNS} completed Script 152 runs; "
            f"observed {len(flags)}."
        )

    verified: list[Path] = []

    for flag in flags:
        if flag.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"Nonzero RUN_COMPLETE: {flag}")

        manifest = flag.parent / "outputs_sha256.txt"
        verified.extend(verify_manifest(manifest))
        verified.extend([flag, manifest])

    return verified


def build_candidate_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    screen = read_tsv(SCREEN152_RANKING)
    selected_kmer = read_tsv(SELECTED_KMER_REGISTRY)
    fused_registry = read_tsv(FUSED_REGISTRY)

    required_screen = {
        "outer_target_code",
        "genome_representation",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    }

    missing_screen = sorted(
        required_screen.difference(screen.columns)
    )

    if missing_screen:
        raise RuntimeError(
            "Missing Script 152 ranking columns: "
            + "|".join(missing_screen)
        )

    if len(screen) != 6:
        raise RuntimeError(
            f"Expected six Script 152 configuration rows; observed {len(screen)}."
        )

    if set(screen["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError("Unexpected outer targets in Script 152 ranking.")

    if set(screen["genome_representation"]) != {
        "common_cross_species_AMR",
        "selected_kmer_plus_common_AMR",
    }:
        raise RuntimeError("Unexpected Script 152 genome representations.")

    for column in [
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    ]:
        screen[column] = pd.to_numeric(
            screen[column],
            errors="coerce",
        )

    if not screen["seed_count"].eq(3).all():
        raise RuntimeError("Not all Script 152 candidates have three seeds.")

    required_kmer = {
        "outer_target_code",
        "genome_representation",
        "selected_kmer_dimension",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    }

    missing_kmer = sorted(
        required_kmer.difference(selected_kmer.columns)
    )

    if missing_kmer:
        raise RuntimeError(
            "Missing selected-kmer columns: "
            + "|".join(missing_kmer)
        )

    if len(selected_kmer) != 3:
        raise RuntimeError("Expected three selected-kmer rows.")

    for column in [
        "selected_kmer_dimension",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    ]:
        selected_kmer[column] = pd.to_numeric(
            selected_kmer[column],
            errors="coerce",
        )

    if not selected_kmer["seed_count"].eq(3).all():
        raise RuntimeError("Selected k-mer metrics do not have three seeds.")

    required_registry = {
        "outer_target_code",
        "selected_kmer_representation",
        "selected_kmer_dimension",
        "common_amr_dimension",
        "fused_dimension",
        "kmer_matrix_path",
        "common_amr_matrix_path",
        "fused_matrix_path",
    }

    missing_registry = sorted(
        required_registry.difference(fused_registry.columns)
    )

    if missing_registry:
        raise RuntimeError(
            "Missing fused-registry columns: "
            + "|".join(missing_registry)
        )

    registry_by_outer = fused_registry.set_index(
        "outer_target_code"
    )

    records: list[dict[str, object]] = []

    for row in selected_kmer.itertuples(index=False):
        outer = str(row.outer_target_code)
        registry_row = registry_by_outer.loc[outer]

        records.append(
            {
                "outer_target_code": outer,
                "candidate_id": "selected_kmer",
                "source_genome_representation": str(
                    row.genome_representation
                ),
                "feature_dimension": int(
                    float(row.selected_kmer_dimension)
                ),
                "matrix_path": str(
                    registry_row["kmer_matrix_path"]
                ),
                "seed_count": int(float(row.seed_count)),
                "bidirectional_macro_rmse_mean": float(
                    row.bidirectional_macro_rmse_mean
                ),
                "bidirectional_macro_rmse_sd": float(
                    row.bidirectional_macro_rmse_sd
                ),
                "metric_source": "Script150_selected_kmer",
            }
        )

    for row in screen.itertuples(index=False):
        outer = str(row.outer_target_code)
        representation = str(row.genome_representation)
        registry_row = registry_by_outer.loc[outer]

        if representation == "common_cross_species_AMR":
            candidate_id = "common_cross_species_AMR"
            dimension = int(float(registry_row["common_amr_dimension"]))
            matrix_path = str(registry_row["common_amr_matrix_path"])

        elif representation == "selected_kmer_plus_common_AMR":
            candidate_id = "selected_kmer_plus_common_AMR"
            dimension = int(float(registry_row["fused_dimension"]))
            matrix_path = str(registry_row["fused_matrix_path"])

        else:
            raise RuntimeError(representation)

        records.append(
            {
                "outer_target_code": outer,
                "candidate_id": candidate_id,
                "source_genome_representation": representation,
                "feature_dimension": dimension,
                "matrix_path": matrix_path,
                "seed_count": int(float(row.seed_count)),
                "bidirectional_macro_rmse_mean": float(
                    row.bidirectional_macro_rmse_mean
                ),
                "bidirectional_macro_rmse_sd": float(
                    row.bidirectional_macro_rmse_sd
                ),
                "metric_source": "Script152_genome_representation_screen",
            }
        )

    candidates = pd.DataFrame(records)

    if len(candidates) != 9:
        raise RuntimeError(
            f"Expected nine genome candidates; observed {len(candidates)}."
        )

    for path_value in candidates["matrix_path"]:
        path = project_path(str(path_value))
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = candidates.sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "feature_dimension",
            "candidate_id",
        ]
    ).reset_index(drop=True)

    candidates["selection_rank"] = (
        candidates.groupby("outer_target_code").cumcount() + 1
    )

    selected = candidates.loc[
        candidates["selection_rank"].eq(1)
    ].copy().reset_index(drop=True)

    if len(selected) != 3:
        raise RuntimeError(
            "Expected one selected genome representation per outer target."
        )

    selected["selection_metric"] = (
        "three_seed_mean_bidirectional_macro_rmse"
    )
    selected["exact_tie_rule"] = (
        "lower_feature_dimension_then_candidate_id"
    )
    selected["outer_target_labels_used"] = "NO"

    return candidates, selected


def prepare_drug_screen(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_plan = read_tsv(FULL_KMER_RUN_PLAN)

    template = full_plan.loc[
        full_plan["genome_representation"].eq("canonical_4mer")
        & full_plan["cross_modal_architecture"].eq(
            "projected_concatenation_MLP"
        )
        & full_plan["drug_representation"].isin(
            NON_MORGAN_DRUG_REPRESENTATIONS
        )
    ].copy()

    if len(template) != 108:
        raise RuntimeError(
            f"Expected 108 non-Morgan template runs; observed {len(template)}."
        )

    selected_lookup = dict(
        zip(
            selected["outer_target_code"],
            selected["candidate_id"],
        )
    )

    template["selected_genome_candidate"] = template[
        "outer_target_code"
    ].map(selected_lookup)

    template["genome_representation"] = (
        "selected_genome_representation"
    )

    template["configuration_id"] = (
        "outer_"
        + template["outer_target_code"].astype(str)
        + "__selected_genome_representation__"
        + template["drug_representation"].astype(str)
        + "__projected_concatenation_MLP"
    )

    template["run_id"] = (
        template["configuration_id"].astype(str)
        + "__"
        + template["source_species_code"].astype(str)
        + "_to_"
        + template["evaluation_species_code"].astype(str)
        + "__seed_"
        + template["seed"].astype(str)
    )

    if template["run_id"].duplicated().any():
        raise RuntimeError("Duplicate drug-screen run IDs.")

    run_plan = template.sort_values(
        [
            "outer_target_code",
            "drug_representation",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    configuration_registry = (
        run_plan[
            [
                "configuration_id",
                "outer_target_code",
                "selected_genome_candidate",
                "genome_representation",
                "drug_representation",
                "cross_modal_architecture",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(configuration_registry) != 18:
        raise RuntimeError(
            f"Expected 18 drug-screen configurations; "
            f"observed {len(configuration_registry)}."
        )

    return run_plan, configuration_registry


def main() -> None:
    for path in [
        SCRIPT151_FREEZE,
        SCREEN152_AGGREGATE_MANIFEST,
        SCREEN152_RANKING,
        SELECTED_KMER_REGISTRY,
        FUSED_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    verify_manifest(SCRIPT151_FREEZE)
    verified_screen_runs = verify_screen152_runs()
    verified_aggregate = verify_manifest(
        SCREEN152_AGGREGATE_MANIFEST
    )

    candidates, selected = build_candidate_table()
    run_plan, configuration_registry = prepare_drug_screen(
        selected
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    candidate_path = (
        TABLE_ROOT
        / "nested_loso_genome_representation_complete_ranking_v1.tsv"
    )

    selected_path = (
        OUTPUT_ROOT
        / "nested_loso_selected_genome_representation_registry_v1.tsv"
    )

    run_plan_path = (
        OUTPUT_ROOT
        / "nested_loso_drug_representation_screen_run_plan_v1.tsv"
    )

    configuration_path = (
        OUTPUT_ROOT
        / "nested_loso_drug_representation_screen_configuration_registry_v1.tsv"
    )

    protocol_path = (
        OUTPUT_ROOT
        / "nested_loso_drug_representation_screen_protocol_v1.tsv"
    )

    protocol = pd.DataFrame(
        [
            {
                "item": "stage_objective",
                "value": (
                    "compare seven drug representations using the selected "
                    "genome representation separately for each outer target"
                ),
            },
            {
                "item": "morgan_candidate",
                "value": (
                    "reuse Morgan metrics from the stage that selected the "
                    "winning genome representation; do not retrain"
                ),
            },
            {
                "item": "new_drug_candidates",
                "value": "|".join(
                    sorted(NON_MORGAN_DRUG_REPRESENTATIONS)
                ),
            },
            {
                "item": "fixed_cross_modal_architecture",
                "value": "projected_concatenation_MLP",
            },
            {
                "item": "new_training_fits",
                "value": 108,
            },
            {
                "item": "primary_metric",
                "value": "bidirectional per-antibiotic macro RMSE",
            },
            {
                "item": "selection_scope",
                "value": "separate per outer target",
            },
            {
                "item": "outer_target_labels_used",
                "value": "NO",
            },
            {
                "item": "full_cartesian_product",
                "value": "NO",
            },
            {
                "item": "models_trained_by_script153",
                "value": "NO",
            },
        ]
    )

    write_tsv(candidates, candidate_path)
    write_tsv(selected, selected_path)
    write_tsv(run_plan, run_plan_path)
    write_tsv(configuration_registry, configuration_path)
    write_tsv(protocol, protocol_path)

    input_manifest_path = (
        OUTPUT_ROOT
        / "script153_input_manifest.tsv"
    )

    input_paths = [
        Path(__file__).resolve(),
        SCRIPT151_FREEZE,
        SCREEN152_AGGREGATE_MANIFEST,
        SCREEN152_RANKING,
        SELECTED_KMER_REGISTRY,
        FUSED_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]

    write_tsv(
        pd.DataFrame(
            [
                {
                    "file_path": str(path.relative_to(PROJECT)),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(
                    input_paths,
                    key=lambda value: value.as_posix(),
                )
            ]
        ),
        input_manifest_path,
    )

    output_paths = [
        candidate_path,
        selected_path,
        run_plan_path,
        configuration_path,
        protocol_path,
        input_manifest_path,
    ]

    write_manifest(output_paths, OUTPUT_MANIFEST)
    verify_manifest(OUTPUT_MANIFEST)

    freeze_paths = [
        Path(__file__).resolve(),
        OUTPUT_MANIFEST,
        *output_paths,
        SCREEN152_AGGREGATE_MANIFEST,
        *verified_aggregate,
    ]

    write_manifest(freeze_paths, FREEZE_MANIFEST)
    verify_manifest(FREEZE_MANIFEST)

    print("===== SCRIPT 153 GENOME REPRESENTATION SELECTION =====")
    print(
        selected[
            [
                "outer_target_code",
                "candidate_id",
                "feature_dimension",
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
                "matrix_path",
            ]
        ].to_string(index=False)
    )

    print()
    print("Verified Script 152 runs:", EXPECTED_COMPLETED_RUNS)
    print("Verified Script 152 run files:", len(verified_screen_runs))
    print("Drug-screen configurations:", len(configuration_registry))
    print("Drug-screen new training fits:", len(run_plan))
    print("Models trained: NO")
    print()
    print(
        "STATUS: SCRIPT 153 GENOME REPRESENTATIONS SELECTED "
        "AND DRUG SCREEN PREPARED"
    )


if __name__ == "__main__":
    main()
