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

SCRIPT153_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script153_successful_genome_selection_core_sha256.txt"
)

SCREEN154_METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "drug_representation_screen_runs_v1"
)

SCREEN154_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "drug_representation_screen_aggregate_v1"
)

SCREEN154_AGGREGATE_MANIFEST = (
    SCREEN154_METADATA_ROOT
    / "aggregate_outputs_sha256.txt"
)

SCREEN154_RANKING = (
    SCREEN154_AGGREGATE_ROOT
    / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)

SELECTED_GENOME_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "drug_representation_screen_v1/"
      "nested_loso_selected_genome_representation_registry_v1.tsv"
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
      "architecture_screen_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "architecture_screen_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script155_outputs_sha256.txt"
)

FREEZE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/"
      "script155_successful_drug_selection_core_sha256.txt"
)

EXPECTED_COMPLETED_RUNS = 108
EXPECTED_OUTERS = {"ec", "kp", "se"}

EXPECTED_NON_MORGAN = {
    "identity_seen_drug_control",
    "RDKit",
    "ChemBERTa_mean",
    "ChemBERTa_first_token_ablation",
    "ChemBERTa_mean_plus_Morgan",
    "ChemBERTa_mean_plus_Morgan_plus_RDKit",
}

NON_PROJECTED_ARCHITECTURES = {
    "dual_tower_interaction",
    "cross_modal_GMU",
    "low_rank_bilinear",
    "drug_to_genome_FiLM",
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


def verify_screen154_runs() -> list[Path]:
    flags = sorted(
        SCREEN154_METADATA_ROOT.glob("*/RUN_COMPLETE")
    )

    if len(flags) != EXPECTED_COMPLETED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETED_RUNS} completed Script 154 runs; "
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


def build_drug_candidate_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    screen = read_tsv(SCREEN154_RANKING)
    selected_genome = read_tsv(SELECTED_GENOME_REGISTRY)

    required_screen = {
        "outer_target_code",
        "drug_representation",
        "cross_modal_architecture",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    }

    missing_screen = sorted(
        required_screen.difference(screen.columns)
    )

    if missing_screen:
        raise RuntimeError(
            "Missing Script 154 ranking columns: "
            + "|".join(missing_screen)
        )

    if len(screen) != 18:
        raise RuntimeError(
            f"Expected 18 Script 154 rows; observed {len(screen)}."
        )

    if set(screen["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError("Unexpected outer targets in Script 154 ranking.")

    if set(screen["drug_representation"]) != EXPECTED_NON_MORGAN:
        raise RuntimeError("Unexpected non-Morgan drug candidates.")

    if set(screen["cross_modal_architecture"]) != {
        "projected_concatenation_MLP"
    }:
        raise RuntimeError(
            "Script 154 did not fix projected-concatenation MLP."
        )

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
        raise RuntimeError(
            "Not all Script 154 candidates have three seeds."
        )

    required_genome = {
        "outer_target_code",
        "candidate_id",
        "matrix_path",
        "feature_dimension",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    }

    missing_genome = sorted(
        required_genome.difference(selected_genome.columns)
    )

    if missing_genome:
        raise RuntimeError(
            "Missing selected-genome registry columns: "
            + "|".join(missing_genome)
        )

    if len(selected_genome) != 3:
        raise RuntimeError(
            "Expected three selected-genome rows."
        )

    for column in [
        "feature_dimension",
        "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    ]:
        selected_genome[column] = pd.to_numeric(
            selected_genome[column],
            errors="coerce",
        )

    records: list[dict[str, object]] = []

    for row in selected_genome.itertuples(index=False):
        records.append(
            {
                "outer_target_code": str(row.outer_target_code),
                "drug_representation": "Morgan",
                "selected_genome_candidate": str(row.candidate_id),
                "selected_genome_matrix_path": str(row.matrix_path),
                "selected_genome_feature_dimension": int(
                    float(row.feature_dimension)
                ),
                "seed_count": int(float(row.seed_count)),
                "bidirectional_macro_rmse_mean": float(
                    row.bidirectional_macro_rmse_mean
                ),
                "bidirectional_macro_rmse_sd": float(
                    row.bidirectional_macro_rmse_sd
                ),
                "metric_source": (
                    "selected-genome-stage Morgan result"
                ),
            }
        )

    genome_lookup = selected_genome.set_index(
        "outer_target_code"
    )

    for row in screen.itertuples(index=False):
        outer = str(row.outer_target_code)
        genome_row = genome_lookup.loc[outer]

        records.append(
            {
                "outer_target_code": outer,
                "drug_representation": str(
                    row.drug_representation
                ),
                "selected_genome_candidate": str(
                    genome_row["candidate_id"]
                ),
                "selected_genome_matrix_path": str(
                    genome_row["matrix_path"]
                ),
                "selected_genome_feature_dimension": int(
                    float(genome_row["feature_dimension"])
                ),
                "seed_count": int(float(row.seed_count)),
                "bidirectional_macro_rmse_mean": float(
                    row.bidirectional_macro_rmse_mean
                ),
                "bidirectional_macro_rmse_sd": float(
                    row.bidirectional_macro_rmse_sd
                ),
                "metric_source": (
                    "Script154 drug-representation screen"
                ),
            }
        )

    candidates = pd.DataFrame(records)

    if len(candidates) != 21:
        raise RuntimeError(
            f"Expected 21 drug candidates; observed {len(candidates)}."
        )

    candidates = candidates.sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "bidirectional_macro_rmse_sd",
            "drug_representation",
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
            "Expected one selected drug representation per outer target."
        )

    selected["selection_metric"] = (
        "three_seed_mean_bidirectional_macro_rmse"
    )
    selected["exact_tie_rule"] = (
        "lower_sd_then_drug_representation_id"
    )
    selected["outer_target_labels_used"] = "NO"

    return candidates, selected


def prepare_architecture_screen(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_plan = read_tsv(FULL_KMER_RUN_PLAN)

    template = full_plan.loc[
        full_plan["genome_representation"].eq("canonical_4mer")
        & full_plan["drug_representation"].eq("Morgan")
        & full_plan["cross_modal_architecture"].isin(
            NON_PROJECTED_ARCHITECTURES
        )
    ].copy()

    if len(template) != 72:
        raise RuntimeError(
            f"Expected 72 architecture-template runs; "
            f"observed {len(template)}."
        )

    selected_drug_lookup = dict(
        zip(
            selected["outer_target_code"],
            selected["drug_representation"],
        )
    )

    selected_genome_lookup = dict(
        zip(
            selected["outer_target_code"],
            selected["selected_genome_candidate"],
        )
    )

    template["genome_representation"] = (
        "selected_genome_representation"
    )

    template["drug_representation"] = template[
        "outer_target_code"
    ].map(selected_drug_lookup)

    template["selected_genome_candidate"] = template[
        "outer_target_code"
    ].map(selected_genome_lookup)

    template["configuration_id"] = (
        "outer_"
        + template["outer_target_code"].astype(str)
        + "__selected_genome_representation__"
        + template["drug_representation"].astype(str)
        + "__"
        + template["cross_modal_architecture"].astype(str)
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
        raise RuntimeError(
            "Duplicate architecture-screen run IDs."
        )

    run_plan = template.sort_values(
        [
            "outer_target_code",
            "cross_modal_architecture",
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

    if len(configuration_registry) != 12:
        raise RuntimeError(
            f"Expected 12 architecture configurations; "
            f"observed {len(configuration_registry)}."
        )

    return run_plan, configuration_registry


def main() -> None:
    for path in [
        SCRIPT153_FREEZE,
        SCREEN154_AGGREGATE_MANIFEST,
        SCREEN154_RANKING,
        SELECTED_GENOME_REGISTRY,
        FULL_KMER_RUN_PLAN,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    verify_manifest(SCRIPT153_FREEZE)
    verified_screen_runs = verify_screen154_runs()
    verified_aggregate = verify_manifest(
        SCREEN154_AGGREGATE_MANIFEST
    )

    candidates, selected = build_drug_candidate_table()
    run_plan, configuration_registry = (
        prepare_architecture_screen(selected)
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    candidate_path = (
        TABLE_ROOT
        / "nested_loso_drug_representation_complete_ranking_v1.tsv"
    )

    selected_path = (
        OUTPUT_ROOT
        / "nested_loso_selected_drug_representation_registry_v1.tsv"
    )

    run_plan_path = (
        OUTPUT_ROOT
        / "nested_loso_architecture_screen_run_plan_v1.tsv"
    )

    configuration_path = (
        OUTPUT_ROOT
        / "nested_loso_architecture_screen_configuration_registry_v1.tsv"
    )

    protocol_path = (
        OUTPUT_ROOT
        / "nested_loso_architecture_screen_protocol_v1.tsv"
    )

    protocol = pd.DataFrame(
        [
            {
                "item": "stage_objective",
                "value": (
                    "compare five cross-modal architectures using the selected "
                    "genome and drug representations separately per outer target"
                ),
            },
            {
                "item": "projected_concatenation_candidate",
                "value": (
                    "reuse selected-drug stage projected-concatenation result; "
                    "do not retrain"
                ),
            },
            {
                "item": "new_architecture_candidates",
                "value": "|".join(
                    sorted(NON_PROJECTED_ARCHITECTURES)
                ),
            },
            {
                "item": "new_training_fits",
                "value": 72,
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
                "item": "models_trained_by_script155",
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
        / "script155_input_manifest.tsv"
    )

    input_paths = [
        Path(__file__).resolve(),
        SCRIPT153_FREEZE,
        SCREEN154_AGGREGATE_MANIFEST,
        SCREEN154_RANKING,
        SELECTED_GENOME_REGISTRY,
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
        SCREEN154_AGGREGATE_MANIFEST,
        *verified_aggregate,
    ]

    write_manifest(freeze_paths, FREEZE_MANIFEST)
    verify_manifest(FREEZE_MANIFEST)

    print("===== SCRIPT 155 DRUG REPRESENTATION SELECTION =====")
    print(
        selected[
            [
                "outer_target_code",
                "drug_representation",
                "selected_genome_candidate",
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
            ]
        ].to_string(index=False)
    )

    print()
    print("Verified Script 154 runs:", EXPECTED_COMPLETED_RUNS)
    print("Verified Script 154 run files:", len(verified_screen_runs))
    print("Architecture configurations:", len(configuration_registry))
    print("Architecture new training fits:", len(run_plan))
    print("Models trained: NO")
    print()
    print(
        "STATUS: SCRIPT 155 DRUG REPRESENTATIONS SELECTED "
        "AND ARCHITECTURE SCREEN PREPARED"
    )


if __name__ == "__main__":
    main()
