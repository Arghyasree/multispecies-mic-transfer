#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT165_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script165_successful_corrective_final_genome_selection_core_sha256.txt"
)

FINAL_SELECTED_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_aggregate_v1/"
      "selected_final_genome_representation_registry.tsv"
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
      "corrective_drug_representation_screen_v2"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_v2"
)

CONFIGURATION_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_drug_representation_configuration_registry_v2.tsv"
)

RUN_PLAN_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_drug_representation_run_plan_v2.tsv"
)

SELECTED_GENOME_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_selected_genome_representation_registry_v2.tsv"
)

PROTOCOL_OUTPUT = (
    OUTPUT_ROOT
    / "corrective_drug_representation_screen_protocol_v2.tsv"
)

INPUT_MANIFEST_OUTPUT = (
    OUTPUT_ROOT
    / "script166_input_manifest.tsv"
)

PLAN_SUMMARY_OUTPUT = (
    RESULT_ROOT
    / "corrective_drug_representation_screen_plan_summary_v2.tsv"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script166_outputs_sha256.txt"
)

FREEZE_OUTPUT = (
    PROJECT
    / "metadata/config_selection/"
      "script166_successful_corrective_drug_screen_preregistration_v2_core_sha256.txt"
)

CONFIRMATION_SEEDS = [
    20260811,
    20260812,
    20260813,
]

DRUG_REPRESENTATION_CANDIDATES = (
    "identity_seen_drug_control",
    "Morgan",
    "RDKit",
    "ChemBERTa_mean",
    "ChemBERTa_first_token_ablation",
    "ChemBERTa_mean_plus_Morgan",
    "ChemBERTa_mean_plus_Morgan_plus_RDKit",
)

EXPECTED_OUTER_TARGETS = {
    "ec",
    "kp",
    "se",
}

EXPECTED_ARCHITECTURE = (
    "projected_concatenation_MLP"
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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def write_sha_manifest(
    paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique = sorted(
        {path.resolve() for path in paths},
        key=lambda value: value.as_posix(),
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in unique:
            try:
                display = path.relative_to(PROJECT)
            except ValueError:
                display = path

            handle.write(
                f"{sha256_file(path)}  {display}\n"
            )


def slug(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_")


def replace_if_present(
    record: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    output = dict(record)

    for key, value in updates.items():
        if key in output:
            output[key] = value

    return output


def first_existing(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise RuntimeError(
        "None of the required columns was found: "
        + "|".join(candidates)
    )


def main() -> None:
    required = [
        SCRIPT165_FREEZE,
        FINAL_SELECTED_PATH,
        FINAL_CONFIGURATION_PATH,
        FULL_KMER_RUN_PLAN,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    frozen_files = verify_manifest(
        SCRIPT165_FREEZE
    )

    selected = read_tsv(
        FINAL_SELECTED_PATH
    )
    final_configurations = read_tsv(
        FINAL_CONFIGURATION_PATH
    )
    full_plan = read_tsv(
        FULL_KMER_RUN_PLAN
    )

    outer_column = first_existing(
        full_plan,
        [
            "outer_target_code",
            "outer_target",
        ],
    )
    drug_column = first_existing(
        full_plan,
        [
            "drug_representation",
            "drug_representation_id",
        ],
    )
    source_column = first_existing(
        full_plan,
        [
            "source_species_code",
            "source_species",
        ],
    )
    evaluation_column = first_existing(
        full_plan,
        [
            "evaluation_species_code",
            "evaluation_species",
        ],
    )
    seed_column = first_existing(
        full_plan,
        ["seed"],
    )

    observed_outer = set(
        full_plan[
            outer_column
        ].astype(str)
    )

    if observed_outer != EXPECTED_OUTER_TARGETS:
        raise RuntimeError(
            "Unexpected full-grid outer targets: "
            f"{sorted(observed_outer)}"
        )

    if set(
        selected[
            "outer_target_code"
        ].astype(str)
    ) != EXPECTED_OUTER_TARGETS:
        raise RuntimeError(
            "Final selected-genome registry does not contain exactly "
            "EC, KP and SE."
        )

    candidate_drugs = list(
        DRUG_REPRESENTATION_CANDIDATES
    )

    observed_drugs = set(
        full_plan[drug_column].astype(str)
    )
    missing_drugs = sorted(
        set(candidate_drugs).difference(observed_drugs)
    )

    if missing_drugs:
        raise RuntimeError(
            "The frozen full-grid plan is missing registered drug "
            "representations: " + "|".join(missing_drugs)
        )

    selected_records: list[dict[str, Any]] = []
    final_rows: dict[str, dict[str, Any]] = {}

    for selected_row in selected.to_dict(
        orient="records"
    ):
        outer = str(
            selected_row[
                "outer_target_code"
            ]
        )
        variant = str(
            selected_row[
                "corrective_genome_variant"
            ]
        )

        matches = final_configurations.loc[
            final_configurations[
                "outer_target_code"
            ].eq(outer)
            & final_configurations[
                "corrective_genome_variant"
            ].eq(variant)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not resolve exactly one frozen final genome "
                f"configuration for outer={outer}, variant={variant}."
            )

        config = matches.iloc[0].to_dict()
        final_rows[outer] = config

        merged = dict(selected_row)

        for key, value in config.items():
            if key not in merged:
                merged[key] = value

        selected_records.append(merged)

    selected_genomes = pd.DataFrame(
        selected_records
    ).sort_values(
        "outer_target_code"
    )

    genome_column = first_existing(
        full_plan,
        ["genome_representation"],
    )
    architecture_column = first_existing(
        full_plan,
        ["cross_modal_architecture"],
    )

    base_plan = (
        full_plan.loc[
            full_plan[genome_column].astype(str).eq(
                "canonical_4mer"
            )
            & full_plan[architecture_column].astype(str).eq(
                EXPECTED_ARCHITECTURE
            )
            & full_plan[drug_column].astype(str).isin(
                candidate_drugs
            )
        ]
        .sort_values(
            [
                outer_column,
                drug_column,
                source_column,
                evaluation_column,
                seed_column,
            ]
        )
        .drop_duplicates(
            [
                outer_column,
                drug_column,
                source_column,
                evaluation_column,
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    expected_base_rows = (
        len(candidate_drugs)
        * 2
        * len(EXPECTED_OUTER_TARGETS)
    )

    if len(base_plan) != expected_base_rows:
        raise RuntimeError(
            "The frozen full-grid plan does not contain exactly two "
            "directions per outer target and drug "
            f"candidate: observed {len(base_plan)}, expected "
            f"{expected_base_rows}."
        )

    run_records: list[dict[str, Any]] = []
    configuration_records: list[dict[str, Any]] = []

    for outer in sorted(EXPECTED_OUTER_TARGETS):
        final = final_rows[outer]
        final_representation = str(
            final[
                "genome_representation"
            ]
        )
        final_variant = str(
            final[
                "corrective_genome_variant"
            ]
        )

        for drug in candidate_drugs:
            configuration_id = (
                f"outer_{outer}__corrective_drug_screen__"
                f"{slug(final_variant)}__{slug(drug)}"
            )

            rows = base_plan.loc[
                base_plan[outer_column].astype(str).eq(outer)
                & base_plan[drug_column].astype(str).eq(drug)
            ]

            if len(rows) != 2:
                raise RuntimeError(
                    "Expected exactly two frozen development directions "
                    f"for outer={outer}, drug={drug}; observed {len(rows)}."
                )

            config_record = rows.iloc[0].to_dict()
            for run_specific_column in [
                "run_id",
                seed_column,
                source_column,
                evaluation_column,
            ]:
                config_record.pop(
                    run_specific_column,
                    None,
                )

            config_updates = {
                "configuration_id": configuration_id,
                "outer_target_code": outer,
                "genome_representation": final_representation,
                "selected_genome_candidate": final_variant,
                "corrective_genome_variant": final_variant,
                "genome_matrix_path": final.get(
                    "genome_matrix_path",
                    "",
                ),
                "shared_hp_id": final.get(
                    "shared_hp_id",
                    "",
                ),
                "low_rank_interaction_rank": final.get(
                    "low_rank_interaction_rank",
                    "0",
                ),
                "latent_width": final.get(
                    "latent_width",
                    "",
                ),
                "genome_hidden_multiplier": final.get(
                    "genome_hidden_multiplier",
                    "",
                ),
                "drug_hidden_multiplier": final.get(
                    "drug_hidden_multiplier",
                    "",
                ),
                "fusion_hidden_multiplier": final.get(
                    "fusion_hidden_multiplier",
                    "",
                ),
                "dropout": final.get(
                    "dropout",
                    "",
                ),
                "learning_rate": final.get(
                    "learning_rate",
                    "",
                ),
                "weight_decay": final.get(
                    "weight_decay",
                    "",
                ),
                "batch_size": final.get(
                    "batch_size",
                    "",
                ),
                "maximum_epochs": final.get(
                    "maximum_epochs",
                    "",
                ),
                "early_stopping_patience": final.get(
                    "early_stopping_patience",
                    "",
                ),
                "minimum_rmse_improvement": final.get(
                    "minimum_rmse_improvement",
                    "",
                ),
                "gradient_clip_norm": final.get(
                    "gradient_clip_norm",
                    "",
                ),
                "cross_modal_architecture": EXPECTED_ARCHITECTURE,
            }

            config_record = replace_if_present(
                config_record,
                config_updates,
            )

            for key, value in config_updates.items():
                if key not in config_record:
                    config_record[key] = value

            config_record[
                "selection_eligible"
            ] = (
                "NO"
                if drug
                == "identity_seen_drug_control"
                else "YES"
            )
            config_record[
                "corrective_analysis_stage"
            ] = (
                "corrective_drug_representation_screen_v2"
            )

            configuration_records.append(
                config_record
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

                for seed in CONFIRMATION_SEEDS:
                    run_id = (
                        f"{configuration_id}__"
                        f"{slug(source)}_to_{slug(evaluation)}__"
                        f"seed_{seed}"
                    )

                    run_updates = dict(
                        config_updates
                    )
                    run_updates.update(
                        {
                            "run_id": run_id,
                            "configuration_id": configuration_id,
                            seed_column: seed,
                        }
                    )

                    run_record = replace_if_present(
                        base,
                        run_updates,
                    )

                    for key, value in run_updates.items():
                        if key not in run_record:
                            run_record[key] = value

                    run_record[
                        "selection_eligible"
                    ] = (
                        "NO"
                        if drug
                        == "identity_seen_drug_control"
                        else "YES"
                    )
                    run_record[
                        "corrective_analysis_stage"
                    ] = (
                        "corrective_drug_representation_screen_v2"
                    )

                    run_records.append(
                        run_record
                    )

    configuration_registry = pd.DataFrame(
        configuration_records
    )
    run_plan = pd.DataFrame(
        run_records
    )

    if configuration_registry[
        "configuration_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate corrective drug configuration IDs."
        )

    if run_plan[
        "run_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate corrective drug run IDs."
        )

    expected_configurations = (
        len(candidate_drugs)
        * len(EXPECTED_OUTER_TARGETS)
    )
    expected_runs = (
        expected_configurations
        * 2
        * len(CONFIRMATION_SEEDS)
    )

    if len(configuration_registry) != expected_configurations:
        raise RuntimeError(
            "Corrective drug configuration count mismatch: "
            f"{len(configuration_registry)} != "
            f"{expected_configurations}."
        )

    if len(run_plan) != expected_runs:
        raise RuntimeError(
            "Corrective drug run count mismatch: "
            f"{len(run_plan)} != {expected_runs}."
        )

    protocol = pd.DataFrame(
        [
            {
                "item": "configuration_selection_design",
                "value": "nested_leave_one_species_out",
            },
            {
                "item": "frozen_genome_representation_source",
                "value": str(
                    FINAL_SELECTED_PATH.relative_to(PROJECT)
                ),
            },
            {
                "item": "fixed_cross_modal_architecture",
                "value": EXPECTED_ARCHITECTURE,
            },
            {
                "item": "drug_representation_candidates",
                "value": "|".join(candidate_drugs),
            },
            {
                "item": "fresh_training_policy",
                "value": (
                    "all seven drug representations, including Morgan, "
                    "are retrained under the same frozen target-specific "
                    "genome configuration and fresh seeds"
                ),
            },
            {
                "item": "morgan_reuse_policy",
                "value": "no previous Morgan metrics are reused",
            },
            {
                "item": "identity_control_selection_policy",
                "value": (
                    "reported but not eligible for final drug-representation selection"
                ),
            },
            {
                "item": "confirmation_seeds",
                "value": "|".join(
                    str(seed)
                    for seed in CONFIRMATION_SEEDS
                ),
            },
            {
                "item": "primary_selection_metric",
                "value": (
                    "per-seed mean of the two directional per-antibiotic macro RMSE values"
                ),
            },
            {
                "item": "primary_reporting",
                "value": (
                    "mean and sample SD with ddof=1 across three paired seeds"
                ),
            },
            {
                "item": "secondary_robustness_metric",
                "value": (
                    "seedwise worst-direction macro RMSE, mean and sample SD"
                ),
            },
            {
                "item": "outer_target_label_policy",
                "value": "outer-target MIC labels excluded",
            },
            {
                "item": "models_trained",
                "value": "none",
            },
        ]
    )

    input_manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(PROJECT)),
                "sha256": sha256_file(path),
            }
            for path in required
        ]
    )

    run_plan["_direction_id"] = (
        run_plan[source_column].astype(str)
        + "_to_"
        + run_plan[evaluation_column].astype(str)
    )

    summary = (
        run_plan.groupby(
            [
                "outer_target_code",
                drug_column,
            ],
            dropna=False,
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
        .reset_index()
    )

    run_plan = run_plan.drop(
        columns=["_direction_id"]
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
        selected_genomes,
        SELECTED_GENOME_OUTPUT,
    )
    write_tsv(
        configuration_registry,
        CONFIGURATION_OUTPUT,
    )
    write_tsv(
        run_plan,
        RUN_PLAN_OUTPUT,
    )
    write_tsv(
        protocol,
        PROTOCOL_OUTPUT,
    )
    write_tsv(
        input_manifest,
        INPUT_MANIFEST_OUTPUT,
    )
    write_tsv(
        summary,
        PLAN_SUMMARY_OUTPUT,
    )

    output_paths = [
        SELECTED_GENOME_OUTPUT,
        CONFIGURATION_OUTPUT,
        RUN_PLAN_OUTPUT,
        PROTOCOL_OUTPUT,
        INPUT_MANIFEST_OUTPUT,
        PLAN_SUMMARY_OUTPUT,
    ]

    write_sha_manifest(
        output_paths,
        OUTPUT_MANIFEST,
    )
    verify_manifest(
        OUTPUT_MANIFEST
    )

    freeze_paths = [
        Path(__file__).resolve(),
        *output_paths,
        OUTPUT_MANIFEST,
        SCRIPT165_FREEZE,
    ]

    write_sha_manifest(
        freeze_paths,
        FREEZE_OUTPUT,
    )
    verify_manifest(
        FREEZE_OUTPUT
    )

    print(
        "===== SCRIPT 166 CORRECTIVE DRUG-REPRESENTATION "
        "SCREEN PREREGISTRATION ====="
    )

    selected_display_columns = [
        column
        for column in [
            "outer_target_code",
            "corrective_genome_variant",
            "shared_hp_id",
            "low_rank_interaction_rank",
            "genome_representation",
        ]
        if column in selected_genomes.columns
    ]

    print(
        selected_genomes[
            selected_display_columns
        ].to_string(index=False)
    )

    print()
    print(
        summary.to_string(index=False)
    )

    print()
    print(
        "Drug representations:",
        len(candidate_drugs),
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
        "Fresh seeds:",
        "|".join(
            str(seed)
            for seed in CONFIRMATION_SEEDS
        ),
    )
    print(
        "Verified Script 165 frozen files:",
        len(frozen_files),
    )
    print(
        "Outer-target MIC labels used: NO"
    )
    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 166 CORRECTIVE DRUG-REPRESENTATION "
        "SCREEN PREREGISTERED"
    )


if __name__ == "__main__":
    main()
