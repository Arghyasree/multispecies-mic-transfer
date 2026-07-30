#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT165_PATH = (
    PROJECT
    / "scripts/"
      "165_run_corrective_final_genome_confirmation.py"
)

EXPECTED_SCRIPT165_SHA256 = (
    "06dcb3639019d1dca51d99be3bc2cf67"
    "b32470d5edce7e31a4764a72de671b65"
)

SCRIPT166_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script166_successful_corrective_drug_screen_preregistration_v2_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_v2/"
      "corrective_drug_representation_run_plan_v2.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_v2/"
      "corrective_drug_representation_configuration_registry_v2.tsv"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_runs_v2"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_runs_v2"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_aggregate_v2"
)

EXPECTED_RUNS = 126

CURRENT_SPEC: dict[str, Any] | None = None
CURRENT_PARAMETER_COUNT: int | None = None


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


def load_script165():
    if not SCRIPT165_PATH.is_file():
        raise FileNotFoundError(SCRIPT165_PATH)

    observed = sha256_file(
        SCRIPT165_PATH
    )

    if observed != EXPECTED_SCRIPT165_SHA256:
        raise RuntimeError(
            "Script 165 SHA mismatch: "
            f"{observed}"
        )

    specification = importlib.util.spec_from_file_location(
        "corrective_final_genome_confirmation_165",
        SCRIPT165_PATH,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load Script 165."
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    return module


final165 = load_script165()
backend = final165.backend


class CorrectiveDrugScreenNetwork(
    final165.FinalGenomeConfirmationNetwork
):
    def __init__(
        self,
        genome_dimension: int,
        drug_matrices: dict[str, np.ndarray],
        spec: dict[str, Any],
    ) -> None:
        if not drug_matrices:
            raise RuntimeError(
                "No drug feature views were supplied."
            )

        self.drug_view_order = tuple(
            drug_matrices.keys()
        )

        total_drug_dimension = int(
            sum(
                matrix.shape[1]
                for matrix in drug_matrices.values()
            )
        )

        super().__init__(
            genome_dimension=genome_dimension,
            drug_dimension=total_drug_dimension,
            spec=spec,
        )

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        observed_order = tuple(
            drug_views.keys()
        )

        if set(observed_order) != set(
            self.drug_view_order
        ):
            raise RuntimeError(
                "Drug-view mismatch. Expected "
                f"{self.drug_view_order}; observed "
                f"{observed_order}."
            )

        genome_latent = self.encode_genome(
            genome
        )

        drug_input = torch.cat(
            [
                drug_views[view_id]
                for view_id
                in self.drug_view_order
            ],
            dim=1,
        )

        drug_latent = self.drug_encoder(
            drug_input
        )

        additive = (
            self.intercept
            + self.genome_head(
                genome_latent
            ).squeeze(1)
            + self.drug_head(
                drug_latent
            ).squeeze(1)
        )

        interaction = self.interaction_head(
            torch.cat(
                [
                    genome_latent,
                    drug_latent,
                ],
                dim=1,
            )
        ).squeeze(1)

        return additive + interaction


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


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    return (
        float(numeric.mean())
        if len(numeric)
        else float("nan")
    )


def sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    return (
        float(numeric.std(ddof=1))
        if len(numeric) >= 2
        else float("nan")
    )


def build_model(
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    target_mean: float,
    device: torch.device,
) -> nn.Module:
    global CURRENT_PARAMETER_COUNT

    if CURRENT_SPEC is None:
        raise RuntimeError(
            "Current corrective drug-screen specification is unset."
        )

    accepted_architectures = {
        "cross_modal_projected_concat",
        "projected_concatenation_MLP",
    }

    if architecture_id not in accepted_architectures:
        raise RuntimeError(
            "Corrective drug screen requires projected concatenation; "
            f"observed {architecture_id!r}."
        )

    model = CorrectiveDrugScreenNetwork(
        genome_dimension=genome_matrix.shape[1],
        drug_matrices=drug_matrices,
        spec=CURRENT_SPEC,
    ).to(device)

    CURRENT_PARAMETER_COUNT = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )
    final165.CURRENT_PARAMETER_COUNT = (
        CURRENT_PARAMETER_COUNT
    )

    with torch.no_grad():
        model.intercept.fill_(
            float(target_mean)
        )

    return model


def patch_run_outputs(
    run_id: str,
    spec: dict[str, Any],
    run_row: pd.Series,
) -> None:
    if CURRENT_PARAMETER_COUNT is None:
        raise RuntimeError(
            "Model parameter count is unavailable."
        )

    result_directory = RESULT_ROOT / run_id
    metadata_directory = METADATA_ROOT / run_id

    summary_path = (
        result_directory
        / "run_summary.tsv"
    )
    configuration_path = (
        metadata_directory
        / "configuration.tsv"
    )
    manifest_path = (
        metadata_directory
        / "outputs_sha256.txt"
    )

    summary = backend.read_tsv(
        summary_path
    )

    additions = {
        "shared_hp_id": spec[
            "shared_hp_id"
        ],
        "corrective_genome_variant": spec[
            "corrective_genome_variant"
        ],
        "low_rank_interaction_rank": spec[
            "low_rank_interaction_rank"
        ],
        "drug_representation": run_row[
            "drug_representation"
        ],
        "model_parameter_count": (
            CURRENT_PARAMETER_COUNT
        ),
        "corrective_analysis_stage": (
            "corrective_drug_representation_screen_v2"
        ),
    }

    for key, value in additions.items():
        summary[key] = value

    backend.write_tsv(
        summary,
        summary_path,
    )

    configuration = backend.read_tsv(
        configuration_path
    )

    addition_frame = pd.DataFrame(
        [
            {
                "item": key,
                "value": value,
            }
            for key, value
            in additions.items()
        ]
    )

    configuration = configuration.loc[
        ~configuration[
            "item"
        ].isin(
            addition_frame["item"]
        )
    ]

    configuration = pd.concat(
        [
            configuration,
            addition_frame,
        ],
        ignore_index=True,
    )

    backend.write_tsv(
        configuration,
        configuration_path,
    )

    output_paths: list[Path] = []

    for line in manifest_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        _, value = line.split(
            maxsplit=1
        )
        output_paths.append(
            project_path(value)
        )

    backend.write_sha_manifest(
        output_paths,
        manifest_path,
    )
    backend.verify_sha_manifest(
        manifest_path
    )


def execute_run(
    run_row: pd.Series,
    observations: pd.DataFrame,
    device: torch.device,
) -> dict[str, object]:
    global CURRENT_SPEC

    representation = str(
        run_row[
            "genome_representation"
        ]
    )

    if representation not in final165.CONFIG_BY_REPRESENTATION:
        raise RuntimeError(
            "Unregistered final genome representation: "
            f"{representation}"
        )

    CURRENT_SPEC = final165.CONFIG_BY_REPRESENTATION[
        representation
    ]
    final165.CURRENT_SPEC = CURRENT_SPEC

    final165.set_current_hyperparameters(
        CURRENT_SPEC
    )

    summary = final165.ORIGINAL_EXECUTE_RUN(
        run_row,
        observations,
        device,
    )

    patch_run_outputs(
        str(run_row["run_id"]),
        CURRENT_SPEC,
        run_row,
    )

    return summary


def aggregate_completed_runs() -> None:
    final165.ORIGINAL_AGGREGATE()

    all_runs_path = (
        AGGREGATE_ROOT
        / "all_direction_seed_metrics.tsv"
    )
    configuration_path = (
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )

    for path in (
        all_runs_path,
        configuration_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = read_tsv(
        RUN_PLAN_PATH
    )
    configuration_registry = read_tsv(
        CONFIGURATION_REGISTRY_PATH
    )

    metadata_columns = [
        column
        for column in [
            "configuration_id",
            "outer_target_code",
            "genome_representation",
            "corrective_genome_variant",
            "shared_hp_id",
            "low_rank_interaction_rank",
            "drug_representation",
            "selection_eligible",
        ]
        if column in configuration_registry.columns
    ]

    metadata = (
        configuration_registry[
            metadata_columns
        ]
        .drop_duplicates(
            "configuration_id"
        )
    )

    all_runs = read_tsv(
        all_runs_path
    )
    all_runs["macro_rmse"] = pd.to_numeric(
        all_runs["macro_rmse"],
        errors="raise",
    )

    worst_seed = (
        all_runs.groupby(
            [
                "outer_target_code",
                "configuration_id",
                "genome_representation",
                "drug_representation",
                "cross_modal_architecture",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            direction_count=(
                "source_species_code",
                "size",
            ),
            seedwise_worst_direction_macro_rmse=(
                "macro_rmse",
                "max",
            ),
        )
        .reset_index()
    )

    if not worst_seed[
        "direction_count"
    ].eq(2).all():
        raise RuntimeError(
            "Every paired seed must contain exactly two development "
            "transfer directions."
        )

    worst_seed_path = (
        AGGREGATE_ROOT
        / "seedwise_worst_direction_macro_rmse.tsv"
    )

    write_tsv(
        worst_seed.sort_values(
            [
                "outer_target_code",
                "configuration_id",
                "seed",
            ]
        ),
        worst_seed_path,
    )

    worst_summary_records: list[dict[str, object]] = []

    grouping_columns = [
        "outer_target_code",
        "configuration_id",
        "genome_representation",
        "drug_representation",
        "cross_modal_architecture",
    ]

    for keys, group in worst_seed.groupby(
        grouping_columns,
        dropna=False,
    ):
        record = {
            key: value
            for key, value
            in zip(
                grouping_columns,
                keys,
            )
        }

        values = group[
            "seedwise_worst_direction_macro_rmse"
        ]

        record["seed_count"] = group[
            "seed"
        ].nunique()
        record[
            "worst_direction_macro_rmse_mean"
        ] = safe_mean(values)
        record[
            "worst_direction_macro_rmse_sd"
        ] = sample_sd(values)

        worst_summary_records.append(
            record
        )

    worst_summary = pd.DataFrame(
        worst_summary_records
    )

    worst_summary_path = (
        AGGREGATE_ROOT
        / "configuration_seedwise_worst_direction_mean_sd.tsv"
    )

    write_tsv(
        worst_summary.sort_values(
            [
                "outer_target_code",
                "configuration_id",
            ]
        ),
        worst_summary_path,
    )

    configuration = read_tsv(
        configuration_path
    )

    configuration = configuration.merge(
        worst_summary[
            [
                "outer_target_code",
                "configuration_id",
                "worst_direction_macro_rmse_mean",
                "worst_direction_macro_rmse_sd",
            ]
        ],
        on=[
            "outer_target_code",
            "configuration_id",
        ],
        how="left",
        validate="one_to_one",
    )

    configuration = configuration.merge(
        metadata,
        on="configuration_id",
        how="left",
        suffixes=("", "_registry"),
        validate="one_to_one",
    )

    parameter_records: list[dict[str, object]] = []

    first_runs = (
        plan.sort_values(
            [
                "configuration_id",
                "run_id",
            ]
        )
        .drop_duplicates(
            "configuration_id"
        )
    )

    for row in first_runs.to_dict(
        orient="records"
    ):
        summary_path = (
            RESULT_ROOT
            / str(row["run_id"])
            / "run_summary.tsv"
        )

        if not summary_path.is_file():
            raise FileNotFoundError(
                summary_path
            )

        run_summary = backend.read_tsv(
            summary_path
        )

        parameter_records.append(
            {
                "configuration_id": row[
                    "configuration_id"
                ],
                "model_parameter_count": int(
                    float(
                        run_summary[
                            "model_parameter_count"
                        ].iloc[0]
                    )
                ),
            }
        )

    parameters = pd.DataFrame(
        parameter_records
    )

    configuration = configuration.merge(
        parameters,
        on="configuration_id",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "bidirectional_macro_rmse_mean",
        "worst_direction_macro_rmse_mean",
    ]:
        configuration[column] = pd.to_numeric(
            configuration[column],
            errors="raise",
        )

    if "selection_eligible" not in configuration.columns:
        configuration[
            "selection_eligible"
        ] = np.where(
            configuration[
                "drug_representation"
            ].eq(
                "identity_seen_drug_control"
            ),
            "NO",
            "YES",
        )

    configuration[
        "selection_rank"
    ] = pd.NA

    eligible = configuration[
        "selection_eligible"
    ].astype(str).str.upper().eq(
        "YES"
    )

    eligible_sorted = configuration.loc[
        eligible
    ].sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "worst_direction_macro_rmse_mean",
            "model_parameter_count",
        ]
    )

    ranks = (
        eligible_sorted.groupby(
            "outer_target_code"
        )
        .cumcount()
        .add(1)
    )

    configuration.loc[
        eligible_sorted.index,
        "selection_rank",
    ] = ranks.astype(int)

    configuration = configuration.sort_values(
        [
            "outer_target_code",
            "selection_eligible",
            "selection_rank",
            "bidirectional_macro_rmse_mean",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    write_tsv(
        configuration,
        configuration_path,
    )

    selected = (
        configuration.loc[
            configuration[
                "selection_rank"
            ].astype(str).eq("1")
        ][
            [
                column
                for column in [
                    "outer_target_code",
                    "genome_representation",
                    "corrective_genome_variant",
                    "shared_hp_id",
                    "low_rank_interaction_rank",
                    "drug_representation",
                    "bidirectional_macro_rmse_mean",
                    "bidirectional_macro_rmse_sd",
                    "worst_direction_macro_rmse_mean",
                    "worst_direction_macro_rmse_sd",
                    "model_parameter_count",
                ]
                if column in configuration.columns
            ]
        ]
        .sort_values(
            "outer_target_code"
        )
        .reset_index(drop=True)
    )

    if len(selected) != 3:
        raise RuntimeError(
            "Expected exactly one eligible drug-representation winner "
            "per outer target."
        )

    selected_path = (
        AGGREGATE_ROOT
        / "selected_corrective_drug_representation_registry.tsv"
    )

    write_tsv(
        selected,
        selected_path,
    )

    protocol_path = (
        METADATA_ROOT
        / "aggregate_protocol.tsv"
    )

    protocol = pd.DataFrame(
        [
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
                "item": "selection_rule",
                "value": (
                    "minimum mean bidirectional macro RMSE among eligible "
                    "non-control drug representations within each outer target"
                ),
            },
            {
                "item": "identity_control_policy",
                "value": (
                    "reported but not eligible for selection"
                ),
            },
            {
                "item": "sample_sd",
                "value": "ddof=1",
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "held-out outer-target MIC labels not used"
                ),
            },
        ]
    )

    write_tsv(
        protocol,
        protocol_path,
    )

    aggregate_paths = [
        AGGREGATE_ROOT
        / "all_direction_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "direction_three_seed_mean_sd.tsv",
        AGGREGATE_ROOT
        / "bidirectional_seed_metrics.tsv",
        configuration_path,
        AGGREGATE_ROOT
        / "all_per_antibiotic_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "per_antibiotic_three_seed_mean_sd.tsv",
        worst_seed_path,
        worst_summary_path,
        selected_path,
        protocol_path,
    ]

    for path in aggregate_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_path = (
        METADATA_ROOT
        / "aggregate_outputs_sha256.txt"
    )

    write_sha_manifest(
        aggregate_paths,
        manifest_path,
    )
    verify_manifest(
        manifest_path
    )


def configure_backend() -> None:
    final165.RUN_PLAN_PATH = RUN_PLAN_PATH
    final165.RESULT_ROOT = RESULT_ROOT
    final165.METADATA_ROOT = METADATA_ROOT
    final165.AGGREGATE_ROOT = AGGREGATE_ROOT
    final165.EXPECTED_RUNS = EXPECTED_RUNS

    backend.__file__ = str(
        Path(__file__).resolve()
    )
    backend.RUN_PLAN_PATH = RUN_PLAN_PATH
    backend.SCRIPT142_OUTPUTS_PATH = SCRIPT166_FREEZE
    backend.SCRIPT142_FROZEN_PATH = SCRIPT166_FREEZE
    backend.IMPLEMENTATION_BUNDLE_PATH = (
        final165.IMPLEMENTATION_SPEC_PATH
    )
    backend.EXPECTED_IMPLEMENTATION_BUNDLE_SHA256 = (
        sha256_file(
            final165.IMPLEMENTATION_SPEC_PATH
        )
    )
    backend.RESULT_ROOT = RESULT_ROOT
    backend.METADATA_ROOT = METADATA_ROOT
    backend.AGGREGATE_ROOT = AGGREGATE_ROOT
    backend.EXPECTED_RUNS = EXPECTED_RUNS
    backend.KMER_PATHS = {
        representation: Path(
            spec[
                "genome_matrix_path"
            ]
        )
        for representation, spec
        in final165.CONFIG_BY_REPRESENTATION.items()
    }
    backend.build_model = build_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer-target",
        choices=[
            "all",
            "ec",
            "kp",
            "se",
        ],
        default="all",
    )
    parser.add_argument(
        "--drug-representation",
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            20260811,
            20260812,
            20260813,
        ],
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )
    parser.add_argument(
        "--aggregate-every",
        type=int,
        default=6,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    required = [
        SCRIPT165_PATH,
        SCRIPT166_FREEZE,
        RUN_PLAN_PATH,
        CONFIGURATION_REGISTRY_PATH,
        final165.IMPLEMENTATION_SPEC_PATH,
        backend.OBSERVATION_INDEX_PATH,
        backend.SCRIPT140_FROZEN_PATH,
        *backend.DRUG_VIEW_PATHS.values(),
        *[
            Path(
                spec[
                    "genome_matrix_path"
                ]
            )
            for spec in (
                final165.CONFIG_BY_REPRESENTATION.values()
            )
        ],
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verify_manifest(
        SCRIPT166_FREEZE
    )

    configure_backend()

    observations, plan = backend.load_inputs()

    if arguments.outer_target != "all":
        plan = plan.loc[
            plan[
                "outer_target_code"
            ].eq(
                arguments.outer_target
            )
        ]

    if arguments.drug_representation != "all":
        plan = plan.loc[
            plan[
                "drug_representation"
            ].eq(
                arguments.drug_representation
            )
        ]

    plan = plan.loc[
        plan["seed"].isin(
            arguments.seeds
        )
    ].copy()

    plan = plan.sort_values(
        [
            "outer_target_code",
            "drug_representation",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    plan["already_complete"] = [
        backend.run_complete(
            str(run_id)
        )
        for run_id
        in plan["run_id"]
    ]

    print(
        "===== SCRIPT 167 RUN PLAN ====="
    )

    print(
        plan.groupby(
            [
                "outer_target_code",
                "drug_representation",
            ]
        )
        .agg(
            planned_runs=(
                "run_id",
                "size",
            ),
            completed_runs=(
                "already_complete",
                "sum",
            ),
        )
        .reset_index()
        .to_string(index=False)
    )

    print()
    print(
        "Selected planned runs:",
        len(plan),
    )
    print(
        "Already complete:",
        int(
            plan[
                "already_complete"
            ].sum()
        ),
    )
    print(
        "New runs remaining:",
        int(
            (~plan["already_complete"]).sum()
        ),
    )

    if arguments.aggregate_only:
        aggregate_completed_runs()
        print(
            "STATUS: SCRIPT 167 AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 167 DRY RUN COMPLETE"
        )
        return

    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device("cuda:0")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    else:
        device = torch.device("cpu")

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    AGGREGATE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_runs = 0

    for row in plan.to_dict(
        orient="records"
    ):
        run_id = str(
            row["run_id"]
        )

        if backend.run_complete(
            run_id
        ):
            print(
                f"SKIP VERIFIED: {run_id}",
                flush=True,
            )
            continue

        execute_run(
            pd.Series(row),
            observations,
            device,
        )

        new_runs += 1

        if (
            not arguments.worker_only
            and arguments.aggregate_every > 0
            and new_runs
            % arguments.aggregate_every
            == 0
        ):
            aggregate_completed_runs()

        if (
            arguments.max_new_runs > 0
            and new_runs
            >= arguments.max_new_runs
        ):
            if not arguments.worker_only:
                aggregate_completed_runs()

            print(
                "STATUS: SCRIPT 167 PARTIAL RUN COMPLETE"
            )
            return

    if arguments.worker_only:
        print(
            "STATUS: SCRIPT 167 WORKER PARTITION COMPLETE"
        )
        return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 167 CORRECTIVE DRUG-REPRESENTATION "
        "SCREEN COMPLETE"
    )


if __name__ == "__main__":
    main()
