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
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()

SCRIPT165_PATH = PROJECT / "src/mic_transfer/final_genome_model.py"

EXPECTED_SCRIPT165_SHA256 = (
    "79878378e3aec6c7af84874955c7985c478d238cb976b060cc67e1eec1170f2f"
)

SCRIPT172_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script172_successful_corrective_architecture_preregistration_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_v2/"
      "corrective_architecture_new_run_plan_v2.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_v2/"
      "corrective_architecture_new_configuration_registry_v2.tsv"
)

ALL_CANDIDATE_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_v2/"
      "corrective_architecture_all_candidate_registry_v2.tsv"
)

DRUG_RANKING_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_aggregate_v1/"
      "complete_corrective_drug_representation_and_fusion_ranking_v1.tsv"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_runs_v2"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_runs_v2"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_aggregate_v2"
)

EXPECTED_RUNS = 90
CROSS_MODAL_BILINEAR_RANK: int

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
        raise FileNotFoundError(
            SCRIPT165_PATH
        )

    observed = sha256_file(
        SCRIPT165_PATH
    )

    if observed != EXPECTED_SCRIPT165_SHA256:
        raise RuntimeError(
            "Script 165 SHA mismatch: "
            f"{observed}"
        )

    specification = importlib.util.spec_from_file_location(
        "corrective_final_genome_confirmation_165_for_architecture",
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


def write_sha_manifest(
    paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            {candidate.resolve() for candidate in paths},
            key=lambda value: value.as_posix(),
        ):
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


CONFIGURATION_REGISTRY = read_tsv(
    CONFIGURATION_REGISTRY_PATH
) if CONFIGURATION_REGISTRY_PATH.is_file() else pd.DataFrame()

CONFIG_BY_ID: dict[str, dict[str, Any]] = {}

if not CONFIGURATION_REGISTRY.empty:
    for record in CONFIGURATION_REGISTRY.to_dict(
        orient="records"
    ):
        configuration_id = str(
            record["configuration_id"]
        )

        if configuration_id in CONFIG_BY_ID:
            raise RuntimeError(
                f"Duplicate architecture configuration ID: {configuration_id}"
            )

        CONFIG_BY_ID[
            configuration_id
        ] = record


class CorrectiveArchitectureNetwork(
    final165.FinalGenomeConfirmationNetwork
):
    def __init__(
        self,
        genome_dimension: int,
        drug_matrices: dict[str, np.ndarray],
        architecture_id: str,
        spec: dict[str, Any],
    ) -> None:
        if not drug_matrices:
            raise RuntimeError(
                "No drug views supplied."
            )

        self.drug_view_order = tuple(
            drug_matrices.keys()
        )
        self.architecture_id = str(
            architecture_id
        )

        self.drug_view_fusion_method = str(
            spec.get(
                "drug_view_fusion_method",
                "",
            )
        )

        if self.drug_view_fusion_method not in {
            "single_view",
            "raw_single_encoder_concatenation",
        }:
            raise RuntimeError(
                "Unsupported selected within-drug fusion method for the "
                "six-way architecture screen: "
                f"{self.drug_view_fusion_method}"
            )

        if (
            self.drug_view_fusion_method == "single_view"
            and len(drug_matrices) != 1
        ):
            raise RuntimeError(
                "single_view requires exactly one drug matrix."
            )

        if (
            self.drug_view_fusion_method
            == "raw_single_encoder_concatenation"
            and len(drug_matrices) < 2
        ):
            raise RuntimeError(
                "raw_single_encoder_concatenation requires multiple views."
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

        latent = int(
            spec["latent_width"]
        )
        dropout = float(
            spec["dropout"]
        )
        fusion_multiplier = int(
            spec[
                "fusion_hidden_multiplier"
            ]
        )
        interaction_hidden = max(
            latent,
            fusion_multiplier * latent,
        )

        if (
            self.architecture_id
            == "additive_linear"
        ):
            # Remove the projected MLP inherited from the parent. The
            # resulting predictor has only independent genome and drug terms.
            self.interaction_head = nn.Identity()

        elif (
            self.architecture_id
            == "dual_tower_explicit_interaction"
        ):
            self.cross_genome_tower = nn.Sequential(
                nn.Linear(
                    latent,
                    latent,
                ),
                nn.LayerNorm(latent),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.cross_drug_tower = nn.Sequential(
                nn.Linear(
                    latent,
                    latent,
                ),
                nn.LayerNorm(latent),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.interaction_head = nn.Sequential(
                nn.Linear(
                    4 * latent,
                    2 * latent,
                ),
                nn.LayerNorm(
                    2 * latent
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    2 * latent,
                    latent,
                ),
                nn.LayerNorm(
                    latent
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    latent,
                    1,
                    bias=False,
                ),
            )

        elif (
            self.architecture_id
            == "cross_modal_gmu"
        ):
            self.gmu_genome_projection = nn.Linear(
                latent,
                latent,
            )
            self.gmu_drug_projection = nn.Linear(
                latent,
                latent,
            )
            self.gmu_gate = nn.Linear(
                2 * latent,
                latent,
            )
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(latent),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    latent,
                    1,
                    bias=False,
                ),
            )

        elif (
            self.architecture_id
            == "cross_modal_low_rank_bilinear"
        ):
            self.cross_genome_bilinear = nn.Linear(
                latent,
                CROSS_MODAL_BILINEAR_RANK,
                bias=False,
            )
            self.cross_drug_bilinear = nn.Linear(
                latent,
                CROSS_MODAL_BILINEAR_RANK,
                bias=False,
            )
            self.cross_bilinear_to_latent = nn.Linear(
                CROSS_MODAL_BILINEAR_RANK,
                latent,
                bias=False,
            )
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(latent),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    latent,
                    1,
                    bias=False,
                ),
            )

        elif (
            self.architecture_id
            == "drug_to_genome_film"
        ):
            self.film_gamma = nn.Linear(
                latent,
                latent,
            )
            self.film_beta = nn.Linear(
                latent,
                latent,
            )
            self.film_norm = nn.LayerNorm(
                latent
            )
            self.interaction_head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    latent,
                    1,
                    bias=False,
                ),
            )

        else:
            raise RuntimeError(
                "Unsupported new architecture ID: "
                f"{self.architecture_id}"
            )

        if self.architecture_id != "additive_linear":
            final_layer = self.interaction_head[-1]
            if not isinstance(
                final_layer,
                nn.Linear,
            ):
                raise RuntimeError(
                    "Interaction head must end in Linear."
                )

            nn.init.zeros_(
                final_layer.weight
            )

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if set(
            drug_views.keys()
        ) != set(
            self.drug_view_order
        ):
            raise RuntimeError(
                "Drug-view mismatch."
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

        if self.architecture_id == "additive_linear":
            return additive

        if (
            self.architecture_id
            == "dual_tower_explicit_interaction"
        ):
            genome_tower = (
                self.cross_genome_tower(
                    genome_latent
                )
            )
            drug_tower = (
                self.cross_drug_tower(
                    drug_latent
                )
            )
            interaction_features = torch.cat(
                [
                    genome_tower,
                    drug_tower,
                    genome_tower
                    * drug_tower,
                    torch.abs(
                        genome_tower
                        - drug_tower
                    ),
                ],
                dim=1,
            )

        elif (
            self.architecture_id
            == "cross_modal_gmu"
        ):
            genome_candidate = torch.tanh(
                self.gmu_genome_projection(
                    genome_latent
                )
            )
            drug_candidate = torch.tanh(
                self.gmu_drug_projection(
                    drug_latent
                )
            )
            gate = torch.sigmoid(
                self.gmu_gate(
                    torch.cat(
                        [
                            genome_latent,
                            drug_latent,
                        ],
                        dim=1,
                    )
                )
            )
            interaction_features = (
                gate * genome_candidate
                + (1.0 - gate)
                * drug_candidate
            )

        elif (
            self.architecture_id
            == "cross_modal_low_rank_bilinear"
        ):
            low_rank = (
                self.cross_genome_bilinear(
                    genome_latent
                )
                * self.cross_drug_bilinear(
                    drug_latent
                )
            )
            interaction_features = (
                self.cross_bilinear_to_latent(
                    low_rank
                )
            )

        elif (
            self.architecture_id
            == "drug_to_genome_film"
        ):
            gamma = torch.tanh(
                self.film_gamma(
                    drug_latent
                )
            )
            beta = self.film_beta(
                drug_latent
            )
            interaction_features = (
                self.film_norm(
                    (1.0 + gamma)
                    * genome_latent
                    + beta
                )
            )

        else:
            raise AssertionError(
                self.architecture_id
            )

        residual = self.interaction_head(
            interaction_features
        ).squeeze(1)

        return additive + residual


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
            "Current corrective architecture specification is unset."
        )

    accepted = {
        "additive_linear",
        "dual_tower_explicit_interaction",
        "cross_modal_gmu",
        "cross_modal_low_rank_bilinear",
        "drug_to_genome_film",
    }

    if architecture_id not in accepted:
        raise RuntimeError(
            "Unexpected architecture ID: "
            f"{architecture_id}"
        )

    model = CorrectiveArchitectureNetwork(
        genome_dimension=genome_matrix.shape[1],
        drug_matrices=drug_matrices,
        architecture_id=architecture_id,
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
            "Model parameter count unavailable."
        )

    result_directory = (
        RESULT_ROOT
        / run_id
    )
    metadata_directory = (
        METADATA_ROOT
        / run_id
    )

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
        "model_parameter_count": (
            CURRENT_PARAMETER_COUNT
        ),
        "corrective_analysis_stage": (
            "corrective_cross_modal_architecture_screen_v2"
        ),
        "cross_modal_bilinear_rank": (
            CROSS_MODAL_BILINEAR_RANK
        ),
        "drug_representation": run_row[
            "drug_representation"
        ],
        "drug_view_fusion_method": run_row[
            "drug_view_fusion_method"
        ],
        "drug_view_low_rank": run_row.get(
            "drug_view_low_rank",
            "0",
        ),
        "cross_modal_architecture": run_row[
            "cross_modal_architecture"
        ],
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

    configuration_id = str(
        run_row[
            "configuration_id"
        ]
    )

    if configuration_id not in CONFIG_BY_ID:
        raise RuntimeError(
            "Unregistered architecture configuration: "
            f"{configuration_id}"
        )

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

    CURRENT_SPEC = dict(
        final165.CONFIG_BY_REPRESENTATION[
            representation
        ]
    )
    CURRENT_SPEC.update(
        CONFIG_BY_ID[
            configuration_id
        ]
    )

    final165.CURRENT_SPEC = (
        CURRENT_SPEC
    )

    final165.set_current_hyperparameters(
        CURRENT_SPEC
    )

    summary = final165.ORIGINAL_EXECUTE_RUN(
        run_row,
        observations,
        device,
    )

    patch_run_outputs(
        str(
            run_row[
                "run_id"
            ]
        ),
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
    new_configuration_path = (
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )

    for path in [
        all_runs_path,
        new_configuration_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = read_tsv(
        RUN_PLAN_PATH
    )
    registry = read_tsv(
        CONFIGURATION_REGISTRY_PATH
    )
    all_candidates = read_tsv(
        ALL_CANDIDATE_REGISTRY_PATH
    )

    all_runs = read_tsv(
        all_runs_path
    )
    all_runs[
        "macro_rmse"
    ] = pd.to_numeric(
        all_runs[
            "macro_rmse"
        ],
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
            "Every paired seed must contain two directions."
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

    worst_records: list[
        dict[str, Any]
    ] = []

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
        record[
            "seed_count"
        ] = group[
            "seed"
        ].nunique()
        record[
            "worst_direction_macro_rmse_mean"
        ] = safe_mean(values)
        record[
            "worst_direction_macro_rmse_sd"
        ] = sample_sd(values)
        worst_records.append(record)

    worst_summary = pd.DataFrame(
        worst_records
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

    new_summary = read_tsv(
        new_configuration_path
    )

    metadata_columns = [
        column
        for column in registry.columns
        if column not in {
            "bidirectional_macro_rmse_mean",
            "bidirectional_macro_rmse_sd",
        }
    ]

    metadata = registry[
        metadata_columns
    ].drop_duplicates(
        "configuration_id"
    )

    new_summary = new_summary.merge(
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

    new_summary = new_summary.merge(
        metadata,
        on="configuration_id",
        how="left",
        suffixes=("", "_registry"),
        validate="one_to_one",
    )

    parameter_records: list[
        dict[str, Any]
    ] = []

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
            / str(
                row["run_id"]
            )
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

    new_summary = new_summary.merge(
        parameters,
        on="configuration_id",
        how="left",
        validate="one_to_one",
    )
    new_summary[
        "result_source"
    ] = "new_script173_training"

    projected_registry = all_candidates.loc[
        all_candidates[
            "cross_modal_architecture"
        ].astype(str).eq(
            "projected_concatenation_MLP"
        )
    ].copy()

    if len(projected_registry) != 3:
        raise RuntimeError(
            "Expected three projected baseline registry rows."
        )

    drug_ranking = read_tsv(
        DRUG_RANKING_PATH
    )

    projected = projected_registry[
        [
            column
            for column in projected_registry.columns
            if column not in {
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
                "worst_direction_macro_rmse_mean",
                "worst_direction_macro_rmse_sd",
                "model_parameter_count",
            }
        ]
    ].merge(
        drug_ranking,
        on=[
            "outer_target_code",
            "drug_representation",
            "drug_view_fusion_method",
        ],
        how="left",
        suffixes=("", "_drug_screen"),
        validate="one_to_one",
    )

    projected[
        "cross_modal_architecture"
    ] = "projected_concatenation_MLP"
    projected[
        "result_source"
    ] = (
        "reused_script171_selected_drug_projected_baseline"
    )

    complete = pd.concat(
        [
            projected,
            new_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    if len(complete) != 18:
        raise RuntimeError(
            f"Expected 18 complete architecture candidates; "
            f"observed {len(complete)}."
        )

    for column in [
        "bidirectional_macro_rmse_mean",
        "worst_direction_macro_rmse_mean",
        "model_parameter_count",
    ]:
        complete[column] = pd.to_numeric(
            complete[column],
            errors="raise",
        )

    complete = complete.sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "worst_direction_macro_rmse_mean",
            "model_parameter_count",
            "cross_modal_architecture",
        ],
        kind="stable",
    ).reset_index(drop=True)

    complete[
        "selection_rank"
    ] = (
        complete.groupby(
            "outer_target_code"
        )
        .cumcount()
        .add(1)
    )

    complete_path = (
        AGGREGATE_ROOT
        / "complete_six_way_architecture_ranking.tsv"
    )
    write_tsv(
        complete,
        complete_path,
    )

    selected = (
        complete.loc[
            complete[
                "selection_rank"
            ].eq(1)
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
                    "drug_view_fusion_method",
                    "drug_view_low_rank",
                    "cross_modal_architecture",
                    "bidirectional_macro_rmse_mean",
                    "bidirectional_macro_rmse_sd",
                    "worst_direction_macro_rmse_mean",
                    "worst_direction_macro_rmse_sd",
                    "model_parameter_count",
                    "result_source",
                ]
                if column in complete.columns
            ]
        ]
        .sort_values(
            "outer_target_code"
        )
        .reset_index(drop=True)
    )

    if len(selected) != 3:
        raise RuntimeError(
            "Expected one architecture winner per outer target."
        )

    selected_path = (
        AGGREGATE_ROOT
        / "selected_corrective_architecture_registry.tsv"
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
                    "minimum mean bidirectional macro RMSE within each outer "
                    "target; exact ties resolved by lower worst-direction "
                    "mean and then lower parameter count"
                ),
            },
            {
                "item": "additive_control_definition",
                "value": (
                    "intercept plus independent linear genome-latent and "
                    "drug-latent effects; no genome-drug interaction term"
                ),
            },
            {
                "item": "projected_baseline_source",
                "value": (
                    "Script 171 final selected drug configuration, with "
                    "the same frozen inputs, within-drug fusion and seeds"
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
        new_configuration_path,
        AGGREGATE_ROOT
        / "all_per_antibiotic_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "per_antibiotic_three_seed_mean_sd.tsv",
        worst_seed_path,
        worst_summary_path,
        complete_path,
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
    backend.ARCHITECTURE_NAME_TO_ID[
        "additive_linear"
    ] = "additive_linear"

    final165.RUN_PLAN_PATH = RUN_PLAN_PATH
    final165.RESULT_ROOT = RESULT_ROOT
    final165.METADATA_ROOT = METADATA_ROOT
    final165.AGGREGATE_ROOT = AGGREGATE_ROOT
    final165.EXPECTED_RUNS = EXPECTED_RUNS

    backend.__file__ = str(
        Path(__file__).resolve()
    )
    backend.RUN_PLAN_PATH = RUN_PLAN_PATH
    backend.SCRIPT142_OUTPUTS_PATH = (
        SCRIPT172_FREEZE
    )
    backend.SCRIPT142_FROZEN_PATH = (
        SCRIPT172_FREEZE
    )
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
    backend.execute_run = execute_run
    backend.aggregate_completed_runs = (
        aggregate_completed_runs
    )

    final165.build_model = build_model
    final165.execute_run = execute_run
    final165.aggregate_completed_runs = (
        aggregate_completed_runs
    )


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
        "--architecture",
        choices=[
            "all",
            "additive_linear",
            "dual_tower_interaction",
            "cross_modal_GMU",
            "low_rank_bilinear",
            "drug_to_genome_FiLM",
        ],
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
        SCRIPT172_FREEZE,
        RUN_PLAN_PATH,
        CONFIGURATION_REGISTRY_PATH,
        ALL_CANDIDATE_REGISTRY_PATH,
        DRUG_RANKING_PATH,
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
        SCRIPT172_FREEZE
    )

    if not CONFIG_BY_ID:
        raise RuntimeError(
            "Architecture configuration registry is empty."
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

    if arguments.architecture != "all":
        plan = plan.loc[
            plan[
                "cross_modal_architecture"
            ].eq(
                arguments.architecture
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
            "cross_modal_architecture",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    plan[
        "already_complete"
    ] = [
        backend.run_complete(
            str(run_id)
        )
        for run_id
        in plan[
            "run_id"
        ]
    ]

    print(
        "===== SCRIPT 173 RUN PLAN ====="
    )

    print(
        plan.groupby(
            [
                "outer_target_code",
                "cross_modal_architecture",
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
        .to_string(
            index=False
        )
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
            (
                ~plan[
                    "already_complete"
                ]
            ).sum()
        ),
    )

    if arguments.aggregate_only:
        aggregate_completed_runs()
        print(
            "STATUS: SCRIPT 173 AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 173 DRY RUN COMPLETE"
        )
        return

    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda:0"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(
            "high"
        )
    else:
        device = torch.device(
            "cpu"
        )

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
            and arguments.aggregate_every
            > 0
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
                "STATUS: SCRIPT 173 PARTIAL RUN COMPLETE"
            )
            return

    if arguments.worker_only:
        print(
            "STATUS: SCRIPT 173 WORKER PARTITION COMPLETE"
        )
        return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 173 CORRECTIVE ARCHITECTURE "
        "SCREEN COMPLETE"
    )


