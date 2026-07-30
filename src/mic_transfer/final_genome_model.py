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

BACKEND_PATH = PROJECT / "src/mic_transfer/model_backend.py"

EXPECTED_BACKEND_SHA256 = (
    "13d11f169cb9b03b7566d5bc8775c0a14295c672a620321aa6ac2a41e8c69628"
)

SCRIPT164_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script164_successful_corrective_final_confirmation_preregistration_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_v1/"
      "corrective_final_genome_run_plan_v1.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_v1/"
      "corrective_final_genome_configuration_registry_v1.tsv"
)

IMPLEMENTATION_SPEC_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_v1/"
      "corrective_genome_variant_specification_v1.tsv"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_runs_v1"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_runs_v1"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_final_genome_confirmation_aggregate_v1"
)

EXPECTED_RUNS = 90

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


def load_backend():
    if not BACKEND_PATH.is_file():
        raise FileNotFoundError(BACKEND_PATH)

    observed = sha256_file(BACKEND_PATH)

    if observed != EXPECTED_BACKEND_SHA256:
        raise RuntimeError(
            "Script 143 backend SHA mismatch: "
            f"{observed}"
        )

    specification = importlib.util.spec_from_file_location(
        "corrective_final_backend_143",
        BACKEND_PATH,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load Script 143 backend."
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    return module


backend = load_backend()


class FreshViewEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        latent_width: int,
        hidden_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()

        hidden_width = max(
            latent_width,
            int(
                hidden_multiplier
                * latent_width
            ),
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_width,
            ),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_width,
                latent_width,
            ),
            nn.LayerNorm(latent_width),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


class FinalGenomeConfirmationNetwork(nn.Module):
    def __init__(
        self,
        genome_dimension: int,
        drug_dimension: int,
        spec: dict[str, Any],
    ) -> None:
        super().__init__()

        self.variant = str(
            spec[
                "corrective_genome_variant"
            ]
        )

        latent = int(
            spec["latent_width"]
        )
        dropout = float(
            spec["dropout"]
        )
        genome_multiplier = int(
            spec[
                "genome_hidden_multiplier"
            ]
        )
        drug_multiplier = int(
            spec[
                "drug_hidden_multiplier"
            ]
        )
        fusion_multiplier = int(
            spec[
                "fusion_hidden_multiplier"
            ]
        )

        self.kmer_dimension = int(
            spec[
                "selected_kmer_dimension"
            ]
        )
        self.amr_dimension = int(
            spec[
                "common_amr_dimension"
            ]
        )
        self.low_rank = int(
            spec[
                "low_rank_interaction_rank"
            ]
        )

        if self.variant in {
            "selected_kmer_only",
            "common_AMR_only",
            "raw_kmer_plus_AMR_single_encoder",
        }:
            self.genome_encoder = FreshViewEncoder(
                input_dimension=genome_dimension,
                latent_width=latent,
                hidden_multiplier=genome_multiplier,
                dropout=dropout,
            )

        elif self.variant in {
            "separate_encoder_projected_kmer_plus_AMR",
            "separate_encoder_low_rank_kmer_plus_AMR",
        }:
            expected = (
                self.kmer_dimension
                + self.amr_dimension
            )

            if genome_dimension != expected:
                raise RuntimeError(
                    "Fused genome dimension mismatch: "
                    f"{genome_dimension} != {expected}"
                )

            self.kmer_encoder = FreshViewEncoder(
                input_dimension=(
                    self.kmer_dimension
                ),
                latent_width=latent,
                hidden_multiplier=genome_multiplier,
                dropout=dropout,
            )

            self.amr_encoder = FreshViewEncoder(
                input_dimension=(
                    self.amr_dimension
                ),
                latent_width=latent,
                hidden_multiplier=genome_multiplier,
                dropout=dropout,
            )

            fusion_hidden = max(
                latent,
                int(
                    fusion_multiplier
                    * latent
                ),
            )

            self.genome_base_fusion = nn.Sequential(
                nn.Linear(
                    2 * latent,
                    fusion_hidden,
                ),
                nn.LayerNorm(
                    fusion_hidden
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    fusion_hidden,
                    latent,
                ),
                nn.LayerNorm(latent),
                nn.GELU(),
            )

            if (
                self.variant
                == "separate_encoder_low_rank_kmer_plus_AMR"
            ):
                if self.low_rank <= 0:
                    raise RuntimeError(
                        "Low-rank final variant requires positive rank."
                    )

                self.kmer_low_rank = nn.Linear(
                    latent,
                    self.low_rank,
                    bias=False,
                )
                self.amr_low_rank = nn.Linear(
                    latent,
                    self.low_rank,
                    bias=False,
                )
                self.low_rank_to_latent = nn.Linear(
                    self.low_rank,
                    latent,
                    bias=False,
                )
                self.genome_residual_norm = nn.LayerNorm(
                    latent
                )

        else:
            raise RuntimeError(
                f"Unsupported final genome variant: {self.variant}"
            )

        self.drug_encoder = FreshViewEncoder(
            input_dimension=drug_dimension,
            latent_width=latent,
            hidden_multiplier=drug_multiplier,
            dropout=dropout,
        )

        self.genome_head = nn.Linear(
            latent,
            1,
            bias=False,
        )
        self.drug_head = nn.Linear(
            latent,
            1,
            bias=False,
        )
        self.intercept = nn.Parameter(
            torch.zeros(
                1,
                dtype=torch.float32,
            )
        )

        interaction_hidden = max(
            latent,
            int(
                fusion_multiplier
                * latent
            ),
        )

        self.interaction_head = nn.Sequential(
            nn.Linear(
                2 * latent,
                interaction_hidden,
            ),
            nn.LayerNorm(
                interaction_hidden
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                interaction_hidden,
                latent,
            ),
            nn.LayerNorm(latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                latent,
                1,
                bias=False,
            ),
        )

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

    def encode_genome(
        self,
        genome: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant in {
            "selected_kmer_only",
            "common_AMR_only",
            "raw_kmer_plus_AMR_single_encoder",
        }:
            return self.genome_encoder(
                genome
            )

        kmer = genome[
            :,
            : self.kmer_dimension,
        ]
        amr = genome[
            :,
            self.kmer_dimension :,
        ]

        kmer_latent = self.kmer_encoder(
            kmer
        )
        amr_latent = self.amr_encoder(
            amr
        )

        base = self.genome_base_fusion(
            torch.cat(
                [
                    kmer_latent,
                    amr_latent,
                ],
                dim=1,
            )
        )

        if (
            self.variant
            == "separate_encoder_projected_kmer_plus_AMR"
        ):
            return base

        interaction = (
            self.kmer_low_rank(
                kmer_latent
            )
            * self.amr_low_rank(
                amr_latent
            )
        )

        residual = self.low_rank_to_latent(
            interaction
        )

        return torch.nn.functional.gelu(
            self.genome_residual_norm(
                base + residual
            )
        )

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if len(drug_views) != 1:
            raise RuntimeError(
                "Final confirmation requires exactly one Morgan view."
            )

        drug_view_id = next(
            iter(drug_views)
        )

        if drug_view_id.lower() != "morgan":
            raise RuntimeError(
                "Final confirmation requires Morgan; "
                f"observed {drug_view_id!r}."
            )

        genome_latent = self.encode_genome(
            genome
        )
        drug_latent = self.drug_encoder(
            drug_views[
                drug_view_id
            ]
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


def load_configuration_registry() -> dict[str, dict[str, Any]]:
    frame = read_tsv(
        CONFIGURATION_REGISTRY_PATH
    )

    if len(frame) != 15:
        raise RuntimeError(
            f"Expected 15 final configuration rows; "
            f"observed {len(frame)}."
        )

    integer_fields = {
        "low_rank_interaction_rank",
        "selected_kmer_dimension",
        "common_amr_dimension",
        "genome_dimension",
        "latent_width",
        "genome_hidden_multiplier",
        "drug_hidden_multiplier",
        "fusion_hidden_multiplier",
        "batch_size",
        "maximum_epochs",
        "early_stopping_patience",
    }

    float_fields = {
        "dropout",
        "learning_rate",
        "weight_decay",
        "minimum_rmse_improvement",
        "gradient_clip_norm",
    }

    registry: dict[str, dict[str, Any]] = {}

    for record in frame.to_dict(
        orient="records"
    ):
        representation = str(
            record[
                "genome_representation"
            ]
        )

        spec: dict[str, Any] = dict(
            record
        )

        for field in integer_fields:
            spec[field] = int(
                float(
                    spec[field]
                )
            )

        for field in float_fields:
            spec[field] = float(
                spec[field]
            )

        spec[
            "genome_matrix_path"
        ] = project_path(
            str(
                spec[
                    "genome_matrix_path"
                ]
            )
        )

        if representation in registry:
            raise RuntimeError(
                f"Duplicate final representation: {representation}"
            )

        registry[
            representation
        ] = spec

    return registry


CONFIG_BY_REPRESENTATION = (
    load_configuration_registry()
)


def set_current_hyperparameters(
    spec: dict[str, Any],
) -> None:
    backend.LATENT_WIDTH = int(
        spec["latent_width"]
    )
    backend.DROPOUT = float(
        spec["dropout"]
    )
    backend.LEARNING_RATE = float(
        spec["learning_rate"]
    )
    backend.WEIGHT_DECAY = float(
        spec["weight_decay"]
    )
    backend.BATCH_SIZE = int(
        spec["batch_size"]
    )
    backend.MAX_EPOCHS = int(
        spec["maximum_epochs"]
    )
    backend.EARLY_STOPPING_PATIENCE = int(
        spec[
            "early_stopping_patience"
        ]
    )
    backend.MINIMUM_RMSE_IMPROVEMENT = float(
        spec[
            "minimum_rmse_improvement"
        ]
    )
    backend.GRADIENT_CLIP_NORM = float(
        spec[
            "gradient_clip_norm"
        ]
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
            "Current final-confirmation specification is unset."
        )

    if (
        architecture_id
        != "cross_modal_projected_concat"
    ):
        raise RuntimeError(
            "Final confirmation requires projected-concatenation "
            "genome-drug architecture."
        )

    if len(drug_matrices) != 1:
        raise RuntimeError(
            "Final confirmation requires exactly one Morgan matrix."
        )

    drug_view_id = next(
        iter(drug_matrices)
    )

    if drug_view_id.lower() != "morgan":
        raise RuntimeError(
            "Final confirmation requires Morgan; "
            f"observed {drug_view_id!r}."
        )

    model = FinalGenomeConfirmationNetwork(
        genome_dimension=(
            genome_matrix.shape[1]
        ),
        drug_dimension=(
            drug_matrices[
                drug_view_id
            ].shape[1]
        ),
        spec=CURRENT_SPEC,
    ).to(device)

    CURRENT_PARAMETER_COUNT = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    with torch.no_grad():
        model.intercept.fill_(
            float(
                target_mean
            )
        )

    return model


def patch_run_outputs(
    run_id: str,
    spec: dict[str, Any],
) -> None:
    if CURRENT_PARAMETER_COUNT is None:
        raise RuntimeError(
            "Model parameter count is unavailable."
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
        "selected_kmer_dimension": spec[
            "selected_kmer_dimension"
        ],
        "common_amr_dimension": spec[
            "common_amr_dimension"
        ],
        "model_parameter_count": (
            CURRENT_PARAMETER_COUNT
        ),
        "corrective_analysis_stage": (
            "Stage_C_fresh_seed_final_genome_confirmation"
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


ORIGINAL_EXECUTE_RUN = (
    backend.execute_run
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

    if representation not in CONFIG_BY_REPRESENTATION:
        raise RuntimeError(
            f"Unregistered final representation: {representation}"
        )

    CURRENT_SPEC = CONFIG_BY_REPRESENTATION[
        representation
    ]

    set_current_hyperparameters(
        CURRENT_SPEC
    )

    summary = ORIGINAL_EXECUTE_RUN(
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
    )

    return summary


ORIGINAL_AGGREGATE = (
    backend.aggregate_completed_runs
)


def safe_mean(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    return (
        float(
            numeric.mean()
        )
        if len(numeric)
        else float("nan")
    )


def sample_sd(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    return (
        float(
            numeric.std(
                ddof=1
            )
        )
        if len(numeric) >= 2
        else float("nan")
    )


def aggregate_completed_runs() -> None:
    ORIGINAL_AGGREGATE()

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

    metadata = (
        plan[
            [
                "configuration_id",
                "outer_target_code",
                "shared_hp_id",
                "corrective_genome_variant",
                "low_rank_interaction_rank",
            ]
        ]
        .drop_duplicates()
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
        errors="coerce",
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

    worst_seed_path = (
        AGGREGATE_ROOT
        / "seedwise_worst_direction_macro_rmse.tsv"
    )

    backend.write_tsv(
        worst_seed.sort_values(
            [
                "outer_target_code",
                "configuration_id",
                "seed",
            ]
        ),
        worst_seed_path,
    )

    worst_summary_records: list[
        dict[str, object]
    ] = []

    for keys, group in worst_seed.groupby(
        [
            "outer_target_code",
            "configuration_id",
            "genome_representation",
            "drug_representation",
            "cross_modal_architecture",
        ],
        dropna=False,
    ):
        record = {
            key: value
            for key, value
            in zip(
                [
                    "outer_target_code",
                    "configuration_id",
                    "genome_representation",
                    "drug_representation",
                    "cross_modal_architecture",
                ],
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

    backend.write_tsv(
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
        on=[
            "outer_target_code",
            "configuration_id",
        ],
        how="left",
        validate="one_to_one",
    )

    parameter_records: list[dict[str, object]] = []

    for configuration_id in configuration[
        "configuration_id"
    ].astype(str):
        matches = sorted(
            RESULT_ROOT.glob(
                f"{configuration_id}__*/run_summary.tsv"
            )
        )

        if not matches:
            continue

        run_summary = backend.read_tsv(
            matches[0]
        )

        if (
            "model_parameter_count"
            not in run_summary.columns
        ):
            continue

        parameter_records.append(
            {
                "configuration_id": configuration_id,
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
    ).drop_duplicates(
        "configuration_id"
    )

    if len(parameters) != 15:
        raise RuntimeError(
            "Could not recover parameter counts for all 15 "
            "final configurations."
        )

    configuration = configuration.merge(
        parameters,
        on="configuration_id",
        how="left",
        validate="one_to_one",
    )

    numeric_columns = [
        "bidirectional_macro_rmse_mean",
        "worst_direction_macro_rmse_mean",
    ]

    for column in numeric_columns:
        configuration[column] = pd.to_numeric(
            configuration[column],
            errors="raise",
        )

    configuration = configuration.sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "worst_direction_macro_rmse_mean",
            "model_parameter_count",
        ]
    ).reset_index(drop=True)

    configuration[
        "selection_rank"
    ] = (
        configuration.groupby(
            "outer_target_code"
        )
        .cumcount()
        .add(1)
    )

    backend.write_tsv(
        configuration,
        configuration_path,
    )

    selected = (
        configuration.loc[
            configuration[
                "selection_rank"
            ].eq(1)
        ][
            [
                "outer_target_code",
                "corrective_genome_variant",
                "shared_hp_id",
                "low_rank_interaction_rank",
                "bidirectional_macro_rmse_mean",
                "bidirectional_macro_rmse_sd",
                "worst_direction_macro_rmse_mean",
                "worst_direction_macro_rmse_sd",
                "model_parameter_count",
            ]
        ]
        .sort_values(
            "outer_target_code"
        )
        .reset_index(drop=True)
    )

    if len(selected) != 3:
        raise RuntimeError(
            "Expected one final genome winner per outer target."
        )

    selected_path = (
        AGGREGATE_ROOT
        / "selected_final_genome_representation_registry.tsv"
    )

    backend.write_tsv(
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
                "item": "final_selection_rule",
                "value": (
                    "minimum mean bidirectional macro RMSE within each outer "
                    "target; exact numerical ties resolved by lower worst-"
                    "direction mean and then lower parameter count"
                ),
            },
            {
                "item": "confirmation_seed_count",
                "value": "3",
            },
            {
                "item": "sample_sd",
                "value": "ddof=1",
            },
            {
                "item": "parameter_count_reporting",
                "value": "reported for every final genome variant",
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "held-out outer-target MIC labels not used"
                ),
            },
        ]
    )

    backend.write_tsv(
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

    backend.write_sha_manifest(
        aggregate_paths,
        manifest_path,
    )
    backend.verify_sha_manifest(
        manifest_path
    )


def configure_backend() -> None:
    backend.__file__ = str(
        Path(__file__).resolve()
    )
    backend.RUN_PLAN_PATH = (
        RUN_PLAN_PATH
    )
    backend.SCRIPT142_OUTPUTS_PATH = (
        SCRIPT164_FREEZE
    )
    backend.SCRIPT142_FROZEN_PATH = (
        SCRIPT164_FREEZE
    )
    backend.IMPLEMENTATION_BUNDLE_PATH = (
        IMPLEMENTATION_SPEC_PATH
    )
    backend.EXPECTED_IMPLEMENTATION_BUNDLE_SHA256 = (
        sha256_file(
            IMPLEMENTATION_SPEC_PATH
        )
    )
    backend.RESULT_ROOT = (
        RESULT_ROOT
    )
    backend.METADATA_ROOT = (
        METADATA_ROOT
    )
    backend.AGGREGATE_ROOT = (
        AGGREGATE_ROOT
    )
    backend.EXPECTED_RUNS = (
        EXPECTED_RUNS
    )
    backend.KMER_PATHS = {
        representation: Path(
            spec[
                "genome_matrix_path"
            ]
        )
        for representation, spec
        in CONFIG_BY_REPRESENTATION.items()
    }
    backend.build_model = build_model
    backend.execute_run = execute_run
    backend.aggregate_completed_runs = (
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
        "--variant",
        choices=[
            "all",
            "kmer",
            "amr",
            "raw_concat",
            "projected",
            "low_rank",
        ],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            20260805,
            20260806,
            20260807,
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
        BACKEND_PATH,
        SCRIPT164_FREEZE,
        RUN_PLAN_PATH,
        CONFIGURATION_REGISTRY_PATH,
        IMPLEMENTATION_SPEC_PATH,
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
                CONFIG_BY_REPRESENTATION.values()
            )
        ],
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verify_manifest(
        SCRIPT164_FREEZE
    )

    configure_backend()

    observations, plan = (
        backend.load_inputs()
    )

    if arguments.outer_target != "all":
        plan = plan.loc[
            plan[
                "outer_target_code"
            ].eq(
                arguments.outer_target
            )
        ]

    variant_map = {
        "kmer": "selected_kmer_only",
        "amr": "common_AMR_only",
        "raw_concat": (
            "raw_kmer_plus_AMR_single_encoder"
        ),
        "projected": (
            "separate_encoder_projected_kmer_plus_AMR"
        ),
        "low_rank": (
            "separate_encoder_low_rank_kmer_plus_AMR"
        ),
    }

    if arguments.variant != "all":
        plan = plan.loc[
            plan[
                "corrective_genome_variant"
            ].eq(
                variant_map[
                    arguments.variant
                ]
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
            "corrective_genome_variant",
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
        in plan["run_id"]
    ]

    print(
        "===== SCRIPT 165 RUN PLAN ====="
    )

    print(
        plan.groupby(
            [
                "outer_target_code",
                "corrective_genome_variant",
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
            "STATUS: SCRIPT 165 AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 165 DRY RUN COMPLETE"
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

        torch.backends.cuda.matmul.allow_tf32 = (
            True
        )
        torch.backends.cudnn.allow_tf32 = (
            True
        )
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
            arguments.max_new_runs
            > 0
            and new_runs
            >= arguments.max_new_runs
        ):
            if not arguments.worker_only:
                aggregate_completed_runs()

            print(
                "STATUS: SCRIPT 165 PARTIAL RUN COMPLETE"
            )
            return

    if arguments.worker_only:
        print(
            "STATUS: SCRIPT 165 WORKER PARTITION COMPLETE"
        )
        return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 165 CORRECTIVE FINAL "
        "GENOME CONFIRMATION COMPLETE"
    )


