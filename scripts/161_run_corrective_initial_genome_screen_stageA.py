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

BACKEND_PATH = (
    PROJECT
    / "scripts/"
      "143_run_nested_loso_full_kmer_grid.py"
)

EXPECTED_BACKEND_SHA256 = (
    "d82435bc05f13fcc330632e6e8b27460"
    "139d24ab812ef4c46a5741ddebb18b80"
)

SCRIPT160_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script160_successful_corrective_genome_screen_preregistration_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_v1/"
      "corrective_stageA_run_plan_v1.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_v1/"
      "corrective_stageA_configuration_registry_v1.tsv"
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
      "corrective_initial_genome_screen_stageA_runs_v1"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_runs_v1"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_initial_genome_screen_stageA_aggregate_v1"
)

EXPECTED_RUNS = 144

CURRENT_SPEC: dict[str, Any] | None = None


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
        "corrective_stageA_backend_143",
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


class CorrectiveGenomeScreenNetwork(nn.Module):
    def __init__(
        self,
        genome_dimension: int,
        drug_dimension: int,
        spec: dict[str, Any],
    ) -> None:
        super().__init__()

        self.spec = spec
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

        elif (
            self.variant
            == "separate_encoder_projected_kmer_plus_AMR"
        ):
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

            self.genome_fusion = nn.Sequential(
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

        else:
            raise RuntimeError(
                f"Unsupported Stage-A variant: {self.variant}"
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

        fusion_hidden = max(
            latent,
            int(
                fusion_multiplier
                * latent
            ),
        )

        self.interaction_head = nn.Sequential(
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
        if (
            self.variant
            != "separate_encoder_projected_kmer_plus_AMR"
        ):
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

        return self.genome_fusion(
            torch.cat(
                [
                    kmer_latent,
                    amr_latent,
                ],
                dim=1,
            )
        )

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if len(drug_views) != 1:
            raise RuntimeError(
                "Corrective Stage A requires exactly one "
                "Morgan drug view."
            )

        drug_view_id = next(
            iter(drug_views)
        )

        if drug_view_id.lower() != "morgan":
            raise RuntimeError(
                "Corrective Stage A requires the Morgan "
                f"drug view; observed {drug_view_id!r}."
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

    if len(frame) != 36:
        raise RuntimeError(
            f"Expected 36 configuration rows; observed {len(frame)}."
        )

    integer_fields = {
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

        spec: dict[
            str,
            Any,
        ] = dict(record)

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
                f"Duplicate genome representation: {representation}"
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
    if CURRENT_SPEC is None:
        raise RuntimeError(
            "Current Stage-A specification is unset."
        )

    if (
        architecture_id
        != "cross_modal_projected_concat"
    ):
        raise RuntimeError(
            "Corrective Stage A requires projected-concatenation architecture."
        )

    if len(drug_matrices) != 1:
        raise RuntimeError(
            "Corrective Stage A requires exactly one "
            "Morgan drug matrix."
        )

    drug_view_id = next(
        iter(drug_matrices)
    )

    if drug_view_id.lower() != "morgan":
        raise RuntimeError(
            "Corrective Stage A requires the Morgan "
            f"drug matrix; observed {drug_view_id!r}."
        )

    model = CorrectiveGenomeScreenNetwork(
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
        "selected_kmer_dimension": spec[
            "selected_kmer_dimension"
        ],
        "common_amr_dimension": spec[
            "common_amr_dimension"
        ],
        "genome_hidden_multiplier": spec[
            "genome_hidden_multiplier"
        ],
        "drug_hidden_multiplier": spec[
            "drug_hidden_multiplier"
        ],
        "fusion_hidden_multiplier": spec[
            "fusion_hidden_multiplier"
        ],
        "maximum_epochs": spec[
            "maximum_epochs"
        ],
        "early_stopping_patience": spec[
            "early_stopping_patience"
        ],
        "minimum_rmse_improvement": spec[
            "minimum_rmse_improvement"
        ],
        "gradient_clip_norm": spec[
            "gradient_clip_norm"
        ],
        "corrective_analysis_stage": "Stage_A_shared_hyperparameter_screen",
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
            f"Unregistered Stage-A representation: {representation}"
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
    paired_path = (
        AGGREGATE_ROOT
        / "bidirectional_seed_metrics.tsv"
    )
    configuration_path = (
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )

    for path in (
        all_runs_path,
        paired_path,
        configuration_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

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

    backend.write_tsv(
        configuration,
        configuration_path,
    )

    paired = read_tsv(
        paired_path
    )

    paired[
        "bidirectional_macro_rmse"
    ] = pd.to_numeric(
        paired[
            "bidirectional_macro_rmse"
        ],
        errors="coerce",
    )

    paired = paired.merge(
        metadata[
            [
                "outer_target_code",
                "configuration_id",
                "shared_hp_id",
                "corrective_genome_variant",
            ]
        ],
        on=[
            "outer_target_code",
            "configuration_id",
        ],
        how="left",
        validate="many_to_one",
    )

    balanced_seed = (
        paired.groupby(
            [
                "outer_target_code",
                "shared_hp_id",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            variant_count=(
                "corrective_genome_variant",
                "nunique",
            ),
            balanced_bidirectional_macro_rmse=(
                "bidirectional_macro_rmse",
                "mean",
            ),
        )
        .reset_index()
    )

    balanced_seed_complete = balanced_seed.loc[
        balanced_seed[
            "variant_count"
        ].eq(4)
    ].copy()

    balanced_seed_path = (
        AGGREGATE_ROOT
        / "shared_bundle_balanced_seed_metrics.tsv"
    )

    backend.write_tsv(
        balanced_seed.sort_values(
            [
                "outer_target_code",
                "shared_hp_id",
                "seed",
            ]
        ),
        balanced_seed_path,
    )

    ranking_records: list[
        dict[str, object]
    ] = []

    for keys, group in balanced_seed_complete.groupby(
        [
            "outer_target_code",
            "shared_hp_id",
        ],
        dropna=False,
    ):
        outer, hp_id = keys

        row = {
            "outer_target_code": outer,
            "shared_hp_id": hp_id,
            "seed_count": group[
                "seed"
            ].nunique(),
            "balanced_bidirectional_macro_rmse_mean": (
                safe_mean(
                    group[
                        "balanced_bidirectional_macro_rmse"
                    ]
                )
            ),
            "balanced_bidirectional_macro_rmse_sd": (
                sample_sd(
                    group[
                        "balanced_bidirectional_macro_rmse"
                    ]
                )
            ),
        }

        hp_metadata = metadata.loc[
            metadata[
                "outer_target_code"
            ].eq(outer)
            & metadata[
                "shared_hp_id"
            ].eq(hp_id)
        ].iloc[0]

        for column in [
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
            row[column] = (
                hp_metadata[column]
            )

        ranking_records.append(row)

    ranking_columns = [
        "outer_target_code",
        "shared_hp_id",
        "seed_count",
        "balanced_bidirectional_macro_rmse_mean",
        "balanced_bidirectional_macro_rmse_sd",
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
        "selection_rank",
    ]

    ranking = pd.DataFrame(
        ranking_records
    )

    if len(ranking):
        ranking["selection_rank"] = (
            ranking.groupby(
                "outer_target_code"
            )[
                "balanced_bidirectional_macro_rmse_mean"
            ]
            .rank(
                method="dense",
                ascending=True,
            )
            .astype(int)
        )

        ranking = ranking.sort_values(
            [
                "outer_target_code",
                "selection_rank",
                "shared_hp_id",
            ]
        ).reset_index(drop=True)

    else:
        ranking = pd.DataFrame(
            columns=ranking_columns
        )

    ranking_path = (
        AGGREGATE_ROOT
        / "corrective_shared_hyperparameter_bundle_ranking.tsv"
    )

    backend.write_tsv(
        ranking,
        ranking_path,
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
                "item": "bundle_selection_score",
                "value": (
                    "equal-weight average of bidirectional macro RMSE across "
                    "the four Stage-A genome variants within each outer target "
                    "and seed"
                ),
            },
            {
                "item": "selection_rule",
                "value": (
                    "minimum mean balanced bidirectional macro RMSE; "
                    "worst-direction remains secondary reporting"
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
        balanced_seed_path,
        ranking_path,
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
        SCRIPT160_FREEZE
    )
    backend.SCRIPT142_FROZEN_PATH = (
        SCRIPT160_FREEZE
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
        "--shared-hp-id",
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
        ],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            20260801,
            20260802,
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
        default=8,
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help=(
            "Run only the selected disjoint worker partition and do not "
            "write shared aggregate files. Run one aggregate-only pass "
            "after all workers finish."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    required = [
        BACKEND_PATH,
        SCRIPT160_FREEZE,
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
        SCRIPT160_FREEZE
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

    if arguments.shared_hp_id != "all":
        plan = plan.loc[
            plan[
                "shared_hp_id"
            ].eq(
                arguments.shared_hp_id
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
            "shared_hp_id",
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
        "===== SCRIPT 161 RUN PLAN ====="
    )

    print(
        plan.groupby(
            [
                "outer_target_code",
                "shared_hp_id",
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
            "STATUS: SCRIPT 161 "
            "AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 161 "
            "DRY RUN COMPLETE"
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
                "STATUS: SCRIPT 161 "
                "PARTIAL RUN COMPLETE"
            )
            return

    if arguments.worker_only:
        print(
            "STATUS: SCRIPT 161 WORKER PARTITION COMPLETE"
        )
        return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 161 CORRECTIVE "
        "INITIAL GENOME SCREEN STAGE A COMPLETE"
    )


if __name__ == "__main__":
    main()
