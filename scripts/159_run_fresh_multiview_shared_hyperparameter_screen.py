#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
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

SCRIPT158_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script158_successful_fresh_multiview_preregistration_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_v1/"
      "fresh_shared_hyperparameter_screen_run_plan_v1.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_v1/"
      "fresh_shared_hyperparameter_screen_configuration_registry_v1.tsv"
)

IMPLEMENTATION_SPEC_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_v1/"
      "fresh_multiview_implementation_specification_v1.tsv"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_runs_v1"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_runs_v1"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "fresh_multiview_hyperparameter_screen_aggregate_v1"
)

EXPECTED_RUNS = 96
EXPECTED_GENOME_ROWS = 21_394

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
        "fresh_multiview_backend_143",
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
            nn.LayerNorm(
                hidden_width
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_width,
                latent_width,
            ),
            nn.LayerNorm(
                latent_width
            ),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            inputs
        )


class FreshDrugEncoder(nn.Module):
    def __init__(
        self,
        view_dimensions: dict[str, int],
        latent_width: int,
        hidden_multiplier: int,
        fusion_hidden_multiplier: int,
        dropout: float,
        pairwise_rank: int,
    ) -> None:
        super().__init__()

        if not view_dimensions:
            raise ValueError(
                "At least one drug view is required."
            )

        self.view_ids = tuple(
            view_dimensions
        )

        self.encoders = nn.ModuleDict(
            {
                view_id: FreshViewEncoder(
                    input_dimension=dimension,
                    latent_width=latent_width,
                    hidden_multiplier=hidden_multiplier,
                    dropout=dropout,
                )
                for view_id, dimension
                in view_dimensions.items()
            }
        )

        self.pairs = tuple(
            (
                self.view_ids[left],
                self.view_ids[right],
            )
            for left in range(
                len(self.view_ids)
            )
            for right in range(
                left + 1,
                len(self.view_ids),
            )
        )

        if len(self.view_ids) > 1:
            fusion_hidden = max(
                latent_width,
                int(
                    fusion_hidden_multiplier
                    * latent_width
                ),
            )

            self.base_fusion = nn.Sequential(
                nn.Linear(
                    len(self.view_ids)
                    * latent_width,
                    fusion_hidden,
                ),
                nn.LayerNorm(
                    fusion_hidden
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    fusion_hidden,
                    latent_width,
                ),
                nn.LayerNorm(
                    latent_width
                ),
                nn.GELU(),
            )

            self.low_rank = nn.ModuleDict(
                {
                    view_id: nn.Linear(
                        latent_width,
                        pairwise_rank,
                        bias=False,
                    )
                    for view_id
                    in self.view_ids
                }
            )

            self.pairwise_to_latent = (
                nn.Linear(
                    len(self.pairs)
                    * pairwise_rank,
                    latent_width,
                    bias=False,
                )
            )

            self.residual_norm = (
                nn.LayerNorm(
                    latent_width
                )
            )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        latents = {
            view_id: self.encoders[
                view_id
            ](
                inputs[
                    view_id
                ]
            )
            for view_id
            in self.view_ids
        }

        if len(self.view_ids) == 1:
            return latents[
                self.view_ids[0]
            ]

        base = self.base_fusion(
            torch.cat(
                [
                    latents[
                        view_id
                    ]
                    for view_id
                    in self.view_ids
                ],
                dim=1,
            )
        )

        low_rank = {
            view_id: self.low_rank[
                view_id
            ](
                latents[
                    view_id
                ]
            )
            for view_id
            in self.view_ids
        }

        pairwise = torch.cat(
            [
                low_rank[left]
                * low_rank[right]
                for left, right
                in self.pairs
            ],
            dim=1,
        )

        residual = (
            self.pairwise_to_latent(
                pairwise
            )
        )

        return torch.nn.functional.gelu(
            self.residual_norm(
                base
                + residual
            )
        )


class FreshControlledNetwork(nn.Module):
    def __init__(
        self,
        genome_dimension: int,
        drug_view_dimensions: dict[str, int],
        architecture_id: str,
        spec: dict[str, Any],
    ) -> None:
        super().__init__()

        self.spec = spec
        self.variant = str(
            spec[
                "fresh_genome_variant"
            ]
        )
        self.architecture_id = (
            architecture_id
        )

        latent = int(
            spec[
                "latent_width"
            ]
        )
        dropout = float(
            spec[
                "dropout"
            ]
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
        drug_rank = int(
            spec[
                "drug_pairwise_rank"
            ]
        )

        if self.variant == "fresh_common_AMR_only":
            self.genome_encoder = (
                FreshViewEncoder(
                    input_dimension=genome_dimension,
                    latent_width=latent,
                    hidden_multiplier=genome_multiplier,
                    dropout=dropout,
                )
            )
            self.kmer_dimension = 0
            self.amr_dimension = (
                genome_dimension
            )

        elif (
            self.variant
            == "fresh_selected_kmer_plus_AMR_projected_concat"
        ):
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

            expected = (
                self.kmer_dimension
                + self.amr_dimension
            )

            if genome_dimension != expected:
                raise RuntimeError(
                    "Fused genome dimension mismatch: "
                    f"{genome_dimension} != {expected}"
                )

            self.kmer_encoder = (
                FreshViewEncoder(
                    input_dimension=(
                        self.kmer_dimension
                    ),
                    latent_width=latent,
                    hidden_multiplier=genome_multiplier,
                    dropout=dropout,
                )
            )

            self.amr_encoder = (
                FreshViewEncoder(
                    input_dimension=(
                        self.amr_dimension
                    ),
                    latent_width=latent,
                    hidden_multiplier=genome_multiplier,
                    dropout=dropout,
                )
            )

            fusion_hidden = max(
                latent,
                int(
                    fusion_multiplier
                    * latent
                ),
            )

            self.genome_base_fusion = (
                nn.Sequential(
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
                    nn.LayerNorm(
                        latent
                    ),
                    nn.GELU(),
                )
            )

        else:
            raise RuntimeError(
                f"Unexpected genome variant: {self.variant}"
            )

        self.drug_encoder = (
            FreshDrugEncoder(
                view_dimensions=(
                    drug_view_dimensions
                ),
                latent_width=latent,
                hidden_multiplier=(
                    drug_multiplier
                ),
                fusion_hidden_multiplier=(
                    fusion_multiplier
                ),
                dropout=dropout,
                pairwise_rank=drug_rank,
            )
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

        if (
            architecture_id
            == "cross_modal_projected_concat"
        ):
            self.interaction_head = (
                nn.Sequential(
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
            )

        elif (
            architecture_id
            == "dual_tower_explicit_interaction"
        ):
            self.cross_genome_tower = (
                nn.Sequential(
                    nn.Linear(
                        latent,
                        latent,
                    ),
                    nn.LayerNorm(latent),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.cross_drug_tower = (
                nn.Sequential(
                    nn.Linear(
                        latent,
                        latent,
                    ),
                    nn.LayerNorm(latent),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.interaction_head = (
                nn.Sequential(
                    nn.Linear(
                        4 * latent,
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
            )

        elif (
            architecture_id
            == "cross_modal_gmu"
        ):
            self.gmu_genome_projection = (
                nn.Linear(
                    latent,
                    latent,
                )
            )
            self.gmu_drug_projection = (
                nn.Linear(
                    latent,
                    latent,
                )
            )
            self.gmu_gate = nn.Linear(
                2 * latent,
                latent,
            )
            self.interaction_head = (
                nn.Sequential(
                    nn.LayerNorm(latent),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        latent,
                        1,
                        bias=False,
                    ),
                )
            )

        elif (
            architecture_id
            == "cross_modal_low_rank_bilinear"
        ):
            cross_rank = max(
                4,
                min(
                    drug_rank,
                    latent,
                ),
            )
            self.cross_genome_bilinear = (
                nn.Linear(
                    latent,
                    cross_rank,
                    bias=False,
                )
            )
            self.cross_drug_bilinear = (
                nn.Linear(
                    latent,
                    cross_rank,
                    bias=False,
                )
            )
            self.cross_bilinear_to_latent = (
                nn.Linear(
                    cross_rank,
                    latent,
                    bias=False,
                )
            )
            self.interaction_head = (
                nn.Sequential(
                    nn.LayerNorm(latent),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        latent,
                        1,
                        bias=False,
                    ),
                )
            )

        elif (
            architecture_id
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
            self.film_norm = (
                nn.LayerNorm(
                    latent
                )
            )
            self.interaction_head = (
                nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        latent,
                        1,
                        bias=False,
                    ),
                )
            )

        else:
            raise RuntimeError(
                f"Unsupported architecture ID: {architecture_id}"
            )

        final_layer = (
            self.interaction_head[-1]
        )

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
        if self.variant == "fresh_common_AMR_only":
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

        kmer_latent = (
            self.kmer_encoder(
                kmer
            )
        )
        amr_latent = (
            self.amr_encoder(
                amr
            )
        )

        return self.genome_base_fusion(
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
        genome_latent = self.encode_genome(
            genome
        )
        drug_latent = self.drug_encoder(
            drug_views
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

        if (
            self.architecture_id
            == "cross_modal_projected_concat"
        ):
            interaction_features = (
                torch.cat(
                    [
                        genome_latent,
                        drug_latent,
                    ],
                    dim=1,
                )
            )

        elif (
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
            interaction_features = (
                torch.cat(
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
                gate
                * genome_candidate
                + (
                    1.0
                    - gate
                )
                * drug_candidate
            )

        elif (
            self.architecture_id
            == "cross_modal_low_rank_bilinear"
        ):
            interaction_features = (
                self.cross_bilinear_to_latent(
                    self.cross_genome_bilinear(
                        genome_latent
                    )
                    * self.cross_drug_bilinear(
                        drug_latent
                    )
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
                    (
                        1.0
                        + gamma
                    )
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

    if len(frame) != 24:
        raise RuntimeError(
            f"Expected 24 configuration rows; observed {len(frame)}."
        )

    registry: dict[
        str,
        dict[str, Any],
    ] = {}

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
        "drug_pairwise_rank",
    }

    float_fields = {
        "dropout",
        "learning_rate",
        "weight_decay",
        "minimum_rmse_improvement",
        "gradient_clip_norm",
    }

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
                    spec[
                        field
                    ]
                )
            )

        for field in float_fields:
            spec[field] = float(
                spec[
                    field
                ]
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
                "Duplicate genome representation: "
                f"{representation}"
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
        spec[
            "latent_width"
        ]
    )
    backend.DROPOUT = float(
        spec[
            "dropout"
        ]
    )
    backend.LEARNING_RATE = float(
        spec[
            "learning_rate"
        ]
    )
    backend.WEIGHT_DECAY = float(
        spec[
            "weight_decay"
        ]
    )
    backend.BATCH_SIZE = int(
        spec[
            "batch_size"
        ]
    )
    backend.MAX_EPOCHS = int(
        spec[
            "maximum_epochs"
        ]
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
    backend.BILINEAR_RANK = int(
        spec[
            "drug_pairwise_rank"
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
            "Current fresh hyperparameter specification is unset."
        )

    model = FreshControlledNetwork(
        genome_dimension=(
            genome_matrix.shape[1]
        ),
        drug_view_dimensions={
            key: value.shape[1]
            for key, value
            in drug_matrices.items()
        },
        architecture_id=architecture_id,
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
        "fresh_genome_variant": spec[
            "fresh_genome_variant"
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
        "drug_pairwise_rank": spec[
            "drug_pairwise_rank"
        ],
        "selected_kmer_dimension": spec[
            "selected_kmer_dimension"
        ],
        "common_amr_dimension": spec[
            "common_amr_dimension"
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
            addition_frame[
                "item"
            ]
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

    output_paths: list[
        Path
    ] = []

    for line in manifest_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        _, value = line.split(
            maxsplit=1
        )

        output_paths.append(
            project_path(
                value
            )
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
            "Unregistered fresh genome representation: "
            f"{representation}"
        )

    CURRENT_SPEC = (
        CONFIG_BY_REPRESENTATION[
            representation
        ]
    )

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

    config_metadata = (
        plan[
            [
                "configuration_id",
                "outer_target_code",
                "shared_hp_id",
                "fresh_genome_variant",
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
                "drug_pairwise_rank",
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
        ] = safe_mean(
            values
        )
        record[
            "worst_direction_macro_rmse_sd"
        ] = sample_sd(
            values
        )

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
        config_metadata,
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
        config_metadata[
            [
                "outer_target_code",
                "configuration_id",
                "shared_hp_id",
                "fresh_genome_variant",
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
                "fresh_genome_variant",
                "nunique",
            ),
            balanced_bidirectional_macro_rmse=(
                "bidirectional_macro_rmse",
                "mean",
            ),
        )
        .reset_index()
    )

    worst_seed_metadata = (
        worst_seed.merge(
            config_metadata[
                [
                    "outer_target_code",
                    "configuration_id",
                    "shared_hp_id",
                    "fresh_genome_variant",
                ]
            ],
            on=[
                "outer_target_code",
                "configuration_id",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    robust_bundle_seed = (
        worst_seed_metadata.groupby(
            [
                "outer_target_code",
                "shared_hp_id",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            bundle_seedwise_worst_macro_rmse=(
                "seedwise_worst_direction_macro_rmse",
                "max",
            )
        )
        .reset_index()
    )

    balanced_seed = balanced_seed.merge(
        robust_bundle_seed,
        on=[
            "outer_target_code",
            "shared_hp_id",
            "seed",
        ],
        how="left",
        validate="one_to_one",
    )

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

    for keys, group in balanced_seed.groupby(
        [
            "outer_target_code",
            "shared_hp_id",
        ],
        dropna=False,
    ):
        outer, hp_id = keys

        record: dict[
            str,
            object,
        ] = {
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
            "bundle_seedwise_worst_macro_rmse_mean": (
                safe_mean(
                    group[
                        "bundle_seedwise_worst_macro_rmse"
                    ]
                )
            ),
            "bundle_seedwise_worst_macro_rmse_sd": (
                sample_sd(
                    group[
                        "bundle_seedwise_worst_macro_rmse"
                    ]
                )
            ),
        }

        hp_metadata = config_metadata.loc[
            config_metadata[
                "outer_target_code"
            ].eq(outer)
            & config_metadata[
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
            "drug_pairwise_rank",
        ]:
            record[column] = (
                hp_metadata[
                    column
                ]
            )

        ranking_records.append(
            record
        )

    ranking = pd.DataFrame(
        ranking_records
    )

    ranking[
        "selection_rank"
    ] = (
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
            "bundle_seedwise_worst_macro_rmse_mean",
            "shared_hp_id",
        ]
    ).reset_index(drop=True)

    ranking_path = (
        AGGREGATE_ROOT
        / "shared_hyperparameter_bundle_balanced_ranking.tsv"
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
                "item": "shared_bundle_selection_score",
                "value": (
                    "within each outer target and tuning seed, average the "
                    "bidirectional macro RMSE of AMR-only and projected "
                    "multiview variants; then report mean and sample SD "
                    "across the two tuning seeds"
                ),
            },
            {
                "item": "secondary_robustness_metric",
                "value": (
                    "seedwise worst-direction macro RMSE: max of the two "
                    "development directions within each seed, followed by "
                    "mean and sample SD across seeds"
                ),
            },
            {
                "item": "bundle_robustness_score",
                "value": (
                    "within each outer target and tuning seed, maximum "
                    "seedwise worst-direction macro RMSE across the two "
                    "screen variants"
                ),
            },
            {
                "item": "selection_rule",
                "value": (
                    "minimum mean balanced bidirectional macro RMSE; "
                    "worst-direction score is secondary reporting only "
                    "and may be consulted only for an exact numerical tie"
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
            raise FileNotFoundError(
                path
            )

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
        SCRIPT158_FREEZE
    )
    backend.SCRIPT142_FROZEN_PATH = (
        SCRIPT158_FREEZE
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
            "amr",
            "projected",
        ],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            20260729,
            20260730,
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

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    required = [
        BACKEND_PATH,
        SCRIPT158_FREEZE,
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
            raise FileNotFoundError(
                path
            )

    verify_manifest(
        SCRIPT158_FREEZE
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

    if arguments.variant == "amr":
        plan = plan.loc[
            plan[
                "fresh_genome_variant"
            ].eq(
                "fresh_common_AMR_only"
            )
        ]

    elif arguments.variant == "projected":
        plan = plan.loc[
            plan[
                "fresh_genome_variant"
            ].eq(
                "fresh_selected_kmer_plus_AMR_projected_concat"
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
            "fresh_genome_variant",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    plan[
        "already_complete"
    ] = [
        backend.run_complete(
            str(
                run_id
            )
        )
        for run_id
        in plan[
            "run_id"
        ]
    ]

    print(
        "===== SCRIPT 159 RUN PLAN ====="
    )

    print(
        plan.groupby(
            [
                "outer_target_code",
                "shared_hp_id",
                "fresh_genome_variant",
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
            "STATUS: SCRIPT 159 "
            "AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 159 "
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
            row[
                "run_id"
            ]
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
            arguments.aggregate_every
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
            aggregate_completed_runs()
            print(
                "STATUS: SCRIPT 159 "
                "PARTIAL RUN COMPLETE"
            )
            return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 159 FRESH "
        "SHARED HYPERPARAMETER SCREEN COMPLETE"
    )


if __name__ == "__main__":
    main()
