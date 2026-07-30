#!/usr/bin/env python3

"""Reusable model, scaling, metric, and training utilities.

Numerical settings are loaded from the frozen shared-hyperparameter
registry for each outer target before a model is instantiated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

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

OBSERVATION_INDEX_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "configuration_splits_v1/"
      "nested_loso_observation_feature_index_v1.tsv"
)
RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "model_selection_backend_v1/"
      "nested_loso_model_selection_run_plan_v1.tsv"
)
SCRIPT140_FROZEN_PATH = (
    PROJECT
    / "metadata/config_selection/"
      "script140_successful_run_core_sha256.txt"
)
SCRIPT142_OUTPUTS_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "model_selection_backend_v1/"
      "script142_outputs_sha256.txt"
)
SCRIPT142_FROZEN_PATH = (
    PROJECT
    / "metadata/config_selection/"
      "script142_successful_preregistration_core_sha256.txt"
)
IMPLEMENTATION_BUNDLE_PATH = (
    PROJECT / "config/model_implementation_spec.tsv"
)
EXPECTED_IMPLEMENTATION_BUNDLE_SHA256 = ""

KMER_PATHS = {
    "canonical_4mer": PROJECT / (
        "features/genome_representation/nested_loso_v1/"
        "canonical_kmer/"
        "nested_loso_all_species_canonical_4mer_"
        "relative_frequency_v1.npy"
    ),
    "canonical_5mer": PROJECT / (
        "features/genome_representation/nested_loso_v1/"
        "canonical_kmer/"
        "nested_loso_all_species_canonical_5mer_"
        "relative_frequency_v1.npy"
    ),
    "canonical_6mer": PROJECT / (
        "features/genome_representation/nested_loso_v1/"
        "canonical_kmer/"
        "nested_loso_all_species_canonical_6mer_"
        "relative_frequency_v1.npy"
    ),
    "canonical_7mer": PROJECT / (
        "features/genome_representation/nested_loso_v1/"
        "canonical_kmer/"
        "nested_loso_all_species_canonical_7mer_"
        "relative_frequency_v1.npy"
    ),
    "canonical_8mer": PROJECT / (
        "features/genome_representation/nested_loso_v1/"
        "canonical_kmer/"
        "nested_loso_all_species_canonical_8mer_"
        "relative_frequency_v1.npy"
    ),
}

DRUG_VIEW_PATHS = {
    "morgan": PROJECT / "features/drug/morgan_radius2_2048_chiral.npy",
    "rdkit": PROJECT / "features/drug/rdkit_descriptors.npy",
    "chemberta_mean": PROJECT / "features/drug/chemberta_mean.npy",
}

DRUG_REPRESENTATION_VIEWS = {
    "Morgan": ("morgan",),
    "RDKit": ("rdkit",),
    "ChemBERTa_mean": ("chemberta_mean",),
    "ChemBERTa_mean_plus_Morgan": (
        "chemberta_mean",
        "morgan",
    ),
    "ChemBERTa_mean_plus_Morgan_plus_RDKit": (
        "chemberta_mean",
        "morgan",
        "rdkit",
    ),
}

ARCHITECTURE_NAME_TO_ID = {
    "projected_concatenation_MLP": "cross_modal_projected_concat",
    "dual_tower_interaction": "dual_tower_explicit_interaction",
    "cross_modal_GMU": "cross_modal_gmu",
    "low_rank_bilinear": "cross_modal_low_rank_bilinear",
    "drug_to_genome_FiLM": "drug_to_genome_film",
}

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "model_selection_backend_runs_v1"
)
METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "model_selection_backend_runs_v1"
)
AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "model_selection_backend_aggregate_v1"
)

EXPECTED_OBSERVATION_ROWS: int
EXPECTED_GENOME_ROWS: int
EXPECTED_DRUG_ROWS: int
EXPECTED_RUNS: int
VALIDATION_FOLD = "fold_05"

# Numerical settings are assigned from the frozen outer-target configuration
# before model construction. No pilot defaults are retained in this module.
LATENT_WIDTH: int
DROPOUT: float
LEARNING_RATE: float
WEIGHT_DECAY: float
BATCH_SIZE: int
MAX_EPOCHS: int
EARLY_STOPPING_PATIENCE: int
MINIMUM_RMSE_IMPROVEMENT: float
GRADIENT_CLIP_NORM: float
BILINEAR_RANK: int

METRIC_IDS = (
    "rmse",
    "mae",
    "r2",
    "pearson",
    "spearman",
    "one_tier_accuracy",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer-target",
        choices=["all", "ec", "kp", "se"],
        default="all",
    )
    parser.add_argument(
        "--genome-representation",
        choices=["all", *KMER_PATHS.keys()],
        default="all",
    )
    parser.add_argument(
        "--drug-representation",
        choices=["all", *DRUG_REPRESENTATION_VIEWS.keys()],
        default="all",
    )
    parser.add_argument(
        "--architecture",
        choices=["all", *ARCHITECTURE_NAME_TO_ID.keys()],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260726, 20260727, 20260728],
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
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--aggregate-every",
        type=int,
        default=25,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Empty SHA manifest: {path}")

    for line in lines:
        if not line.strip():
            continue
        expected, path_text = line.split(maxsplit=1)
        candidate = Path(path_text.strip())
        if not candidate.is_absolute():
            candidate = PROJECT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != expected:
            raise RuntimeError(f"SHA mismatch: {candidate}")


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


def write_sha_manifest(paths: Iterable[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in sorted(set(paths), key=lambda value: value.as_posix()):
            relative = candidate.relative_to(PROJECT)
            handle.write(f"{sha256_file(candidate)}  {relative}\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_sd(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    return float(array.std(ddof=1))


def safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(array.mean())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))

    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0.0:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(error * error) / denominator)

    if (
        len(target) < 2
        or float(np.std(target)) == 0.0
        or float(np.std(prediction)) == 0.0
    ):
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(target, prediction)[0, 1])

    if len(target) < 2:
        spearman = float("nan")
    else:
        target_rank = (
            pd.Series(target).rank(method="average").to_numpy(dtype=np.float64)
        )
        prediction_rank = (
            pd.Series(prediction).rank(method="average").to_numpy(dtype=np.float64)
        )
        if (
            float(np.std(target_rank)) == 0.0
            or float(np.std(prediction_rank)) == 0.0
        ):
            spearman = float("nan")
        else:
            spearman = float(
                np.corrcoef(target_rank, prediction_rank)[0, 1]
            )

    one_tier = float(np.mean(np.abs(error) <= 1.0))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "one_tier_accuracy": one_tier,
    }


def per_antibiotic_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    target = frame["mic_target_log2_mg_per_l"].to_numpy(dtype=np.float64)
    antibiotics = frame["normalized_antibiotic"].astype(str).to_numpy()

    records: list[dict[str, object]] = []
    for antibiotic in sorted(np.unique(antibiotics).tolist()):
        mask = antibiotics == antibiotic
        records.append(
            {
                "normalized_antibiotic": antibiotic,
                "observations": int(mask.sum()),
                **regression_metrics(target[mask], predictions[mask]),
            }
        )
    return pd.DataFrame(records)


def macro_metrics(per_drug: pd.DataFrame) -> dict[str, float | int]:
    record: dict[str, float | int] = {}
    for metric in METRIC_IDS:
        values = pd.to_numeric(per_drug[metric], errors="coerce")
        record[f"macro_{metric}"] = float(values.mean(skipna=True))
        record[f"macro_{metric}_valid_antibiotics"] = int(
            values.notna().sum()
        )
    return record


# ---------------------------------------------------------------------------
# Source-only weighted standardisation
# ---------------------------------------------------------------------------


def weighted_standardizer(
    matrix: np.ndarray,
    observation_rows: np.ndarray,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows, counts = np.unique(
        observation_rows.astype(np.int64, copy=False),
        return_counts=True,
    )
    if len(rows) == 0:
        raise RuntimeError("Cannot fit scaler using zero rows.")

    dimension = matrix.shape[1]
    weighted_sum = np.zeros(dimension, dtype=np.float64)
    weighted_square_sum = np.zeros(dimension, dtype=np.float64)
    total_weight = float(counts.sum())

    for start in range(0, len(rows), chunk_size):
        selected_rows = rows[start : start + chunk_size]
        selected_counts = counts[start : start + chunk_size].astype(
            np.float64,
            copy=False,
        )
        values = np.asarray(matrix[selected_rows], dtype=np.float32)
        weighted_sum += np.sum(
            values * selected_counts[:, None],
            axis=0,
            dtype=np.float64,
        )
        weighted_square_sum += np.sum(
            values * values * selected_counts[:, None],
            axis=0,
            dtype=np.float64,
        )

    mean = weighted_sum / total_weight
    variance = weighted_square_sum / total_weight - mean * mean
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    zero_mask = scale < 1e-8
    zero_count = int(zero_mask.sum())
    scale[zero_mask] = 1.0

    return (
        mean.astype(np.float32),
        scale.astype(np.float32),
        zero_count,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class ViewEncoder(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        hidden_dimensions: int,
        latent_dimensions: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimensions, hidden_dimensions),
            nn.LayerNorm(hidden_dimensions),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions, latent_dimensions),
            nn.LayerNorm(latent_dimensions),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def drug_hidden_width(view_id: str, latent_width: int) -> int:
    if view_id == "morgan":
        return 2 * latent_width
    if view_id in {"chemberta_mean", "chemberta_first"}:
        return latent_width
    if view_id in {"rdkit", "identity"}:
        return max(32, latent_width // 2)
    raise ValueError(view_id)


class DrugRepresentationEncoder(nn.Module):
    """Encode one or more registered drug views.

    """

    def __init__(
        self,
        view_dimensions: dict[str, int],
        latent_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not view_dimensions:
            raise ValueError("At least one drug view is required.")

        self.view_ids = tuple(view_dimensions)
        self.encoders = nn.ModuleDict(
            {
                view_id: ViewEncoder(
                    input_dimensions=dimension,
                    hidden_dimensions=drug_hidden_width(
                        view_id,
                        latent_width,
                    ),
                    latent_dimensions=latent_width,
                    dropout=dropout,
                )
                for view_id, dimension in view_dimensions.items()
            }
        )

        self.pairs = tuple(
            (self.view_ids[left], self.view_ids[right])
            for left in range(len(self.view_ids))
            for right in range(left + 1, len(self.view_ids))
        )

        if len(self.view_ids) > 1:
            fusion_hidden = 2 * latent_width
            self.base_fusion = nn.Sequential(
                nn.Linear(
                    len(self.view_ids) * latent_width,
                    fusion_hidden,
                ),
                nn.LayerNorm(fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, latent_width),
                nn.LayerNorm(latent_width),
                nn.GELU(),
            )

            self.low_rank = nn.ModuleDict(
                {
                    view_id: nn.Linear(
                        latent_width,
                        BILINEAR_RANK,
                        bias=False,
                    )
                    for view_id in self.view_ids
                }
            )
            self.pairwise_to_latent = nn.Linear(
                len(self.pairs) * BILINEAR_RANK,
                latent_width,
                bias=False,
            )
            self.residual_norm = nn.LayerNorm(latent_width)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        latents = {
            view_id: self.encoders[view_id](inputs[view_id])
            for view_id in self.view_ids
        }

        if len(self.view_ids) == 1:
            return latents[self.view_ids[0]]

        base = self.base_fusion(
            torch.cat(
                [latents[view_id] for view_id in self.view_ids],
                dim=1,
            )
        )
        low_rank = {
            view_id: self.low_rank[view_id](latents[view_id])
            for view_id in self.view_ids
        }
        pairwise = torch.cat(
            [
                low_rank[left] * low_rank[right]
                for left, right in self.pairs
            ],
            dim=1,
        )
        residual = self.pairwise_to_latent(pairwise)
        return torch.nn.functional.gelu(
            self.residual_norm(base + residual)
        )


class FullGridCrossModalNetwork(nn.Module):
    """Cross-modal genome–antibiotic architecture.

    The cross-modal blocks implement the registered interaction families.
    Each candidate starts from the same additive genome-plus-drug anchor;
    the final interaction layer is zero-initialised.
    """

    def __init__(
        self,
        genome_dimension: int,
        drug_view_dimensions: dict[str, int],
        architecture_id: str,
    ) -> None:
        super().__init__()
        if architecture_id not in ARCHITECTURE_NAME_TO_ID.values():
            raise ValueError(architecture_id)

        self.architecture_id = architecture_id
        self.genome_encoder = ViewEncoder(
            input_dimensions=genome_dimension,
            hidden_dimensions=4 * LATENT_WIDTH,
            latent_dimensions=LATENT_WIDTH,
            dropout=DROPOUT,
        )
        self.drug_encoder = DrugRepresentationEncoder(
            view_dimensions=drug_view_dimensions,
            latent_width=LATENT_WIDTH,
            dropout=DROPOUT,
        )

        self.genome_head = nn.Linear(LATENT_WIDTH, 1, bias=False)
        self.drug_head = nn.Linear(LATENT_WIDTH, 1, bias=False)
        self.intercept = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        if architecture_id == "cross_modal_projected_concat":
            self.interaction_head = nn.Sequential(
                nn.Linear(2 * LATENT_WIDTH, 2 * LATENT_WIDTH),
                nn.LayerNorm(2 * LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(2 * LATENT_WIDTH, LATENT_WIDTH),
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(LATENT_WIDTH, 1, bias=False),
            )

        elif architecture_id == "dual_tower_explicit_interaction":
            self.cross_genome_tower = nn.Sequential(
                nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            self.cross_drug_tower = nn.Sequential(
                nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            self.interaction_head = nn.Sequential(
                nn.Linear(4 * LATENT_WIDTH, 2 * LATENT_WIDTH),
                nn.LayerNorm(2 * LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(2 * LATENT_WIDTH, LATENT_WIDTH),
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(LATENT_WIDTH, 1, bias=False),
            )

        elif architecture_id == "cross_modal_gmu":
            self.gmu_genome_projection = nn.Linear(
                LATENT_WIDTH,
                LATENT_WIDTH,
            )
            self.gmu_drug_projection = nn.Linear(
                LATENT_WIDTH,
                LATENT_WIDTH,
            )
            self.gmu_gate = nn.Linear(
                2 * LATENT_WIDTH,
                LATENT_WIDTH,
            )
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(LATENT_WIDTH, 1, bias=False),
            )

        elif architecture_id == "cross_modal_low_rank_bilinear":
            self.cross_genome_bilinear = nn.Linear(
                LATENT_WIDTH,
                BILINEAR_RANK,
                bias=False,
            )
            self.cross_drug_bilinear = nn.Linear(
                LATENT_WIDTH,
                BILINEAR_RANK,
                bias=False,
            )
            self.cross_bilinear_to_latent = nn.Linear(
                BILINEAR_RANK,
                LATENT_WIDTH,
                bias=False,
            )
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(LATENT_WIDTH),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(LATENT_WIDTH, 1, bias=False),
            )

        elif architecture_id == "drug_to_genome_film":
            self.film_gamma = nn.Linear(LATENT_WIDTH, LATENT_WIDTH)
            self.film_beta = nn.Linear(LATENT_WIDTH, LATENT_WIDTH)
            self.film_norm = nn.LayerNorm(LATENT_WIDTH)
            self.interaction_head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(LATENT_WIDTH, 1, bias=False),
            )

        else:
            raise AssertionError(architecture_id)

        final_layer = self.interaction_head[-1]
        if not isinstance(final_layer, nn.Linear):
            raise RuntimeError("Interaction head must end in Linear.")
        nn.init.zeros_(final_layer.weight)

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        genome_latent = self.genome_encoder(genome)
        drug_latent = self.drug_encoder(drug_views)

        additive_prediction = (
            self.intercept
            + self.genome_head(genome_latent).squeeze(1)
            + self.drug_head(drug_latent).squeeze(1)
        )

        if self.architecture_id == "cross_modal_projected_concat":
            interaction_features = torch.cat(
                [genome_latent, drug_latent],
                dim=1,
            )

        elif self.architecture_id == "dual_tower_explicit_interaction":
            genome_tower = self.cross_genome_tower(genome_latent)
            drug_tower = self.cross_drug_tower(drug_latent)
            interaction_features = torch.cat(
                [
                    genome_tower,
                    drug_tower,
                    genome_tower * drug_tower,
                    torch.abs(genome_tower - drug_tower),
                ],
                dim=1,
            )

        elif self.architecture_id == "cross_modal_gmu":
            genome_candidate = torch.tanh(
                self.gmu_genome_projection(genome_latent)
            )
            drug_candidate = torch.tanh(
                self.gmu_drug_projection(drug_latent)
            )
            gate = torch.sigmoid(
                self.gmu_gate(
                    torch.cat([genome_latent, drug_latent], dim=1)
                )
            )
            interaction_features = (
                gate * genome_candidate
                + (1.0 - gate) * drug_candidate
            )

        elif self.architecture_id == "cross_modal_low_rank_bilinear":
            low_rank_interaction = (
                self.cross_genome_bilinear(genome_latent)
                * self.cross_drug_bilinear(drug_latent)
            )
            interaction_features = self.cross_bilinear_to_latent(
                low_rank_interaction
            )

        elif self.architecture_id == "drug_to_genome_film":
            gamma = torch.tanh(self.film_gamma(drug_latent))
            beta = self.film_beta(drug_latent)
            interaction_features = self.film_norm(
                (1.0 + gamma) * genome_latent + beta
            )

        else:
            raise AssertionError(self.architecture_id)

        interaction_residual = self.interaction_head(
            interaction_features
        ).squeeze(1)
        return additive_prediction + interaction_residual


# ---------------------------------------------------------------------------
# Data batching and training
# ---------------------------------------------------------------------------


def batch_positions(
    length: int,
    batch_size: int,
    generator: np.random.Generator,
    shuffle: bool,
) -> Iterable[np.ndarray]:
    positions = np.arange(length, dtype=np.int64)
    if shuffle:
        generator.shuffle(positions)
    for start in range(0, length, batch_size):
        yield positions[start : start + batch_size]


def to_scaled_tensor(
    matrix: np.ndarray,
    rows: np.ndarray,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = torch.from_numpy(
        np.asarray(matrix[rows], dtype=np.float32)
    ).to(device=device, dtype=torch.float32)
    return (values - mean) / scale


def make_batch(
    frame: pd.DataFrame,
    positions: np.ndarray,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    genome_mean: torch.Tensor,
    genome_scale: torch.Tensor,
    drug_means: dict[str, torch.Tensor],
    drug_scales: dict[str, torch.Tensor],
    device: torch.device,
    include_target: bool,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor | None,
]:
    genome_rows = frame["genome_feature_row"].to_numpy(dtype=np.int64)[
        positions
    ]
    drug_rows = frame["drug_feature_row"].to_numpy(dtype=np.int64)[
        positions
    ]

    genome = to_scaled_tensor(
        genome_matrix,
        genome_rows,
        genome_mean,
        genome_scale,
        device,
    )
    drug_views = {
        view_id: to_scaled_tensor(
            matrix,
            drug_rows,
            drug_means[view_id],
            drug_scales[view_id],
            device,
        )
        for view_id, matrix in drug_matrices.items()
    }

    target_tensor = None
    if include_target:
        target = frame["mic_target_log2_mg_per_l"].to_numpy(
            dtype=np.float32
        )[positions]
        target_tensor = torch.from_numpy(target).to(
            device=device,
            dtype=torch.float32,
        )

    return genome, drug_views, target_tensor


@torch.no_grad()
def predict(
    model: nn.Module,
    frame: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    genome_mean: torch.Tensor,
    genome_scale: torch.Tensor,
    drug_means: dict[str, torch.Tensor],
    drug_scales: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(frame), dtype=np.float32)
    generator = np.random.default_rng(0)

    for positions in batch_positions(
        len(frame),
        BATCH_SIZE,
        generator,
        shuffle=False,
    ):
        genome, drug_views, _ = make_batch(
            frame,
            positions,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            device,
            include_target=False,
        )
        predictions[positions] = (
            model(genome, drug_views)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    return predictions


def scaler_tensors(
    genome_mean_np: np.ndarray,
    genome_scale_np: np.ndarray,
    drug_means_np: dict[str, np.ndarray],
    drug_scales_np: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    return (
        torch.from_numpy(genome_mean_np).to(device),
        torch.from_numpy(genome_scale_np).to(device),
        {
            key: torch.from_numpy(value).to(device)
            for key, value in drug_means_np.items()
        },
        {
            key: torch.from_numpy(value).to(device)
            for key, value in drug_scales_np.items()
        },
    )


def fit_scalers(
    frame: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, int],
]:
    genome_mean, genome_scale, genome_zero = weighted_standardizer(
        genome_matrix,
        frame["genome_feature_row"].to_numpy(dtype=np.int64),
    )

    drug_means: dict[str, np.ndarray] = {}
    drug_scales: dict[str, np.ndarray] = {}
    zero_counts = {"genome": genome_zero}

    drug_rows = frame["drug_feature_row"].to_numpy(dtype=np.int64)
    for view_id, matrix in drug_matrices.items():
        mean, scale, zero_count = weighted_standardizer(
            matrix,
            drug_rows,
            chunk_size=64,
        )
        drug_means[view_id] = mean
        drug_scales[view_id] = scale
        zero_counts[view_id] = zero_count

    return (
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
        zero_counts,
    )


def build_model(
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    target_mean: float,
    device: torch.device,
) -> FullGridCrossModalNetwork:
    model = FullGridCrossModalNetwork(
        genome_dimension=genome_matrix.shape[1],
        drug_view_dimensions={
            key: value.shape[1]
            for key, value in drug_matrices.items()
        },
        architecture_id=architecture_id,
    ).to(device)
    with torch.no_grad():
        model.intercept.fill_(float(target_mean))
    return model


def select_epoch_source_only(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    seed: int,
    device: torch.device,
) -> tuple[int, pd.DataFrame, dict[str, int]]:
    set_seed(seed)
    (
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        zero_counts,
    ) = fit_scalers(training, genome_matrix, drug_matrices)

    (
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
    ) = scaler_tensors(
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        device,
    )

    model = build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        float(training["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.MSELoss()

    best_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    records: list[dict[str, object]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        generator = np.random.default_rng(seed + epoch)
        squared_error_sum = 0.0
        observations = 0

        for positions in batch_positions(
            len(training),
            BATCH_SIZE,
            generator,
            shuffle=True,
        ):
            genome, drug_views, target = make_batch(
                training,
                positions,
                genome_matrix,
                drug_matrices,
                genome_mean,
                genome_scale,
                drug_means,
                drug_scales,
                device,
                include_target=True,
            )
            assert target is not None

            optimizer.zero_grad(set_to_none=True)
            prediction = model(genome, drug_views)
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite gradient norm.")
            optimizer.step()

            squared_error_sum += float(loss.detach().cpu()) * len(target)
            observations += len(target)

        validation_prediction = predict(
            model,
            validation,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            device,
        )
        validation_target = validation["mic_target_log2_mg_per_l"].to_numpy(
            dtype=np.float64
        )
        metrics = regression_metrics(
            validation_target,
            validation_prediction,
        )
        current_rmse = metrics["rmse"]

        if current_rmse < best_rmse - MINIMUM_RMSE_IMPROVEMENT:
            best_rmse = current_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        records.append(
            {
                "epoch": epoch,
                "training_rmse": math.sqrt(
                    squared_error_sum / observations
                ),
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
                "best_validation_rmse_so_far": best_rmse,
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    if best_epoch < 1:
        raise RuntimeError("No best epoch selected.")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_epoch, pd.DataFrame(records), zero_counts


def fit_full_source_and_predict(
    source: pd.DataFrame,
    evaluation: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, int],
    int,
]:
    set_seed(seed)
    (
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        zero_counts,
    ) = fit_scalers(source, genome_matrix, drug_matrices)

    (
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
    ) = scaler_tensors(
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        device,
    )

    model = build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        float(source["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.MSELoss()
    records: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        generator = np.random.default_rng(seed + 100_000 + epoch)
        squared_error_sum = 0.0
        observations = 0

        for positions in batch_positions(
            len(source),
            BATCH_SIZE,
            generator,
            shuffle=True,
        ):
            genome, drug_views, target = make_batch(
                source,
                positions,
                genome_matrix,
                drug_matrices,
                genome_mean,
                genome_scale,
                drug_means,
                drug_scales,
                device,
                include_target=True,
            )
            assert target is not None

            optimizer.zero_grad(set_to_none=True)
            prediction = model(genome, drug_views)
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite full-source loss.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite full-source gradient norm.")
            optimizer.step()

            squared_error_sum += float(loss.detach().cpu()) * len(target)
            observations += len(target)

        records.append(
            {
                "epoch": epoch,
                "training_rmse": math.sqrt(
                    squared_error_sum / observations
                ),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )

    evaluation_prediction = predict(
        model,
        evaluation,
        genome_matrix,
        drug_matrices,
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
        device,
    )

    scaler_arrays: dict[str, np.ndarray] = {
        "genome__mean": genome_mean_np,
        "genome__scale": genome_scale_np,
    }
    for view_id in drug_matrices:
        scaler_arrays[f"{view_id}__mean"] = drug_means_np[view_id]
        scaler_arrays[f"{view_id}__scale"] = drug_scales_np[view_id]

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        evaluation_prediction,
        pd.DataFrame(records),
        scaler_arrays,
        zero_counts,
        parameter_count,
    )


# ---------------------------------------------------------------------------
# Run I/O
# ---------------------------------------------------------------------------


def run_output_directory(run_id: str) -> Path:
    return RESULT_ROOT / run_id


def run_metadata_directory(run_id: str) -> Path:
    return METADATA_ROOT / run_id


def run_complete(run_id: str) -> bool:
    metadata_directory = run_metadata_directory(run_id)
    flag = metadata_directory / "RUN_COMPLETE"
    manifest = metadata_directory / "outputs_sha256.txt"
    if not flag.is_file() or not manifest.is_file():
        return False
    try:
        verify_sha_manifest(manifest)
    except Exception:
        return False
    return flag.read_text(encoding="utf-8").strip() == "0"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_sha_manifest(SCRIPT140_FROZEN_PATH)
    verify_sha_manifest(SCRIPT142_OUTPUTS_PATH)
    verify_sha_manifest(SCRIPT142_FROZEN_PATH)

    observed_bundle_sha = sha256_file(IMPLEMENTATION_BUNDLE_PATH)
    if observed_bundle_sha != EXPECTED_IMPLEMENTATION_BUNDLE_SHA256:
        raise RuntimeError(
            "Implementation-specification SHA mismatch: "
            f"{observed_bundle_sha}"
        )

    observations = read_tsv(OBSERVATION_INDEX_PATH)
    run_plan = read_tsv(RUN_PLAN_PATH)

    if len(observations) != EXPECTED_OBSERVATION_ROWS:
        raise RuntimeError(
            f"Observation row mismatch: {len(observations)}"
        )
    if len(run_plan) != EXPECTED_RUNS:
        raise RuntimeError(f"Run-plan mismatch: {len(run_plan)}")

    for column in (
        "configuration_observation_row",
        "genome_feature_row",
        "drug_feature_row",
        "mic_target_log2_mg_per_l",
    ):
        observations[column] = pd.to_numeric(
            observations[column],
            errors="raise",
        )

    observations["genome_feature_row"] = observations[
        "genome_feature_row"
    ].astype(np.int64)
    observations["drug_feature_row"] = observations[
        "drug_feature_row"
    ].astype(np.int64)
    observations["mic_target_log2_mg_per_l"] = observations[
        "mic_target_log2_mg_per_l"
    ].astype(np.float32)
    run_plan["seed"] = pd.to_numeric(run_plan["seed"], errors="raise").astype(
        int
    )

    return observations, run_plan


def load_feature_matrices(
    genome_representation: str,
    drug_representation: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    genome_matrix = np.load(
        KMER_PATHS[genome_representation],
        mmap_mode="r",
        allow_pickle=False,
    )
    if genome_matrix.shape[0] != EXPECTED_GENOME_ROWS:
        raise RuntimeError(f"Genome matrix shape mismatch: {genome_matrix.shape}")

    drug_matrices: dict[str, np.ndarray] = {}
    for view_id in DRUG_REPRESENTATION_VIEWS[drug_representation]:
        matrix = np.load(
            DRUG_VIEW_PATHS[view_id],
            mmap_mode="r",
            allow_pickle=False,
        )
        if matrix.shape[0] != EXPECTED_DRUG_ROWS:
            raise RuntimeError(f"Drug matrix shape mismatch: {view_id} {matrix.shape}")
        drug_matrices[view_id] = matrix

    return genome_matrix, drug_matrices


def execute_run(
    run_row: pd.Series,
    observations: pd.DataFrame,
    device: torch.device,
) -> dict[str, object]:
    run_id = str(run_row["run_id"])
    if run_complete(run_id):
        print(f"SKIP VERIFIED: {run_id}", flush=True)
        return read_tsv(run_output_directory(run_id) / "run_summary.tsv").iloc[
            0
        ].to_dict()

    outer_target = str(run_row["outer_target_code"])
    genome_representation = str(run_row["genome_representation"])
    drug_representation = str(run_row["drug_representation"])
    architecture_name = str(run_row["cross_modal_architecture"])
    architecture_id = ARCHITECTURE_NAME_TO_ID[architecture_name]
    source_species = str(run_row["source_species_code"])
    evaluation_species = str(run_row["evaluation_species_code"])
    seed = int(run_row["seed"])

    outer = observations.loc[
        observations["outer_target_code"].eq(outer_target)
    ].copy()
    source = outer.loc[
        outer["development_species_code"].eq(source_species)
    ].copy().reset_index(drop=True)
    evaluation = outer.loc[
        outer["development_species_code"].eq(evaluation_species)
    ].copy().reset_index(drop=True)

    validation = source.loc[
        source["within_species_fold"].eq(VALIDATION_FOLD)
    ].copy().reset_index(drop=True)
    early_training = source.loc[
        ~source["within_species_fold"].eq(VALIDATION_FOLD)
    ].copy().reset_index(drop=True)

    if any(frame.empty for frame in (source, evaluation, validation, early_training)):
        raise RuntimeError(f"Empty split for {run_id}")

    source_drugs = sorted(source["normalized_antibiotic"].unique().tolist())
    evaluation_drugs = sorted(
        evaluation["normalized_antibiotic"].unique().tolist()
    )
    if source_drugs != evaluation_drugs:
        raise RuntimeError(f"Drug mismatch for {run_id}")

    genome_matrix, drug_matrices = load_feature_matrices(
        genome_representation,
        drug_representation,
    )

    started = time.perf_counter()
    print(
        f"START {run_id} source={len(source)} evaluation={len(evaluation)}",
        flush=True,
    )

    best_epoch, validation_history, validation_zero_counts = (
        select_epoch_source_only(
            early_training,
            validation,
            genome_matrix,
            drug_matrices,
            architecture_id,
            seed,
            device,
        )
    )

    (
        predictions,
        source_history,
        scaler_arrays,
        source_zero_counts,
        parameter_count,
    ) = fit_full_source_and_predict(
        source,
        evaluation,
        genome_matrix,
        drug_matrices,
        architecture_id,
        seed,
        best_epoch,
        device,
    )

    target = evaluation["mic_target_log2_mg_per_l"].to_numpy(
        dtype=np.float64
    )
    pooled = regression_metrics(target, predictions)
    per_drug = per_antibiotic_metrics(evaluation, predictions)
    macro = macro_metrics(per_drug)
    elapsed_seconds = time.perf_counter() - started

    summary_record: dict[str, object] = {
        "run_id": run_id,
        "outer_target_code": outer_target,
        "configuration_id": str(run_row["configuration_id"]),
        "genome_representation": genome_representation,
        "drug_representation": drug_representation,
        "cross_modal_architecture": architecture_name,
        "architecture_id": architecture_id,
        "source_species_code": source_species,
        "evaluation_species_code": evaluation_species,
        "seed": seed,
        "source_observations": len(source),
        "source_unique_genomes": source["genome_id"].nunique(),
        "evaluation_observations": len(evaluation),
        "evaluation_unique_genomes": evaluation["genome_id"].nunique(),
        "antibiotic_count": len(source_drugs),
        "source_validation_fold": VALIDATION_FOLD,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "latent_width": LATENT_WIDTH,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": elapsed_seconds,
        **{f"pooled_{key}": value for key, value in pooled.items()},
        **macro,
    }

    result_directory = run_output_directory(run_id)
    metadata_directory = run_metadata_directory(run_id)
    result_directory.mkdir(parents=True, exist_ok=True)
    metadata_directory.mkdir(parents=True, exist_ok=True)

    summary_path = result_directory / "run_summary.tsv"
    per_drug_path = result_directory / "per_antibiotic_metrics.tsv"
    validation_history_path = result_directory / "validation_training_history.tsv"
    source_history_path = result_directory / "full_source_training_history.tsv"
    scaler_path = result_directory / "source_input_scalers.npz"
    configuration_path = metadata_directory / "configuration.tsv"
    input_manifest_path = metadata_directory / "input_manifest.tsv"

    per_drug.insert(0, "run_id", run_id)
    per_drug.insert(1, "outer_target_code", outer_target)
    per_drug.insert(2, "configuration_id", str(run_row["configuration_id"]))
    per_drug.insert(3, "source_species_code", source_species)
    per_drug.insert(4, "evaluation_species_code", evaluation_species)
    per_drug.insert(5, "seed", seed)

    write_tsv(pd.DataFrame([summary_record]), summary_path)
    write_tsv(per_drug, per_drug_path)
    write_tsv(validation_history, validation_history_path)
    write_tsv(source_history, source_history_path)
    np.savez_compressed(scaler_path, **scaler_arrays)

    configuration_records = {
        "run_id": run_id,
        "genome_representation": genome_representation,
        "drug_representation": drug_representation,
        "drug_views": "|".join(drug_matrices),
        "cross_modal_architecture": architecture_name,
        "architecture_id": architecture_id,
        "latent_width": LATENT_WIDTH,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "bilinear_rank": BILINEAR_RANK,
        "epoch_selection": (
            "source fold_05 only; model then freshly refit on complete source"
        ),
        "evaluation_label_policy": (
            "opposite development-species labels used only for final evaluation"
        ),
        "training_loss": "unweighted observation-level MSE",
        "scaling": "source-observation-weighted mean and population SD",
        "mean_sd_reporting": "sample SD ddof=1 across three seeds",
        "validation_zero_variance_counts": json.dumps(
            validation_zero_counts,
            sort_keys=True,
        ),
        "source_zero_variance_counts": json.dumps(
            source_zero_counts,
            sort_keys=True,
        ),
        "model_state_saved": "no",
    }
    write_tsv(
        pd.DataFrame(
            [
                {"item": key, "value": value}
                for key, value in configuration_records.items()
            ]
        ),
        configuration_path,
    )

    input_paths = [
        Path(__file__).resolve(),
        OBSERVATION_INDEX_PATH,
        RUN_PLAN_PATH,
        SCRIPT140_FROZEN_PATH,
        SCRIPT142_OUTPUTS_PATH,
        SCRIPT142_FROZEN_PATH,
        IMPLEMENTATION_BUNDLE_PATH,
        KMER_PATHS[genome_representation],
        *[DRUG_VIEW_PATHS[view_id] for view_id in drug_matrices],
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
        summary_path,
        per_drug_path,
        validation_history_path,
        source_history_path,
        scaler_path,
        configuration_path,
        input_manifest_path,
    ]
    manifest_path = metadata_directory / "outputs_sha256.txt"
    write_sha_manifest(output_paths, manifest_path)
    verify_sha_manifest(manifest_path)
    (metadata_directory / "RUN_COMPLETE").write_text(
        "0\n",
        encoding="utf-8",
    )

    print(
        f"COMPLETE {run_id} macro_rmse={macro['macro_rmse']:.6f} "
        f"best_epoch={best_epoch} elapsed_min={elapsed_seconds / 60.0:.2f}",
        flush=True,
    )

    del genome_matrix
    del drug_matrices
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_record


# ---------------------------------------------------------------------------
# Aggregation with mean and sample SD for all metrics
# ---------------------------------------------------------------------------


def aggregate_completed_runs() -> None:
    summaries: list[pd.DataFrame] = []
    per_drug_frames: list[pd.DataFrame] = []

    for summary_path in sorted(RESULT_ROOT.glob("*/run_summary.tsv")):
        run_id = summary_path.parent.name
        if not run_complete(run_id):
            continue
        summaries.append(read_tsv(summary_path))
        per_drug_frames.append(
            read_tsv(summary_path.parent / "per_antibiotic_metrics.tsv")
        )

    if not summaries:
        print("No completed runs available for aggregation.", flush=True)
        return

    runs = pd.concat(summaries, ignore_index=True)
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)

    metric_columns = [
        column
        for column in runs.columns
        if column.startswith("pooled_") or column.startswith("macro_")
    ]
    for column in metric_columns:
        runs[column] = pd.to_numeric(runs[column], errors="coerce")

    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    all_runs_path = AGGREGATE_ROOT / "all_direction_seed_metrics.tsv"
    write_tsv(
        runs.sort_values(
            [
                "outer_target_code",
                "configuration_id",
                "source_species_code",
                "seed",
            ]
        ),
        all_runs_path,
    )

    identity_columns = [
        "outer_target_code",
        "configuration_id",
        "genome_representation",
        "drug_representation",
        "cross_modal_architecture",
        "source_species_code",
        "evaluation_species_code",
    ]
    aggregate_metrics = [
        column
        for column in metric_columns
        if not column.endswith("_valid_antibiotics")
    ]

    direction_records: list[dict[str, object]] = []
    for keys, group in runs.groupby(identity_columns, dropna=False):
        record = {key: value for key, value in zip(identity_columns, keys)}
        record["seed_count"] = group["seed"].nunique()
        record["parameter_count"] = int(
            pd.to_numeric(group["parameter_count"], errors="raise").max()
        )
        for metric in aggregate_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            record[f"{metric}_mean"] = safe_mean(values)
            record[f"{metric}_sd"] = sample_sd(values)
        direction_records.append(record)

    direction_summary = pd.DataFrame(direction_records)
    direction_path = AGGREGATE_ROOT / "direction_three_seed_mean_sd.tsv"
    write_tsv(
        direction_summary.sort_values(
            [
                "outer_target_code",
                "configuration_id",
                "source_species_code",
            ]
        ),
        direction_path,
    )

    paired_seed_records: list[dict[str, object]] = []
    paired_grouping = [
        "outer_target_code",
        "configuration_id",
        "genome_representation",
        "drug_representation",
        "cross_modal_architecture",
        "seed",
    ]
    for keys, group in runs.groupby(paired_grouping, dropna=False):
        if len(group) != 2:
            continue
        record = {key: value for key, value in zip(paired_grouping, keys)}
        record["direction_count"] = len(group)
        record["parameter_count"] = int(
            pd.to_numeric(group["parameter_count"], errors="raise").max()
        )
        record["directions"] = "|".join(
            sorted(
                (
                    group["source_species_code"].astype(str)
                    + "_to_"
                    + group["evaluation_species_code"].astype(str)
                ).tolist()
            )
        )
        for metric in aggregate_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            record[f"bidirectional_{metric}"] = safe_mean(values)
        paired_seed_records.append(record)

    paired_seed_columns = [
        *paired_grouping,
        "direction_count",
        "parameter_count",
        "directions",
        *[f"bidirectional_{metric}" for metric in aggregate_metrics],
    ]
    paired_seed = pd.DataFrame(
        paired_seed_records,
        columns=paired_seed_columns,
    )
    paired_seed_path = AGGREGATE_ROOT / "bidirectional_seed_metrics.tsv"
    if paired_seed.empty:
        write_tsv(paired_seed, paired_seed_path)
    else:
        write_tsv(
            paired_seed.sort_values(
                ["outer_target_code", "configuration_id", "seed"]
            ),
            paired_seed_path,
        )

    config_records: list[dict[str, object]] = []
    config_grouping = [
        "outer_target_code",
        "configuration_id",
        "genome_representation",
        "drug_representation",
        "cross_modal_architecture",
    ]
    bidirectional_columns = [
        column
        for column in paired_seed.columns
        if column.startswith("bidirectional_")
    ]

    if not paired_seed.empty:
        for keys, group in paired_seed.groupby(config_grouping, dropna=False):
            record = {key: value for key, value in zip(config_grouping, keys)}
            record["seed_count"] = group["seed"].nunique()
            record["parameter_count"] = int(
                pd.to_numeric(group["parameter_count"], errors="raise").max()
            )
            for metric in bidirectional_columns:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                record[f"{metric}_mean"] = safe_mean(values)
                record[f"{metric}_sd"] = sample_sd(values)
            config_records.append(record)

    config_summary_columns = [
        *config_grouping,
        "seed_count",
        "parameter_count",
        *[
            suffix
            for metric in bidirectional_columns
            for suffix in (f"{metric}_mean", f"{metric}_sd")
        ],
    ]
    config_summary = pd.DataFrame(
        config_records,
        columns=config_summary_columns,
    )
    primary = "bidirectional_macro_rmse_mean"
    if primary in config_summary.columns:
        config_summary["provisional_rank"] = (
            config_summary.groupby("outer_target_code")[primary]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
        config_summary = config_summary.sort_values(
            ["outer_target_code", "provisional_rank", "parameter_count"]
        )

    config_path = (
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )
    write_tsv(config_summary, config_path)

    per_drug_paths: list[Path] = []
    if per_drug_frames:
        per_drug = pd.concat(per_drug_frames, ignore_index=True)
        per_drug["seed"] = pd.to_numeric(
            per_drug["seed"],
            errors="raise",
        ).astype(int)
        for metric in METRIC_IDS:
            per_drug[metric] = pd.to_numeric(
                per_drug[metric],
                errors="coerce",
            )

        all_per_drug_path = AGGREGATE_ROOT / "all_per_antibiotic_seed_metrics.tsv"
        write_tsv(
            per_drug.sort_values(
                [
                    "outer_target_code",
                    "configuration_id",
                    "source_species_code",
                    "normalized_antibiotic",
                    "seed",
                ]
            ),
            all_per_drug_path,
        )

        per_drug_records: list[dict[str, object]] = []
        group_columns = [
            "outer_target_code",
            "configuration_id",
            "source_species_code",
            "evaluation_species_code",
            "normalized_antibiotic",
        ]
        for keys, group in per_drug.groupby(group_columns, dropna=False):
            record = {key: value for key, value in zip(group_columns, keys)}
            record["seed_count"] = group["seed"].nunique()
            record["observations"] = int(
                pd.to_numeric(group["observations"], errors="raise").max()
            )
            for metric in METRIC_IDS:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                record[f"{metric}_mean"] = safe_mean(values)
                record[f"{metric}_sd"] = sample_sd(values)
            per_drug_records.append(record)

        per_drug_summary_path = (
            AGGREGATE_ROOT / "per_antibiotic_three_seed_mean_sd.tsv"
        )
        write_tsv(
            pd.DataFrame(per_drug_records).sort_values(group_columns),
            per_drug_summary_path,
        )
        per_drug_paths = [all_per_drug_path, per_drug_summary_path]

    protocol_path = METADATA_ROOT / "aggregate_protocol.tsv"
    write_tsv(
        pd.DataFrame(
            [
                {
                    "item": "primary_selection_metric",
                    "value": "bidirectional per-antibiotic macro RMSE",
                },
                {
                    "item": "direction_reporting",
                    "value": "mean and sample SD ddof=1 across three seeds",
                },
                {
                    "item": "bidirectional_pairing",
                    "value": (
                        "average the two directional metrics within each seed; "
                        "then mean and sample SD across paired seeds"
                    ),
                },
                {
                    "item": "metrics",
                    "value": "|".join(METRIC_IDS),
                },
                {
                    "item": "metric_levels",
                    "value": "pooled|per_antibiotic|per_antibiotic_macro",
                },
                {
                    "item": "configuration_status",
                    "value": (
                        "provisional k-mer-grid ranking only; AMR and kmer+AMR "
                        "candidates remain pending"
                    ),
                },
            ]
        ),
        protocol_path,
    )

    aggregate_paths = [
        all_runs_path,
        direction_path,
        paired_seed_path,
        config_path,
        protocol_path,
        *per_drug_paths,
    ]
    manifest_path = METADATA_ROOT / "aggregate_outputs_sha256.txt"
    write_sha_manifest(aggregate_paths, manifest_path)
    verify_sha_manifest(manifest_path)

    print(
        f"AGGREGATION COMPLETE: verified_runs={len(runs)} "
        f"paired_seed_rows={len(paired_seed)} configs={len(config_summary)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    arguments = parse_arguments()

    required_paths = [
        OBSERVATION_INDEX_PATH,
        RUN_PLAN_PATH,
        SCRIPT140_FROZEN_PATH,
        SCRIPT142_OUTPUTS_PATH,
        SCRIPT142_FROZEN_PATH,
        IMPLEMENTATION_BUNDLE_PATH,
        *KMER_PATHS.values(),
        *DRUG_VIEW_PATHS.values(),
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    observations, plan = load_inputs()

    if arguments.outer_target != "all":
        plan = plan.loc[
            plan["outer_target_code"].eq(arguments.outer_target)
        ]
    if arguments.genome_representation != "all":
        plan = plan.loc[
            plan["genome_representation"].eq(
                arguments.genome_representation
            )
        ]
    if arguments.drug_representation != "all":
        plan = plan.loc[
            plan["drug_representation"].eq(
                arguments.drug_representation
            )
        ]
    if arguments.architecture != "all":
        plan = plan.loc[
            plan["cross_modal_architecture"].eq(arguments.architecture)
        ]
    plan = plan.loc[plan["seed"].isin(arguments.seeds)].copy()
    plan = plan.sort_values(
        [
            "outer_target_code",
            "genome_representation",
            "drug_representation",
            "cross_modal_architecture",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    plan["already_complete"] = [
        run_complete(str(run_id))
        for run_id in plan["run_id"]
    ]

    print("===== SCRIPT 143 RUN PLAN =====")
    print(
        plan.groupby(
            [
                "outer_target_code",
                "genome_representation",
            ]
        )
        .agg(
            planned_runs=("run_id", "size"),
            completed_runs=("already_complete", "sum"),
        )
        .reset_index()
        .to_string(index=False)
    )
    print()
    print("Selected planned runs:", len(plan))
    print("Already complete:", int(plan["already_complete"].sum()))
    print("New runs remaining:", int((~plan["already_complete"]).sum()))

    if arguments.aggregate_only:
        aggregate_completed_runs()
        print("STATUS: SCRIPT 143 AGGREGATE-ONLY COMPLETE")
        return

    if arguments.dry_run:
        print("STATUS: SCRIPT 143 DRY RUN COMPLETE")
        return

    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)

    new_runs = 0
    for row in plan.to_dict(orient="records"):
        run_id = str(row["run_id"])
        if run_complete(run_id):
            print(f"SKIP VERIFIED: {run_id}", flush=True)
            continue

        execute_run(pd.Series(row), observations, device)
        new_runs += 1

        if (
            arguments.aggregate_every > 0
            and new_runs % arguments.aggregate_every == 0
        ):
            aggregate_completed_runs()

        if (
            arguments.max_new_runs > 0
            and new_runs >= arguments.max_new_runs
        ):
            aggregate_completed_runs()
            print("STATUS: SCRIPT 143 PARTIAL RUN COMPLETE")
            return

    aggregate_completed_runs()
    print("STATUS: SCRIPT 143 MODEL-SELECTION GRID COMPLETE")


