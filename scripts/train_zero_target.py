#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()

SCRIPT173_PATH = PROJECT / "src/mic_transfer/final_architectures.py"
PREREG_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script176_successful_zero_shot_preregistration_core_sha256.txt"
)
RUN_PLAN_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/zero_shot_source_checkpoints_v1/"
      "zero_shot_source_checkpoint_run_plan_v1.tsv"
)
SOURCE_REGISTRY_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1/"
      "final_transfer_source_regime_registry_v1.tsv"
)
SELECTED_CONFIGURATION_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_aggregate_v2/"
      "selected_corrective_architecture_registry.tsv"
)
OBSERVATION_INDEX_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1/"
      "final_transfer_observation_feature_index_v1.tsv.gz"
)
GENOME_FOLD_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1/"
      "target_genome_disjoint_fold_registry_v1.tsv.gz"
)
QUERY_MEMBERSHIP_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1/"
      "target_query_membership_v1.tsv.gz"
)
RESULT_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_runs_v1"
)
METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_runs_v1"
)
AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_aggregate_v1"
)
AGGREGATE_METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_aggregate_v1"
)

EXPECTED_RUNS = 27
VALIDATION_FOLD = "fold_05"
MODEL_SEEDS = (20260815, 20260816, 20260817)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else PROJECT / path


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
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_sha_manifest(paths: Iterable[Path], output_path: Path) -> None:
    unique = sorted({path.resolve() for path in paths}, key=lambda p: p.as_posix())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for path in unique:
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                display = path.relative_to(PROJECT)
            except ValueError:
                display = path
            handle.write(f"{sha256_file(path)}  {display}\n")


def verify_sha_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    verified: list[Path] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed SHA line {line_number}: {path}")
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


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def first_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise RuntimeError(f"Could not locate {label}; columns={list(frame.columns)}")


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


arch173 = load_module(SCRIPT173_PATH, "final_architecture_173_for_zero_shot")
final165 = arch173.final165
backend = arch173.backend


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_sha_manifest(PREREG_FREEZE)
    plan = read_tsv(RUN_PLAN_PATH)
    observations = read_tsv(OBSERVATION_INDEX_PATH)
    folds = read_tsv(GENOME_FOLD_PATH)

    if len(plan) != EXPECTED_RUNS:
        raise RuntimeError(f"Expected {EXPECTED_RUNS} runs; observed {len(plan)}")
    if sorted(pd.to_numeric(plan["seed"], errors="raise").astype(int).unique()) != list(MODEL_SEEDS):
        raise RuntimeError("Unexpected zero-shot model seeds.")

    require_columns(
        observations,
        [
            "observation_id",
            "species_code",
            "genome_id",
            "genome_group_id",
            "normalized_antibiotic",
            "mic_target_log2_mg_per_l",
            "genome_feature_row",
            "drug_feature_row",
        ],
        "final-transfer observation index",
    )
    for column in [
        "mic_target_log2_mg_per_l",
        "genome_feature_row",
        "drug_feature_row",
    ]:
        observations[column] = pd.to_numeric(observations[column], errors="raise")
    observations["mic_target_log2_mg_per_l"] = observations[
        "mic_target_log2_mg_per_l"
    ].astype(np.float32)
    observations["genome_feature_row"] = observations["genome_feature_row"].astype(np.int64)
    observations["drug_feature_row"] = observations["drug_feature_row"].astype(np.int64)

    # Script 175 writes the per-genome fold registry using
    # `species_code`. Accept a target-prefixed historical alias too.
    fold_species_column = first_column(
        folds,
        [
            "species_code",
            "target_species_code",
        ],
        "target genome-fold species column",
    )

    require_columns(
        folds,
        [
            fold_species_column,
            "genome_group_id",
            "genome_disjoint_fold",
        ],
        "target genome-disjoint fold registry",
    )

    fold_merge = (
        folds[
            [
                fold_species_column,
                "genome_group_id",
                "genome_disjoint_fold",
            ]
        ]
        .rename(
            columns={
                fold_species_column: "species_code",
            }
        )
        .copy()
    )

    for column in [
        "species_code",
        "genome_group_id",
        "genome_disjoint_fold",
    ]:
        fold_merge[column] = (
            fold_merge[column]
            .astype(str)
            .str.strip()
        )

    if (
        fold_merge[
            [
                "species_code",
                "genome_group_id",
                "genome_disjoint_fold",
            ]
        ]
        .eq("")
        .any()
        .any()
    ):
        raise RuntimeError(
            "Target genome-disjoint fold registry contains "
            "blank keys or fold assignments."
        )

    # The registry is genome-level. Duplicate-profile genome groups
    # can consequently occur on multiple genome rows, but every such
    # group must have one and only one frozen fold.
    conflicting_fold_counts = (
        fold_merge.groupby(
            [
                "species_code",
                "genome_group_id",
            ],
            sort=False,
        )["genome_disjoint_fold"]
        .nunique()
    )

    if conflicting_fold_counts.gt(1).any():
        examples = (
            conflicting_fold_counts.loc[
                conflicting_fold_counts.gt(1)
            ]
            .head(10)
            .index
            .tolist()
        )

        raise RuntimeError(
            "A target genome group is assigned to multiple "
            f"genome-disjoint folds: {examples}"
        )

    fold_merge = (
        fold_merge.drop_duplicates(
            [
                "species_code",
                "genome_group_id",
                "genome_disjoint_fold",
            ]
        )
        .reset_index(drop=True)
    )

    for column in [
        "species_code",
        "genome_group_id",
    ]:
        observations[column] = (
            observations[column]
            .astype(str)
            .str.strip()
        )

    if "genome_disjoint_fold" in observations.columns:
        # The final observation index already contains the frozen
        # Script 175 assignment. Audit it against the standalone
        # genome-level registry rather than replacing it.
        observations[
            "genome_disjoint_fold"
        ] = (
            observations[
                "genome_disjoint_fold"
            ]
            .astype(str)
            .str.strip()
        )

        fold_audit = fold_merge.rename(
            columns={
                "genome_disjoint_fold":
                "genome_disjoint_fold_registry",
            }
        )

        observations = observations.merge(
            fold_audit,
            on=[
                "species_code",
                "genome_group_id",
            ],
            how="left",
            validate="many_to_one",
        )

        missing_registry_assignment = (
            observations[
                "genome_disjoint_fold_registry"
            ]
            .isna()
            |
            observations[
                "genome_disjoint_fold_registry"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if missing_registry_assignment.any():
            examples = (
                observations.loc[
                    missing_registry_assignment,
                    [
                        "species_code",
                        "genome_group_id",
                    ],
                ]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )

            raise RuntimeError(
                "Some observation-index genome groups are absent "
                f"from the frozen genome-fold registry: {examples}"
            )

        fold_mismatch = (
            observations[
                "genome_disjoint_fold"
            ].astype(str)
            !=
            observations[
                "genome_disjoint_fold_registry"
            ].astype(str)
        )

        if fold_mismatch.any():
            examples = (
                observations.loc[
                    fold_mismatch,
                    [
                        "species_code",
                        "genome_group_id",
                        "genome_disjoint_fold",
                        "genome_disjoint_fold_registry",
                    ],
                ]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )

            raise RuntimeError(
                "Observation-index and standalone registry fold "
                f"assignments disagree: {examples}"
            )

        observations = observations.drop(
            columns=[
                "genome_disjoint_fold_registry",
            ]
        )

    else:
        observations = observations.merge(
            fold_merge,
            on=[
                "species_code",
                "genome_group_id",
            ],
            how="left",
            validate="many_to_one",
        )

    if (
        observations["genome_disjoint_fold"]
        .isna()
        .any()
        or
        observations[
            "genome_disjoint_fold"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Missing source-validation genome-fold assignment."
        )

    plan["seed"] = pd.to_numeric(plan["seed"], errors="raise").astype(int)
    return observations, plan, folds


def load_feature_matrices(
    run_row: pd.Series,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any], str]:
    representation = str(run_row["genome_representation"])
    if representation not in final165.CONFIG_BY_REPRESENTATION:
        raise RuntimeError(f"Unregistered genome representation: {representation}")

    spec = dict(final165.CONFIG_BY_REPRESENTATION[representation])
    for key, value in run_row.to_dict().items():
        spec[key] = value

    integer_fields = {
        "low_rank_interaction_rank",
        "drug_view_low_rank",
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
    for field in integer_fields:
        if field in spec and str(spec[field]).strip() != "":
            spec[field] = int(float(spec[field]))
    for field in float_fields:
        if field in spec and str(spec[field]).strip() != "":
            spec[field] = float(spec[field])

    final165.CURRENT_SPEC = spec
    final165.set_current_hyperparameters(spec)

    genome_path = Path(spec["genome_matrix_path"])
    if not genome_path.is_absolute():
        genome_path = PROJECT / genome_path
    genome_matrix = np.load(genome_path, mmap_mode="r", allow_pickle=False)

    drug_representation = str(run_row["drug_representation"])
    if drug_representation not in backend.DRUG_REPRESENTATION_VIEWS:
        raise RuntimeError(f"Unknown drug representation: {drug_representation}")
    drug_matrices: dict[str, np.ndarray] = {}
    for view_id in backend.DRUG_REPRESENTATION_VIEWS[drug_representation]:
        path = backend.DRUG_VIEW_PATHS[view_id]
        drug_matrices[view_id] = np.load(path, mmap_mode="r", allow_pickle=False)

    architecture_name = str(run_row["cross_modal_architecture"])
    if architecture_name not in backend.ARCHITECTURE_NAME_TO_ID:
        raise RuntimeError(f"Unknown selected architecture: {architecture_name}")
    architecture_id = backend.ARCHITECTURE_NAME_TO_ID[architecture_name]

    return genome_matrix, drug_matrices, spec, architecture_id


def build_model(
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    spec: dict[str, Any],
    target_mean: float,
    device: torch.device,
) -> nn.Module:
    model = arch173.CorrectiveArchitectureNetwork(
        genome_dimension=genome_matrix.shape[1],
        drug_matrices=drug_matrices,
        architecture_id=architecture_id,
        spec=spec,
    ).to(device)
    with torch.no_grad():
        model.intercept.fill_(float(target_mean))
    return model


def source_species_frames(frame: pd.DataFrame, source_codes: list[str]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for species_code in source_codes:
        subset = frame.loc[frame["species_code"].eq(species_code)].copy().reset_index(drop=True)
        if subset.empty:
            raise RuntimeError(f"Empty source species: {species_code}")
        output[species_code] = subset
    return output


def cyclic_batch(
    length: int,
    batch_size: int,
    rng: np.random.Generator,
    state: dict[str, Any],
) -> np.ndarray:
    if length <= 0:
        raise RuntimeError("Cannot batch an empty frame.")
    permutation = state.get("permutation")
    cursor = int(state.get("cursor", 0))
    if permutation is None or cursor >= len(permutation):
        permutation = rng.permutation(length).astype(np.int64, copy=False)
        cursor = 0
    end = min(cursor + batch_size, len(permutation))
    positions = permutation[cursor:end]
    state["permutation"] = permutation
    state["cursor"] = end
    return positions

def train_one_epoch_species_balanced(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    species_frames: dict[str, pd.DataFrame],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    genome_mean: torch.Tensor,
    genome_scale: torch.Tensor,
    drug_means: dict[str, torch.Tensor],
    drug_scales: dict[str, torch.Tensor],
    seed: int,
    epoch: int,
    batch_size: int,
    gradient_clip_norm: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    species_codes = sorted(species_frames)
    rng = np.random.default_rng(seed + 100_000 * epoch)
    squared_error = {code: 0.0 for code in species_codes}
    observations = {code: 0 for code in species_codes}
    loss_function = nn.MSELoss()

    if len(species_codes) == 1:
        code = species_codes[0]
        frame = species_frames[code]
        permutation = rng.permutation(len(frame)).astype(np.int64, copy=False)
        for start in range(0, len(permutation), batch_size):
            positions = permutation[start : start + batch_size]
            genome, drug_views, target = backend.make_batch(
                frame, positions, genome_matrix, drug_matrices,
                genome_mean, genome_scale, drug_means, drug_scales,
                device, include_target=True,
            )
            assert target is not None
            optimizer.zero_grad(set_to_none=True)
            prediction = model(genome, drug_views)
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite source training loss.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite source training gradient norm.")
            optimizer.step()
            squared_error[code] += float(loss.detach().cpu()) * len(target)
            observations[code] += len(target)
    else:
        maximum_length = max(len(frame) for frame in species_frames.values())
        steps = max(1, math.ceil(maximum_length / batch_size))
        states = {code: {} for code in species_codes}
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for code in species_codes:
                frame = species_frames[code]
                positions = cyclic_batch(len(frame), batch_size, rng, states[code])
                genome, drug_views, target = backend.make_batch(
                    frame, positions, genome_matrix, drug_matrices,
                    genome_mean, genome_scale, drug_means, drug_scales,
                    device, include_target=True,
                )
                assert target is not None
                prediction = model(genome, drug_views)
                loss = loss_function(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite source training loss.")
                (loss / len(species_codes)).backward()
                squared_error[code] += float(loss.detach().cpu()) * len(target)
                observations[code] += len(target)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Non-finite source training gradient norm.")
            optimizer.step()

    return {
        code: math.sqrt(squared_error[code] / observations[code])
        for code in species_codes
    }


@torch.no_grad()
def validation_objective(
    model: nn.Module,
    validation_frames: dict[str, pd.DataFrame],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    genome_mean: torch.Tensor,
    genome_scale: torch.Tensor,
    drug_means: dict[str, torch.Tensor],
    drug_scales: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for species_code, frame in sorted(validation_frames.items()):
        predictions = backend.predict(
            model,
            frame,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            device,
        )
        per_drug = backend.per_antibiotic_metrics(frame, predictions)
        macro = backend.macro_metrics(per_drug)
        scores[species_code] = float(macro["macro_rmse"])
    return float(np.mean(list(scores.values()))), scores


def fit_scalers_and_tensors(
    frame: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    device: torch.device,
):
    arrays = backend.fit_scalers(frame, genome_matrix, drug_matrices)
    genome_mean_np, genome_scale_np, drug_means_np, drug_scales_np, zero_counts = arrays
    tensors = backend.scaler_tensors(
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        device,
    )
    return arrays, tensors, zero_counts


def select_epoch(
    early_training: pd.DataFrame,
    validation: pd.DataFrame,
    source_codes: list[str],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[int, pd.DataFrame, dict[str, int]]:
    set_seed(seed)
    arrays, tensors, zero_counts = fit_scalers_and_tensors(
        early_training, genome_matrix, drug_matrices, device
    )
    del arrays
    genome_mean, genome_scale, drug_means, drug_scales = tensors
    model = build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        float(early_training["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    training_frames = source_species_frames(early_training, source_codes)
    validation_frames = source_species_frames(validation, source_codes)

    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []

    for epoch in range(1, int(spec["maximum_epochs"]) + 1):
        started = time.perf_counter()
        train_rmse = train_one_epoch_species_balanced(
            model,
            optimizer,
            training_frames,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            seed,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        objective, validation_scores = validation_objective(
            model,
            validation_frames,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            device,
        )
        if objective < best_objective - float(spec["minimum_rmse_improvement"]):
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record: dict[str, Any] = {
            "epoch": epoch,
            "validation_equal_species_macro_rmse": objective,
            "best_validation_objective_so_far": best_objective,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": time.perf_counter() - started,
        }
        for code, value in train_rmse.items():
            record[f"training_rmse_{code}"] = value
        for code, value in validation_scores.items():
            record[f"validation_macro_rmse_{code}"] = value
        records.append(record)

        if epochs_without_improvement >= int(spec["early_stopping_patience"]):
            break

    if best_epoch < 1:
        raise RuntimeError("No source-only best epoch selected.")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, pd.DataFrame(records), zero_counts


def fit_full_source(
    source: pd.DataFrame,
    source_codes: list[str],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    spec: dict[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
):
    set_seed(seed)
    arrays, tensors, zero_counts = fit_scalers_and_tensors(
        source, genome_matrix, drug_matrices, device
    )
    genome_mean_np, genome_scale_np, drug_means_np, drug_scales_np, _ = arrays
    genome_mean, genome_scale, drug_means, drug_scales = tensors
    model = build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        float(source["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    source_frames = source_species_frames(source, source_codes)
    records: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_rmse = train_one_epoch_species_balanced(
            model,
            optimizer,
            source_frames,
            genome_matrix,
            drug_matrices,
            genome_mean,
            genome_scale,
            drug_means,
            drug_scales,
            seed + 10_000_000,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        record: dict[str, Any] = {"epoch": epoch, "epoch_seconds": time.perf_counter() - started}
        for code, value in train_rmse.items():
            record[f"training_rmse_{code}"] = value
        records.append(record)

    scaler_arrays: dict[str, np.ndarray] = {
        "genome__mean": genome_mean_np,
        "genome__scale": genome_scale_np,
    }
    for view_id in drug_matrices:
        scaler_arrays[f"{view_id}__mean"] = drug_means_np[view_id]
        scaler_arrays[f"{view_id}__scale"] = drug_scales_np[view_id]

    return (
        model,
        pd.DataFrame(records),
        scaler_arrays,
        tensors,
        zero_counts,
    )


def run_result_directory(run_id: str) -> Path:
    return RESULT_ROOT / run_id


def run_metadata_directory(run_id: str) -> Path:
    return METADATA_ROOT / run_id


def run_complete(run_id: str) -> bool:
    metadata = run_metadata_directory(run_id)
    flag = metadata / "RUN_COMPLETE"
    manifest = metadata / "outputs_sha256.txt"
    if not flag.is_file() or not manifest.is_file():
        return False
    try:
        verify_sha_manifest(manifest)
    except Exception:
        return False
    return flag.read_text(encoding="utf-8").strip() == "0"


def panel_metric_records(
    target: pd.DataFrame,
    predictions: np.ndarray,
    source_drugs: set[str],
) -> pd.DataFrame:
    target = target.copy()
    target["prediction"] = predictions
    panels = {
        "full_target_panel": pd.Series(True, index=target.index),
        "source_shared_drugs": target["normalized_antibiotic"].isin(source_drugs),
        "source_unseen_in_mic_training": ~target["normalized_antibiotic"].isin(source_drugs),
    }
    records: list[dict[str, Any]] = []
    for panel_id, mask in panels.items():
        subset = target.loc[mask].copy().reset_index(drop=True)
        if subset.empty:
            records.append(
                {
                    "panel_id": panel_id,
                    "observations": 0,
                    "unique_antibiotics": 0,
                    "macro_rmse": float("nan"),
                }
            )
            continue
        per_drug = backend.per_antibiotic_metrics(
            subset, subset["prediction"].to_numpy(dtype=np.float32)
        )
        pooled = backend.regression_metrics(
            subset["mic_target_log2_mg_per_l"].to_numpy(dtype=np.float64),
            subset["prediction"].to_numpy(dtype=np.float64),
        )
        macro = backend.macro_metrics(per_drug)
        records.append(
            {
                "panel_id": panel_id,
                "observations": len(subset),
                "unique_antibiotics": subset["normalized_antibiotic"].nunique(),
                **{f"pooled_{key}": value for key, value in pooled.items()},
                **macro,
            }
        )
    return pd.DataFrame(records)


def execute_run(
    row: pd.Series,
    observations: pd.DataFrame,
    device: torch.device,
) -> None:
    run_id = str(row["run_id"])
    if run_complete(run_id):
        print(f"SKIP VERIFIED: {run_id}", flush=True)
        return

    source_codes = [value for value in str(row["source_species_codes"]).split("|") if value]
    target_code = str(row["outer_target_code"])
    seed = int(row["seed"])
    source = observations.loc[observations["species_code"].isin(source_codes)].copy().reset_index(drop=True)
    target = observations.loc[observations["species_code"].eq(target_code)].copy().reset_index(drop=True)
    validation = source.loc[source["genome_disjoint_fold"].eq(VALIDATION_FOLD)].copy().reset_index(drop=True)
    early_training = source.loc[~source["genome_disjoint_fold"].eq(VALIDATION_FOLD)].copy().reset_index(drop=True)

    if any(frame.empty for frame in [source, target, validation, early_training]):
        raise RuntimeError(f"Empty zero-shot split for {run_id}")
    for code in source_codes:
        if not validation["species_code"].eq(code).any():
            raise RuntimeError(f"Validation fold missing source species {code}: {run_id}")
        if not early_training["species_code"].eq(code).any():
            raise RuntimeError(f"Early training missing source species {code}: {run_id}")

    genome_matrix, drug_matrices, spec, architecture_id = load_feature_matrices(row)
    started = time.perf_counter()
    print(
        f"START {run_id} source={len(source)} target={len(target)} "
        f"source_species={'|'.join(source_codes)}",
        flush=True,
    )

    best_epoch, validation_history, validation_zero_counts = select_epoch(
        early_training,
        validation,
        source_codes,
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        seed,
        device,
    )
    (
        model,
        full_history,
        scaler_arrays,
        scaler_tensors,
        full_zero_counts,
    ) = fit_full_source(
        source,
        source_codes,
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        seed,
        best_epoch,
        device,
    )
    genome_mean, genome_scale, drug_means, drug_scales = scaler_tensors
    predictions = backend.predict(
        model,
        target,
        genome_matrix,
        drug_matrices,
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
        device,
    )

    per_drug = backend.per_antibiotic_metrics(target, predictions)
    pooled = backend.regression_metrics(
        target["mic_target_log2_mg_per_l"].to_numpy(dtype=np.float64), predictions
    )
    macro = backend.macro_metrics(per_drug)
    source_drugs = set(source["normalized_antibiotic"].astype(str))
    species_drug_sets = [
        set(group["normalized_antibiotic"].astype(str))
        for _, group in observations.groupby("species_code", sort=True)
    ]
    panels = panel_metric_records(target, predictions, source_drugs)

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    elapsed = time.perf_counter() - started
    summary = {
        "run_id": run_id,
        "outer_target_code": target_code,
        "source_regime_id": str(row["source_regime_id"]),
        "source_species_codes": "|".join(source_codes),
        "seed": seed,
        "source_observations": len(source),
        "source_unique_genomes": source["genome_id"].nunique(),
        "source_antibiotic_count": source["normalized_antibiotic"].nunique(),
        "target_observations": len(target),
        "target_unique_genomes": target["genome_id"].nunique(),
        "target_antibiotic_count": target["normalized_antibiotic"].nunique(),
        "source_validation_fold": VALIDATION_FOLD,
        "source_validation_policy": (
            "fold_05 from each source species; epoch objective is equal-species "
            "mean of per-antibiotic macro RMSE"
        ),
        "multi_source_sampling": (
            "equal species contribution per optimiser step via cyclic balanced batches"
        ),
        "best_epoch": best_epoch,
        "model_parameter_count": parameter_count,
        "genome_representation": str(row["genome_representation"]),
        "drug_representation": str(row["drug_representation"]),
        "drug_view_fusion_method": str(row["drug_view_fusion_method"]),
        "cross_modal_architecture": str(row["cross_modal_architecture"]),
        "latent_width": spec["latent_width"],
        "dropout": spec["dropout"],
        "learning_rate": spec["learning_rate"],
        "weight_decay": spec["weight_decay"],
        "batch_size": spec["batch_size"],
        "elapsed_seconds": elapsed,
        **{f"pooled_{key}": value for key, value in pooled.items()},
        **macro,
    }

    result_dir = run_result_directory(run_id)
    metadata_dir = run_metadata_directory(run_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "run_summary.tsv"
    per_drug_path = result_dir / "per_antibiotic_metrics.tsv"
    panel_path = result_dir / "panel_metrics.tsv"
    prediction_path = result_dir / "target_predictions.tsv"
    validation_history_path = result_dir / "source_validation_training_history.tsv"
    full_history_path = result_dir / "full_source_training_history.tsv"
    scaler_path = result_dir / "source_input_scalers.npz"
    checkpoint_path = result_dir / "source_checkpoint_state_dict.pt"
    configuration_path = metadata_dir / "configuration.json"
    input_manifest_path = metadata_dir / "input_manifest.tsv"
    output_manifest_path = metadata_dir / "outputs_sha256.txt"
    complete_path = metadata_dir / "RUN_COMPLETE"

    write_tsv(pd.DataFrame([summary]), summary_path)
    per_drug.insert(0, "run_id", run_id)
    write_tsv(per_drug, per_drug_path)
    panels.insert(0, "run_id", run_id)
    write_tsv(panels, panel_path)

    prediction_columns = [
        "observation_id",
        "species_code",
        "genome_id",
        "normalized_antibiotic",
        "mic_target_log2_mg_per_l",
        "genome_feature_row",
        "drug_feature_row",
    ]
    prediction_frame = target[prediction_columns].copy()
    prediction_frame.insert(0, "run_id", run_id)
    prediction_frame.insert(1, "source_regime_id", str(row["source_regime_id"]))
    prediction_frame.insert(2, "seed", seed)
    prediction_frame["zero_shot_prediction"] = predictions
    write_tsv(prediction_frame, prediction_path)
    write_tsv(validation_history, validation_history_path)
    write_tsv(full_history, full_history_path)
    np.savez_compressed(scaler_path, **scaler_arrays)

    checkpoint = {
        "format_version": 1,
        "run_id": run_id,
        "outer_target_code": target_code,
        "source_regime_id": str(row["source_regime_id"]),
        "source_species_codes": source_codes,
        "seed": seed,
        "best_epoch": best_epoch,
        "architecture_id": architecture_id,
        "configuration": spec,
        "drug_view_order": list(drug_matrices),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(checkpoint, checkpoint_path)
    write_json(
        {
            "run_plan_row": row.to_dict(),
            "resolved_specification": spec,
            "architecture_id": architecture_id,
            "drug_views": list(drug_matrices),
            "validation_zero_variance_counts": validation_zero_counts,
            "full_source_zero_variance_counts": full_zero_counts,
            "target_label_use": "evaluation metrics only; zero target labels used for training or epoch selection",
        },
        configuration_path,
    )

    input_paths = [
        Path(__file__).resolve(),
        SCRIPT173_PATH,
        RUN_PLAN_PATH,
        SOURCE_REGISTRY_PATH,
        SELECTED_CONFIGURATION_PATH,
        OBSERVATION_INDEX_PATH,
        GENOME_FOLD_PATH,
        PREREG_FREEZE,
        Path(spec["genome_matrix_path"]) if Path(spec["genome_matrix_path"]).is_absolute() else PROJECT / Path(spec["genome_matrix_path"]),
        *[backend.DRUG_VIEW_PATHS[view_id] for view_id in drug_matrices],
    ]
    manifest_rows = [
        {
            "file_path": str(path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted({path.resolve() for path in input_paths}, key=lambda p: p.as_posix())
    ]
    write_tsv(pd.DataFrame(manifest_rows), input_manifest_path)

    output_paths = [
        summary_path,
        per_drug_path,
        panel_path,
        prediction_path,
        validation_history_path,
        full_history_path,
        scaler_path,
        checkpoint_path,
        configuration_path,
        input_manifest_path,
    ]
    write_sha_manifest(output_paths, output_manifest_path)
    verify_sha_manifest(output_manifest_path)
    complete_path.write_text("0\n", encoding="utf-8")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"COMPLETE {run_id} best_epoch={best_epoch} "
        f"target_macro_rmse={float(macro['macro_rmse']):.6f}",
        flush=True,
    )


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else float("nan")


def sample_sd(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) >= 2 else float("nan")


def metric_summary(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["seed_count"] = group["seed"].nunique() if "seed" in group else len(group)
        for metric in metric_columns:
            record[f"{metric}_mean"] = safe_mean(group[metric])
            record[f"{metric}_sd"] = sample_sd(group[metric])
        records.append(record)
    return pd.DataFrame(records)


def aggregate_completed_runs() -> None:
    plan = read_tsv(RUN_PLAN_PATH)
    missing = [run_id for run_id in plan["run_id"] if not run_complete(str(run_id))]
    if missing:
        raise RuntimeError(f"Cannot aggregate: {len(missing)} zero-shot runs incomplete; first={missing[:3]}")

    summaries: list[pd.DataFrame] = []
    per_drug_frames: list[pd.DataFrame] = []
    panel_frames: list[pd.DataFrame] = []
    prediction_paths: dict[str, Path] = {}
    for run_id in plan["run_id"].astype(str):
        result_dir = run_result_directory(run_id)
        summaries.append(read_tsv(result_dir / "run_summary.tsv"))
        per_drug_frames.append(read_tsv(result_dir / "per_antibiotic_metrics.tsv"))
        panel_frames.append(read_tsv(result_dir / "panel_metrics.tsv"))
        prediction_paths[run_id] = result_dir / "target_predictions.tsv"

    all_summary = pd.concat(summaries, ignore_index=True)
    all_per_drug = pd.concat(per_drug_frames, ignore_index=True)
    all_panel = pd.concat(panel_frames, ignore_index=True)
    for frame in [all_summary, all_per_drug, all_panel]:
        if "seed" in frame.columns:
            frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)

    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    AGGREGATE_METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_summary_path = AGGREGATE_ROOT / "all_zero_shot_run_metrics.tsv"
    all_per_drug_path = AGGREGATE_ROOT / "all_zero_shot_per_antibiotic_metrics.tsv"
    all_panel_path = AGGREGATE_ROOT / "all_zero_shot_panel_metrics.tsv"
    write_tsv(all_summary, all_summary_path)
    write_tsv(all_per_drug, all_per_drug_path)
    write_tsv(all_panel, all_panel_path)

    metric_columns = [
        column
        for column in all_panel.columns
        if column.startswith("macro_") or column.startswith("pooled_")
    ]
    panel_seed_summary = metric_summary(
        all_panel,
        ["outer_target_code", "source_regime_id", "panel_id"],
        metric_columns,
    )
    panel_seed_summary_path = AGGREGATE_ROOT / "zero_shot_panel_three_seed_mean_sd.tsv"
    write_tsv(panel_seed_summary, panel_seed_summary_path)

    query_membership = read_tsv(QUERY_MEMBERSHIP_PATH)
    observation_column = first_column(query_membership, ["observation_id"], "query observation ID")
    target_column = first_column(
        query_membership,
        ["target_species_code", "species_code"],
        "query target species",
    )
    protocol_column = first_column(
        query_membership,
        ["target_protocol", "protocol_id", "protocol"],
        "query protocol",
    )
    query_id_column = first_column(
        query_membership,
        ["query_id", "query_fold_id", "fold_id"],
        "query identifier",
    )
    query_membership = query_membership[
        [observation_column, target_column, protocol_column, query_id_column]
    ].rename(
        columns={
            observation_column: "observation_id",
            target_column: "outer_target_code",
            protocol_column: "target_protocol",
            query_id_column: "query_id",
        }
    )

    query_records: list[dict[str, Any]] = []
    plan_lookup = plan.set_index("run_id")
    for run_id, prediction_path in prediction_paths.items():
        prediction = read_tsv(prediction_path)
        row = plan_lookup.loc[run_id]
        target_code = str(row["outer_target_code"])
        membership = query_membership.loc[
            query_membership["outer_target_code"].eq(target_code)
        ]
        joined = membership.merge(
            prediction,
            on="observation_id",
            how="left",
            validate="many_to_one",
        )
        if joined["zero_shot_prediction"].eq("").any() or joined["zero_shot_prediction"].isna().any():
            raise RuntimeError(f"Missing prediction after query join: {run_id}")
        joined["zero_shot_prediction"] = pd.to_numeric(
            joined["zero_shot_prediction"], errors="raise"
        )
        joined["mic_target_log2_mg_per_l"] = pd.to_numeric(
            joined["mic_target_log2_mg_per_l"], errors="raise"
        )
        for (protocol, query_id), subset in joined.groupby(
            ["target_protocol", "query_id"], sort=True
        ):
            subset = subset.reset_index(drop=True)
            per_drug = backend.per_antibiotic_metrics(
                subset, subset["zero_shot_prediction"].to_numpy(dtype=np.float32)
            )
            pooled = backend.regression_metrics(
                subset["mic_target_log2_mg_per_l"].to_numpy(dtype=np.float64),
                subset["zero_shot_prediction"].to_numpy(dtype=np.float64),
            )
            macro = backend.macro_metrics(per_drug)
            query_records.append(
                {
                    "run_id": run_id,
                    "outer_target_code": target_code,
                    "source_regime_id": str(row["source_regime_id"]),
                    "seed": int(row["seed"]),
                    "target_protocol": protocol,
                    "query_id": query_id,
                    "query_observations": len(subset),
                    "query_unique_genomes": subset["genome_id"].nunique(),
                    "query_unique_antibiotics": subset["normalized_antibiotic"].nunique(),
                    **{f"pooled_{key}": value for key, value in pooled.items()},
                    **macro,
                }
            )

    query_metrics = pd.DataFrame(query_records)
    query_metrics_path = AGGREGATE_ROOT / "zero_shot_query_seed_metrics.tsv"
    write_tsv(query_metrics, query_metrics_path)

    query_metric_columns = [
        column
        for column in query_metrics.columns
        if column.startswith("macro_") or column.startswith("pooled_")
    ]
    fold_seed_mean_records: list[dict[str, Any]] = []
    group_columns = [
        "outer_target_code",
        "source_regime_id",
        "target_protocol",
        "query_id",
    ]
    for keys, group in query_metrics.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys))
        record["seed_count"] = group["seed"].nunique()
        for metric in query_metric_columns:
            record[metric] = safe_mean(group[metric])
        fold_seed_mean_records.append(record)
    fold_seed_means = pd.DataFrame(fold_seed_mean_records)
    fold_seed_means_path = AGGREGATE_ROOT / "zero_shot_query_seed_averaged_fold_metrics.tsv"
    write_tsv(fold_seed_means, fold_seed_means_path)

    protocol_summary_records: list[dict[str, Any]] = []
    protocol_groups = ["outer_target_code", "source_regime_id", "target_protocol"]
    for keys, group in fold_seed_means.groupby(protocol_groups, sort=True):
        record = dict(zip(protocol_groups, keys))
        record["query_count"] = group["query_id"].nunique()
        for metric in query_metric_columns:
            record[f"{metric}_mean_across_queries"] = safe_mean(group[metric])
            record[f"{metric}_sd_across_queries"] = sample_sd(group[metric])
        protocol_summary_records.append(record)
    protocol_summary = pd.DataFrame(protocol_summary_records)
    protocol_summary_path = AGGREGATE_ROOT / "zero_shot_protocol_fold_level_summary.tsv"
    write_tsv(protocol_summary, protocol_summary_path)

    protocol = pd.DataFrame(
        [
            {"item": "completed_checkpoint_runs", "value": EXPECTED_RUNS},
            {"item": "target_label_use", "value": "zero target labels used for source training or epoch selection"},
            {"item": "source_epoch_selection", "value": "fold_05 genome-disjoint validation in each source species"},
            {"item": "joint_source_validation_objective", "value": "equal-species mean of per-antibiotic macro RMSE"},
            {"item": "joint_source_training", "value": "cyclic balanced batches; equal source-species loss contribution"},
            {"item": "fold_uncertainty", "value": "average three seeds inside each query fold, then mean and sample SD across query folds"},
            {"item": "worst_direction_metric", "value": "not used in final one-way transfer"},
        ]
    )
    protocol_path = AGGREGATE_METADATA_ROOT / "aggregate_protocol.tsv"
    write_tsv(protocol, protocol_path)

    output_paths = [
        all_summary_path,
        all_per_drug_path,
        all_panel_path,
        panel_seed_summary_path,
        query_metrics_path,
        fold_seed_means_path,
        protocol_summary_path,
        protocol_path,
    ]
    manifest_path = AGGREGATE_METADATA_ROOT / "aggregate_outputs_sha256.txt"
    write_sha_manifest(output_paths, manifest_path)
    verify_sha_manifest(manifest_path)

    selected_display = panel_seed_summary.loc[
        panel_seed_summary["panel_id"].eq("full_target_panel"),
        [
            "outer_target_code",
            "source_regime_id",
            "macro_rmse_mean",
            "macro_rmse_sd",
        ],
    ].sort_values(["outer_target_code", "macro_rmse_mean"])
    print("===== ZERO-SHOT FULL-TARGET PANEL =====")
    print(selected_display.to_string(index=False))
    print(
        f"AGGREGATION COMPLETE: verified_runs={EXPECTED_RUNS} "
        f"query_seed_rows={len(query_metrics)}",
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-target", choices=["all", "ec", "kp", "se"], default="all")
    parser.add_argument("--source-regime", default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--max-new-runs", type=int, default=0)
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    observations, plan, _ = load_inputs()

    if args.aggregate_only:
        aggregate_completed_runs()
        print("STATUS: SCRIPT 177 ZERO-SHOT AGGREGATE-ONLY COMPLETE")
        return

    if args.outer_target != "all":
        plan = plan.loc[plan["outer_target_code"].eq(args.outer_target)]
    if args.source_regime != "all":
        plan = plan.loc[plan["source_regime_id"].eq(args.source_regime)]
    plan = plan.loc[plan["seed"].isin(args.seeds)].reset_index(drop=True)
    if plan.empty:
        raise RuntimeError("No zero-shot runs selected.")

    if args.device == "cuda":
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

    new_runs = 0
    for _, row in plan.iterrows():
        if run_complete(str(row["run_id"])):
            print(f"SKIP VERIFIED: {row['run_id']}", flush=True)
            continue
        execute_run(row, observations, device)
        new_runs += 1
        if args.max_new_runs > 0 and new_runs >= args.max_new_runs:
            print("STATUS: SCRIPT 177 PARTIAL ZERO-SHOT RUN COMPLETE")
            return

    if args.worker_only:
        print("STATUS: SCRIPT 177 ZERO-SHOT WORKER PARTITION COMPLETE")
        return

    aggregate_completed_runs()
    print("STATUS: SCRIPT 177 ZERO-SHOT CHECKPOINTS AND EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
