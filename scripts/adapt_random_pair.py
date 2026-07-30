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

SCRIPT177_PATH = PROJECT / "scripts/train_zero_target.py"
EXPECTED_SCRIPT177_SHA256 = (
    "c9a916b48e19486729fe078c250efe72407533c368fee3b5b2d77e8f6bbde1d7"
)
PREREG_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script179_successful_random_pair_few_shot_preregistration_core_sha256.txt"
)
RUN_PLAN_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/random_pair_few_shot_v1/"
      "random_pair_few_shot_run_plan_v1.tsv"
)
QUERY_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1/"
      "target_query_membership_v1.tsv.gz"
)
SUPPORT_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1/"
      "target_nested_support_membership_v1.tsv.gz"
)
ZERO_PLAN_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/zero_shot_source_checkpoints_v1/"
      "zero_shot_source_checkpoint_run_plan_v1.tsv"
)
ZERO_RESULT_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_runs_v1"
)
ZERO_METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "zero_shot_source_checkpoints_runs_v1"
)
RESULT_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "random_pair_few_shot_runs_v1"
)
METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "random_pair_few_shot_runs_v1"
)

EXPECTED_RUNS = 540
INNER_SPLIT_SEED = 20260818
PRETRAINED_LR_MULTIPLIER = 0.1
PRETRAINED_MAX_EPOCHS = 100
PRETRAINED_PATIENCE = 12
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted({path.resolve() for path in paths}, key=lambda p: p.as_posix())
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
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
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


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stable_int(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


if not SCRIPT177_PATH.is_file():
    raise FileNotFoundError(SCRIPT177_PATH)
if sha256_file(SCRIPT177_PATH) != EXPECTED_SCRIPT177_SHA256:
    raise RuntimeError(
        "Frozen Script 177 SHA mismatch: "
        f"{sha256_file(SCRIPT177_PATH)}"
    )
zero177 = load_module(SCRIPT177_PATH, "frozen_zero_shot_177_for_random_pair_few_shot")
backend = zero177.backend


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_sha_manifest(PREREG_FREEZE)
    observations, _, _ = zero177.load_inputs()
    plan = read_tsv(RUN_PLAN_PATH)
    zero_plan = read_tsv(ZERO_PLAN_PATH)
    query = read_tsv(QUERY_PATH)
    support = read_tsv(SUPPORT_PATH)

    require_columns(
        plan,
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "source_species_codes",
            "query_id",
            "support_budget_percent",
            "seed",
            "zero_shot_run_id",
            "configuration_zero_shot_run_id",
        ],
        "random-pair few-shot run plan",
    )
    if len(plan) != EXPECTED_RUNS or plan["run_id"].duplicated().any():
        raise RuntimeError(f"Invalid random-pair few-shot run plan: {len(plan)}")
    plan["support_budget_percent"] = pd.to_numeric(
        plan["support_budget_percent"], errors="raise"
    ).astype(int)
    plan["seed"] = pd.to_numeric(plan["seed"], errors="raise").astype(int)
    if sorted(plan["seed"].unique()) != list(MODEL_SEEDS):
        raise RuntimeError("Unexpected random-pair few-shot seeds.")

    zero_plan["seed"] = pd.to_numeric(zero_plan["seed"], errors="raise").astype(int)
    if zero_plan["run_id"].duplicated().any():
        raise RuntimeError("Duplicate zero-shot run IDs.")

    require_columns(
        query,
        ["target_species_code", "target_protocol", "query_id", "observation_id"],
        "query membership",
    )
    require_columns(
        support,
        [
            "target_species_code",
            "target_protocol",
            "query_id",
            "support_budget_percent",
            "observation_id",
            "support_rank",
        ],
        "support membership",
    )
    query = query.loc[query["target_protocol"].eq("random_pair")].copy()
    support = support.loc[support["target_protocol"].eq("random_pair")].copy()
    support["support_budget_percent"] = pd.to_numeric(
        support["support_budget_percent"], errors="raise"
    ).astype(int)
    support["support_rank"] = pd.to_numeric(
        support["support_rank"], errors="raise"
    ).astype(int)

    observation_ids = set(observations["observation_id"].astype(str))
    missing_query = sorted(set(query["observation_id"].astype(str)) - observation_ids)
    missing_support = sorted(set(support["observation_id"].astype(str)) - observation_ids)
    if missing_query or missing_support:
        raise RuntimeError(
            "Membership references missing observations: "
            f"query={missing_query[:3]} support={missing_support[:3]}"
        )

    return observations, plan, zero_plan, query, support


def run_result_directory(run_id: str) -> Path:
    return RESULT_ROOT / run_id


def run_metadata_directory(run_id: str) -> Path:
    return METADATA_ROOT / run_id


def run_complete(run_id: str) -> bool:
    metadata_dir = run_metadata_directory(run_id)
    complete = metadata_dir / "RUN_COMPLETE"
    manifest = metadata_dir / "outputs_sha256.txt"
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != "0":
        return False
    try:
        verify_sha_manifest(manifest)
    except Exception:
        return False
    return True


def zero_paths(zero_run_id: str) -> dict[str, Path]:
    result_dir = ZERO_RESULT_ROOT / zero_run_id
    metadata_dir = ZERO_METADATA_ROOT / zero_run_id
    paths = {
        "checkpoint": result_dir / "source_checkpoint_state_dict.pt",
        "scalers": result_dir / "source_input_scalers.npz",
        "predictions": result_dir / "target_predictions.tsv",
        "summary": result_dir / "run_summary.tsv",
        "manifest": metadata_dir / "outputs_sha256.txt",
        "complete": metadata_dir / "RUN_COMPLETE",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing zero-shot {key}: {path}")
    if paths["complete"].read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError(f"Zero-shot run is not complete: {zero_run_id}")
    verify_sha_manifest(paths["manifest"])
    return paths


def deterministic_inner_split(
    support: pd.DataFrame,
    target_code: str,
    query_id: str,
    budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    audit_records: list[dict[str, object]] = []

    for antibiotic, group in support.groupby("normalized_antibiotic", sort=True):
        group = group.copy()
        group["inner_hash"] = [
            stable_int(
                f"inner|{INNER_SPLIT_SEED}|{target_code}|{query_id}|"
                f"b{budget}|{antibiotic}|{observation_id}"
            )
            for observation_id in group["observation_id"].astype(str)
        ]
        group = group.sort_values(
            ["inner_hash", "observation_id"], kind="stable"
        ).reset_index(drop=True)
        count = len(group)
        if count >= 2:
            validation_count = min(count - 1, max(1, int(math.floor(0.20 * count + 0.5))))
        else:
            validation_count = 0
        validation = group.iloc[:validation_count].drop(columns=["inner_hash"])
        training = group.iloc[validation_count:].drop(columns=["inner_hash"])
        if training.empty:
            raise RuntimeError(
                f"Empty inner training split: {target_code}/{query_id}/{budget}/{antibiotic}"
            )
        training_parts.append(training)
        if not validation.empty:
            validation_parts.append(validation)
        audit_records.append(
            {
                "normalized_antibiotic": antibiotic,
                "support_observations": count,
                "inner_training_observations": len(training),
                "inner_validation_observations": len(validation),
            }
        )

    inner_training = pd.concat(training_parts, ignore_index=True)
    inner_validation = pd.concat(validation_parts, ignore_index=True)
    audit = pd.DataFrame(audit_records)
    if inner_training.empty or inner_validation.empty:
        raise RuntimeError(
            f"Empty inner training/validation set: {target_code}/{query_id}/{budget}"
        )
    overlap = set(inner_training["observation_id"]).intersection(
        set(inner_validation["observation_id"])
    )
    if overlap:
        raise RuntimeError("Inner training/validation observation overlap.")
    if len(inner_training) + len(inner_validation) != len(support):
        raise RuntimeError("Inner split row-count mismatch.")
    return inner_training, inner_validation, audit


def metric_bundle(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    per_drug = backend.per_antibiotic_metrics(frame, predictions)
    pooled = backend.regression_metrics(
        frame["mic_target_log2_mg_per_l"].to_numpy(dtype=np.float64),
        predictions.astype(np.float64, copy=False),
    )
    macro = backend.macro_metrics(per_drug)
    return (
        {
            **{f"pooled_{key}": value for key, value in pooled.items()},
            **macro,
        },
        per_drug,
    )


def validation_macro_rmse(
    model: nn.Module,
    validation: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    scaler_tensors,
    device: torch.device,
) -> float:
    genome_mean, genome_scale, drug_means, drug_scales = scaler_tensors
    predictions = backend.predict(
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
    per_drug = backend.per_antibiotic_metrics(validation, predictions)
    macro = backend.macro_metrics(per_drug)
    return float(macro["macro_rmse"])


def train_target_only_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    frame: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    scaler_tensors,
    seed: int,
    epoch: int,
    batch_size: int,
    gradient_clip_norm: float,
    device: torch.device,
) -> float:
    model.train()
    genome_mean, genome_scale, drug_means, drug_scales = scaler_tensors
    rng = np.random.default_rng(seed + 100_000 * epoch)
    permutation = rng.permutation(len(frame)).astype(np.int64, copy=False)
    squared_error = 0.0
    observations = 0
    loss_function = nn.MSELoss()
    for start in range(0, len(permutation), batch_size):
        positions = permutation[start : start + batch_size]
        genome, drug_views, target = backend.make_batch(
            frame,
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
            raise RuntimeError("Non-finite target-only training loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Non-finite target-only gradient norm.")
        optimizer.step()
        squared_error += float(loss.detach().cpu()) * len(target)
        observations += len(target)
    return math.sqrt(squared_error / observations)


def train_pretrained_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    target_frame: pd.DataFrame,
    source_frames: dict[str, pd.DataFrame],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    scaler_tensors,
    seed: int,
    epoch: int,
    batch_size: int,
    gradient_clip_norm: float,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.train()
    genome_mean, genome_scale, drug_means, drug_scales = scaler_tensors
    rng = np.random.default_rng(seed + 100_000 * epoch)
    target_permutation = rng.permutation(len(target_frame)).astype(np.int64, copy=False)
    source_states = {code: {} for code in source_frames}
    target_squared_error = 0.0
    target_observations = 0
    source_squared_error = {code: 0.0 for code in source_frames}
    source_observations = {code: 0 for code in source_frames}
    loss_function = nn.MSELoss()
    source_codes = sorted(source_frames)

    for start in range(0, len(target_permutation), batch_size):
        target_positions = target_permutation[start : start + batch_size]
        target_genome, target_drugs, target = backend.make_batch(
            target_frame,
            target_positions,
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
        target_prediction = model(target_genome, target_drugs)
        target_loss = loss_function(target_prediction, target)
        if not torch.isfinite(target_loss):
            raise RuntimeError("Non-finite target-support loss.")
        (0.5 * target_loss).backward()
        target_squared_error += float(target_loss.detach().cpu()) * len(target)
        target_observations += len(target)

        for code in source_codes:
            source_frame = source_frames[code]
            positions = zero177.cyclic_batch(
                len(source_frame), batch_size, rng, source_states[code]
            )
            source_genome, source_drugs, source_target = backend.make_batch(
                source_frame,
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
            assert source_target is not None
            source_prediction = model(source_genome, source_drugs)
            source_loss = loss_function(source_prediction, source_target)
            if not torch.isfinite(source_loss):
                raise RuntimeError("Non-finite source-replay loss.")
            (0.5 * source_loss / len(source_codes)).backward()
            source_squared_error[code] += float(source_loss.detach().cpu()) * len(source_target)
            source_observations[code] += len(source_target)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Non-finite pretrained adaptation gradient norm.")
        optimizer.step()

    target_rmse = math.sqrt(target_squared_error / target_observations)
    source_rmse = {
        code: math.sqrt(source_squared_error[code] / source_observations[code])
        for code in source_codes
    }
    return target_rmse, source_rmse


def load_source_scalers(
    path: Path,
    drug_matrices: dict[str, np.ndarray],
    device: torch.device,
):
    with np.load(path, allow_pickle=False) as archive:
        genome_mean_np = np.asarray(archive["genome__mean"], dtype=np.float32)
        genome_scale_np = np.asarray(archive["genome__scale"], dtype=np.float32)
        drug_means_np: dict[str, np.ndarray] = {}
        drug_scales_np: dict[str, np.ndarray] = {}
        for view_id in drug_matrices:
            drug_means_np[view_id] = np.asarray(
                archive[f"{view_id}__mean"], dtype=np.float32
            )
            drug_scales_np[view_id] = np.asarray(
                archive[f"{view_id}__scale"], dtype=np.float32
            )
    tensors = backend.scaler_tensors(
        genome_mean_np,
        genome_scale_np,
        drug_means_np,
        drug_scales_np,
        device,
    )
    arrays = {
        "genome__mean": genome_mean_np,
        "genome__scale": genome_scale_np,
    }
    for view_id in drug_matrices:
        arrays[f"{view_id}__mean"] = drug_means_np[view_id]
        arrays[f"{view_id}__scale"] = drug_scales_np[view_id]
    return arrays, tensors


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    require = ["state_dict", "architecture_id", "configuration", "drug_view_order"]
    missing = [key for key in require if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Source checkpoint missing keys: {missing}")
    return checkpoint


def fresh_pretrained_model(
    checkpoint: dict[str, Any],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
    architecture_id: str,
    device: torch.device,
) -> nn.Module:
    if str(checkpoint["architecture_id"]) != architecture_id:
        raise RuntimeError("Checkpoint architecture differs from frozen run plan.")
    if list(checkpoint["drug_view_order"]) != list(drug_matrices):
        raise RuntimeError("Checkpoint drug-view order differs from frozen matrices.")
    model = zero177.build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        target_mean=0.0,
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def select_pretrained_epoch(
    checkpoint: dict[str, Any],
    inner_training: pd.DataFrame,
    inner_validation: pd.DataFrame,
    source: pd.DataFrame,
    source_codes: list[str],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
    architecture_id: str,
    scaler_tensors,
    seed: int,
    device: torch.device,
) -> tuple[int, pd.DataFrame]:
    set_seed(seed + 30_000_000)
    model = fresh_pretrained_model(
        checkpoint,
        genome_matrix,
        drug_matrices,
        spec,
        architecture_id,
        device,
    )
    learning_rate = float(spec["learning_rate"]) * PRETRAINED_LR_MULTIPLIER
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(spec["weight_decay"]),
    )
    source_frames = zero177.source_species_frames(source, source_codes)
    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []
    minimum_improvement = float(spec.get("minimum_rmse_improvement", 0.0))

    for epoch in range(1, PRETRAINED_MAX_EPOCHS + 1):
        started = time.perf_counter()
        target_rmse, source_rmse = train_pretrained_epoch(
            model,
            optimizer,
            inner_training,
            source_frames,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            seed + 30_000_000,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        objective = validation_macro_rmse(
            model,
            inner_validation,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            device,
        )
        if objective < best_objective - minimum_improvement:
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "inner_validation_macro_rmse": objective,
            "best_inner_validation_macro_rmse": best_objective,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "target_support_training_rmse": target_rmse,
            "epoch_seconds": time.perf_counter() - started,
        }
        for code, value in source_rmse.items():
            record[f"source_replay_rmse_{code}"] = value
        records.append(record)
        if epochs_without_improvement >= PRETRAINED_PATIENCE:
            break

    if best_epoch < 1:
        raise RuntimeError("No pretrained adaptation epoch selected.")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, pd.DataFrame(records)


def fit_pretrained_full_support(
    checkpoint: dict[str, Any],
    support: pd.DataFrame,
    source: pd.DataFrame,
    source_codes: list[str],
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
    architecture_id: str,
    scaler_tensors,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    set_seed(seed + 31_000_000)
    model = fresh_pretrained_model(
        checkpoint,
        genome_matrix,
        drug_matrices,
        spec,
        architecture_id,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]) * PRETRAINED_LR_MULTIPLIER,
        weight_decay=float(spec["weight_decay"]),
    )
    source_frames = zero177.source_species_frames(source, source_codes)
    records: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        target_rmse, source_rmse = train_pretrained_epoch(
            model,
            optimizer,
            support,
            source_frames,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            seed + 31_000_000,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "target_support_training_rmse": target_rmse,
            "epoch_seconds": time.perf_counter() - started,
        }
        for code, value in source_rmse.items():
            record[f"source_replay_rmse_{code}"] = value
        records.append(record)
    return model, pd.DataFrame(records)


def select_scratch_epoch(
    inner_training: pd.DataFrame,
    inner_validation: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
    architecture_id: str,
    seed: int,
    device: torch.device,
) -> tuple[int, pd.DataFrame, dict[str, int]]:
    set_seed(seed + 40_000_000)
    arrays, scaler_tensors, zero_counts = zero177.fit_scalers_and_tensors(
        inner_training, genome_matrix, drug_matrices, device
    )
    del arrays
    model = zero177.build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        float(inner_training["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    best_epoch = -1
    best_objective = float("inf")
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []
    minimum_improvement = float(spec.get("minimum_rmse_improvement", 0.0))
    maximum_epochs = int(spec["maximum_epochs"])
    patience = int(spec["early_stopping_patience"])

    for epoch in range(1, maximum_epochs + 1):
        started = time.perf_counter()
        training_rmse = train_target_only_epoch(
            model,
            optimizer,
            inner_training,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            seed + 40_000_000,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        objective = validation_macro_rmse(
            model,
            inner_validation,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            device,
        )
        if objective < best_objective - minimum_improvement:
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        records.append(
            {
                "epoch": epoch,
                "inner_validation_macro_rmse": objective,
                "best_inner_validation_macro_rmse": best_objective,
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "target_support_training_rmse": training_rmse,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
        if epochs_without_improvement >= patience:
            break

    if best_epoch < 1:
        raise RuntimeError("No target-only scratch epoch selected.")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, pd.DataFrame(records), zero_counts


def fit_scratch_full_support(
    support: pd.DataFrame,
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
    architecture_id: str,
    seed: int,
    epochs: int,
    device: torch.device,
):
    set_seed(seed + 41_000_000)
    arrays, scaler_tensors, zero_counts = zero177.fit_scalers_and_tensors(
        support, genome_matrix, drug_matrices, device
    )
    genome_mean_np, genome_scale_np, drug_means_np, drug_scales_np, _ = arrays
    model = zero177.build_model(
        genome_matrix,
        drug_matrices,
        architecture_id,
        spec,
        float(support["mic_target_log2_mg_per_l"].mean()),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    records: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training_rmse = train_target_only_epoch(
            model,
            optimizer,
            support,
            genome_matrix,
            drug_matrices,
            scaler_tensors,
            seed + 41_000_000,
            epoch,
            int(spec["batch_size"]),
            float(spec["gradient_clip_norm"]),
            device,
        )
        records.append(
            {
                "epoch": epoch,
                "target_support_training_rmse": training_rmse,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
    scaler_arrays: dict[str, np.ndarray] = {
        "genome__mean": genome_mean_np,
        "genome__scale": genome_scale_np,
    }
    for view_id in drug_matrices:
        scaler_arrays[f"{view_id}__mean"] = drug_means_np[view_id]
        scaler_arrays[f"{view_id}__scale"] = drug_scales_np[view_id]
    return model, pd.DataFrame(records), scaler_arrays, scaler_tensors, zero_counts


def execute_run(
    row: pd.Series,
    observations: pd.DataFrame,
    zero_plan_lookup: pd.DataFrame,
    query_membership: pd.DataFrame,
    support_membership: pd.DataFrame,
    device: torch.device,
) -> None:
    run_id = str(row["run_id"])
    if run_complete(run_id):
        print(f"SKIP VERIFIED: {run_id}", flush=True)
        return

    model_kind = str(row["model_kind"])
    target_code = str(row["outer_target_code"])
    query_id = str(row["query_id"])
    budget = int(row["support_budget_percent"])
    seed = int(row["seed"])
    configuration_zero_run_id = str(row["configuration_zero_shot_run_id"])

    if configuration_zero_run_id not in zero_plan_lookup.index:
        raise RuntimeError(
            f"Unknown zero-shot configuration run: {configuration_zero_run_id}"
        )
    configuration_row = zero_plan_lookup.loc[configuration_zero_run_id]
    if isinstance(configuration_row, pd.DataFrame):
        raise RuntimeError("Duplicate zero-shot configuration run ID.")

    target_observations = observations.loc[
        observations["species_code"].eq(target_code)
    ].copy()
    target_lookup = target_observations.set_index("observation_id", verify_integrity=True)

    query_ids = query_membership.loc[
        query_membership["target_species_code"].eq(target_code)
        & query_membership["query_id"].eq(query_id),
        "observation_id",
    ].astype(str).tolist()
    support_ids = support_membership.loc[
        support_membership["target_species_code"].eq(target_code)
        & support_membership["query_id"].eq(query_id)
        & support_membership["support_budget_percent"].eq(budget),
        "observation_id",
    ].astype(str).tolist()

    if not query_ids or not support_ids:
        raise RuntimeError(f"Empty query/support membership: {run_id}")
    if set(query_ids).intersection(support_ids):
        raise RuntimeError(f"Query/support observation leakage: {run_id}")

    query = target_lookup.loc[query_ids].reset_index()
    support = target_lookup.loc[support_ids].reset_index()
    if query["normalized_antibiotic"].nunique() != target_observations["normalized_antibiotic"].nunique():
        raise RuntimeError(f"Random-pair query lacks target antibiotics: {run_id}")

    inner_training, inner_validation, inner_audit = deterministic_inner_split(
        support,
        target_code,
        query_id,
        budget,
    )

    genome_matrix, drug_matrices, spec, architecture_id = zero177.load_feature_matrices(
        configuration_row
    )
    started = time.perf_counter()
    print(
        f"START {run_id} kind={model_kind} support={len(support)} "
        f"query={len(query)}",
        flush=True,
    )

    source_checkpoint_path: Path | None = None
    source_scaler_path: Path | None = None
    zero_prediction_path: Path | None = None
    source: pd.DataFrame | None = None
    source_codes: list[str] = []
    zero_metrics: dict[str, float | int] = {}
    zero_per_drug: pd.DataFrame | None = None

    if model_kind == "source_pretrained_few_shot":
        zero_run_id = str(row["zero_shot_run_id"])
        zero_assets = zero_paths(zero_run_id)
        source_checkpoint_path = zero_assets["checkpoint"]
        source_scaler_path = zero_assets["scalers"]
        zero_prediction_path = zero_assets["predictions"]
        checkpoint = load_checkpoint_state(source_checkpoint_path)
        scaler_arrays, scaler_tensors = load_source_scalers(
            source_scaler_path,
            drug_matrices,
            device,
        )
        source_codes = [
            code for code in str(row["source_species_codes"]).split("|") if code
        ]
        source = observations.loc[
            observations["species_code"].isin(source_codes)
        ].copy().reset_index(drop=True)
        if source.empty or not source_codes:
            raise RuntimeError(f"Empty source replay data: {run_id}")

        best_epoch, selection_history = select_pretrained_epoch(
            checkpoint,
            inner_training,
            inner_validation,
            source,
            source_codes,
            genome_matrix,
            drug_matrices,
            spec,
            architecture_id,
            scaler_tensors,
            seed,
            device,
        )
        model, full_history = fit_pretrained_full_support(
            checkpoint,
            support,
            source,
            source_codes,
            genome_matrix,
            drug_matrices,
            spec,
            architecture_id,
            scaler_tensors,
            seed,
            best_epoch,
            device,
        )
        scratch_zero_counts: dict[str, int] = {}

        zero_prediction = read_tsv(zero_prediction_path)
        require_columns(
            zero_prediction,
            ["observation_id", "zero_shot_prediction"],
            "zero-shot target predictions",
        )
        zero_prediction["zero_shot_prediction"] = pd.to_numeric(
            zero_prediction["zero_shot_prediction"], errors="raise"
        )
        zero_query = query[["observation_id"]].merge(
            zero_prediction[["observation_id", "zero_shot_prediction"]],
            on="observation_id",
            how="left",
            validate="one_to_one",
        )
        if zero_query["zero_shot_prediction"].isna().any():
            raise RuntimeError(f"Missing matched zero-shot query prediction: {run_id}")
        zero_metrics, zero_per_drug = metric_bundle(
            query,
            zero_query["zero_shot_prediction"].to_numpy(dtype=np.float32),
        )

    elif model_kind == "target_only_scratch":
        best_epoch, selection_history, inner_zero_counts = select_scratch_epoch(
            inner_training,
            inner_validation,
            genome_matrix,
            drug_matrices,
            spec,
            architecture_id,
            seed,
            device,
        )
        (
            model,
            full_history,
            scaler_arrays,
            scaler_tensors,
            scratch_zero_counts,
        ) = fit_scratch_full_support(
            support,
            genome_matrix,
            drug_matrices,
            spec,
            architecture_id,
            seed,
            best_epoch,
            device,
        )
        scratch_zero_counts = {
            "inner_selection": inner_zero_counts,
            "full_support": scratch_zero_counts,
        }
    else:
        raise RuntimeError(f"Unknown model kind: {model_kind}")

    genome_mean, genome_scale, drug_means, drug_scales = scaler_tensors
    predictions = backend.predict(
        model,
        query,
        genome_matrix,
        drug_matrices,
        genome_mean,
        genome_scale,
        drug_means,
        drug_scales,
        device,
    )
    metrics, per_drug = metric_bundle(query, predictions)
    elapsed = time.perf_counter() - started
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    summary: dict[str, Any] = {
        "run_id": run_id,
        "model_kind": model_kind,
        "outer_target_code": target_code,
        "source_regime_id": str(row["source_regime_id"]),
        "source_species_codes": "|".join(source_codes),
        "query_id": query_id,
        "support_budget_percent": budget,
        "seed": seed,
        "support_observations": len(support),
        "support_unique_genomes": support["genome_id"].nunique(),
        "support_unique_antibiotics": support["normalized_antibiotic"].nunique(),
        "inner_training_observations": len(inner_training),
        "inner_validation_observations": len(inner_validation),
        "query_observations": len(query),
        "query_unique_genomes": query["genome_id"].nunique(),
        "query_unique_antibiotics": query["normalized_antibiotic"].nunique(),
        "best_epoch": best_epoch,
        "model_parameter_count": parameter_count,
        "genome_representation": str(row["genome_representation"]),
        "drug_representation": str(row["drug_representation"]),
        "drug_view_fusion_method": str(row["drug_view_fusion_method"]),
        "cross_modal_architecture": str(row["cross_modal_architecture"]),
        "base_learning_rate": float(spec["learning_rate"]),
        "effective_learning_rate": (
            float(spec["learning_rate"]) * PRETRAINED_LR_MULTIPLIER
            if model_kind == "source_pretrained_few_shot"
            else float(spec["learning_rate"])
        ),
        "weight_decay": float(spec["weight_decay"]),
        "batch_size": int(spec["batch_size"]),
        "scaler_policy": (
            "frozen_source_scalers"
            if model_kind == "source_pretrained_few_shot"
            else "target_support_only_scalers"
        ),
        "source_replay_policy": (
            "0.5 target support + 0.5 equal-species source replay"
            if model_kind == "source_pretrained_few_shot"
            else "none"
        ),
        "query_label_use": "evaluation only",
        "elapsed_seconds": elapsed,
        **metrics,
    }
    if zero_metrics:
        for key, value in zero_metrics.items():
            summary[f"zero_shot_{key}"] = value
        summary["few_shot_gain_macro_rmse"] = (
            float(zero_metrics["macro_rmse"]) - float(metrics["macro_rmse"])
        )
        summary["few_shot_gain_macro_mae"] = (
            float(zero_metrics["macro_mae"]) - float(metrics["macro_mae"])
        )

    result_dir = run_result_directory(run_id)
    metadata_dir = run_metadata_directory(run_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "run_summary.tsv"
    per_drug_path = result_dir / "per_antibiotic_metrics.tsv"
    zero_per_drug_path = result_dir / "matched_zero_shot_per_antibiotic_metrics.tsv"
    prediction_path = result_dir / "query_predictions.tsv"
    selection_history_path = result_dir / "inner_selection_history.tsv"
    full_history_path = result_dir / "full_support_refit_history.tsv"
    inner_audit_path = result_dir / "inner_support_split_audit.tsv"
    scaler_path = result_dir / "final_input_scalers.npz"
    checkpoint_path = result_dir / "final_model_state_dict.pt"
    configuration_path = metadata_dir / "configuration.json"
    input_manifest_path = metadata_dir / "input_manifest.tsv"
    output_manifest_path = metadata_dir / "outputs_sha256.txt"
    complete_path = metadata_dir / "RUN_COMPLETE"

    write_tsv(pd.DataFrame([summary]), summary_path)
    per_drug.insert(0, "run_id", run_id)
    per_drug.insert(1, "model_kind", model_kind)
    write_tsv(per_drug, per_drug_path)
    output_paths = [summary_path, per_drug_path]
    if zero_per_drug is not None:
        zero_per_drug.insert(0, "run_id", run_id)
        write_tsv(zero_per_drug, zero_per_drug_path)
        output_paths.append(zero_per_drug_path)

    prediction_frame = query[
        [
            "observation_id",
            "genome_id",
            "normalized_antibiotic",
            "mic_target_log2_mg_per_l",
        ]
    ].copy()
    prediction_frame.insert(0, "run_id", run_id)
    prediction_frame["prediction"] = predictions
    write_tsv(prediction_frame, prediction_path)
    write_tsv(selection_history, selection_history_path)
    write_tsv(full_history, full_history_path)
    write_tsv(inner_audit, inner_audit_path)
    np.savez_compressed(scaler_path, **scaler_arrays)

    checkpoint_output = {
        "format_version": 1,
        "run_id": run_id,
        "model_kind": model_kind,
        "outer_target_code": target_code,
        "source_regime_id": str(row["source_regime_id"]),
        "query_id": query_id,
        "support_budget_percent": budget,
        "seed": seed,
        "best_epoch": best_epoch,
        "architecture_id": architecture_id,
        "configuration": spec,
        "drug_view_order": list(drug_matrices),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(checkpoint_output, checkpoint_path)

    write_json(
        {
            "run_plan_row": row.to_dict(),
            "resolved_specification": spec,
            "architecture_id": architecture_id,
            "drug_views": list(drug_matrices),
            "pretrained_learning_rate_multiplier": PRETRAINED_LR_MULTIPLIER,
            "pretrained_maximum_epochs": PRETRAINED_MAX_EPOCHS,
            "pretrained_patience": PRETRAINED_PATIENCE,
            "inner_split_seed": INNER_SPLIT_SEED,
            "scratch_zero_variance_counts": scratch_zero_counts,
            "target_query_label_use": "evaluation only; never used for training, scaling, or epoch selection",
        },
        configuration_path,
    )

    input_paths = [
        Path(__file__).resolve(),
        SCRIPT177_PATH,
        PREREG_FREEZE,
        RUN_PLAN_PATH,
        QUERY_PATH,
        SUPPORT_PATH,
        ZERO_PLAN_PATH,
    ]
    genome_path = Path(spec["genome_matrix_path"])
    if not genome_path.is_absolute():
        genome_path = PROJECT / genome_path
    input_paths.append(genome_path)
    input_paths.extend(backend.DRUG_VIEW_PATHS[view_id] for view_id in drug_matrices)
    if source_checkpoint_path is not None:
        input_paths.extend(
            [source_checkpoint_path, source_scaler_path, zero_prediction_path]
        )
    manifest_rows: list[dict[str, object]] = []
    for path in sorted({candidate.resolve() for candidate in input_paths if candidate is not None}, key=lambda p: p.as_posix()):
        try:
            display = path.relative_to(PROJECT)
        except ValueError:
            display = path
        manifest_rows.append(
            {
                "file_path": str(display),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_tsv(pd.DataFrame(manifest_rows), input_manifest_path)

    output_paths.extend(
        [
            prediction_path,
            selection_history_path,
            full_history_path,
            inner_audit_path,
            scaler_path,
            checkpoint_path,
            configuration_path,
            input_manifest_path,
        ]
    )
    write_sha_manifest(output_paths, output_manifest_path)
    verify_sha_manifest(output_manifest_path)
    complete_path.write_text("0\n", encoding="utf-8")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"COMPLETE {run_id} best_epoch={best_epoch} "
        f"macro_rmse={float(metrics['macro_rmse']):.6f}",
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-target", choices=["all", "ec", "kp", "se"], default="all")
    parser.add_argument(
        "--model-kind",
        choices=["all", "source_pretrained_few_shot", "target_only_scratch"],
        default="all",
    )
    parser.add_argument("--source-regime", default="all")
    parser.add_argument("--query-id", default="all")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--max-new-runs", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    observations, plan, zero_plan, query, support = load_inputs()

    if args.outer_target != "all":
        plan = plan.loc[plan["outer_target_code"].eq(args.outer_target)]
    if args.model_kind != "all":
        plan = plan.loc[plan["model_kind"].eq(args.model_kind)]
    if args.source_regime != "all":
        plan = plan.loc[plan["source_regime_id"].eq(args.source_regime)]
    if args.query_id != "all":
        plan = plan.loc[plan["query_id"].eq(args.query_id)]
    plan = plan.loc[
        plan["support_budget_percent"].isin(args.budgets)
        & plan["seed"].isin(args.seeds)
    ].reset_index(drop=True)
    if plan.empty:
        raise RuntimeError("No random-pair few-shot runs selected.")

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
    zero_plan_lookup = zero_plan.set_index("run_id", verify_integrity=True)

    new_runs = 0
    for _, row in plan.iterrows():
        if run_complete(str(row["run_id"])):
            print(f"SKIP VERIFIED: {row['run_id']}", flush=True)
            continue
        execute_run(
            row,
            observations,
            zero_plan_lookup,
            query,
            support,
            device,
        )
        new_runs += 1
        if args.max_new_runs > 0 and new_runs >= args.max_new_runs:
            print("STATUS: SCRIPT 180 PARTIAL RANDOM-PAIR FEW-SHOT RUN COMPLETE")
            return

    print("STATUS: SCRIPT 180 SELECTED RANDOM-PAIR FEW-SHOT PARTITION COMPLETE")


if __name__ == "__main__":
    main()
