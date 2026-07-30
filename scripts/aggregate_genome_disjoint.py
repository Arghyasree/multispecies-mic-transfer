#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()

PREREG_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script182_successful_genome_disjoint_few_shot_preregistration_core_sha256.txt"
)
RUN_PLAN_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/genome_disjoint_few_shot_v1/"
      "genome_disjoint_few_shot_run_plan_v1.tsv"
)
RUN_RESULT_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "genome_disjoint_few_shot_runs_v1"
)
RUN_METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "genome_disjoint_few_shot_runs_v1"
)
AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/"
      "genome_disjoint_few_shot_aggregate_v1"
)
AGGREGATE_METADATA_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "genome_disjoint_few_shot_aggregate_v1"
)
SUCCESS_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script184_successful_genome_disjoint_few_shot_aggregation_core_sha256.txt"
)

EXPECTED_RUNS = 540
MODEL_SEEDS = (20260815, 20260816, 20260817)
SUPPORT_BUDGETS = (1, 5, 10)
BEST_ZERO_SHOT_SINGLE = {
    "ec": "kp_to_ec",
    "kp": "ec_to_kp",
    "se": "ec_to_se",
}
MULTI_SOURCE_REGIME = {
    "ec": "kp_plus_se_to_ec",
    "kp": "se_plus_ec_to_kp",
    "se": "kp_plus_ec_to_se",
}


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


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else float("nan")


def sample_sd(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) >= 2 else float("nan")


def run_complete(run_id: str) -> bool:
    metadata_dir = RUN_METADATA_ROOT / run_id
    complete = metadata_dir / "RUN_COMPLETE"
    manifest = metadata_dir / "outputs_sha256.txt"
    if not complete.is_file() or complete.read_text(encoding="utf-8").strip() != "0":
        return False
    try:
        verify_sha_manifest(manifest)
    except Exception:
        return False
    return True


def numeric_metrics(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "pooled_",
        "macro_",
        "zero_shot_pooled_",
        "zero_shot_macro_",
        "few_shot_gain_",
    )
    excluded_suffixes = ("_valid_antibiotics",)
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes)
        and not column.endswith(excluded_suffixes)
    ]


def seed_average_by_fold(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["seed_count"] = group["seed"].nunique()
        for metric in metric_columns:
            record[metric] = safe_mean(group[metric])
        records.append(record)
    output = pd.DataFrame(records)
    if not output["seed_count"].eq(3).all():
        raise RuntimeError("Not every genome-disjoint fold has three model seeds.")
    return output


def fold_mean_sd(
    fold_frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in fold_frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["fold_count"] = group["query_id"].nunique()
        for metric in metric_columns:
            record[f"{metric}_mean_across_folds"] = safe_mean(group[metric])
            record[f"{metric}_sd_across_folds"] = sample_sd(group[metric])
        records.append(record)
    output = pd.DataFrame(records)
    if not output["fold_count"].eq(5).all():
        raise RuntimeError("Not every genome-disjoint summary has five folds.")
    return output


def main() -> None:
    verify_sha_manifest(PREREG_FREEZE)
    plan = read_tsv(RUN_PLAN_PATH)
    require_columns(
        plan,
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "query_id",
            "support_budget_percent",
            "seed",
        ],
        "genome-disjoint few-shot run plan",
    )
    plan["support_budget_percent"] = pd.to_numeric(
        plan["support_budget_percent"], errors="raise"
    ).astype(int)
    plan["seed"] = pd.to_numeric(plan["seed"], errors="raise").astype(int)
    if len(plan) != EXPECTED_RUNS or plan["run_id"].duplicated().any():
        raise RuntimeError(f"Invalid run plan: {len(plan)}")

    incomplete = [
        run_id
        for run_id in plan["run_id"].astype(str)
        if not run_complete(run_id)
    ]
    if incomplete:
        raise RuntimeError(
            f"Cannot aggregate: {len(incomplete)} runs incomplete; first={incomplete[:3]}"
        )

    summaries: list[pd.DataFrame] = []
    per_drug_frames: list[pd.DataFrame] = []
    run_manifests: list[Path] = []
    for run_id in plan["run_id"].astype(str):
        result_dir = RUN_RESULT_ROOT / run_id
        metadata_dir = RUN_METADATA_ROOT / run_id
        summaries.append(read_tsv(result_dir / "run_summary.tsv"))
        per_drug_frames.append(read_tsv(result_dir / "per_antibiotic_metrics.tsv"))
        run_manifests.append(metadata_dir / "outputs_sha256.txt")

    all_runs = pd.concat(summaries, ignore_index=True)
    all_per_drug = pd.concat(per_drug_frames, ignore_index=True)
    require_columns(
        all_runs,
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "query_id",
            "support_budget_percent",
            "seed",
            "macro_rmse",
            "macro_mae",
        ],
        "all genome-disjoint run summaries",
    )
    all_runs["support_budget_percent"] = pd.to_numeric(
        all_runs["support_budget_percent"], errors="raise"
    ).astype(int)
    all_runs["seed"] = pd.to_numeric(all_runs["seed"], errors="raise").astype(int)

    plan_keys = plan[
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "query_id",
            "support_budget_percent",
            "seed",
        ]
    ].copy()
    audit = all_runs[
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "query_id",
            "support_budget_percent",
            "seed",
        ]
    ].merge(
        plan_keys,
        on="run_id",
        how="outer",
        suffixes=("_result", "_plan"),
        indicator=True,
        validate="one_to_one",
    )
    if not audit["_merge"].eq("both").all():
        raise RuntimeError("Run-summary/run-plan membership mismatch.")
    for column in [
        "model_kind",
        "outer_target_code",
        "source_regime_id",
        "query_id",
        "support_budget_percent",
        "seed",
    ]:
        if not audit[f"{column}_result"].astype(str).eq(
            audit[f"{column}_plan"].astype(str)
        ).all():
            raise RuntimeError(f"Run-summary metadata mismatch: {column}")

    run_metadata = all_runs[
        [
            "run_id",
            "model_kind",
            "outer_target_code",
            "source_regime_id",
            "query_id",
            "support_budget_percent",
            "seed",
        ]
    ].copy()
    all_per_drug = all_per_drug.drop(
        columns=[
            column
            for column in ["model_kind"]
            if column in all_per_drug.columns
        ]
    ).merge(
        run_metadata,
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    if all_per_drug[
        ["model_kind", "outer_target_code", "query_id", "seed"]
    ].isna().any().any():
        raise RuntimeError("Per-antibiotic rows lack run metadata.")

    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    AGGREGATE_METADATA_ROOT.mkdir(parents=True, exist_ok=True)

    all_runs_path = AGGREGATE_ROOT / "all_genome_disjoint_few_shot_run_metrics.tsv"
    all_per_drug_path = AGGREGATE_ROOT / "all_genome_disjoint_few_shot_per_antibiotic_metrics.tsv"
    write_tsv(all_runs, all_runs_path)
    write_tsv(all_per_drug, all_per_drug_path)

    scratch = all_runs.loc[
        all_runs["model_kind"].eq("target_only_scratch")
    ].copy()
    pretrained = all_runs.loc[
        all_runs["model_kind"].eq("source_pretrained_few_shot")
    ].copy()
    if len(scratch) != 135 or len(pretrained) != 405:
        raise RuntimeError(
            f"Unexpected model-kind counts: scratch={len(scratch)} pretrained={len(pretrained)}"
        )

    comparison_metrics = [
        metric
        for metric in [
            "macro_rmse",
            "macro_mae",
            "macro_r2",
            "macro_pearson",
            "macro_spearman",
            "macro_one_tier_accuracy",
            "pooled_rmse",
            "pooled_mae",
            "pooled_r2",
            "pooled_pearson",
            "pooled_spearman",
            "pooled_one_tier_accuracy",
        ]
        if metric in all_runs.columns
    ]
    scratch_lookup = scratch[
        [
            "outer_target_code",
            "query_id",
            "support_budget_percent",
            "seed",
            *comparison_metrics,
        ]
    ].rename(
        columns={metric: f"scratch_{metric}" for metric in comparison_metrics}
    )
    comparison = pretrained.merge(
        scratch_lookup,
        on=[
            "outer_target_code",
            "query_id",
            "support_budget_percent",
            "seed",
        ],
        how="left",
        validate="many_to_one",
    )
    if comparison[[f"scratch_{metric}" for metric in comparison_metrics]].isna().any().any():
        raise RuntimeError("Missing matched target-only scratch control.")
    for metric in ["macro_rmse", "macro_mae", "pooled_rmse", "pooled_mae"]:
        if metric in comparison_metrics:
            comparison[f"pretraining_gain_{metric}"] = (
                pd.to_numeric(comparison[f"scratch_{metric}"], errors="raise")
                - pd.to_numeric(comparison[metric], errors="raise")
            )
    comparison_path = AGGREGATE_ROOT / "paired_pretrained_vs_scratch_run_metrics.tsv"
    write_tsv(comparison, comparison_path)

    run_metric_columns = numeric_metrics(all_runs)
    pretrained_metric_columns = numeric_metrics(comparison) + [
        column
        for column in comparison.columns
        if column.startswith("pretraining_gain_")
    ]
    pretrained_metric_columns = sorted(set(pretrained_metric_columns))

    scratch_fold = seed_average_by_fold(
        scratch,
        [
            "outer_target_code",
            "model_kind",
            "support_budget_percent",
            "query_id",
        ],
        run_metric_columns,
    )
    pretrained_fold = seed_average_by_fold(
        comparison,
        [
            "outer_target_code",
            "source_regime_id",
            "model_kind",
            "support_budget_percent",
            "query_id",
        ],
        pretrained_metric_columns,
    )
    scratch_fold_path = AGGREGATE_ROOT / "target_only_scratch_seed_averaged_fold_metrics.tsv"
    pretrained_fold_path = AGGREGATE_ROOT / "pretrained_few_shot_seed_averaged_fold_metrics.tsv"
    write_tsv(scratch_fold, scratch_fold_path)
    write_tsv(pretrained_fold, pretrained_fold_path)

    scratch_summary = fold_mean_sd(
        scratch_fold,
        ["outer_target_code", "model_kind", "support_budget_percent"],
        run_metric_columns,
    )
    pretrained_summary = fold_mean_sd(
        pretrained_fold,
        [
            "outer_target_code",
            "source_regime_id",
            "model_kind",
            "support_budget_percent",
        ],
        pretrained_metric_columns,
    )
    scratch_summary_path = AGGREGATE_ROOT / "target_only_scratch_genome_disjoint_mean_sd.tsv"
    pretrained_summary_path = AGGREGATE_ROOT / "pretrained_few_shot_genome_disjoint_mean_sd.tsv"
    write_tsv(scratch_summary, scratch_summary_path)
    write_tsv(pretrained_summary, pretrained_summary_path)

    per_drug_metric_columns = [
        metric
        for metric in [
            "rmse",
            "mae",
            "r2",
            "pearson",
            "spearman",
            "one_tier_accuracy",
        ]
        if metric in all_per_drug.columns
    ]
    per_drug_fold_records: list[dict[str, Any]] = []
    per_drug_groups = [
        "outer_target_code",
        "source_regime_id",
        "model_kind",
        "support_budget_percent",
        "query_id",
        "normalized_antibiotic",
    ]
    for keys, group in all_per_drug.groupby(per_drug_groups, sort=True, dropna=False):
        record = dict(zip(per_drug_groups, keys))
        record["seed_count"] = group["seed"].nunique()
        record["observations_mean_across_seeds"] = safe_mean(group["observations"])
        for metric in per_drug_metric_columns:
            record[metric] = safe_mean(group[metric])
        per_drug_fold_records.append(record)
    per_drug_fold = pd.DataFrame(per_drug_fold_records)
    if not per_drug_fold["seed_count"].eq(3).all():
        raise RuntimeError("Not every per-antibiotic fold has three seeds.")

    per_drug_summary_records: list[dict[str, Any]] = []
    per_drug_summary_groups = [
        "outer_target_code",
        "source_regime_id",
        "model_kind",
        "support_budget_percent",
        "normalized_antibiotic",
    ]
    for keys, group in per_drug_fold.groupby(
        per_drug_summary_groups, sort=True, dropna=False
    ):
        record = dict(zip(per_drug_summary_groups, keys))
        record["fold_count"] = group["query_id"].nunique()
        record["observations_mean_across_folds"] = safe_mean(
            group["observations_mean_across_seeds"]
        )
        for metric in per_drug_metric_columns:
            record[f"{metric}_mean_across_folds"] = safe_mean(group[metric])
            record[f"{metric}_sd_across_folds"] = sample_sd(group[metric])
        per_drug_summary_records.append(record)
    per_drug_summary = pd.DataFrame(per_drug_summary_records)
    if not per_drug_summary["fold_count"].eq(5).all():
        raise RuntimeError("Not every per-antibiotic summary has five folds.")
    per_drug_fold_path = AGGREGATE_ROOT / "per_antibiotic_seed_averaged_fold_metrics.tsv"
    per_drug_summary_path = AGGREGATE_ROOT / "per_antibiotic_genome_disjoint_mean_sd.tsv"
    write_tsv(per_drug_fold, per_drug_fold_path)
    write_tsv(per_drug_summary, per_drug_summary_path)

    multisource_cell_records: list[pd.DataFrame] = []
    for target in sorted(BEST_ZERO_SHOT_SINGLE):
        single_regime = BEST_ZERO_SHOT_SINGLE[target]
        multi_regime = MULTI_SOURCE_REGIME[target]
        single = pretrained.loc[
            pretrained["outer_target_code"].eq(target)
            & pretrained["source_regime_id"].eq(single_regime),
            [
                "outer_target_code",
                "query_id",
                "support_budget_percent",
                "seed",
                "macro_rmse",
                "macro_mae",
            ],
        ].rename(
            columns={
                "macro_rmse": "reference_single_macro_rmse",
                "macro_mae": "reference_single_macro_mae",
            }
        )
        multi = pretrained.loc[
            pretrained["outer_target_code"].eq(target)
            & pretrained["source_regime_id"].eq(multi_regime),
            [
                "outer_target_code",
                "query_id",
                "support_budget_percent",
                "seed",
                "macro_rmse",
                "macro_mae",
            ],
        ].rename(
            columns={
                "macro_rmse": "multisource_macro_rmse",
                "macro_mae": "multisource_macro_mae",
            }
        )
        paired = single.merge(
            multi,
            on=[
                "outer_target_code",
                "query_id",
                "support_budget_percent",
                "seed",
            ],
            how="inner",
            validate="one_to_one",
        )
        for column in [
            "reference_single_macro_rmse",
            "multisource_macro_rmse",
            "reference_single_macro_mae",
            "multisource_macro_mae",
        ]:
            paired[column] = pd.to_numeric(paired[column], errors="raise")
        paired["reference_single_source_regime"] = single_regime
        paired["multisource_regime"] = multi_regime
        paired["multisource_gain_macro_rmse"] = (
            paired["reference_single_macro_rmse"]
            - paired["multisource_macro_rmse"]
        )
        paired["multisource_gain_macro_mae"] = (
            paired["reference_single_macro_mae"]
            - paired["multisource_macro_mae"]
        )
        multisource_cell_records.append(paired)
    multisource_cells = pd.concat(multisource_cell_records, ignore_index=True)
    multisource_cell_path = AGGREGATE_ROOT / "paired_multisource_gain_run_metrics.tsv"
    write_tsv(multisource_cells, multisource_cell_path)

    multisource_fold = seed_average_by_fold(
        multisource_cells,
        [
            "outer_target_code",
            "reference_single_source_regime",
            "multisource_regime",
            "support_budget_percent",
            "query_id",
        ],
        [
            "reference_single_macro_rmse",
            "multisource_macro_rmse",
            "multisource_gain_macro_rmse",
            "reference_single_macro_mae",
            "multisource_macro_mae",
            "multisource_gain_macro_mae",
        ],
    )
    multisource_summary = fold_mean_sd(
        multisource_fold,
        [
            "outer_target_code",
            "reference_single_source_regime",
            "multisource_regime",
            "support_budget_percent",
        ],
        [
            "reference_single_macro_rmse",
            "multisource_macro_rmse",
            "multisource_gain_macro_rmse",
            "reference_single_macro_mae",
            "multisource_macro_mae",
            "multisource_gain_macro_mae",
        ],
    )
    multisource_fold_path = AGGREGATE_ROOT / "multisource_gain_seed_averaged_fold_metrics.tsv"
    multisource_summary_path = AGGREGATE_ROOT / "multisource_gain_genome_disjoint_mean_sd.tsv"
    write_tsv(multisource_fold, multisource_fold_path)
    write_tsv(multisource_summary, multisource_summary_path)

    protocol = pd.DataFrame(
        [
            {
                "item": "completed_experiment_cells",
                "value": EXPECTED_RUNS,
            },
            {
                "item": "primary_reporting_unit",
                "value": "five genome-disjoint query folds",
            },
            {
                "item": "seed_handling",
                "value": "average three model seeds inside each fold",
            },
            {
                "item": "uncertainty",
                "value": "mean and sample SD across five seed-averaged folds; ddof=1",
            },
            {
                "item": "few_shot_gain",
                "value": "matched zero-shot macro RMSE minus pretrained few-shot macro RMSE",
            },
            {
                "item": "pretraining_gain",
                "value": "matched target-only scratch macro RMSE minus pretrained few-shot macro RMSE",
            },
            {
                "item": "multisource_reference",
                "value": "best single-source regime from the frozen zero-shot full-target result",
            },
            {
                "item": "target_query_labels",
                "value": "evaluation only",
            },
        ]
    )
    protocol_path = AGGREGATE_METADATA_ROOT / "aggregate_protocol_v1.tsv"
    write_tsv(protocol, protocol_path)

    input_manifest_rows: list[dict[str, object]] = []
    input_paths = [
        Path(__file__).resolve(),
        PREREG_FREEZE,
        RUN_PLAN_PATH,
        *run_manifests,
    ]
    for path in sorted({candidate.resolve() for candidate in input_paths}, key=lambda p: p.as_posix()):
        try:
            display = path.relative_to(PROJECT)
        except ValueError:
            display = path
        input_manifest_rows.append(
            {
                "path": str(display),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    input_manifest_path = AGGREGATE_METADATA_ROOT / "script184_input_manifest.tsv"
    write_tsv(pd.DataFrame(input_manifest_rows), input_manifest_path)

    output_paths = [
        all_runs_path,
        all_per_drug_path,
        comparison_path,
        scratch_fold_path,
        pretrained_fold_path,
        scratch_summary_path,
        pretrained_summary_path,
        per_drug_fold_path,
        per_drug_summary_path,
        multisource_cell_path,
        multisource_fold_path,
        multisource_summary_path,
        protocol_path,
        input_manifest_path,
    ]
    output_manifest_path = AGGREGATE_METADATA_ROOT / "aggregate_outputs_sha256.txt"
    write_sha_manifest(output_paths, output_manifest_path)
    verify_sha_manifest(output_manifest_path)

    write_sha_manifest(
        [
            Path(__file__).resolve(),
            PREREG_FREEZE,
            RUN_PLAN_PATH,
            output_manifest_path,
            pretrained_summary_path,
            scratch_summary_path,
            per_drug_summary_path,
            multisource_summary_path,
            protocol_path,
            input_manifest_path,
        ],
        SUCCESS_FREEZE,
    )
    verify_sha_manifest(SUCCESS_FREEZE)

    display_columns = [
        "outer_target_code",
        "source_regime_id",
        "support_budget_percent",
        "macro_rmse_mean_across_folds",
        "macro_rmse_sd_across_folds",
        "few_shot_gain_macro_rmse_mean_across_folds",
        "few_shot_gain_macro_rmse_sd_across_folds",
        "pretraining_gain_macro_rmse_mean_across_folds",
        "pretraining_gain_macro_rmse_sd_across_folds",
    ]
    display_columns = [column for column in display_columns if column in pretrained_summary.columns]
    print("===== GENOME-DISJOINT PRETRAINED FEW-SHOT SUMMARY =====")
    print(
        pretrained_summary[display_columns]
        .sort_values(
            ["outer_target_code", "support_budget_percent", "macro_rmse_mean_across_folds"],
            kind="stable",
        )
        .to_string(index=False)
    )
    print()
    print("Every reported metric includes mean and sample SD across five seed-averaged folds.")
    print("STATUS: SCRIPT 184 GENOME-DISJOINT FEW-SHOT RESULTS AGGREGATED AND FROZEN")


if __name__ == "__main__":
    main()
