#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
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

SCRIPT177_PATH = PROJECT / "scripts/train_zero_target.py"

EXPECTED_SCRIPT177_SHA256 = (
    "c9a916b48e19486729fe078c250efe72407533c368fee3b5b2d77e8f6bbde1d7"
)

SCRIPT176_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script176_successful_zero_shot_preregistration_core_sha256.txt"
)

SCRIPT175_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script175_successful_final_transfer_splits_core_sha256.txt"
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

SUCCESS_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script178_successful_zero_shot_aggregation_core_sha256.txt"
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


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def write_sha_manifest(
    paths: Iterable[Path],
    output_path: Path,
) -> None:
    unique = sorted(
        {path.resolve() for path in paths},
        key=lambda value: value.as_posix(),
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in unique:
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                display = path.relative_to(PROJECT)
            except ValueError:
                display = path
            handle.write(
                f"{sha256_file(path)}  {display}\n"
            )


def verify_sha_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)

    verified: list[Path] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8",
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue

        parts = line.split(
            maxsplit=1,
        )

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


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(
        set(columns) - set(frame.columns)
    )
    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )


def first_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise RuntimeError(
        f"Could not locate {label}; "
        f"columns={list(frame.columns)}"
    )


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return (
        float(numeric.mean())
        if len(numeric)
        else float("nan")
    )


def sample_sd(series: pd.Series) -> float:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return (
        float(numeric.std(ddof=1))
        if len(numeric) >= 2
        else float("nan")
    )


def metric_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
    replicate_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for keys, group in frame.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        record = dict(
            zip(group_columns, keys)
        )

        record["replicate_count"] = (
            group[replicate_column].nunique()
            if replicate_column in group.columns
            else len(group)
        )

        for metric in metric_columns:
            record[f"{metric}_mean"] = safe_mean(
                group[metric]
            )
            record[f"{metric}_sd"] = sample_sd(
                group[metric]
            )

        records.append(record)

    return pd.DataFrame(records)


def assert_mean_sd_columns(
    frame: pd.DataFrame,
    metrics: list[str],
    mean_suffix: str,
    sd_suffix: str,
    label: str,
) -> None:
    missing: list[str] = []

    for metric in metrics:
        mean_column = (
            f"{metric}{mean_suffix}"
        )
        sd_column = (
            f"{metric}{sd_suffix}"
        )

        if mean_column not in frame.columns:
            missing.append(mean_column)

        if sd_column not in frame.columns:
            missing.append(sd_column)

    if missing:
        raise RuntimeError(
            f"{label} lacks mean/SD columns: {missing}"
        )


def load_script177():
    if not SCRIPT177_PATH.is_file():
        raise FileNotFoundError(
            SCRIPT177_PATH
        )

    observed = sha256_file(
        SCRIPT177_PATH
    )

    if observed != EXPECTED_SCRIPT177_SHA256:
        raise RuntimeError(
            "Script 177 SHA mismatch: "
            f"expected={EXPECTED_SCRIPT177_SHA256} "
            f"observed={observed}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "frozen_zero_shot_runner_177_for_aggregation",
            SCRIPT177_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load frozen Script 177."
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(
        module
    )

    return module


def attach_run_metadata(
    frame: pd.DataFrame,
    run_metadata: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    require_columns(
        frame,
        ["run_id"],
        label,
    )

    metadata_columns = [
        "outer_target_code",
        "source_regime_id",
        "seed",
    ]

    overlapping = [
        column
        for column in metadata_columns
        if column in frame.columns
    ]

    if overlapping:
        audit = frame[
            ["run_id", *overlapping]
        ].merge(
            run_metadata[
                ["run_id", *overlapping]
            ],
            on="run_id",
            how="left",
            suffixes=(
                "_result",
                "_plan",
            ),
            validate="many_to_one",
        )

        mismatches: list[str] = []

        for column in overlapping:
            result_column = (
                f"{column}_result"
            )
            plan_column = (
                f"{column}_plan"
            )

            unequal = (
                audit[result_column]
                .astype(str)
                !=
                audit[plan_column]
                .astype(str)
            )

            if unequal.any():
                mismatches.append(column)

        if mismatches:
            raise RuntimeError(
                f"{label} metadata disagrees with run plan: "
                f"{mismatches}"
            )

    output = frame.drop(
        columns=overlapping,
    ).merge(
        run_metadata,
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    if (
        output[
            [
                "outer_target_code",
                "source_regime_id",
                "seed",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            f"{label} contains run IDs absent from the plan."
        )

    output["seed"] = pd.to_numeric(
        output["seed"],
        errors="raise",
    ).astype(int)

    return output


def main() -> None:
    script177 = load_script177()

    verify_sha_manifest(
        SCRIPT175_FREEZE
    )
    verify_sha_manifest(
        SCRIPT176_FREEZE
    )

    plan = script177.read_tsv(
        script177.RUN_PLAN_PATH
    )

    require_columns(
        plan,
        [
            "run_id",
            "outer_target_code",
            "source_regime_id",
            "seed",
        ],
        "zero-shot run plan",
    )

    if plan["run_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate run IDs in zero-shot run plan."
        )

    plan["seed"] = pd.to_numeric(
        plan["seed"],
        errors="raise",
    ).astype(int)

    if len(plan) != script177.EXPECTED_RUNS:
        raise RuntimeError(
            "Unexpected zero-shot run count: "
            f"{len(plan)}"
        )

    incomplete = [
        run_id
        for run_id in plan["run_id"].astype(str)
        if not script177.run_complete(run_id)
    ]

    if incomplete:
        raise RuntimeError(
            "Cannot aggregate: "
            f"{len(incomplete)} runs are incomplete; "
            f"first={incomplete[:3]}"
        )

    run_metadata = (
        plan[
            [
                "run_id",
                "outer_target_code",
                "source_regime_id",
                "seed",
            ]
        ]
        .copy()
    )

    summaries: list[pd.DataFrame] = []
    per_drug_frames: list[pd.DataFrame] = []
    panel_frames: list[pd.DataFrame] = []
    prediction_paths: dict[str, Path] = {}
    run_output_manifests: list[Path] = []

    for run_id in plan["run_id"].astype(str):
        result_dir = (
            script177.run_result_directory(
                run_id
            )
        )
        metadata_dir = (
            script177.run_metadata_directory(
                run_id
            )
        )

        output_manifest = (
            metadata_dir
            / "outputs_sha256.txt"
        )

        verify_sha_manifest(
            output_manifest
        )

        run_output_manifests.append(
            output_manifest
        )

        summaries.append(
            script177.read_tsv(
                result_dir
                / "run_summary.tsv"
            )
        )
        per_drug_frames.append(
            script177.read_tsv(
                result_dir
                / "per_antibiotic_metrics.tsv"
            )
        )
        panel_frames.append(
            script177.read_tsv(
                result_dir
                / "panel_metrics.tsv"
            )
        )

        prediction_paths[run_id] = (
            result_dir
            / "target_predictions.tsv"
        )

    all_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    all_per_drug = attach_run_metadata(
        pd.concat(
            per_drug_frames,
            ignore_index=True,
        ),
        run_metadata,
        "per-antibiotic result table",
    )

    all_panel = attach_run_metadata(
        pd.concat(
            panel_frames,
            ignore_index=True,
        ),
        run_metadata,
        "panel result table",
    )

    all_summary = attach_run_metadata(
        all_summary,
        run_metadata,
        "run-summary table",
    )

    AGGREGATE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    AGGREGATE_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summary_path = (
        AGGREGATE_ROOT
        / "all_zero_shot_run_metrics.tsv"
    )
    all_per_drug_path = (
        AGGREGATE_ROOT
        / "all_zero_shot_per_antibiotic_metrics.tsv"
    )
    all_panel_path = (
        AGGREGATE_ROOT
        / "all_zero_shot_panel_metrics.tsv"
    )

    write_tsv(
        all_summary,
        all_summary_path,
    )
    write_tsv(
        all_per_drug,
        all_per_drug_path,
    )
    write_tsv(
        all_panel,
        all_panel_path,
    )

    panel_metric_columns = [
        column
        for column in all_panel.columns
        if (
            column.startswith("macro_")
            or column.startswith("pooled_")
        )
        and not column.endswith(
            "_valid_antibiotics"
        )
    ]

    panel_seed_summary = metric_summary(
        all_panel,
        [
            "outer_target_code",
            "source_regime_id",
            "panel_id",
        ],
        panel_metric_columns,
        "seed",
    )

    assert_mean_sd_columns(
        panel_seed_summary,
        panel_metric_columns,
        "_mean",
        "_sd",
        "panel three-seed summary",
    )

    panel_seed_summary_path = (
        AGGREGATE_ROOT
        / "zero_shot_panel_three_seed_mean_sd.tsv"
    )

    write_tsv(
        panel_seed_summary,
        panel_seed_summary_path,
    )

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

    per_drug_seed_summary = metric_summary(
        all_per_drug,
        [
            "outer_target_code",
            "source_regime_id",
            "normalized_antibiotic",
        ],
        per_drug_metric_columns,
        "seed",
    )

    per_drug_counts = (
        all_per_drug.groupby(
            [
                "outer_target_code",
                "source_regime_id",
                "normalized_antibiotic",
            ],
            sort=True,
            dropna=False,
        )["observations"]
        .agg(
            observations_min="min",
            observations_max="max",
        )
        .reset_index()
    )

    per_drug_counts[
        "observations_min"
    ] = pd.to_numeric(
        per_drug_counts[
            "observations_min"
        ],
        errors="raise",
    ).astype(int)

    per_drug_counts[
        "observations_max"
    ] = pd.to_numeric(
        per_drug_counts[
            "observations_max"
        ],
        errors="raise",
    ).astype(int)

    if (
        per_drug_counts[
            "observations_min"
        ]
        !=
        per_drug_counts[
            "observations_max"
        ]
    ).any():
        raise RuntimeError(
            "Per-antibiotic observation counts vary "
            "across model seeds."
        )

    per_drug_counts = (
        per_drug_counts.drop(
            columns=[
                "observations_max",
            ]
        )
        .rename(
            columns={
                "observations_min":
                "observations",
            }
        )
    )

    per_drug_seed_summary = (
        per_drug_seed_summary.merge(
            per_drug_counts,
            on=[
                "outer_target_code",
                "source_regime_id",
                "normalized_antibiotic",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    assert_mean_sd_columns(
        per_drug_seed_summary,
        per_drug_metric_columns,
        "_mean",
        "_sd",
        "per-antibiotic three-seed summary",
    )

    per_drug_seed_summary_path = (
        AGGREGATE_ROOT
        / "zero_shot_per_antibiotic_three_seed_mean_sd.tsv"
    )

    write_tsv(
        per_drug_seed_summary,
        per_drug_seed_summary_path,
    )

    query_membership = (
        script177.read_tsv(
            script177.QUERY_MEMBERSHIP_PATH
        )
    )

    observation_column = first_column(
        query_membership,
        ["observation_id"],
        "query observation ID",
    )

    target_column = first_column(
        query_membership,
        [
            "target_species_code",
            "species_code",
        ],
        "query target species",
    )

    protocol_column = first_column(
        query_membership,
        [
            "target_protocol",
            "protocol_id",
            "protocol",
        ],
        "query protocol",
    )

    query_id_column = first_column(
        query_membership,
        [
            "query_id",
            "query_fold_id",
            "fold_id",
        ],
        "query identifier",
    )

    query_membership = (
        query_membership[
            [
                observation_column,
                target_column,
                protocol_column,
                query_id_column,
            ]
        ]
        .rename(
            columns={
                observation_column:
                "observation_id",
                target_column:
                "outer_target_code",
                protocol_column:
                "target_protocol",
                query_id_column:
                "query_id",
            }
        )
    )

    for column in [
        "observation_id",
        "outer_target_code",
        "target_protocol",
        "query_id",
    ]:
        query_membership[column] = (
            query_membership[column]
            .astype(str)
            .str.strip()
        )

    if (
        query_membership[
            [
                "observation_id",
                "outer_target_code",
                "target_protocol",
                "query_id",
            ]
        ]
        .eq("")
        .any()
        .any()
    ):
        raise RuntimeError(
            "Query-membership table contains blank keys."
        )

    plan_lookup = plan.set_index(
        "run_id"
    )

    query_records: list[
        dict[str, Any]
    ] = []

    for run_id, prediction_path in (
        prediction_paths.items()
    ):
        prediction = (
            script177.read_tsv(
                prediction_path
            )
        )

        require_columns(
            prediction,
            [
                "observation_id",
                "genome_id",
                "normalized_antibiotic",
                "mic_target_log2_mg_per_l",
                "zero_shot_prediction",
            ],
            f"prediction table {run_id}",
        )

        row = plan_lookup.loc[
            run_id
        ]

        target_code = str(
            row[
                "outer_target_code"
            ]
        )

        membership = (
            query_membership.loc[
                query_membership[
                    "outer_target_code"
                ].eq(target_code)
            ]
        )

        joined = membership.merge(
            prediction,
            on="observation_id",
            how="left",
            validate="many_to_one",
        )

        missing_prediction = (
            joined[
                "zero_shot_prediction"
            ]
            .isna()
            |
            joined[
                "zero_shot_prediction"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if missing_prediction.any():
            raise RuntimeError(
                "Missing prediction after query join: "
                f"{run_id}"
            )

        joined[
            "zero_shot_prediction"
        ] = pd.to_numeric(
            joined[
                "zero_shot_prediction"
            ],
            errors="raise",
        )

        joined[
            "mic_target_log2_mg_per_l"
        ] = pd.to_numeric(
            joined[
                "mic_target_log2_mg_per_l"
            ],
            errors="raise",
        )

        for (
            protocol,
            query_id,
        ), subset in joined.groupby(
            [
                "target_protocol",
                "query_id",
            ],
            sort=True,
        ):
            subset = subset.reset_index(
                drop=True,
            )

            per_drug = (
                script177.backend
                .per_antibiotic_metrics(
                    subset,
                    subset[
                        "zero_shot_prediction"
                    ].to_numpy(
                        dtype=np.float32,
                    ),
                )
            )

            pooled = (
                script177.backend
                .regression_metrics(
                    subset[
                        "mic_target_log2_mg_per_l"
                    ].to_numpy(
                        dtype=np.float64,
                    ),
                    subset[
                        "zero_shot_prediction"
                    ].to_numpy(
                        dtype=np.float64,
                    ),
                )
            )

            macro = (
                script177.backend
                .macro_metrics(
                    per_drug
                )
            )

            query_records.append(
                {
                    "run_id":
                    run_id,
                    "outer_target_code":
                    target_code,
                    "source_regime_id":
                    str(
                        row[
                            "source_regime_id"
                        ]
                    ),
                    "seed":
                    int(
                        row[
                            "seed"
                        ]
                    ),
                    "target_protocol":
                    str(protocol),
                    "query_id":
                    str(query_id),
                    "query_observations":
                    len(subset),
                    "query_unique_genomes":
                    subset[
                        "genome_id"
                    ].nunique(),
                    "query_unique_antibiotics":
                    subset[
                        "normalized_antibiotic"
                    ].nunique(),
                    **{
                        f"pooled_{key}":
                        value
                        for key, value
                        in pooled.items()
                    },
                    **macro,
                }
            )

    query_metrics = pd.DataFrame(
        query_records
    )

    query_metrics_path = (
        AGGREGATE_ROOT
        / "zero_shot_query_seed_metrics.tsv"
    )

    write_tsv(
        query_metrics,
        query_metrics_path,
    )

    query_metric_columns = [
        column
        for column in query_metrics.columns
        if (
            column.startswith("macro_")
            or column.startswith("pooled_")
        )
        and not column.endswith(
            "_valid_antibiotics"
        )
    ]

    fold_seed_mean_records: list[
        dict[str, Any]
    ] = []

    fold_group_columns = [
        "outer_target_code",
        "source_regime_id",
        "target_protocol",
        "query_id",
    ]

    for keys, group in query_metrics.groupby(
        fold_group_columns,
        sort=True,
    ):
        record = dict(
            zip(
                fold_group_columns,
                keys,
            )
        )

        record["seed_count"] = (
            group[
                "seed"
            ].nunique()
        )

        for metric in query_metric_columns:
            record[
                metric
            ] = safe_mean(
                group[
                    metric
                ]
            )

        fold_seed_mean_records.append(
            record
        )

    fold_seed_means = pd.DataFrame(
        fold_seed_mean_records
    )

    if not fold_seed_means[
        "seed_count"
    ].eq(3).all():
        raise RuntimeError(
            "Not every query fold has three model seeds."
        )

    fold_seed_means_path = (
        AGGREGATE_ROOT
        / "zero_shot_query_seed_averaged_fold_metrics.tsv"
    )

    write_tsv(
        fold_seed_means,
        fold_seed_means_path,
    )

    protocol_summary_records: list[
        dict[str, Any]
    ] = []

    protocol_groups = [
        "outer_target_code",
        "source_regime_id",
        "target_protocol",
    ]

    for keys, group in fold_seed_means.groupby(
        protocol_groups,
        sort=True,
    ):
        record = dict(
            zip(
                protocol_groups,
                keys,
            )
        )

        record["query_count"] = (
            group[
                "query_id"
            ].nunique()
        )

        for metric in query_metric_columns:
            record[
                f"{metric}_mean_across_queries"
            ] = safe_mean(
                group[
                    metric
                ]
            )

            record[
                f"{metric}_sd_across_queries"
            ] = sample_sd(
                group[
                    metric
                ]
            )

        protocol_summary_records.append(
            record
        )

    protocol_summary = pd.DataFrame(
        protocol_summary_records
    )

    assert_mean_sd_columns(
        protocol_summary,
        query_metric_columns,
        "_mean_across_queries",
        "_sd_across_queries",
        "protocol fold-level summary",
    )

    protocol_summary_path = (
        AGGREGATE_ROOT
        / "zero_shot_protocol_fold_level_summary.tsv"
    )

    write_tsv(
        protocol_summary,
        protocol_summary_path,
    )

    protocol = pd.DataFrame(
        [
            {
                "item":
                "completed_checkpoint_runs",
                "value":
                script177.EXPECTED_RUNS,
            },
            {
                "item":
                "target_label_use",
                "value":
                "zero target labels used for source training or epoch selection",
            },
            {
                "item":
                "source_epoch_selection",
                "value":
                "fold_05 genome-disjoint validation in each source species",
            },
            {
                "item":
                "joint_source_validation_objective",
                "value":
                "equal-species mean of per-antibiotic macro RMSE",
            },
            {
                "item":
                "joint_source_training",
                "value":
                "cyclic balanced batches; equal source-species loss contribution",
            },
            {
                "item":
                "full_panel_uncertainty",
                "value":
                "mean and sample SD across three model seeds",
            },
            {
                "item":
                "per_antibiotic_uncertainty",
                "value":
                "mean and sample SD across three model seeds",
            },
            {
                "item":
                "fold_uncertainty",
                "value":
                "average three seeds inside each query fold, then mean and sample SD across query folds",
            },
            {
                "item":
                "standard_deviation",
                "value":
                "sample SD with ddof=1",
            },
            {
                "item":
                "worst_direction_metric",
                "value":
                "not used in final one-way transfer",
            },
        ]
    )

    protocol_path = (
        AGGREGATE_METADATA_ROOT
        / "aggregate_protocol_v2.tsv"
    )

    write_tsv(
        protocol,
        protocol_path,
    )

    input_paths = [
        Path(__file__).resolve(),
        SCRIPT177_PATH,
        SCRIPT175_FREEZE,
        SCRIPT176_FREEZE,
        script177.RUN_PLAN_PATH,
        script177.QUERY_MEMBERSHIP_PATH,
        *run_output_manifests,
    ]

    input_manifest_path = (
        AGGREGATE_METADATA_ROOT
        / "script178_input_manifest.tsv"
    )

    input_rows = []

    for path in sorted(
        {
            candidate.resolve()
            for candidate in input_paths
        },
        key=lambda candidate:
        candidate.as_posix(),
    ):
        try:
            display = path.relative_to(
                PROJECT
            )
        except ValueError:
            display = path

        input_rows.append(
            {
                "path":
                str(display),
                "sha256":
                sha256_file(path),
            }
        )

    write_tsv(
        pd.DataFrame(
            input_rows
        ),
        input_manifest_path,
    )

    output_paths = [
        all_summary_path,
        all_per_drug_path,
        all_panel_path,
        panel_seed_summary_path,
        per_drug_seed_summary_path,
        query_metrics_path,
        fold_seed_means_path,
        protocol_summary_path,
        protocol_path,
        input_manifest_path,
    ]

    output_manifest_path = (
        AGGREGATE_METADATA_ROOT
        / "aggregate_outputs_sha256.txt"
    )

    write_sha_manifest(
        output_paths,
        output_manifest_path,
    )

    verify_sha_manifest(
        output_manifest_path
    )

    write_sha_manifest(
        [
            Path(__file__).resolve(),
            SCRIPT177_PATH,
            SCRIPT175_FREEZE,
            SCRIPT176_FREEZE,
            input_manifest_path,
            output_manifest_path,
            panel_seed_summary_path,
            per_drug_seed_summary_path,
            protocol_summary_path,
            protocol_path,
        ],
        SUCCESS_FREEZE,
    )

    verify_sha_manifest(
        SUCCESS_FREEZE
    )

    selected_display = (
        panel_seed_summary.loc[
            panel_seed_summary[
                "panel_id"
            ].eq(
                "full_target_panel"
            ),
            [
                "outer_target_code",
                "source_regime_id",
                "macro_rmse_mean",
                "macro_rmse_sd",
            ],
        ]
        .sort_values(
            [
                "outer_target_code",
                "macro_rmse_mean",
            ]
        )
    )

    print(
        "===== ZERO-SHOT FULL-TARGET PANEL ====="
    )
    print(
        selected_display.to_string(
            index=False,
        )
    )
    print()
    print(
        "Full-panel and per-antibiotic metrics: "
        "mean ± sample SD across 3 seeds."
    )
    print(
        "Random-pair/genome-disjoint/drug-held-out metrics: "
        "3 seeds averaged inside each query, then "
        "mean ± sample SD across queries."
    )
    print()
    print(
        "STATUS: SCRIPT 178 ZERO-SHOT RESULTS "
        "AGGREGATED AND FROZEN"
    )


if __name__ == "__main__":
    main()
