#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "metadata/audits/"
    "shortlist_genome_taxonomy_qc_audit.tsv"
)

OUTPUT_ROOT = Path(
    "metadata/qc"
)

RESULT_ROOT = Path(
    "results/tables"
)

ACCEPTED_TAXONOMY_CATEGORIES = {
    "exact_species",
    "lineage_species_species_blank",
    "genus_only_species_blank",
}

QC_CRITERIA = [
    (
        "passes_taxonomy_rule",
        "taxonomy_not_supported",
    ),
    (
        "passes_source_identity_rule",
        "source_name_conflict",
    ),
    (
        "passes_genome_quality_rule",
        "genome_quality_not_good",
    ),
    (
        "passes_completeness_rule",
        "checkm_completeness_missing_or_lt95",
    ),
    (
        "passes_contamination_rule",
        "checkm_contamination_missing_or_gt5",
    ),
    (
        "passes_contig_count_rule",
        "contigs_missing_or_gt500",
    ),
    (
        "passes_n50_rule",
        "contig_n50_missing_or_lt20000",
    ),
]

ROBUST_Z_THRESHOLD = 8.0


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                16 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def clean_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def parse_bool(
    series: pd.Series,
) -> pd.Series:
    return (
        clean_text(series)
        .str.casefold()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def robust_absolute_z(
    values: pd.Series,
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    observed = values.dropna().astype(float)

    if observed.empty:
        return result

    median = observed.median()

    mad = (
        observed
        .sub(median)
        .abs()
        .median()
    )

    if (
        pd.isna(mad)
        or mad <= 0
    ):
        result.loc[
            observed.index
        ] = 0.0

        return result

    result.loc[
        observed.index
    ] = (
        0.6744897501960817
        * observed.sub(median).abs()
        / mad
    )

    return result


def build_failure_reasons(
    frame: pd.DataFrame,
) -> pd.Series:
    reasons: list[list[str]] = [
        []
        for _ in range(len(frame))
    ]

    for criterion, reason in QC_CRITERIA:
        failed_positions = np.flatnonzero(
            ~frame[
                criterion
            ].to_numpy(
                dtype=bool
            )
        )

        for position in failed_positions:
            reasons[position].append(
                reason
            )

    return pd.Series(
        [
            "|".join(items)
            for items in reasons
        ],
        index=frame.index,
        dtype=str,
    )


def build_species_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for species, group in frame.groupby(
        "provisional_species",
        sort=True,
    ):
        passing = int(
            group[
                "passes_baseline_metadata_qc"
            ].sum()
        )

        total = len(group)

        rows.append(
            {
                "provisional_species":
                    species,
                "total_genomes":
                    total,
                "passing_genomes":
                    passing,
                "excluded_genomes":
                    total - passing,
                "passing_fraction":
                    passing / total,
            }
        )

    return pd.DataFrame(rows)


def build_reason_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for species, group in frame.groupby(
        "provisional_species",
        sort=True,
    ):
        total = len(group)

        for criterion, reason in QC_CRITERIA:
            failed = int(
                (
                    ~group[criterion]
                ).sum()
            )

            rows.append(
                {
                    "provisional_species":
                        species,
                    "failure_reason":
                        reason,
                    "affected_genomes":
                        failed,
                    "total_genomes":
                        total,
                    "affected_fraction":
                        failed / total,
                }
            )

    return pd.DataFrame(rows)


def build_combination_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    failures = frame.loc[
        ~frame[
            "passes_baseline_metadata_qc"
        ]
    ].copy()

    if failures.empty:
        return pd.DataFrame(
            columns=[
                "provisional_species",
                "failure_reason_combination",
                "genomes",
            ]
        )

    return (
        failures.groupby(
            [
                "provisional_species",
                "baseline_failure_reasons",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "baseline_failure_reasons":
                    "failure_reason_combination",
                "size":
                    "genomes",
            }
        )
        .sort_values(
            [
                "provisional_species",
                "genomes",
                "failure_reason_combination",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_pass_quantiles(
    passing: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    metrics = [
        (
            "genome_length",
            "genome_length_numeric",
        ),
        (
            "gc_content",
            "gc_content_numeric",
        ),
    ]

    for species, group in passing.groupby(
        "provisional_species",
        sort=True,
    ):
        for metric, column in metrics:
            values = (
                group[column]
                .dropna()
                .astype(float)
            )

            row = {
                "provisional_species":
                    species,
                "metric":
                    metric,
                "total_passing_genomes":
                    len(group),
                "present_values":
                    len(values),
            }

            if values.empty:
                row.update(
                    {
                        "minimum": np.nan,
                        "q001": np.nan,
                        "q005": np.nan,
                        "q01": np.nan,
                        "q05": np.nan,
                        "q25": np.nan,
                        "median": np.nan,
                        "q75": np.nan,
                        "q95": np.nan,
                        "q99": np.nan,
                        "q995": np.nan,
                        "q999": np.nan,
                        "maximum": np.nan,
                        "mean": np.nan,
                        "mad": np.nan,
                    }
                )
            else:
                median = values.median()

                row.update(
                    {
                        "minimum":
                            values.min(),
                        "q001":
                            values.quantile(0.001),
                        "q005":
                            values.quantile(0.005),
                        "q01":
                            values.quantile(0.01),
                        "q05":
                            values.quantile(0.05),
                        "q25":
                            values.quantile(0.25),
                        "median":
                            median,
                        "q75":
                            values.quantile(0.75),
                        "q95":
                            values.quantile(0.95),
                        "q99":
                            values.quantile(0.99),
                        "q995":
                            values.quantile(0.995),
                        "q999":
                            values.quantile(0.999),
                        "maximum":
                            values.max(),
                        "mean":
                            values.mean(),
                        "mad":
                            (
                                values
                                .sub(median)
                                .abs()
                                .median()
                            ),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    print(
        "===== BUILD BASELINE METADATA-QC "
        "MANIFEST ====="
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Script 07 audit: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if frame[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Input audit contains duplicate genome IDs."
        )

    numeric_fields = [
        "checkm_completeness",
        "checkm_contamination",
        "contigs",
        "contig_n50",
        "genome_length",
        "gc_content",
    ]

    for field in numeric_fields:
        frame[
            f"{field}_baseline_numeric"
        ] = pd.to_numeric(
            frame[field],
            errors="coerce",
        )

    source_conflict = parse_bool(
        frame[
            "source_name_conflict"
        ]
    )

    frame[
        "passes_taxonomy_rule"
    ] = frame[
        "taxonomy_support_category"
    ].isin(
        ACCEPTED_TAXONOMY_CATEGORIES
    )

    frame[
        "passes_source_identity_rule"
    ] = ~source_conflict

    frame[
        "passes_genome_quality_rule"
    ] = (
        clean_text(
            frame[
                "genome_quality"
            ]
        )
        .str.casefold()
        .eq("good")
    )

    frame[
        "passes_completeness_rule"
    ] = frame[
        "checkm_completeness_baseline_numeric"
    ].ge(95)

    frame[
        "passes_contamination_rule"
    ] = frame[
        "checkm_contamination_baseline_numeric"
    ].le(5)

    frame[
        "passes_contig_count_rule"
    ] = frame[
        "contigs_baseline_numeric"
    ].le(500)

    frame[
        "passes_n50_rule"
    ] = frame[
        "contig_n50_baseline_numeric"
    ].ge(20_000)

    criterion_columns = [
        criterion
        for criterion, _ in QC_CRITERIA
    ]

    frame[
        "passes_baseline_metadata_qc"
    ] = frame[
        criterion_columns
    ].all(axis=1)

    frame[
        "baseline_failure_reasons"
    ] = build_failure_reasons(
        frame
    )

    frame[
        "genome_length_robust_abs_z"
    ] = np.nan

    frame[
        "gc_content_robust_abs_z"
    ] = np.nan

    passing_mask = frame[
        "passes_baseline_metadata_qc"
    ]

    for _, group in frame.loc[
        passing_mask
    ].groupby(
        "provisional_species",
        sort=True,
    ):
        frame.loc[
            group.index,
            "genome_length_robust_abs_z",
        ] = robust_absolute_z(
            group[
                "genome_length_baseline_numeric"
            ]
        )

        frame.loc[
            group.index,
            "gc_content_robust_abs_z",
        ] = robust_absolute_z(
            group[
                "gc_content_baseline_numeric"
            ]
        )

    frame[
        "robust_genome_length_outlier"
    ] = (
        frame[
            "genome_length_robust_abs_z"
        ].gt(
            ROBUST_Z_THRESHOLD
        )
    )

    frame[
        "robust_gc_content_outlier"
    ] = (
        frame[
            "gc_content_robust_abs_z"
        ].gt(
            ROBUST_Z_THRESHOLD
        )
    )

    frame[
        "robust_length_or_gc_outlier"
    ] = (
        frame[
            "robust_genome_length_outlier"
        ]
        |
        frame[
            "robust_gc_content_outlier"
        ]
    )

    passing = frame.loc[
        passing_mask
    ].copy()

    failures = frame.loc[
        ~passing_mask
    ].copy()

    robust_outliers = frame.loc[
        passing_mask
        & frame[
            "robust_length_or_gc_outlier"
        ]
    ].copy()

    robust_outliers = (
        robust_outliers.sort_values(
            [
                "provisional_species",
                "genome_length_robust_abs_z",
                "gc_content_robust_abs_z",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    species_summary = (
        build_species_summary(
            frame
        )
    )

    reason_summary = (
        build_reason_summary(
            frame
        )
    )

    combination_summary = (
        build_combination_summary(
            frame
        )
    )

    pass_quantiles = (
        build_pass_quantiles(
            passing
        )
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        OUTPUT_ROOT
        / "shortlist_baseline_metadata_qc_manifest.tsv"
    )

    pass_ids_path = (
        OUTPUT_ROOT
        / "shortlist_baseline_metadata_qc_pass_ids.txt"
    )

    failures_path = (
        OUTPUT_ROOT
        / "shortlist_baseline_metadata_qc_failures.tsv"
    )

    outliers_path = (
        OUTPUT_ROOT
        / "shortlist_baseline_metadata_qc_robust_outliers.tsv"
    )

    species_summary_path = (
        RESULT_ROOT
        / "shortlist_baseline_metadata_qc_species_summary.tsv"
    )

    reason_summary_path = (
        RESULT_ROOT
        / "shortlist_baseline_metadata_qc_reason_summary.tsv"
    )

    combination_summary_path = (
        RESULT_ROOT
        / "shortlist_baseline_metadata_qc_failure_combinations.tsv"
    )

    pass_quantiles_path = (
        RESULT_ROOT
        / "shortlist_baseline_metadata_qc_pass_length_gc_quantiles.tsv"
    )

    frame.to_csv(
        manifest_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    failures.to_csv(
        failures_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    robust_outliers.to_csv(
        outliers_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    species_summary.to_csv(
        species_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    reason_summary.to_csv(
        reason_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    combination_summary.to_csv(
        combination_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    pass_quantiles.to_csv(
        pass_quantiles_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    passing_ids = (
        passing.sort_values(
            [
                "provisional_species",
                "genome_id",
            ],
            kind="stable",
        )[
            "genome_id"
        ]
        .astype(str)
        .tolist()
    )

    pass_ids_path.write_text(
        "\n".join(
            passing_ids
        )
        + "\n",
        encoding="utf-8",
    )

    output_paths = [
        manifest_path,
        pass_ids_path,
        failures_path,
        outliers_path,
        species_summary_path,
        reason_summary_path,
        combination_summary_path,
        pass_quantiles_path,
    ]

    checksum_path = (
        OUTPUT_ROOT
        / "script08_outputs_sha256.txt"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            output_paths,
            key=lambda item: item.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.as_posix()}\n"
            )

    print(
        "Input genomes:",
        f"{len(frame):,}",
    )

    print(
        "Baseline metadata-QC passing:",
        f"{len(passing):,}",
    )

    print(
        "Baseline metadata-QC excluded:",
        f"{len(failures):,}",
    )

    print(
        "Passing genomes with robust "
        "length/GC flags:",
        f"{len(robust_outliers):,}",
    )

    print()
    print(
        "===== SPECIES SUMMARY ====="
    )

    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== FAILURE-REASON SUMMARY ====="
    )

    print(
        reason_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== ROBUST LENGTH/GC OUTLIERS ====="
    )

    display_columns = [
        "genome_id",
        "provisional_species",
        "genome_name",
        "species",
        "genome_length",
        "gc_content",
        "genome_length_robust_abs_z",
        "gc_content_robust_abs_z",
        "robust_genome_length_outlier",
        "robust_gc_content_outlier",
    ]

    if robust_outliers.empty:
        print("None")
    else:
        print(
            robust_outliers[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "STATUS: BASELINE METADATA-QC "
        "MANIFEST COMPLETE"
    )


if __name__ == "__main__":
    main()
