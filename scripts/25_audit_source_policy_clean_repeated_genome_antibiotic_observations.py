#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/interim/mic/"
    "postqc_monotherapy_source_policy_clean_candidates.tsv"
)

RESULT_ROOT = Path(
    "results/tables"
)

AUDIT_ROOT = Path(
    "metadata/mic_audit"
)

EXPECTED_ROWS = 310_048
EXPECTED_IDENTITIES = 83
EXPECTED_PAIRS = 286_614
EXPECTED_SINGLETON_PAIRS = 264_449
EXPECTED_REPEATED_PAIRS = 22_165
EXPECTED_REPEATED_RECORDS = 45_599
EXPECTED_MAX_RECORDS_PER_PAIR = 3

KEYS = [
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
]

READ_COLUMNS = [
    "id",
    "genome_id",
    "genome_name",
    "provisional_species",
    "normalized_antibiotic",
    "normalized_sign",
    "mic_value",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "laboratory_typing_method",
    "laboratory_typing_platform",
    "pmid",
    "date_inserted",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def join_text(
    values: pd.Series,
    blank_label: str = "<blank>",
) -> str:
    result = sorted(
        {
            str(value).strip()
            if str(value).strip()
            else blank_label
            for value in values
        }
    )

    return "|".join(result)


def join_numbers(values: pd.Series) -> str:
    result = sorted(
        {
            float(value)
            for value in values
        }
    )

    return "|".join(
        f"{value:g}"
        for value in result
    )


def record_interval(
    sign: str,
    value: float,
) -> tuple[
    float,
    bool,
    float,
    bool,
]:
    if sign in {"", "="}:
        return value, True, value, True

    if sign == "<":
        return -math.inf, False, value, False

    if sign == "<=":
        return -math.inf, False, value, True

    if sign == ">":
        return value, False, math.inf, False

    if sign == ">=":
        return value, True, math.inf, False

    raise ValueError(
        f"Unsupported sign: {sign!r}"
    )


def format_bound(value: float) -> str:
    if math.isinf(value):
        return ""

    return f"{value:g}"


def analyse_group(
    group: pd.DataFrame,
) -> dict[str, object]:
    intervals = [
        record_interval(
            str(sign),
            float(value),
        )
        for sign, value in zip(
            group["normalized_sign"],
            group["mic_value"],
            strict=True,
        )
    ]

    lower = max(
        interval[0]
        for interval in intervals
    )

    upper = min(
        interval[2]
        for interval in intervals
    )

    lower_ties = [
        interval
        for interval in intervals
        if np.isclose(
            interval[0],
            lower,
            rtol=0,
            atol=1e-12,
        )
        or (
            math.isinf(interval[0])
            and math.isinf(lower)
        )
    ]

    upper_ties = [
        interval
        for interval in intervals
        if np.isclose(
            interval[2],
            upper,
            rtol=0,
            atol=1e-12,
        )
        or (
            math.isinf(interval[2])
            and math.isinf(upper)
        )
    ]

    lower_closed = (
        all(
            interval[1]
            for interval in lower_ties
        )
        if not math.isinf(lower)
        else False
    )

    upper_closed = (
        all(
            interval[3]
            for interval in upper_ties
        )
        if not math.isinf(upper)
        else False
    )

    equal_bounds = (
        not math.isinf(lower)
        and not math.isinf(upper)
        and np.isclose(
            lower,
            upper,
            rtol=0,
            atol=1e-12,
        )
    )

    compatible = (
        lower < upper
        or (
            equal_bounds
            and lower_closed
            and upper_closed
        )
    )

    exact_mask = group[
        "normalized_sign"
    ].isin({"", "="})

    left_mask = group[
        "normalized_sign"
    ].isin({"<", "<="})

    right_mask = group[
        "normalized_sign"
    ].isin({">", ">="})

    semantic_tokens = {
        (
            "exact"
            if sign in {"", "="}
            else sign,
            float(value),
        )
        for sign, value in zip(
            group["normalized_sign"],
            group["mic_value"],
            strict=True,
        )
    }

    if not compatible:
        duplicate_class = (
            "conflicting_constraints"
        )
        reduced_type = "empty_intersection"

    elif len(semantic_tokens) == 1:
        duplicate_class = (
            "identical_or_equivalent_duplicate"
        )

        if exact_mask.all():
            reduced_type = "exact"
        elif left_mask.all():
            reduced_type = "left_censored"
        else:
            reduced_type = "right_censored"

    elif exact_mask.any():
        duplicate_class = (
            "compatible_exact_with_censoring"
        )
        reduced_type = "exact"

    else:
        duplicate_class = (
            "compatible_censored_constraints"
        )

        if (
            not math.isinf(lower)
            and not math.isinf(upper)
        ):
            reduced_type = "bounded_interval"
        elif not math.isinf(lower):
            reduced_type = "right_censored"
        else:
            reduced_type = "left_censored"

    left_bracket = (
        "["
        if lower_closed
        else "("
    )

    right_bracket = (
        "]"
        if upper_closed
        else ")"
    )

    lower_text = (
        "-inf"
        if math.isinf(lower)
        else format_bound(lower)
    )

    upper_text = (
        "inf"
        if math.isinf(upper)
        else format_bound(upper)
    )

    return {
        "records":
            len(group),
        "unique_values":
            group["mic_value"].nunique(),
        "unique_signs":
            group["normalized_sign"].nunique(),
        "exact_records":
            int(exact_mask.sum()),
        "left_censored_records":
            int(left_mask.sum()),
        "right_censored_records":
            int(right_mask.sum()),
        "observed_values":
            join_numbers(group["mic_value"]),
        "observed_signs":
            join_text(
                group["normalized_sign"]
            ),
        "methods":
            join_text(
                group[
                    "laboratory_typing_method"
                ]
            ),
        "platforms":
            join_text(
                group[
                    "laboratory_typing_platform"
                ]
            ),
        "pmids":
            join_text(
                group["pmid"]
            ),
        "intersection_lower":
            format_bound(lower),
        "intersection_lower_closed":
            lower_closed,
        "intersection_upper":
            format_bound(upper),
        "intersection_upper_closed":
            upper_closed,
        "intersection_notation":
            (
                f"{left_bracket}"
                f"{lower_text},"
                f"{upper_text}"
                f"{right_bracket}"
            ),
        "constraints_compatible":
            compatible,
        "duplicate_class":
            duplicate_class,
        "reduced_constraint_type":
            reduced_type,
    }


def main() -> None:
    print(
        "===== AUDIT REPEATED GENOME-"
        "ANTIBIOTIC OBSERVATIONS ====="
    )

    frame = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=READ_COLUMNS,
        low_memory=False,
    )

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows; "
            f"found {len(frame):,}."
        )

    if frame["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate record IDs found."
        )

    for column in READ_COLUMNS:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    frame["mic_value"] = pd.to_numeric(
        frame["mic_value"],
        errors="raise",
    )

    supported_signs = {
        "",
        "=",
        "<",
        "<=",
        ">",
        ">=",
    }

    unexpected_signs = sorted(
        set(frame["normalized_sign"])
        - supported_signs
    )

    if unexpected_signs:
        raise RuntimeError(
            "Unexpected signs: "
            + ", ".join(unexpected_signs)
        )

    if (
        frame[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_IDENTITIES
    ):
        raise RuntimeError(
            "Unexpected antibiotic-identity "
            "count."
        )

    pair_counts = (
        frame.groupby(
            KEYS,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "records",
            }
        )
    )

    repeated_keys = pair_counts.loc[
        pair_counts["records"].gt(1),
        KEYS,
    ]

    repeated_records = frame.merge(
        repeated_keys,
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    observed_pairs = len(
        pair_counts
    )

    observed_singletons = int(
        pair_counts[
            "records"
        ].eq(1).sum()
    )

    observed_repeated_pairs = int(
        pair_counts[
            "records"
        ].gt(1).sum()
    )

    observed_repeated_records = len(
        repeated_records
    )

    observed_maximum = int(
        pair_counts[
            "records"
        ].max()
    )

    expected_structure = {
        "genome_antibiotic_pairs":
            EXPECTED_PAIRS,
        "singleton_pairs":
            EXPECTED_SINGLETON_PAIRS,
        "repeated_pairs":
            EXPECTED_REPEATED_PAIRS,
        "records_in_repeated_pairs":
            EXPECTED_REPEATED_RECORDS,
        "maximum_records_per_pair":
            EXPECTED_MAX_RECORDS_PER_PAIR,
    }

    observed_structure = {
        "genome_antibiotic_pairs":
            observed_pairs,
        "singleton_pairs":
            observed_singletons,
        "repeated_pairs":
            observed_repeated_pairs,
        "records_in_repeated_pairs":
            observed_repeated_records,
        "maximum_records_per_pair":
            observed_maximum,
    }

    if observed_structure != expected_structure:
        raise RuntimeError(
            "Mapping-clean pair structure differs "
            "from the frozen structural probe. "
            f"Expected {expected_structure}; "
            f"observed {observed_structure}."
        )

    audit_rows: list[
        dict[str, object]
    ] = []

    for key, group in repeated_records.groupby(
        KEYS,
        sort=True,
        observed=True,
    ):
        result = analyse_group(group)

        row = dict(
            zip(
                KEYS,
                key,
                strict=True,
            )
        )

        row.update(result)
        audit_rows.append(row)

    repeated_audit = pd.DataFrame(
        audit_rows
    )

    if repeated_audit.empty:
        raise RuntimeError(
            "No repeated pairs were found."
        )

    conflict_groups = repeated_audit.loc[
        ~repeated_audit[
            "constraints_compatible"
        ]
    ].copy()

    conflict_keys = conflict_groups[
        KEYS
    ]

    conflict_records = frame.merge(
        conflict_keys,
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    conflict_records = conflict_records.merge(
        conflict_groups[
            KEYS
            + [
                "records",
                "observed_values",
                "observed_signs",
                "intersection_notation",
                "duplicate_class",
            ]
        ],
        on=KEYS,
        how="left",
        validate="many_to_one",
        suffixes=(
            "_record",
            "_group",
        ),
    )

    singleton_pairs = int(
        pair_counts["records"].eq(1).sum()
    )

    repeated_pairs = len(
        repeated_audit
    )

    compatible_pairs = int(
        repeated_audit[
            "constraints_compatible"
        ].sum()
    )

    conflicting_pairs = len(
        conflict_groups
    )

    overall_summary = pd.DataFrame(
        [
            {
                "metric":
                    "input_records",
                "value":
                    len(frame),
            },
            {
                "metric":
                    "genome_antibiotic_pairs",
                "value":
                    len(pair_counts),
            },
            {
                "metric":
                    "singleton_pairs",
                "value":
                    singleton_pairs,
            },
            {
                "metric":
                    "repeated_pairs",
                "value":
                    repeated_pairs,
            },
            {
                "metric":
                    "records_in_repeated_pairs",
                "value":
                    len(repeated_records),
            },
            {
                "metric":
                    "compatible_repeated_pairs",
                "value":
                    compatible_pairs,
            },
            {
                "metric":
                    "conflicting_repeated_pairs",
                "value":
                    conflicting_pairs,
            },
            {
                "metric":
                    "records_in_conflicting_pairs",
                "value":
                    len(conflict_records),
            },
            {
                "metric":
                    "maximum_records_per_pair",
                "value":
                    int(
                        pair_counts[
                            "records"
                        ].max()
                    ),
            },
        ]
    )

    class_summary = (
        repeated_audit.groupby(
            [
                "duplicate_class",
                "reduced_constraint_type",
                "constraints_compatible",
            ],
            as_index=False,
        )
        .agg(
            genome_antibiotic_pairs=(
                "genome_id",
                "size",
            ),
            records=(
                "records",
                "sum",
            ),
        )
        .sort_values(
            "genome_antibiotic_pairs",
            ascending=False,
        )
    )

    size_distribution = (
        pair_counts.groupby(
            "records",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "genome_antibiotic_pairs",
            }
        )
        .sort_values("records")
    )

    species_base = (
        pair_counts.groupby(
            "provisional_species",
            as_index=False,
        )
        .agg(
            genome_antibiotic_pairs=(
                "genome_id",
                "size",
            ),
            input_records=(
                "records",
                "sum",
            ),
            singleton_pairs=(
                "records",
                lambda values: int(
                    values.eq(1).sum()
                ),
            ),
            repeated_pairs=(
                "records",
                lambda values: int(
                    values.gt(1).sum()
                ),
            ),
            maximum_records_per_pair=(
                "records",
                "max",
            ),
        )
    )

    species_repeated = (
        repeated_audit.groupby(
            "provisional_species",
            as_index=False,
        )
        .agg(
            records_in_repeated_pairs=(
                "records",
                "sum",
            ),
            compatible_repeated_pairs=(
                "constraints_compatible",
                "sum",
            ),
            conflicting_repeated_pairs=(
                "constraints_compatible",
                lambda values: int(
                    (~values).sum()
                ),
            ),
        )
    )

    species_summary = species_base.merge(
        species_repeated,
        on="provisional_species",
        how="left",
        validate="one_to_one",
    ).fillna(0)

    antibiotic_summary = (
        repeated_audit.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            repeated_pairs=(
                "genome_id",
                "size",
            ),
            repeated_records=(
                "records",
                "sum",
            ),
            compatible_pairs=(
                "constraints_compatible",
                "sum",
            ),
            conflicting_pairs=(
                "constraints_compatible",
                lambda values: int(
                    (~values).sum()
                ),
            ),
            maximum_records_per_pair=(
                "records",
                "max",
            ),
        )
        .sort_values(
            [
                "conflicting_pairs",
                "repeated_pairs",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        RESULT_ROOT
        / "postqc_source_policy_clean_monotherapy_duplicate_overall_summary.tsv":
            overall_summary,

        RESULT_ROOT
        / "postqc_source_policy_clean_monotherapy_duplicate_class_summary.tsv":
            class_summary,

        RESULT_ROOT
        / "postqc_source_policy_clean_monotherapy_duplicate_group_size_distribution.tsv":
            size_distribution,

        RESULT_ROOT
        / "postqc_source_policy_clean_monotherapy_duplicate_species_summary.tsv":
            species_summary,

        RESULT_ROOT
        / "postqc_source_policy_clean_monotherapy_duplicate_antibiotic_summary.tsv":
            antibiotic_summary,

        AUDIT_ROOT
        / "postqc_source_policy_clean_monotherapy_repeated_group_audit.tsv":
            repeated_audit,

        AUDIT_ROOT
        / "postqc_source_policy_clean_monotherapy_conflicting_records.tsv":
            conflict_records,
    }

    for path, table in outputs.items():
        table.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )

    checksum_path = (
        AUDIT_ROOT
        / "script25_outputs_sha256.txt"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            outputs,
            key=lambda item: item.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.as_posix()}\n"
            )

    print(
        "Input records:",
        f"{len(frame):,}",
    )

    print(
        "Genome-antibiotic pairs:",
        f"{len(pair_counts):,}",
    )

    print(
        "Repeated pairs:",
        f"{repeated_pairs:,}",
    )

    print(
        "Compatible repeated pairs:",
        f"{compatible_pairs:,}",
    )

    print(
        "Conflicting repeated pairs:",
        f"{conflicting_pairs:,}",
    )

    print()
    print("===== OVERALL SUMMARY =====")
    print(
        overall_summary.to_string(
            index=False
        )
    )

    print()
    print("===== DUPLICATE CLASSES =====")
    print(
        class_summary.to_string(
            index=False
        )
    )

    print()
    print("===== SPECIES SUMMARY =====")
    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: REPEATED GENOME-ANTIBIOTIC "
        "AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
