#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/mic/"
    "postqc_monotherapy_mapping_clean_candidates.tsv"
)

GROUP_AUDIT_PATH = Path(
    "metadata/mic_audit/"
    "postqc_mapping_clean_monotherapy_"
    "repeated_group_audit.tsv"
)

RESULT_ROOT = Path(
    "results/tables"
)

AUDIT_ROOT = Path(
    "metadata/mic_audit"
)

EXPECTED_INPUT_ROWS = 313_910

TARGET_PMIDS = {
    "32205351",
    "31266463",
}

KEYS = [
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
]

SOURCE_KEYS = [
    "pmid",
    *KEYS,
]

READ_COLUMNS = [
    "id",
    "genome_id",
    "genome_name",
    "provisional_species",
    "antibiotic",
    "normalized_antibiotic",
    "normalized_sign",
    "mic_value",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "laboratory_typing_method",
    "laboratory_typing_platform",
    "vendor",
    "testing_standard",
    "testing_standard_year",
    "pmid",
    "date_inserted",
]


def clean(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def join_unique(
    values: pd.Series,
) -> str:
    items = sorted(
        {
            str(value).strip()
            if str(value).strip()
            else "<blank>"
            for value in values
        }
    )

    return "|".join(items)


def join_numbers(
    values: pd.Series,
) -> str:
    items = sorted(
        {
            float(value)
            for value in values
        }
    )

    return "|".join(
        f"{value:g}"
        for value in items
    )


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


def assign_method_class(
    frame: pd.DataFrame,
) -> pd.Series:
    method = (
        frame[
            "laboratory_typing_method"
        ]
        .str.casefold()
    )

    platform = (
        frame[
            "laboratory_typing_platform"
        ]
        .str.casefold()
    )

    result = pd.Series(
        "other_or_unspecified",
        index=frame.index,
        dtype="object",
    )

    blank_mask = (
        method.eq("")
        & platform.eq("")
    )

    broth_mask = method.str.contains(
        "broth",
        regex=False,
    )

    agar_mask = method.str.contains(
        "agar",
        regex=False,
    )

    disk_mask = method.str.contains(
        "disk",
        regex=False,
    )

    etest_mask = (
        platform.str.contains(
            "e-test",
            regex=False,
        )
        |
        platform.str.contains(
            "etest",
            regex=False,
        )
    )

    result.loc[
        blank_mask
    ] = "blank_method_context"

    result.loc[
        broth_mask
    ] = "broth_dilution"

    result.loc[
        agar_mask
    ] = "agar_dilution"

    result.loc[
        disk_mask
    ] = "disk_labelled_nonetest"

    result.loc[
        etest_mask
    ] = "gradient_diffusion_e_test"

    return result


def main() -> None:
    print(
        "===== AUDIT TARGETED MIC SOURCE "
        "IDENTITY AND METHODS ====="
    )

    frame = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=READ_COLUMNS,
        low_memory=False,
    )

    if len(frame) != EXPECTED_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_ROWS:,} "
            f"records; found {len(frame):,}."
        )

    if frame["id"].duplicated().any():
        raise RuntimeError(
            "Input contains duplicate record IDs."
        )

    for column in READ_COLUMNS:
        frame[column] = clean(
            frame[column]
        )

    frame["mic_value"] = pd.to_numeric(
        frame["mic_value"],
        errors="raise",
    )

    target = frame.loc[
        frame["pmid"].isin(
            TARGET_PMIDS
        )
    ].copy()

    observed_pmids = set(
        target["pmid"]
    )

    if observed_pmids != TARGET_PMIDS:
        raise RuntimeError(
            "Expected target PMIDs "
            f"{sorted(TARGET_PMIDS)}; found "
            f"{sorted(observed_pmids)}."
        )

    target["insertion_date"] = (
        target["date_inserted"]
        .str.slice(
            0,
            10,
        )
    )

    target["method_class"] = (
        assign_method_class(
            target
        )
    )

    group_audit = pd.read_csv(
        GROUP_AUDIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_group_columns = set(
        KEYS
        + [
            "records",
            "constraints_compatible",
            "duplicate_class",
            "reduced_constraint_type",
            "intersection_notation",
        ]
    )

    missing_group_columns = sorted(
        required_group_columns
        - set(group_audit.columns)
    )

    if missing_group_columns:
        raise RuntimeError(
            "Missing repeated-group columns: "
            + ", ".join(
                missing_group_columns
            )
        )

    for column in required_group_columns:
        group_audit[column] = clean(
            group_audit[column]
        )

    group_meta = group_audit[
        KEYS
        + [
            "records",
            "constraints_compatible",
            "duplicate_class",
            "reduced_constraint_type",
            "intersection_notation",
        ]
    ].rename(
        columns={
            "records":
                "global_pair_records",
        }
    )

    source_pair_audit = (
        target.groupby(
            SOURCE_KEYS,
            as_index=False,
            dropna=False,
        )
        .agg(
            source_records=(
                "id",
                "size",
            ),
            unique_genome_names=(
                "genome_name",
                "nunique",
            ),
            genome_names=(
                "genome_name",
                join_unique,
            ),
            raw_antibiotic_labels=(
                "antibiotic",
                join_unique,
            ),
            unique_mic_values=(
                "mic_value",
                "nunique",
            ),
            observed_values=(
                "mic_value",
                join_numbers,
            ),
            observed_signs=(
                "normalized_sign",
                join_unique,
            ),
            method_classes=(
                "method_class",
                join_unique,
            ),
            methods=(
                "laboratory_typing_method",
                join_unique,
            ),
            platforms=(
                "laboratory_typing_platform",
                join_unique,
            ),
            vendors=(
                "vendor",
                join_unique,
            ),
            testing_standards=(
                "testing_standard",
                join_unique,
            ),
            testing_standard_years=(
                "testing_standard_year",
                join_unique,
            ),
            insertion_dates=(
                "insertion_date",
                join_unique,
            ),
            record_ids=(
                "id",
                join_unique,
            ),
        )
    )

    source_pair_audit = (
        source_pair_audit.merge(
            group_meta,
            on=KEYS,
            how="left",
            validate="many_to_one",
        )
    )

    source_pair_audit[
        "repeated_within_source"
    ] = source_pair_audit[
        "source_records"
    ].gt(1)

    source_pair_audit[
        "global_pair_status"
    ] = "singleton_in_full_cohort"

    compatible_mask = source_pair_audit[
        "constraints_compatible"
    ].eq("True")

    conflict_mask = source_pair_audit[
        "constraints_compatible"
    ].eq("False")

    source_pair_audit.loc[
        compatible_mask,
        "global_pair_status",
    ] = "compatible_repeated_full_cohort"

    source_pair_audit.loc[
        conflict_mask,
        "global_pair_status",
    ] = "conflicting_repeated_full_cohort"


    overall_rows = []

    for pmid in sorted(
        TARGET_PMIDS
    ):
        source_records = target.loc[
            target["pmid"].eq(pmid)
        ]

        source_pairs = source_pair_audit.loc[
            source_pair_audit[
                "pmid"
            ].eq(pmid)
        ]

        overall_rows.append(
            {
                "pmid":
                    pmid,
                "records":
                    len(source_records),
                "unique_genomes":
                    source_records[
                        "genome_id"
                    ].nunique(),
                "raw_antibiotic_labels":
                    source_records[
                        "antibiotic"
                    ].nunique(),
                "normalized_antibiotics":
                    source_records[
                        "normalized_antibiotic"
                    ].nunique(),
                "genome_antibiotic_pairs":
                    len(source_pairs),
                "pairs_repeated_within_source":
                    int(
                        source_pairs[
                            "repeated_within_source"
                        ].sum()
                    ),
                "globally_compatible_pairs":
                    int(
                        source_pairs[
                            "global_pair_status"
                        ].eq(
                            "compatible_repeated_full_cohort"
                        ).sum()
                    ),
                "globally_conflicting_pairs":
                    int(
                        source_pairs[
                            "global_pair_status"
                        ].eq(
                            "conflicting_repeated_full_cohort"
                        ).sum()
                    ),
                "maximum_records_per_source_pair":
                    int(
                        source_pairs[
                            "source_records"
                        ].max()
                    ),
                "maximum_unique_values_per_pair":
                    int(
                        source_pairs[
                            "unique_mic_values"
                        ].max()
                    ),
            }
        )

    overall_summary = pd.DataFrame(
        overall_rows
    )

    context_summary = (
        target.groupby(
            [
                "pmid",
                "provisional_species",
                "antibiotic",
                "normalized_antibiotic",
                "insertion_date",
                "method_class",
                "laboratory_typing_method",
                "laboratory_typing_platform",
                "vendor",
                "testing_standard",
                "testing_standard_year",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            records=(
                "id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            genome_antibiotic_pairs=(
                "genome_id",
                "nunique",
            ),
            unique_values=(
                "mic_value",
                "nunique",
            ),
            minimum_value=(
                "mic_value",
                "min",
            ),
            maximum_value=(
                "mic_value",
                "max",
            ),
            exact_records=(
                "normalized_sign",
                lambda values: int(
                    values.isin(
                        {
                            "",
                            "=",
                        }
                    ).sum()
                ),
            ),
            left_censored_records=(
                "normalized_sign",
                lambda values: int(
                    values.isin(
                        {
                            "<",
                            "<=",
                        }
                    ).sum()
                ),
            ),
            right_censored_records=(
                "normalized_sign",
                lambda values: int(
                    values.isin(
                        {
                            ">",
                            ">=",
                        }
                    ).sum()
                ),
            ),
        )
        .sort_values(
            [
                "pmid",
                "records",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
    )

    drug_summary = (
        target.groupby(
            [
                "pmid",
                "provisional_species",
                "antibiotic",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            records=(
                "id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            unique_values=(
                "mic_value",
                "nunique",
            ),
            minimum_value=(
                "mic_value",
                "min",
            ),
            maximum_value=(
                "mic_value",
                "max",
            ),
            method_classes=(
                "method_class",
                join_unique,
            ),
            methods=(
                "laboratory_typing_method",
                join_unique,
            ),
            insertion_dates=(
                "insertion_date",
                join_unique,
            ),
        )
        .sort_values(
            [
                "pmid",
                "records",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
    )

    method_profile_summary = (
        source_pair_audit.groupby(
            [
                "pmid",
                "provisional_species",
                "normalized_antibiotic",
                "method_classes",
                "global_pair_status",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            genome_antibiotic_pairs=(
                "genome_id",
                "size",
            ),
            source_records=(
                "source_records",
                "sum",
            ),
            maximum_records_per_pair=(
                "source_records",
                "max",
            ),
            maximum_unique_values_per_pair=(
                "unique_mic_values",
                "max",
            ),
        )
        .sort_values(
            [
                "pmid",
                "genome_antibiotic_pairs",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
    )

    source_pair_audit = (
        source_pair_audit.sort_values(
            [
                "pmid",
                "global_pair_status",
                "source_records",
                "unique_mic_values",
                "genome_id",
                "normalized_antibiotic",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    target = target.sort_values(
        [
            "pmid",
            "genome_id",
            "normalized_antibiotic",
            "insertion_date",
            "method_class",
            "mic_value",
            "normalized_sign",
            "id",
        ],
        kind="stable",
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
        / "targeted_mic_source_overall_summary.tsv":
            overall_summary,

        RESULT_ROOT
        / "targeted_mic_source_context_summary.tsv":
            context_summary,

        RESULT_ROOT
        / "targeted_mic_source_drug_summary.tsv":
            drug_summary,

        RESULT_ROOT
        / "targeted_mic_source_method_profile_summary.tsv":
            method_profile_summary,

        AUDIT_ROOT
        / "targeted_mic_source_pair_audit.tsv":
            source_pair_audit,

        AUDIT_ROOT
        / "targeted_mic_source_all_records.tsv":
            target,
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
        / "script22_outputs_sha256.txt"
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
        "Target records:",
        f"{len(target):,}",
    )

    print()
    print(
        "===== OVERALL SUMMARY ====="
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== DRUG SUMMARY ====="
    )

    print(
        drug_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== CONTEXT SUMMARY ====="
    )

    print(
        context_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== METHOD-PROFILE SUMMARY ====="
    )

    print(
        method_profile_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: TARGETED MIC SOURCE "
        "IDENTITY/METHOD AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
