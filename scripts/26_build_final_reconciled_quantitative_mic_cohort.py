#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "data/interim/mic/"
    "postqc_monotherapy_source_policy_clean_candidates.tsv"
)

AUDIT = Path(
    "metadata/mic_audit/"
    "postqc_source_policy_clean_monotherapy_"
    "repeated_group_audit.tsv"
)

OUT = Path("data/processed/mic")
META = Path("metadata/mic_audit")
TABLES = Path("results/tables")

KEYS = [
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
]

EXPECTED = {
    "input_rows": 310_048,
    "pairs": 286_614,
    "singletons": 264_449,
    "repeated_pairs": 22_165,
    "repeated_records": 45_599,
    "compatible_pairs": 21_348,
    "compatible_records": 43_566,
    "conflict_pairs": 817,
    "conflict_records": 2_033,
    "final_rows": 285_797,
}

EXPECTED_SPECIES = {
    "Acinetobacter baumannii": 15_003,
    "Escherichia coli": 100_311,
    "Klebsiella pneumoniae": 71_566,
    "Pseudomonas aeruginosa": 6_186,
    "Salmonella enterica": 92_731,
}


def clean(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_bool(
    series: pd.Series,
    name: str,
) -> pd.Series:
    parsed = (
        clean(series)
        .str.casefold()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if parsed.isna().any():
        raise RuntimeError(
            f"Cannot parse boolean column {name}."
        )

    return parsed.astype(bool)


def join_unique(
    series: pd.Series,
) -> str:
    values = sorted(
        {
            (
                str(value).strip()
                if str(value).strip()
                else "<blank>"
            )
            for value in series
        }
    )

    return "|".join(values)


def format_number(
    value: float,
) -> str:
    if pd.isna(value):
        return ""

    return f"{float(value):g}"


def interval_notation(
    lower: float,
    lower_closed: bool,
    upper: float,
    upper_closed: bool,
) -> str:
    return (
        ("[" if lower_closed else "(")
        + (
            "-inf"
            if pd.isna(lower)
            else format_number(lower)
        )
        + ","
        + (
            "inf"
            if pd.isna(upper)
            else format_number(upper)
        )
        + ("]" if upper_closed else ")")
    )


def make_pair_id(
    species: str,
    genome: str,
    antibiotic: str,
) -> str:
    payload = "\x1f".join(
        [
            species,
            genome,
            antibiotic,
        ]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return f"micpair_{digest}"


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


def main() -> None:
    print(
        "===== BUILD FINAL RECONCILED "
        "QUANTITATIVE MIC COHORT ====="
    )

    frame = pd.read_csv(
        INPUT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "id",
        "genome_id",
        "genome_name",
        "taxon_id",
        "antibiotic",
        "measurement",
        "measurement_sign",
        "measurement_value",
        "normalized_unit",
        "laboratory_typing_method",
        "laboratory_typing_method_version",
        "laboratory_typing_platform",
        "vendor",
        "testing_standard",
        "testing_standard_year",
        "pmid",
        "date_inserted",
        "date_modified",
        "insertion_date",
        "provisional_species",
        "normalized_sign",
        "mic_value",
        "normalized_antibiotic",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing input columns: "
            + ", ".join(missing)
        )

    if len(frame) != EXPECTED["input_rows"]:
        raise RuntimeError(
            f"Expected {EXPECTED['input_rows']:,} "
            f"rows; found {len(frame):,}."
        )

    if frame["id"].duplicated().any():
        raise RuntimeError(
            "Input record IDs are not unique."
        )

    for column in frame.columns:
        frame[column] = clean(
            frame[column]
        )

    frame["mic_value_numeric"] = (
        pd.to_numeric(
            frame["mic_value"],
            errors="raise",
        )
    )

    if frame[
        "mic_value_numeric"
    ].le(0).any():
        raise RuntimeError(
            "Non-positive MIC value found."
        )

    if frame[
        "normalized_unit"
    ].eq("").any():
        raise RuntimeError(
            "Blank normalized MIC unit found."
        )

    pair_unit_counts = (
        frame.groupby(
            KEYS
        )[
            "normalized_unit"
        ]
        .nunique(
            dropna=False
        )
    )

    if not pair_unit_counts.eq(1).all():
        raise RuntimeError(
            "A genome-antibiotic pair has "
            "multiple normalized units."
        )

    pair_sizes = (
        frame.groupby(
            KEYS,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "pair_records",
            }
        )
    )

    observed_structure = {
        "pairs":
            len(pair_sizes),
        "singletons":
            int(
                pair_sizes[
                    "pair_records"
                ].eq(1).sum()
            ),
        "repeated_pairs":
            int(
                pair_sizes[
                    "pair_records"
                ].gt(1).sum()
            ),
        "repeated_records":
            int(
                pair_sizes.loc[
                    pair_sizes[
                        "pair_records"
                    ].gt(1),
                    "pair_records",
                ].sum()
            ),
    }

    expected_structure = {
        key: EXPECTED[key]
        for key in observed_structure
    }

    if observed_structure != expected_structure:
        raise RuntimeError(
            "Pair structure changed. "
            f"Expected {expected_structure}; "
            f"observed {observed_structure}."
        )

    audit = pd.read_csv(
        AUDIT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    audit_required = {
        *KEYS,
        "records",
        "unique_values",
        "unique_signs",
        "exact_records",
        "left_censored_records",
        "right_censored_records",
        "observed_values",
        "observed_signs",
        "intersection_lower",
        "intersection_lower_closed",
        "intersection_upper",
        "intersection_upper_closed",
        "intersection_notation",
        "constraints_compatible",
        "duplicate_class",
        "reduced_constraint_type",
    }

    missing = sorted(
        audit_required - set(audit.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing repeated-audit columns: "
            + ", ".join(missing)
        )

    for column in audit.columns:
        audit[column] = clean(
            audit[column]
        )

    if len(audit) != EXPECTED[
        "repeated_pairs"
    ]:
        raise RuntimeError(
            "Unexpected repeated-pair audit size."
        )

    if audit.duplicated(
        subset=KEYS
    ).any():
        raise RuntimeError(
            "Repeated-pair audit contains "
            "duplicate keys."
        )

    audit["records_numeric"] = (
        pd.to_numeric(
            audit["records"],
            errors="raise",
        )
    )

    audit["compatible"] = parse_bool(
        audit[
            "constraints_compatible"
        ],
        "constraints_compatible",
    )

    audit["lower"] = pd.to_numeric(
        audit[
            "intersection_lower"
        ].replace(
            "",
            np.nan,
        ),
        errors="coerce",
    )

    audit["upper"] = pd.to_numeric(
        audit[
            "intersection_upper"
        ].replace(
            "",
            np.nan,
        ),
        errors="coerce",
    )

    audit["lower_closed"] = parse_bool(
        audit[
            "intersection_lower_closed"
        ],
        "intersection_lower_closed",
    )

    audit["upper_closed"] = parse_bool(
        audit[
            "intersection_upper_closed"
        ],
        "intersection_upper_closed",
    )

    repeated_sizes = pair_sizes.loc[
        pair_sizes[
            "pair_records"
        ].gt(1)
    ]

    check = audit.merge(
        repeated_sizes,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not check["_merge"].eq(
        "both"
    ).all():
        raise RuntimeError(
            "Repeated-pair keys differ between "
            "the input and audit."
        )

    if not check[
        "records_numeric"
    ].eq(
        check[
            "pair_records"
        ]
    ).all():
        raise RuntimeError(
            "Repeated-pair record counts differ "
            "between input and audit."
        )

    compatible = audit.loc[
        audit["compatible"]
    ].copy()

    conflicts = audit.loc[
        ~audit["compatible"]
    ].copy()

    observed_audit_counts = {
        "compatible_pairs":
            len(compatible),
        "compatible_records":
            int(
                compatible[
                    "records_numeric"
                ].sum()
            ),
        "conflict_pairs":
            len(conflicts),
        "conflict_records":
            int(
                conflicts[
                    "records_numeric"
                ].sum()
            ),
    }

    expected_audit_counts = {
        key: EXPECTED[key]
        for key in observed_audit_counts
    }

    if (
        observed_audit_counts
        != expected_audit_counts
    ):
        raise RuntimeError(
            "Repeated-audit counts changed. "
            f"Observed: {observed_audit_counts}"
        )

    compatible_types = set(
        compatible[
            "reduced_constraint_type"
        ]
    )

    if not compatible_types.issubset(
        {
            "exact",
            "left_censored",
            "right_censored",
        }
    ):
        raise RuntimeError(
            "Unexpected compatible constraint type."
        )

    if set(
        conflicts[
            "reduced_constraint_type"
        ]
    ) != {
        "empty_intersection",
    }:
        raise RuntimeError(
            "Conflicting pairs are not all "
            "empty intersections."
        )

    frame["source_context"] = (
        frame[
            [
                "pmid",
                "insertion_date",
                "laboratory_typing_method",
                "laboratory_typing_platform",
                "vendor",
                "testing_standard",
                "testing_standard_year",
            ]
        ]
        .replace(
            "",
            "<blank>",
        )
        .agg(
            " || ".join,
            axis=1,
        )
    )

    provenance = (
        frame.groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            source_record_count=(
                "id",
                "size",
            ),
            source_record_ids=(
                "id",
                join_unique,
            ),
            source_genome_names=(
                "genome_name",
                join_unique,
            ),
            source_taxon_ids=(
                "taxon_id",
                join_unique,
            ),
            source_antibiotic_labels=(
                "antibiotic",
                join_unique,
            ),
            normalized_unit=(
                "normalized_unit",
                join_unique,
            ),
            source_measurements=(
                "measurement",
                join_unique,
            ),
            source_measurement_signs=(
                "measurement_sign",
                join_unique,
            ),
            source_measurement_values=(
                "measurement_value",
                join_unique,
            ),
            source_normalized_signs=(
                "normalized_sign",
                join_unique,
            ),
            source_mic_values=(
                "mic_value",
                join_unique,
            ),
            source_methods=(
                "laboratory_typing_method",
                join_unique,
            ),
            source_method_versions=(
                "laboratory_typing_method_version",
                join_unique,
            ),
            source_platforms=(
                "laboratory_typing_platform",
                join_unique,
            ),
            source_vendors=(
                "vendor",
                join_unique,
            ),
            source_testing_standards=(
                "testing_standard",
                join_unique,
            ),
            source_testing_standard_years=(
                "testing_standard_year",
                join_unique,
            ),
            source_pmids=(
                "pmid",
                join_unique,
            ),
            source_insertion_dates=(
                "insertion_date",
                join_unique,
            ),
            source_context_count=(
                "source_context",
                "nunique",
            ),
            source_contexts=(
                "source_context",
                join_unique,
            ),
        )
    )

    provenance["pair_id"] = [
        make_pair_id(
            species,
            genome,
            antibiotic,
        )
        for species, genome, antibiotic
        in provenance[
            KEYS
        ].itertuples(
            index=False,
            name=None,
        )
    ]

    if provenance[
        "pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Pair-ID collision detected."
        )

    singleton_keys = pair_sizes.loc[
        pair_sizes[
            "pair_records"
        ].eq(1),
        KEYS,
    ]

    singleton = frame.merge(
        singleton_keys,
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    if len(singleton) != EXPECTED[
        "singletons"
    ]:
        raise RuntimeError(
            "Unexpected number of singleton "
            f"records after key selection: "
            f"{len(singleton):,}."
        )

    if singleton.duplicated(
        subset=KEYS
    ).any():
        raise RuntimeError(
            "Singleton selection produced "
            "duplicate genome-antibiotic keys."
        )

    if set(
        singleton["id"]
    ) != set(
        frame.loc[
            frame.set_index(
                KEYS
            ).index.isin(
                singleton_keys.set_index(
                    KEYS
                ).index
            ),
            "id",
        ]
    ):
        raise RuntimeError(
            "Singleton selection did not preserve "
            "the expected source-record IDs."
        )

    sign = singleton[
        "normalized_sign"
    ]

    exact = sign.isin(
        {
            "",
            "=",
        }
    )

    left = sign.isin(
        {
            "<",
            "<=",
        }
    )

    right = sign.isin(
        {
            ">",
            ">=",
        }
    )

    if not (
        exact
        | left
        | right
    ).all():
        raise RuntimeError(
            "Unexpected singleton censoring sign."
        )

    single = singleton[
        KEYS
    ].copy()

    value = singleton[
        "mic_value_numeric"
    ]

    single[
        "reconciliation_status"
    ] = "singleton_retained"

    single[
        "duplicate_class"
    ] = "singleton"

    single[
        "reduced_constraint_type"
    ] = pd.Series(
        np.select(
            [
                exact,
                left,
                right,
            ],
            [
                "exact",
                "left_censored",
                "right_censored",
            ],
            default="__invalid_constraint_type__",
        ),
        index=single.index,
        dtype="object",
    )

    if single[
        "reduced_constraint_type"
    ].eq(
        "__invalid_constraint_type__"
    ).any():
        invalid_signs = sorted(
            set(
                single.loc[
                    single[
                        "reduced_constraint_type"
                    ].eq(
                        "__invalid_constraint_type__"
                    ),
                    "normalized_sign",
                ]
            )
        )

        raise RuntimeError(
            "Singleton rows contain unsupported "
            "censoring signs: "
            f"{invalid_signs}"
        )

    single[
        "reduced_sign"
    ] = np.where(
        exact,
        "=",
        sign,
    )

    single[
        "reduced_mic_value"
    ] = value

    single[
        "intersection_lower"
    ] = np.where(
        exact | right,
        value,
        np.nan,
    )

    single[
        "intersection_upper"
    ] = np.where(
        exact | left,
        value,
        np.nan,
    )

    single[
        "intersection_lower_closed"
    ] = np.where(
        exact,
        True,
        np.where(
            right,
            sign.eq(">="),
            False,
        ),
    ).astype(bool)

    single[
        "intersection_upper_closed"
    ] = np.where(
        exact,
        True,
        np.where(
            left,
            sign.eq("<="),
            False,
        ),
    ).astype(bool)

    single[
        "source_unique_values"
    ] = 1

    single[
        "source_unique_signs"
    ] = 1

    single[
        "source_exact_records"
    ] = exact.astype(int)

    single[
        "source_left_censored_records"
    ] = left.astype(int)

    single[
        "source_right_censored_records"
    ] = right.astype(int)

    single[
        "source_observed_values"
    ] = value.map(
        format_number
    )

    single[
        "source_observed_signs"
    ] = sign.replace(
        "",
        "<blank>",
    )

    single[
        "intersection_notation"
    ] = [
        interval_notation(
            lower_bound,
            lower_closed,
            upper_bound,
            upper_closed,
        )
        for (
            lower_bound,
            lower_closed,
            upper_bound,
            upper_closed,
        ) in single[
            [
                "intersection_lower",
                "intersection_lower_closed",
                "intersection_upper",
                "intersection_upper_closed",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ]

    repeated = compatible[
        KEYS
        + [
            "records_numeric",
            "unique_values",
            "unique_signs",
            "exact_records",
            "left_censored_records",
            "right_censored_records",
            "observed_values",
            "observed_signs",
            "duplicate_class",
            "reduced_constraint_type",
            "lower",
            "lower_closed",
            "upper",
            "upper_closed",
            "intersection_notation",
        ]
    ].copy()

    repeated = repeated.rename(
        columns={
            "records_numeric":
                "audit_source_record_count",
            "unique_values":
                "source_unique_values",
            "unique_signs":
                "source_unique_signs",
            "exact_records":
                "source_exact_records",
            "left_censored_records":
                "source_left_censored_records",
            "right_censored_records":
                "source_right_censored_records",
            "observed_values":
                "source_observed_values",
            "observed_signs":
                "source_observed_signs",
            "lower":
                "intersection_lower",
            "lower_closed":
                "intersection_lower_closed",
            "upper":
                "intersection_upper",
            "upper_closed":
                "intersection_upper_closed",
            "intersection_notation":
                "audit_intersection_notation",
        }
    )

    for column in [
        "source_unique_values",
        "source_unique_signs",
        "source_exact_records",
        "source_left_censored_records",
        "source_right_censored_records",
    ]:
        repeated[column] = pd.to_numeric(
            repeated[column],
            errors="raise",
        )

    repeated[
        "reconciliation_status"
    ] = "compatible_repeated_collapsed"

    repeated_exact = repeated[
        "reduced_constraint_type"
    ].eq("exact")

    repeated_left = repeated[
        "reduced_constraint_type"
    ].eq("left_censored")

    repeated_right = repeated[
        "reduced_constraint_type"
    ].eq("right_censored")

    if repeated.loc[
        repeated_exact,
        [
            "intersection_lower",
            "intersection_upper",
        ],
    ].isna().any().any():
        raise RuntimeError(
            "Exact repeated constraints have "
            "missing bounds."
        )

    if not np.isclose(
        repeated.loc[
            repeated_exact,
            "intersection_lower",
        ],
        repeated.loc[
            repeated_exact,
            "intersection_upper",
        ],
    ).all():
        raise RuntimeError(
            "Exact repeated constraints have "
            "unequal bounds."
        )

    if not repeated.loc[
        repeated_exact,
        "intersection_lower_closed",
    ].all():
        raise RuntimeError(
            "Exact repeated lower bounds "
            "are not closed."
        )

    if not repeated.loc[
        repeated_exact,
        "intersection_upper_closed",
    ].all():
        raise RuntimeError(
            "Exact repeated upper bounds "
            "are not closed."
        )

    if repeated.loc[
        repeated_left,
        "intersection_upper",
    ].isna().any():
        raise RuntimeError(
            "Left-censored repeated constraints "
            "lack an upper bound."
        )

    if repeated.loc[
        repeated_right,
        "intersection_lower",
    ].isna().any():
        raise RuntimeError(
            "Right-censored repeated constraints "
            "lack a lower bound."
        )

    repeated.loc[
        repeated_left,
        "intersection_lower",
    ] = np.nan

    repeated.loc[
        repeated_left,
        "intersection_lower_closed",
    ] = False

    repeated.loc[
        repeated_right,
        "intersection_upper",
    ] = np.nan

    repeated.loc[
        repeated_right,
        "intersection_upper_closed",
    ] = False

    repeated[
        "reduced_sign"
    ] = "="

    repeated.loc[
        repeated_left,
        "reduced_sign",
    ] = np.where(
        repeated.loc[
            repeated_left,
            "intersection_upper_closed",
        ],
        "<=",
        "<",
    )

    repeated.loc[
        repeated_right,
        "reduced_sign",
    ] = np.where(
        repeated.loc[
            repeated_right,
            "intersection_lower_closed",
        ],
        ">=",
        ">",
    )

    repeated[
        "reduced_mic_value"
    ] = np.where(
        repeated_exact
        | repeated_right,
        repeated[
            "intersection_lower"
        ],
        repeated[
            "intersection_upper"
        ],
    )

    repeated[
        "intersection_notation"
    ] = [
        interval_notation(
            lower_bound,
            lower_closed,
            upper_bound,
            upper_closed,
        )
        for (
            lower_bound,
            lower_closed,
            upper_bound,
            upper_closed,
        ) in repeated[
            [
                "intersection_lower",
                "intersection_lower_closed",
                "intersection_upper",
                "intersection_upper_closed",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ]

    constraints = pd.concat(
        [
            single,
            repeated,
        ],
        ignore_index=True,
        sort=False,
    )

    if len(constraints) != EXPECTED[
        "final_rows"
    ]:
        raise RuntimeError(
            "Unexpected final retained "
            "constraint count."
        )

    if constraints.duplicated(
        subset=KEYS
    ).any():
        raise RuntimeError(
            "Retained constraints contain "
            "duplicate keys."
        )

    final = constraints.merge(
        provenance,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    final[
        "observation_id"
    ] = final[
        "pair_id"
    ]

    final[
        "constraint_origin"
    ] = np.where(
        final[
            "reconciliation_status"
        ].eq(
            "singleton_retained"
        ),
        "single_source_record",
        "intersection_of_compatible_records",
    )

    repeated_final = final[
        "reconciliation_status"
    ].eq(
        "compatible_repeated_collapsed"
    )

    if not final.loc[
        repeated_final,
        "audit_source_record_count",
    ].eq(
        final.loc[
            repeated_final,
            "source_record_count",
        ]
    ).all():
        raise RuntimeError(
            "Repeated source counts do not "
            "match provenance."
        )

    if final[
        "reduced_mic_value"
    ].isna().any():
        raise RuntimeError(
            "Missing reduced MIC value."
        )

    if final[
        "reduced_mic_value"
    ].le(0).any():
        raise RuntimeError(
            "Non-positive reduced MIC value."
        )

    if final.duplicated(
        subset=KEYS
    ).any():
        raise RuntimeError(
            "Final table contains duplicate "
            "genome-antibiotic pairs."
        )

    observed_species = (
        final[
            "provisional_species"
        ]
        .value_counts()
        .to_dict()
    )

    if observed_species != EXPECTED_SPECIES:
        raise RuntimeError(
            "Unexpected final species counts. "
            f"Observed: {observed_species}"
        )

    conflict_pairs = conflicts.merge(
        provenance,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    conflict_pairs[
        "reconciliation_exclusion_reason"
    ] = (
        "conflicting_constraints_"
        "empty_intersection"
    )

    if len(conflict_pairs) != EXPECTED[
        "conflict_pairs"
    ]:
        raise RuntimeError(
            "Unexpected conflict-pair table size."
        )

    flow = pd.DataFrame(
        [
            [
                "source_policy_clean_input_records",
                EXPECTED["input_rows"],
                0,
            ],
            [
                "singleton_records_retained",
                EXPECTED["singletons"],
                EXPECTED["singletons"],
            ],
            [
                "compatible_repeated_records_collapsed",
                EXPECTED["compatible_records"],
                EXPECTED["compatible_pairs"],
            ],
            [
                "redundant_compatible_records_removed",
                (
                    EXPECTED["compatible_records"]
                    - EXPECTED["compatible_pairs"]
                ),
                0,
            ],
            [
                "conflicting_records_excluded",
                EXPECTED["conflict_records"],
                0,
            ],
            [
                "final_reconciled_observations",
                (
                    EXPECTED["input_rows"]
                    - EXPECTED["conflict_records"]
                ),
                EXPECTED["final_rows"],
            ],
        ],
        columns=[
            "stage",
            "source_records",
            "pair_level_observations",
        ],
    )

    species_summary = (
        final.groupby(
            "provisional_species",
            as_index=False,
        )
        .agg(
            observations=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
            exact_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "exact"
                    ).sum()
                ),
            ),
            left_censored_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "left_censored"
                    ).sum()
                ),
            ),
            right_censored_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "right_censored"
                    ).sum()
                ),
            ),
            collapsed_compatible_observations=(
                "reconciliation_status",
                lambda values: int(
                    values.eq(
                        "compatible_repeated_collapsed"
                    ).sum()
                ),
            ),
            contributing_source_records=(
                "source_record_count",
                "sum",
            ),
        )
        .sort_values(
            "observations",
            ascending=False,
        )
    )

    censoring_summary = (
        final.groupby(
            [
                "reduced_constraint_type",
                "reduced_sign",
            ],
            as_index=False,
        )
        .agg(
            observations=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
        )
        .sort_values(
            "observations",
            ascending=False,
        )
    )

    species_antibiotic_summary = (
        final.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            observations=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            exact_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "exact"
                    ).sum()
                ),
            ),
            left_censored_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "left_censored"
                    ).sum()
                ),
            ),
            right_censored_observations=(
                "reduced_constraint_type",
                lambda values: int(
                    values.eq(
                        "right_censored"
                    ).sum()
                ),
            ),
            collapsed_compatible_observations=(
                "reconciliation_status",
                lambda values: int(
                    values.eq(
                        "compatible_repeated_collapsed"
                    ).sum()
                ),
            ),
            contributing_source_records=(
                "source_record_count",
                "sum",
            ),
        )
        .sort_values(
            [
                "provisional_species",
                "observations",
                "normalized_antibiotic",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
    )

    final = final.sort_values(
        KEYS,
        kind="stable",
    ).reset_index(
        drop=True
    )

    conflict_pairs = (
        conflict_pairs.sort_values(
            KEYS,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    META.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        OUT
        / "multispecies_monotherapy_"
        "quantitative_mic_reconciled.tsv"
    )

    conflict_path = (
        META
        / "final_conflicting_genome_"
        "antibiotic_pairs.tsv"
    )

    flow_path = (
        TABLES
        / "final_mic_reconciliation_flow.tsv"
    )

    species_path = (
        TABLES
        / "final_mic_species_summary.tsv"
    )

    censoring_path = (
        TABLES
        / "final_mic_censoring_summary.tsv"
    )

    species_antibiotic_path = (
        TABLES
        / "final_mic_species_antibiotic_summary.tsv"
    )

    outputs = {
        final_path:
            final,
        conflict_path:
            conflict_pairs,
        flow_path:
            flow,
        species_path:
            species_summary,
        censoring_path:
            censoring_summary,
        species_antibiotic_path:
            species_antibiotic_summary,
    }

    for path, table in outputs.items():
        table.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )

    checksum_path = (
        META
        / "script26_outputs_sha256.txt"
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
        "Input source records:",
        f"{len(frame):,}",
    )

    print(
        "Singleton observations retained:",
        f"{EXPECTED['singletons']:,}",
    )

    print(
        "Compatible repeated observations "
        "collapsed:",
        f"{EXPECTED['compatible_pairs']:,}",
    )

    print(
        "Conflicting pairs excluded:",
        f"{EXPECTED['conflict_pairs']:,}",
    )

    print(
        "Final reconciled observations:",
        f"{len(final):,}",
    )

    print(
        "Final represented genomes:",
        f"{final['genome_id'].nunique():,}",
    )

    print(
        "Final antibiotic identities:",
        f"{final['normalized_antibiotic'].nunique():,}",
    )

    print()
    print(
        "===== RECONCILIATION FLOW ====="
    )
    print(
        flow.to_string(
            index=False
        )
    )

    print()
    print(
        "===== FINAL SPECIES SUMMARY ====="
    )
    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== FINAL CENSORING SUMMARY ====="
    )
    print(
        censoring_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: FINAL RECONCILED "
        "QUANTITATIVE MIC COHORT COMPLETE"
    )


if __name__ == "__main__":
    main()
