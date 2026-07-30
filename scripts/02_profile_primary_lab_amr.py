#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_GLOB = (
    "bvbrc_primary_laboratory_amr_*.tsv"
)

INPUT_ROOT = Path(
    "data/raw/amr"
)

OUTPUT_ROOT = Path(
    "results/tables"
)

METADATA_ROOT = Path(
    "metadata/profiling"
)

CHUNK_SIZE = 100_000

EXPECTED_COLUMNS = [
    "id",
    "genome_id",
    "genome_name",
    "taxon_id",
    "antibiotic",
    "evidence",
    "resistant_phenotype",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "measurement_unit",
    "laboratory_typing_method",
    "laboratory_typing_method_version",
    "laboratory_typing_platform",
    "vendor",
    "testing_standard",
    "testing_standard_year",
    "pmid",
    "date_inserted",
    "date_modified",
]

NUMERIC_PATTERN = re.compile(
    r"[+-]?"
    r"(?:"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][+-]?\d+)?"
)

THRESHOLDS = [
    (
        "g100_u50",
        100,
        50,
    ),
    (
        "g200_u100",
        200,
        100,
    ),
    (
        "g500_u200",
        500,
        200,
    ),
    (
        "g1000_u200",
        1_000,
        200,
    ),
]

OVERALL_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_overall_summary.tsv"
)

FIELD_MISSINGNESS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_field_missingness.tsv"
)

UNIT_COUNTS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_measurement_units.tsv"
)

SIGN_COUNTS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_measurement_signs.tsv"
)

METHOD_COUNTS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_laboratory_methods.tsv"
)

PLATFORM_COUNTS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_laboratory_platforms.tsv"
)

STANDARD_COUNTS_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_testing_standards.tsv"
)

ANTIBIOTIC_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_antibiotic_summary.tsv"
)

TAXON_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_taxon_summary.tsv"
)

TAXON_DRUG_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_taxon_antibiotic_coverage.tsv"
)

THRESHOLD_SWEEP_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_candidate_taxon_threshold_sweep.tsv"
)

DUPLICATE_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "bvbrc_primary_lab_amr_duplicate_pair_summary.tsv"
)

DEFINITION_PATH = (
    METADATA_ROOT
    / "primary_lab_amr_profile_definition.txt"
)

OUTPUT_CHECKSUM_PATH = (
    METADATA_ROOT
    / "primary_lab_amr_profile_outputs_sha256.txt"
)


def make_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "measurement_present_rows": 0,
        "scalar_numeric_rows": 0,
        "paired_rows": 0,
        "quantitative_candidate_rows": 0,
        "mic_like_method_candidate_rows": 0,
        "exact_rows": 0,
        "blank_sign_rows": 0,
        "left_censored_rows": 0,
        "right_censored_rows": 0,
        "other_sign_rows": 0,
        "phenotype_present_rows": 0,
        "genomes": set(),
        "candidate_genomes": set(),
        "antibiotics": set(),
        "candidate_antibiotics": set(),
        "labels": Counter(),
        "mic_values": set(),
    }


def sha256_file(path: Path) -> str:
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


def normalize_unit(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .str.strip()
        .str.lower()
        .str.replace(
            "µ",
            "u",
            regex=False,
        )
        .str.replace(
            "μ",
            "u",
            regex=False,
        )
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )

    replacements = {
        "milligram/liter": "mg/l",
        "milligrams/liter": "mg/l",
        "milligram/litre": "mg/l",
        "milligrams/litre": "mg/l",
        "mgperliter": "mg/l",
        "mgperlitre": "mg/l",
        "microgram/ml": "ug/ml",
        "micrograms/ml": "ug/ml",
        "mcg/ml": "ug/ml",
    }

    return normalized.replace(
        replacements
    )


def normalize_sign(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .str.strip()
        .str.replace(
            "≤",
            "<=",
            regex=False,
        )
        .str.replace(
            "≥",
            ">=",
            regex=False,
        )
        .replace(
            {
                "==": "=",
            }
        )
    )

    return normalized


def provisional_binomial(
    genome_name: str,
) -> str:
    words = genome_name.strip().split()

    if len(words) < 2:
        return ""

    first = words[0].strip(
        "[](),;"
    )

    second = words[1].strip(
        "[](),;"
    )

    if not first or not second:
        return ""

    return f"{first} {second}"


def update_group_stats(
    stats: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    stats["rows"] += len(frame)

    stats["measurement_present_rows"] += int(
        frame["_measurement_present"].sum()
    )

    stats["scalar_numeric_rows"] += int(
        frame["_scalar_numeric"].sum()
    )

    stats["paired_rows"] += int(
        frame["_paired"].sum()
    )

    stats[
        "quantitative_candidate_rows"
    ] += int(
        frame["_quantitative_candidate"].sum()
    )

    stats[
        "mic_like_method_candidate_rows"
    ] += int(
        frame[
            "_mic_like_method_candidate"
        ].sum()
    )

    stats["exact_rows"] += int(
        frame["_candidate_exact"].sum()
    )

    stats["blank_sign_rows"] += int(
        frame["_candidate_blank_sign"].sum()
    )

    stats["left_censored_rows"] += int(
        frame["_candidate_left"].sum()
    )

    stats["right_censored_rows"] += int(
        frame["_candidate_right"].sum()
    )

    stats["other_sign_rows"] += int(
        frame["_candidate_other_sign"].sum()
    )

    stats["phenotype_present_rows"] += int(
        frame["_phenotype_present"].sum()
    )

    stats["genomes"].update(
        frame.loc[
            frame["genome_id"].ne(""),
            "genome_id",
        ].tolist()
    )

    candidate_frame = frame.loc[
        frame["_quantitative_candidate"]
    ]

    stats["candidate_genomes"].update(
        candidate_frame.loc[
            candidate_frame[
                "genome_id"
            ].ne(""),
            "genome_id",
        ].tolist()
    )

    stats["antibiotics"].update(
        frame.loc[
            frame["antibiotic"].ne(""),
            "antibiotic",
        ].tolist()
    )

    stats[
        "candidate_antibiotics"
    ].update(
        candidate_frame.loc[
            candidate_frame[
                "antibiotic"
            ].ne(""),
            "antibiotic",
        ].tolist()
    )

    labels = frame.loc[
        frame[
            "_provisional_binomial"
        ].ne(""),
        "_provisional_binomial",
    ].value_counts()

    stats["labels"].update(
        {
            str(label): int(count)
            for label, count
            in labels.items()
        }
    )

    candidate_values = (
        candidate_frame[
            "_numeric_value"
        ]
        .dropna()
        .astype(float)
        .tolist()
    )

    stats["mic_values"].update(
        candidate_values
    )


def stats_to_row(
    stats: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = int(
        stats[
            "quantitative_candidate_rows"
        ]
    )

    censored_rows = int(
        stats["left_censored_rows"]
        + stats["right_censored_rows"]
    )

    uncensored_rows = int(
        stats["exact_rows"]
        + stats["blank_sign_rows"]
    )

    mic_values = sorted(
        float(value)
        for value in stats["mic_values"]
        if math.isfinite(float(value))
    )

    return {
        "rows": int(stats["rows"]),
        "unique_genomes": len(
            stats["genomes"]
        ),
        "unique_antibiotics": len(
            stats["antibiotics"]
        ),
        "measurement_present_rows": int(
            stats[
                "measurement_present_rows"
            ]
        ),
        "scalar_numeric_rows": int(
            stats["scalar_numeric_rows"]
        ),
        "paired_rows": int(
            stats["paired_rows"]
        ),
        "quantitative_candidate_rows": (
            candidate_rows
        ),
        "quantitative_candidate_genomes": len(
            stats["candidate_genomes"]
        ),
        "quantitative_candidate_antibiotics": len(
            stats["candidate_antibiotics"]
        ),
        "mic_like_method_candidate_rows": int(
            stats[
                "mic_like_method_candidate_rows"
            ]
        ),
        "exact_sign_rows": int(
            stats["exact_rows"]
        ),
        "blank_sign_rows": int(
            stats["blank_sign_rows"]
        ),
        "uncensored_rows": uncensored_rows,
        "left_censored_rows": int(
            stats["left_censored_rows"]
        ),
        "right_censored_rows": int(
            stats["right_censored_rows"]
        ),
        "censored_rows": censored_rows,
        "other_sign_rows": int(
            stats["other_sign_rows"]
        ),
        "candidate_censoring_fraction": (
            censored_rows / candidate_rows
            if candidate_rows
            else math.nan
        ),
        "distinct_candidate_mic_values": len(
            mic_values
        ),
        "minimum_candidate_mic": (
            mic_values[0]
            if mic_values
            else math.nan
        ),
        "maximum_candidate_mic": (
            mic_values[-1]
            if mic_values
            else math.nan
        ),
        "phenotype_present_rows": int(
            stats["phenotype_present_rows"]
        ),
    }


def main() -> None:
    print(
        "===== PROFILE BV-BRC PRIMARY "
        "LABORATORY AMR TABLE ====="
    )

    input_paths = sorted(
        INPUT_ROOT.glob(
            INPUT_GLOB
        )
    )

    if len(input_paths) != 1:
        raise ValueError(
            "Expected exactly one primary "
            "laboratory AMR TSV; found "
            f"{len(input_paths)}: {input_paths}"
        )

    input_path = input_paths[0]

    print("Input:", input_path)
    print(
        "Input size:",
        f"{input_path.stat().st_size:,}",
        "bytes",
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = pd.read_csv(
        input_path,
        sep="\t",
        nrows=0,
    ).columns.tolist()

    if header != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected input header.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Observed: {header}"
        )

    total_rows = 0

    nonblank_counts = Counter()

    unit_counts = Counter()
    normalized_unit_counts = Counter()

    sign_counts = Counter()
    normalized_sign_counts = Counter()

    method_counts = Counter()
    platform_counts = Counter()
    standard_counts = Counter()

    raw_antibiotic_counts = Counter()

    overall_stats = make_stats()

    taxon_stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(make_stats)

    taxon_drug_stats: dict[
        tuple[str, str],
        dict[str, Any],
    ] = defaultdict(make_stats)

    anti_stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(make_stats)

    antibiotic_stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(make_stats)

    genome_drug_pair_counts = Counter()

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        for column in EXPECTED_COLUMNS:
            chunk[column] = (
                chunk[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            nonblank_counts[column] += int(
                chunk[column].ne("").sum()
            )

        total_rows += len(chunk)

        measurement = chunk[
            "measurement"
        ]

        measurement_value = chunk[
            "measurement_value"
        ]

        measurement_sign = normalize_sign(
            chunk["measurement_sign"]
        )

        measurement_unit = chunk[
            "measurement_unit"
        ]

        normalized_unit = normalize_unit(
            measurement_unit
        )

        scalar_numeric = (
            measurement_value
            .str.fullmatch(
                NUMERIC_PATTERN,
                na=False,
            )
        )

        numeric_value = pd.to_numeric(
            measurement_value.where(
                scalar_numeric
            ),
            errors="coerce",
        )

        paired = (
            measurement_value.str.contains(
                "/",
                regex=False,
            )
            |
            measurement.str.contains(
                "/",
                regex=False,
            )
        )

        measurement_present = (
            measurement.ne("")
            |
            measurement_value.ne("")
        )

        equivalent_mg_l = (
            normalized_unit.isin(
                {
                    "mg/l",
                    "ug/ml",
                }
            )
        )

        supported_sign = (
            measurement_sign.isin(
                {
                    "",
                    "=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                }
            )
        )

        quantitative_candidate = (
            scalar_numeric
            & equivalent_mg_l
            & supported_sign
            & ~paired
        )

        method_lower = (
            chunk[
                "laboratory_typing_method"
            ]
            .str.lower()
        )

        mic_like_method = (
            method_lower.str.contains(
                r"\bmic\b"
                r"|minimum inhibitory"
                r"|microdilution"
                r"|broth dilution"
                r"|agar dilution"
                r"|e-?test",
                regex=True,
                na=False,
            )
        )

        mic_like_method_candidate = (
            quantitative_candidate
            & mic_like_method
        )

        candidate_exact = (
            quantitative_candidate
            & measurement_sign.eq("=")
        )

        candidate_blank_sign = (
            quantitative_candidate
            & measurement_sign.eq("")
        )

        candidate_left = (
            quantitative_candidate
            & measurement_sign.isin(
                {
                    "<",
                    "<=",
                }
            )
        )

        candidate_right = (
            quantitative_candidate
            & measurement_sign.isin(
                {
                    ">",
                    ">=",
                }
            )
        )

        candidate_other_sign = (
            quantitative_candidate
            & ~(
                candidate_exact
                | candidate_blank_sign
                | candidate_left
                | candidate_right
            )
        )

        phenotype_present = chunk[
            "resistant_phenotype"
        ].ne("")

        chunk[
            "_measurement_present"
        ] = measurement_present

        chunk[
            "_scalar_numeric"
        ] = scalar_numeric

        chunk[
            "_numeric_value"
        ] = numeric_value

        chunk[
            "_paired"
        ] = paired

        chunk[
            "_quantitative_candidate"
        ] = quantitative_candidate

        chunk[
            "_mic_like_method_candidate"
        ] = mic_like_method_candidate

        chunk[
            "_candidate_exact"
        ] = candidate_exact

        chunk[
            "_candidate_blank_sign"
        ] = candidate_blank_sign

        chunk[
            "_candidate_left"
        ] = candidate_left

        chunk[
            "_candidate_right"
        ] = candidate_right

        chunk[
            "_candidate_other_sign"
        ] = candidate_other_sign

        chunk[
            "_phenotype_present"
        ] = phenotype_present

        chunk[
            "_provisional_binomial"
        ] = chunk["genome_name"].map(
            provisional_binomial
        )

        unit_counts.update(
            measurement_unit.value_counts(
                dropna=False
            ).to_dict()
        )

        normalized_unit_counts.update(
            normalized_unit.value_counts(
                dropna=False
            ).to_dict()
        )

        sign_counts.update(
            chunk[
                "measurement_sign"
            ].value_counts(
                dropna=False
            ).to_dict()
        )

        normalized_sign_counts.update(
            measurement_sign.value_counts(
                dropna=False
            ).to_dict()
        )

        method_counts.update(
            chunk[
                "laboratory_typing_method"
            ].value_counts(
                dropna=False
            ).to_dict()
        )

        platform_counts.update(
            chunk[
                "laboratory_typing_platform"
            ].value_counts(
                dropna=False
            ).to_dict()
        )

        standard_counts.update(
            chunk[
                "testing_standard"
            ].value_counts(
                dropna=False
            ).to_dict()
        )

        raw_antibiotic_counts.update(
            chunk["antibiotic"].value_counts(
                dropna=False
            ).to_dict()
        )

        update_group_stats(
            overall_stats,
            chunk,
        )

        for taxon_id, group in chunk.groupby(
            "taxon_id",
            sort=False,
            dropna=False,
        ):
            update_group_stats(
                taxon_stats[str(taxon_id)],
                group,
            )

        for antibiotic, group in chunk.groupby(
            "antibiotic",
            sort=False,
            dropna=False,
        ):
            update_group_stats(
                antibiotic_stats[
                    str(antibiotic)
                ],
                group,
            )

        for (
            taxon_id,
            antibiotic,
        ), group in chunk.groupby(
            [
                "taxon_id",
                "antibiotic",
            ],
            sort=False,
            dropna=False,
        ):
            update_group_stats(
                taxon_drug_stats[
                    (
                        str(taxon_id),
                        str(antibiotic),
                    )
                ],
                group,
            )

        pair_frame = chunk.loc[
            chunk["genome_id"].ne("")
            & chunk["antibiotic"].ne(""),
            [
                "genome_id",
                "antibiotic",
            ],
        ]

        pair_value_counts = (
            pair_frame.value_counts(
                sort=False
            )
        )

        for (
            genome_id,
            antibiotic,
        ), count in pair_value_counts.items():
            genome_drug_pair_counts[
                (
                    str(genome_id),
                    str(antibiotic),
                )
            ] += int(count)

        print(
            "Processed chunk",
            chunk_number,
            "- cumulative rows:",
            f"{total_rows:,}",
            flush=True,
        )

    overall_row = stats_to_row(
        overall_stats
    )

    overall_rows = [
        {
            "metric": "input_rows",
            "value": total_rows,
        },
        {
            "metric": "unique_record_ids",
            "value": nonblank_counts["id"],
        },
    ]

    for key, value in overall_row.items():
        overall_rows.append(
            {
                "metric": key,
                "value": value,
            }
        )

    unique_pairs = len(
        genome_drug_pair_counts
    )

    repeated_pairs = sum(
        count > 1
        for count
        in genome_drug_pair_counts.values()
    )

    rows_in_repeated_pairs = sum(
        count
        for count
        in genome_drug_pair_counts.values()
        if count > 1
    )

    maximum_pair_multiplicity = max(
        genome_drug_pair_counts.values(),
        default=0,
    )

    overall_rows.extend(
        [
            {
                "metric": (
                    "unique_genome_antibiotic_pairs"
                ),
                "value": unique_pairs,
            },
            {
                "metric": (
                    "repeated_genome_antibiotic_pairs"
                ),
                "value": repeated_pairs,
            },
            {
                "metric": (
                    "rows_in_repeated_pairs"
                ),
                "value": rows_in_repeated_pairs,
            },
            {
                "metric": (
                    "maximum_pair_multiplicity"
                ),
                "value": maximum_pair_multiplicity,
            },
        ]
    )

    overall_frame = pd.DataFrame(
        overall_rows
    )

    field_missingness_rows = []

    for column in EXPECTED_COLUMNS:
        nonblank = int(
            nonblank_counts[column]
        )

        field_missingness_rows.append(
            {
                "field": column,
                "nonblank_rows": nonblank,
                "blank_rows": (
                    total_rows - nonblank
                ),
                "nonblank_fraction": (
                    nonblank / total_rows
                    if total_rows
                    else math.nan
                ),
            }
        )

    field_missingness_frame = (
        pd.DataFrame(
            field_missingness_rows
        )
    )

    def counter_frame(
        raw_counter: Counter,
        value_column: str,
    ) -> pd.DataFrame:
        rows = [
            {
                value_column: value,
                "rows": int(count),
                "fraction": (
                    int(count) / total_rows
                    if total_rows
                    else math.nan
                ),
            }
            for value, count
            in raw_counter.items()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "rows",
                    value_column,
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    unit_frame = counter_frame(
        unit_counts,
        "measurement_unit",
    )

    normalized_unit_frame = counter_frame(
        normalized_unit_counts,
        "normalized_measurement_unit",
    )

    unit_frame = unit_frame.merge(
        normalized_unit_frame,
        left_on="measurement_unit",
        right_on=(
            "normalized_measurement_unit"
        ),
        how="outer",
        suffixes=(
            "_raw",
            "_normalized",
        ),
    )

    sign_frame = counter_frame(
        sign_counts,
        "measurement_sign",
    )

    normalized_sign_frame = counter_frame(
        normalized_sign_counts,
        "normalized_measurement_sign",
    )

    method_frame = counter_frame(
        method_counts,
        "laboratory_typing_method",
    )

    platform_frame = counter_frame(
        platform_counts,
        "laboratory_typing_platform",
    )

    standard_frame = counter_frame(
        standard_counts,
        "testing_standard",
    )

    antibiotic_rows = []

    for antibiotic, stats in (
        antibiotic_stats.items()
    ):
        row = stats_to_row(stats)

        row["antibiotic"] = antibiotic

        antibiotic_rows.append(row)

    antibiotic_frame = (
        pd.DataFrame(
            antibiotic_rows
        )
        .sort_values(
            [
                "quantitative_candidate_rows",
                "quantitative_candidate_genomes",
                "antibiotic",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    taxon_rows = []

    for taxon_id, stats in (
        taxon_stats.items()
    ):
        row = stats_to_row(stats)

        label_counts = stats["labels"]

        if label_counts:
            dominant_label, label_rows = (
                label_counts.most_common(1)[0]
            )
        else:
            dominant_label = ""
            label_rows = 0

        row["taxon_id"] = taxon_id

        row[
            "provisional_dominant_binomial"
        ] = dominant_label

        row[
            "dominant_binomial_rows"
        ] = int(label_rows)

        row[
            "distinct_provisional_binomials"
        ] = len(label_counts)

        taxon_rows.append(row)

    taxon_frame = (
        pd.DataFrame(
            taxon_rows
        )
        .sort_values(
            [
                "quantitative_candidate_rows",
                "quantitative_candidate_genomes",
                "taxon_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    taxon_drug_rows = []

    for (
        taxon_id,
        antibiotic,
    ), stats in taxon_drug_stats.items():
        row = stats_to_row(stats)

        label_counts = stats["labels"]

        dominant_label = (
            label_counts.most_common(1)[0][0]
            if label_counts
            else ""
        )

        row["taxon_id"] = taxon_id

        row[
            "provisional_dominant_binomial"
        ] = dominant_label

        row["antibiotic"] = antibiotic

        taxon_drug_rows.append(row)

    taxon_drug_frame = (
        pd.DataFrame(
            taxon_drug_rows
        )
        .sort_values(
            [
                "taxon_id",
                "quantitative_candidate_genomes",
                "quantitative_candidate_rows",
                "antibiotic",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    threshold_rows = []

    for (
        threshold_name,
        minimum_genomes,
        minimum_uncensored,
    ) in THRESHOLDS:
        eligible = taxon_drug_frame.loc[
            (
                taxon_drug_frame[
                    "quantitative_candidate_genomes"
                ]
                >= minimum_genomes
            )
            &
            (
                taxon_drug_frame[
                    "uncensored_rows"
                ]
                >= minimum_uncensored
            )
        ]

        eligible_grouped = {
            str(taxon_id): group
            for taxon_id, group
            in eligible.groupby(
                "taxon_id",
                sort=False,
            )
        }

        for _, taxon_row in (
            taxon_frame.iterrows()
        ):
            taxon_id = str(
                taxon_row["taxon_id"]
            )

            group = eligible_grouped.get(
                taxon_id
            )

            if group is None:
                eligible_drugs = 0
                eligible_rows = 0
                eligible_drug_names = ""
            else:
                eligible_drugs = len(group)

                eligible_rows = int(
                    group[
                        "quantitative_candidate_rows"
                    ].sum()
                )

                eligible_drug_names = "|".join(
                    sorted(
                        group[
                            "antibiotic"
                        ].astype(str)
                    )
                )

            threshold_rows.append(
                {
                    "threshold_name": (
                        threshold_name
                    ),
                    "minimum_unique_genomes_per_drug": (
                        minimum_genomes
                    ),
                    "minimum_uncensored_rows_per_drug": (
                        minimum_uncensored
                    ),
                    "taxon_id": taxon_id,
                    "provisional_dominant_binomial": (
                        taxon_row[
                            "provisional_dominant_binomial"
                        ]
                    ),
                    "quantitative_candidate_genomes": (
                        int(
                            taxon_row[
                                "quantitative_candidate_genomes"
                            ]
                        )
                    ),
                    "quantitative_candidate_rows": (
                        int(
                            taxon_row[
                                "quantitative_candidate_rows"
                            ]
                        )
                    ),
                    "eligible_antibiotics": (
                        eligible_drugs
                    ),
                    "eligible_candidate_rows": (
                        eligible_rows
                    ),
                    "eligible_antibiotic_names": (
                        eligible_drug_names
                    ),
                }
            )

    threshold_frame = (
        pd.DataFrame(
            threshold_rows
        )
        .sort_values(
            [
                "threshold_name",
                "eligible_antibiotics",
                "eligible_candidate_rows",
                "quantitative_candidate_genomes",
                "taxon_id",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    duplicate_frame = pd.DataFrame(
        [
            {
                "metric": (
                    "unique_genome_antibiotic_pairs"
                ),
                "value": unique_pairs,
            },
            {
                "metric": (
                    "pairs_with_multiple_records"
                ),
                "value": repeated_pairs,
            },
            {
                "metric": (
                    "rows_in_pairs_with_multiple_records"
                ),
                "value": rows_in_repeated_pairs,
            },
            {
                "metric": (
                    "maximum_records_for_one_pair"
                ),
                "value": maximum_pair_multiplicity,
            },
        ]
    )

    output_frames = {
        OVERALL_PATH: overall_frame,
        FIELD_MISSINGNESS_PATH: (
            field_missingness_frame
        ),
        UNIT_COUNTS_PATH: unit_frame,
        SIGN_COUNTS_PATH: (
            normalized_sign_frame
        ),
        METHOD_COUNTS_PATH: method_frame,
        PLATFORM_COUNTS_PATH: (
            platform_frame
        ),
        STANDARD_COUNTS_PATH: (
            standard_frame
        ),
        ANTIBIOTIC_SUMMARY_PATH: (
            antibiotic_frame
        ),
        TAXON_SUMMARY_PATH: taxon_frame,
        TAXON_DRUG_PATH: (
            taxon_drug_frame
        ),
        THRESHOLD_SWEEP_PATH: (
            threshold_frame
        ),
        DUPLICATE_SUMMARY_PATH: (
            duplicate_frame
        ),
    }

    for path, frame in (
        output_frames.items()
    ):
        frame.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )

    definition_lines = [
        (
            "BV-BRC primary laboratory AMR "
            "global profile"
        ),
        "",
        f"Input: {input_path.as_posix()}",
        f"Rows: {total_rows}",
        (
            "Quantitative candidate definition:"
        ),
        (
            "- scalar numeric measurement_value"
        ),
        (
            "- unit equivalent to mg/L "
            "(mg/L or ug/mL)"
        ),
        (
            "- supported sign: blank, =, <, <=, >, >="
        ),
        (
            "- no slash-separated paired value"
        ),
        "",
        (
            "A blank measurement sign is retained "
            "separately and counted as uncensored "
            "for feasibility summaries."
        ),
        (
            "No species has been selected."
        ),
        (
            "Provisional binomials are derived only "
            "from the first two genome-name tokens "
            "and are not final taxonomy assignments."
        ),
        (
            "Official taxonomic names and genome QC "
            "will be retrieved later for shortlisted "
            "taxa."
        ),
        (
            "Computational Method records are absent "
            "from this source file."
        ),
        (
            "Legacy records with missing evidence "
            "are not included."
        ),
        "",
        (
            "STATUS: PRIMARY LABORATORY AMR "
            "PROFILE COMPLETE"
        ),
    ]

    DEFINITION_PATH.write_text(
        "\n".join(
            definition_lines
        ) + "\n",
        encoding="utf-8",
    )

    output_paths = [
        *output_frames.keys(),
        DEFINITION_PATH,
    ]

    with OUTPUT_CHECKSUM_PATH.open(
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

    print()
    print(
        "===== GLOBAL PROFILE SUMMARY ====="
    )

    print(
        overall_frame.to_string(
            index=False
        )
    )

    print()
    print(
        "===== TOP 25 TAXA BY "
        "QUANTITATIVE CANDIDATE ROWS ====="
    )

    display_columns = [
        "taxon_id",
        "provisional_dominant_binomial",
        "quantitative_candidate_rows",
        "quantitative_candidate_genomes",
        "quantitative_candidate_antibiotics",
        "uncensored_rows",
        "censored_rows",
        "candidate_censoring_fraction",
    ]

    print(
        taxon_frame[
            display_columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "===== TOP CANDIDATES UNDER "
        "g500_u200 ====="
    )

    strict_display = (
        threshold_frame.loc[
            threshold_frame[
                "threshold_name"
            ].eq("g500_u200")
        ]
        .head(25)
    )

    print(
        strict_display[
            [
                "taxon_id",
                "provisional_dominant_binomial",
                "quantitative_candidate_genomes",
                "quantitative_candidate_rows",
                "eligible_antibiotics",
                "eligible_candidate_rows",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Profile completion time:",
        datetime.now(
            timezone.utc
        ).isoformat(),
    )

    print(
        "Output checksum manifest:",
        OUTPUT_CHECKSUM_PATH,
    )

    print()
    print(
        "STATUS: PRIMARY LABORATORY "
        "AMR PROFILE COMPLETE"
    )


if __name__ == "__main__":
    main()
