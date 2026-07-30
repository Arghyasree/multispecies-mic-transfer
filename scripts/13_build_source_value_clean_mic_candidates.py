#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


AMR_PATH = Path(
    "data/raw/amr/"
    "bvbrc_primary_laboratory_amr_2026-07-22.tsv"
)

PASS_IDS_PATH = Path(
    "metadata/qc/"
    "shortlist_baseline_metadata_qc_pass_ids.txt"
)

QC_PATH = Path(
    "metadata/qc/"
    "shortlist_baseline_metadata_qc_manifest.tsv"
)

NORMALIZATION_PATH = Path(
    "results/tables/"
    "bvbrc_antibiotic_normalization_audit.tsv"
)

OUTPUT_ROOT = Path(
    "data/interim/mic"
)

AUDIT_ROOT = Path(
    "metadata/mic_audit"
)

RESULT_ROOT = Path(
    "results/tables"
)

NUMBER_PATTERN = re.compile(
    r"[+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)

DEFINITE_ZONE_PMIDS = {
    "38219757",
    "32422315",
    "37327220",
    "37549252",
}

USECOLS = [
    "id",
    "genome_id",
    "genome_name",
    "taxon_id",
    "antibiotic",
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


def clean(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def normalize_sign(
    series: pd.Series,
) -> pd.Series:
    return (
        clean(series)
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


def normalize_unit(
    series: pd.Series,
) -> pd.Series:
    return (
        clean(series)
        .str.casefold()
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
            r"\s+",
            "",
            regex=True,
        )
        .replace(
            {
                "mcg/ml": "ug/ml",
                "microgram/ml": "ug/ml",
                "micrograms/ml": "ug/ml",
                "milligram/liter": "mg/l",
                "milligrams/liter": "mg/l",
                "milligram/litre": "mg/l",
                "milligrams/litre": "mg/l",
            }
        )
    )


def normalize_antibiotic_text(
    series: pd.Series,
) -> pd.Series:
    return (
        clean(series)
        .str.replace(
            "Â",
            "",
            regex=False,
        )
        .str.replace(
            "â",
            "",
            regex=False,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
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


def main() -> None:
    print(
        "===== BUILD SOURCE/VALUE-CLEAN "
        "MIC CANDIDATES ====="
    )

    pass_ids = {
        line.strip()
        for line in PASS_IDS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    if len(pass_ids) != 28_908:
        raise RuntimeError(
            "Expected 28,908 baseline-QC genomes; "
            f"found {len(pass_ids):,}."
        )

    qc = pd.read_csv(
        QC_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "genome_id",
            "provisional_species",
        ],
        low_memory=False,
    )

    species_map = dict(
        zip(
            qc["genome_id"],
            qc["provisional_species"],
            strict=True,
        )
    )

    normalization = pd.read_csv(
        NORMALIZATION_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if normalization[
        "raw_antibiotic"
    ].duplicated().any():
        raise RuntimeError(
            "Antibiotic normalization table "
            "contains duplicate raw names."
        )

    normalization_map = dict(
        zip(
            normalization["raw_antibiotic"],
            normalization[
                "normalized_antibiotic"
            ],
            strict=True,
        )
    )

    frame = pd.read_csv(
        AMR_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=USECOLS,
        low_memory=False,
    )

    frame = frame.loc[
        frame["genome_id"].isin(
            pass_ids
        )
    ].copy()

    for column in USECOLS:
        frame[column] = clean(
            frame[column]
        )

    frame[
        "provisional_species"
    ] = frame[
        "genome_id"
    ].map(
        species_map
    )

    frame[
        "normalized_sign"
    ] = normalize_sign(
        frame["measurement_sign"]
    )

    frame[
        "normalized_unit"
    ] = normalize_unit(
        frame["measurement_unit"]
    )

    scalar = frame[
        "measurement_value"
    ].str.fullmatch(
        NUMBER_PATTERN,
        na=False,
    )

    frame[
        "mic_value"
    ] = pd.to_numeric(
        frame[
            "measurement_value"
        ].where(
            scalar
        ),
        errors="coerce",
    )

    paired_measurement = (
        frame["measurement"]
        .str.contains(
            "/",
            regex=False,
        )
        |
        frame["measurement_value"]
        .str.contains(
            "/",
            regex=False,
        )
    )

    candidate = (
        scalar
        & frame["mic_value"].gt(0)
        & frame["normalized_unit"].isin(
            {
                "mg/l",
                "ug/ml",
            }
        )
        & frame["normalized_sign"].isin(
            {
                "",
                "=",
                "<",
                "<=",
                ">",
                ">=",
            }
        )
        & ~paired_measurement
    )

    work = frame.loc[
        candidate
    ].copy()

    if len(work) != 388_471:
        raise RuntimeError(
            "Expected 388,471 positive candidates; "
            f"found {len(work):,}."
        )

    work[
        "normalized_antibiotic"
    ] = (
        work["antibiotic"]
        .map(
            normalization_map
        )
        .fillna(
            work["antibiotic"]
        )
    )

    work[
        "normalized_antibiotic"
    ] = normalize_antibiotic_text(
        work[
            "normalized_antibiotic"
        ]
    )

    method = (
        work[
            "laboratory_typing_method"
        ]
        .str.casefold()
    )

    platform = (
        work[
            "laboratory_typing_platform"
        ]
        .str.casefold()
    )

    disk_non_e_test = (
        method.eq(
            "disk diffusion"
        )
        &
        ~platform.str.contains(
            r"\be[\s-]*test\b",
            regex=True,
            na=False,
        )
    )

    work[
        "source_exclusion_reason"
    ] = ""

    zone_mask = (
        disk_non_e_test
        & work["pmid"].isin(
            DEFINITE_ZONE_PMIDS
        )
    )

    sentinel_mask = (
        disk_non_e_test
        & work["pmid"].eq(
            "32780112"
        )
        & work["normalized_antibiotic"]
        .str.casefold()
        .eq("cefepime")
        & work["mic_value"].eq(
            1_000_000
        )
    )

    value_46_mask = (
        disk_non_e_test
        & work["pmid"].eq(
            "32780112"
        )
        & work["normalized_antibiotic"]
        .str.casefold()
        .eq("chloramphenicol")
        & work["mic_value"].eq(46)
    )

    work.loc[
        zone_mask,
        "source_exclusion_reason",
    ] = "zone_diameter_source"

    work.loc[
        sentinel_mask,
        "source_exclusion_reason",
    ] = "invalid_sentinel_1000000"

    work.loc[
        value_46_mask,
        "source_exclusion_reason",
    ] = "unverified_source_value_46"

    work[
        "passes_source_value_qc"
    ] = work[
        "source_exclusion_reason"
    ].eq("")

    unexpected_disk = work.loc[
        disk_non_e_test
        & ~work["pmid"].isin(
            DEFINITE_ZONE_PMIDS
            | {
                "32780112",
            }
        )
    ]

    if not unexpected_disk.empty:
        raise RuntimeError(
            "Unexpected non-E-test disk source "
            "encountered."
        )

    exclusions = work.loc[
        ~work[
            "passes_source_value_qc"
        ]
    ].copy()

    passing = work.loc[
        work[
            "passes_source_value_qc"
        ]
    ].copy()

    reason_counts = (
        exclusions[
            "source_exclusion_reason"
        ]
        .value_counts()
        .to_dict()
    )

    expected_reasons = {
        "zone_diameter_source":
            32_296,
        "invalid_sentinel_1000000":
            9,
        "unverified_source_value_46":
            1,
    }

    if reason_counts != expected_reasons:
        raise RuntimeError(
            "Unexpected source exclusion counts: "
            f"{reason_counts}"
        )

    if len(exclusions) != 32_306:
        raise RuntimeError(
            "Expected 32,306 exclusions; "
            f"found {len(exclusions):,}."
        )

    if len(passing) != 356_165:
        raise RuntimeError(
            "Expected 356,165 passing records; "
            f"found {len(passing):,}."
        )

    if passing["id"].duplicated().any():
        raise RuntimeError(
            "Passing table contains duplicate "
            "record IDs."
        )

    qc_summary = pd.DataFrame(
        [
            {
                "stage":
                    "positive_candidates",
                "records":
                    len(work),
            },
            {
                "stage":
                    "zone_diameter_source_excluded",
                "records":
                    int(zone_mask.sum()),
            },
            {
                "stage":
                    "invalid_sentinel_1000000_excluded",
                "records":
                    int(sentinel_mask.sum()),
            },
            {
                "stage":
                    "unverified_source_value_46_excluded",
                "records":
                    int(value_46_mask.sum()),
            },
            {
                "stage":
                    "source_value_qc_passing",
                "records":
                    len(passing),
            },
        ]
    )

    species_summary = (
        passing.groupby(
            "provisional_species",
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
            normalized_antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
        )
        .sort_values(
            "records",
            ascending=False,
        )
    )

    antibiotic_summary = (
        passing.groupby(
            [
                "provisional_species",
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
                "provisional_species",
                "records",
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

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_path = (
        OUTPUT_ROOT
        / "postqc_positive_quantitative_candidates_source_audited.tsv"
    )

    passing_path = (
        OUTPUT_ROOT
        / "postqc_quantitative_candidates_source_clean.tsv"
    )

    exclusion_path = (
        AUDIT_ROOT
        / "postqc_source_value_exclusions.tsv"
    )

    qc_summary_path = (
        RESULT_ROOT
        / "postqc_source_value_qc_summary.tsv"
    )

    species_summary_path = (
        RESULT_ROOT
        / "postqc_source_clean_species_summary.tsv"
    )

    antibiotic_summary_path = (
        RESULT_ROOT
        / "postqc_source_clean_antibiotic_summary.tsv"
    )

    work.to_csv(
        all_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    passing.to_csv(
        passing_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    exclusions.to_csv(
        exclusion_path,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    qc_summary.to_csv(
        qc_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    species_summary.to_csv(
        species_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    antibiotic_summary.to_csv(
        antibiotic_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        all_path,
        passing_path,
        exclusion_path,
        qc_summary_path,
        species_summary_path,
        antibiotic_summary_path,
    ]

    checksum_path = (
        AUDIT_ROOT
        / "script13_outputs_sha256.txt"
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
        "Positive candidates:",
        f"{len(work):,}",
    )

    print(
        "Source/value exclusions:",
        f"{len(exclusions):,}",
    )

    print(
        "Source/value-QC passing:",
        f"{len(passing):,}",
    )

    print()
    print(
        "===== QC FLOW ====="
    )

    print(
        qc_summary.to_string(
            index=False
        )
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
        "STATUS: SOURCE/VALUE-CLEAN MIC "
        "CANDIDATES COMPLETE"
    )


if __name__ == "__main__":
    main()
