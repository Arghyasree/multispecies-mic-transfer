#!/usr/bin/env python3
"""Validate the released 168,363-row quantitative MIC benchmark table."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_COLUMNS = [
    "final_transfer_observation_row",
    "species_code",
    "provisional_species",
    "observation_id",
    "genome_id",
    "normalized_antibiotic",
    "mic_target_log2_mg_per_l",
    "is_exact_observation",
    "is_censored_observation",
    "genome_feature_row",
    "drug_feature_row",
    "identity_feature_row",
    "morgan_feature_row",
    "rdkit_feature_row",
    "chemberta_mean_feature_row",
    "chemberta_first_feature_row",
    "genome_group_id",
    "duplicate_profile_group_id",
    "duplicate_profile_group_size",
    "random_pair_fold",
    "genome_disjoint_fold",
]

CURATION_COLUMNS = [
    "reconciliation_status",
    "constraint_origin",
    "duplicate_class",
    "reduced_constraint_type",
    "reduced_sign",
    "reduced_mic_value",
    "intersection_lower",
    "intersection_lower_closed",
    "intersection_upper",
    "intersection_upper_closed",
    "intersection_notation",
    "mic_target_point_mg_per_l",
    "mic_target_substitution_rule",
    "censoring_direction",
    "censoring_strictness",
    "point_target_version",
    "source_record_count",
    "source_record_ids",
    "source_genome_names",
    "source_taxon_ids",
    "source_antibiotic_labels",
    "normalized_unit",
    "source_measurements",
    "source_measurement_signs",
    "source_measurement_values",
    "source_normalized_signs",
    "source_mic_values",
    "source_methods",
    "source_method_versions",
    "source_platforms",
    "source_vendors",
    "source_testing_standards",
    "source_testing_standard_years",
    "source_pmids",
    "source_insertion_dates",
    "source_context_count",
    "source_contexts",
]

ENRICHED_COLUMNS = [
    "source_measurement_units",
    "source_resistant_phenotypes",
    "source_modification_dates",
    "source_evidence",
]

EXPECTED_COLUMNS = (
    INDEX_COLUMNS
    + CURATION_COLUMNS
    + ENRICHED_COLUMNS
)

EXPECTED_BY_SPECIES = {
    "ec": (68_881, 6_673, 19, 25_742, 43_139),
    "kp": (50_299, 5_602, 17, 13_582, 36_717),
    "se": (49_183, 9_119, 8, 20_644, 28_539),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def boolean(
    series: pd.Series,
    label: str,
) -> pd.Series:
    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "0": False,
                "1": True,
                "false": False,
                "true": True,
                "f": False,
                "t": True,
                "no": False,
                "yes": True,
            }
        )
    )

    require(
        not values.isna().any(),
        f"Invalid Boolean value in {label}.",
    )

    return values.astype(bool)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_public_benchmark(
    root: Path,
    observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    benchmark_path = (
        root
        / "data/benchmark/"
          "final_quantitative_mic_benchmark_v1.tsv.gz"
    )

    checksum_path = benchmark_path.with_name(
        benchmark_path.name + ".sha256"
    )

    schema_path = (
        root
        / "docs/benchmark_schema.md"
    )

    for path in (
        benchmark_path,
        checksum_path,
        schema_path,
    ):
        require(
            path.is_file(),
            f"Required public benchmark file missing: {path}",
        )

    checksum_parts = (
        checksum_path
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )

    require(
        len(checksum_parts) == 2,
        "Malformed benchmark checksum file.",
    )

    expected_digest, expected_name = checksum_parts

    require(
        expected_name == benchmark_path.name,
        "Benchmark checksum filename does not match the artifact.",
    )

    require(
        file_sha256(benchmark_path) == expected_digest,
        "Public benchmark SHA-256 verification failed.",
    )

    frame = read_tsv(benchmark_path)

    require(
        frame.columns.tolist() == EXPECTED_COLUMNS,
        "Public benchmark columns or column order changed.",
    )

    require(
        len(frame) == 168_363,
        f"Public benchmark has {len(frame):,} rows; "
        "expected 168,363.",
    )

    require(
        not frame["observation_id"].duplicated().any(),
        "Public benchmark observation IDs are not unique.",
    )

    require(
        not frame.duplicated(
            [
                "species_code",
                "genome_id",
                "normalized_antibiotic",
            ]
        ).any(),
        "Public benchmark contains repeated "
        "species-genome-antibiotic keys.",
    )

    if observations is None:
        index_path = (
            root
            / "metadata/final_transfer/nested_loso_v1/"
              "splits_v1/"
              "final_transfer_observation_feature_index_v1.tsv.gz"
        )

        require(
            index_path.is_file(),
            f"Observation index missing: {index_path}",
        )

        observations = read_tsv(index_path)

    missing_index_columns = sorted(
        set(INDEX_COLUMNS)
        - set(observations.columns)
    )

    require(
        not missing_index_columns,
        "Observation index lacks columns: "
        f"{missing_index_columns}",
    )

    require(
        len(observations) == len(frame),
        "Public benchmark and observation-index "
        "row counts differ.",
    )

    require(
        frame[INDEX_COLUMNS].equals(
            observations[INDEX_COLUMNS]
        ),
        "Public benchmark is not aligned exactly "
        "to the frozen observation index.",
    )

    point_target = pd.to_numeric(
        frame["mic_target_point_mg_per_l"],
        errors="raise",
    )

    log_target = pd.to_numeric(
        frame["mic_target_log2_mg_per_l"],
        errors="raise",
    )

    require(
        (point_target > 0).all(),
        "Non-positive point MIC target detected.",
    )

    require(
        np.isfinite(
            log_target.to_numpy(dtype=float)
        ).all(),
        "Non-finite log2 MIC target detected.",
    )

    require(
        set(frame["normalized_unit"]) == {"mg/L"},
        "The public normalized unit must be exactly mg/L.",
    )

    exact = boolean(
        frame["is_exact_observation"],
        "is_exact_observation",
    )

    censored = boolean(
        frame["is_censored_observation"],
        "is_censored_observation",
    )

    require(
        (exact ^ censored).all(),
        "Each observation must be exactly one "
        "of exact or censored.",
    )

    require(
        (
            int(exact.sum()),
            int(censored.sum()),
        )
        == (
            59_968,
            108_395,
        ),
        "Unexpected exact/censored totals.",
    )

    species = (
        frame["species_code"]
        .str.strip()
        .str.lower()
    )

    require(
        set(species) == set(EXPECTED_BY_SPECIES),
        "Unexpected public benchmark species codes.",
    )

    for code, expected in EXPECTED_BY_SPECIES.items():
        subset = frame.loc[species.eq(code)]

        observed = (
            len(subset),
            subset["genome_id"].nunique(),
            subset["normalized_antibiotic"].nunique(),
            int(exact.loc[subset.index].sum()),
            int(censored.loc[subset.index].sum()),
        )

        require(
            observed == expected,
            "Public benchmark summary mismatch "
            f"for {code}: {observed}",
        )

    source_counts = pd.to_numeric(
        frame["source_record_count"],
        errors="raise",
    ).astype(int)

    require(
        int(source_counts.sum()) == 179_385,
        "Unexpected number of represented source records.",
    )

    require(
        int(source_counts.gt(1).sum()) == 11_005,
        "Unexpected number of reconciled "
        "multi-record observations.",
    )

    require(
        int(source_counts.max()) == 3,
        "Unexpected maximum source-record count.",
    )

    aligned_columns = (
        ["source_record_ids"]
        + ENRICHED_COLUMNS
    )

    for column in aligned_columns:
        lengths = frame[column].map(
            lambda value: len(value.split("|"))
        )

        require(
            lengths.equals(source_counts),
            f"Source-record alignment failed for {column}.",
        )

    all_source_ids = [
        value
        for cell in frame["source_record_ids"]
        for value in cell.split("|")
    ]

    require(
        len(all_source_ids) == 179_385,
        "Unexpected source-record identifier count.",
    )

    require(
        len(set(all_source_ids)) == 179_385,
        "Source-record identifiers are not unique "
        "across observations.",
    )

    evidence = {
        value
        for cell in frame["source_evidence"]
        for value in cell.split("|")
        if value
    }

    require(
        evidence == {"Laboratory Method"},
        "Unexpected source evidence values: "
        f"{sorted(evidence)}",
    )

    private_markers = (
    "/" + "home" + "/",
    "arghya" + "sree",
    "ISI" + "_Research",
    )

    for marker in private_markers:
        containing = [
            column
            for column in frame.columns
            if frame[column]
            .str.contains(
                marker,
                regex=False,
                na=False,
            )
            .any()
        ]

        require(
            not containing,
            f"Private-path marker {marker!r} "
            f"found in columns: {containing}",
        )

    schema = schema_path.read_text(
        encoding="utf-8"
    )

    undocumented = [
        column
        for column in EXPECTED_COLUMNS
        if f"`{column}`" not in schema
    ]

    require(
        not undocumented,
        "Public benchmark columns missing from schema: "
        f"{undocumented}",
    )

    root_readme = (
        root / "README.md"
    ).read_text(encoding="utf-8")

    data_readme = (
        root / "data/README.md"
    ).read_text(encoding="utf-8")

    required_links = [
        "data/benchmark/"
        "final_quantitative_mic_benchmark_v1.tsv.gz",
        "docs/benchmark_schema.md",
    ]

    for link in required_links:
        require(
            link in root_readme,
            f"Root README lacks public benchmark link: {link}",
        )

    require(
        "benchmark/"
        "final_quantitative_mic_benchmark_v1.tsv.gz"
        in data_readme,
        "Data README lacks the public benchmark link.",
    )

    require(
        "docs/benchmark_schema.md" in data_readme,
        "Data README lacks the benchmark-schema link.",
    )

    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    frame = verify_public_benchmark(root)

    print("Public benchmark verification: PASS")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns)}")
    print(
        "Genomes: "
        f"{frame['genome_id'].nunique():,}"
    )
    print(
        "Antibiotics: "
        f"{frame['normalized_antibiotic'].nunique()}"
    )


if __name__ == "__main__":
    main()