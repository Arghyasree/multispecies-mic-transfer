#!/usr/bin/env python3
"""Export the frozen 168,363-row public quantitative MIC benchmark."""

import csv
import hashlib
import os
from pathlib import Path

import pandas as pd


SOURCE = Path(os.environ["MIC_TRANSFER_SOURCE_PROJECT"]).resolve()
RELEASE = Path(os.environ["MIC_TRANSFER_RELEASE_PROJECT"]).resolve()

RICH_PATH = SOURCE / (
    "data/processed/modelling/"
    "multispecies_taxonomy_verified_finalized_panel_mic_cohort.tsv"
)
RAW_PATH = SOURCE / (
    "data/raw/amr/bvbrc_primary_laboratory_amr_2026-07-22.tsv"
)
INDEX_PATH = RELEASE / (
    "metadata/final_transfer/nested_loso_v1/splits_v1/"
    "final_transfer_observation_feature_index_v1.tsv.gz"
)
OUTPUT_PATH = RELEASE / (
    "data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz"
)

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

RICH_COLUMNS = [
    "observation_id",
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
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
    "mic_target_log2_mg_per_l",
    "mic_target_substitution_rule",
    "censoring_direction",
    "censoring_strictness",
    "is_exact_observation",
    "is_censored_observation",
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

RAW_COLUMNS = [
    "id",
    "measurement_unit",
    "resistant_phenotype",
    "date_modified",
    "evidence",
]

EXPECTED = {
    "ec": (68_881, 6_673, 19, 25_742, 43_139),
    "kp": (50_299, 5_602, 17, 13_582, 36_717),
    "se": (49_183, 9_119, 8, 20_644, 28_539),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_columns(frame, columns, path):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"Missing columns in {path}: {missing}")


def boolean(series, name):
    values = series.astype(str).str.strip().str.lower().map(
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
    require(not values.isna().any(), f"Invalid Boolean value in {name}.")
    return values.astype(bool)


for path in (RICH_PATH, RAW_PATH, INDEX_PATH):
    require(path.is_file(), f"Required input is missing: {path}")

print(f"Reading final index: {INDEX_PATH}")
index = pd.read_csv(
    INDEX_PATH,
    sep="\t",
    dtype=str,
    na_filter=False,
    compression="gzip",
)
require_columns(index, INDEX_COLUMNS, INDEX_PATH)
index = index[INDEX_COLUMNS].copy()
require(len(index) == 168_363, f"Unexpected index rows: {len(index):,}")
require(
    not index["observation_id"].duplicated().any(),
    "Duplicate observation IDs in final index.",
)

print(f"Reading rich cohort: {RICH_PATH}")
rich = pd.read_csv(
    RICH_PATH,
    sep="\t",
    dtype=str,
    na_filter=False,
    usecols=lambda name: name in set(RICH_COLUMNS),
    low_memory=False,
)
require_columns(rich, RICH_COLUMNS, RICH_PATH)
rich = rich[RICH_COLUMNS].copy()
require(
    not rich["observation_id"].duplicated().any(),
    "Duplicate observation IDs in rich cohort.",
)

final_ids = index["observation_id"].tolist()
rich_ids = set(rich["observation_id"])
missing = [value for value in final_ids if value not in rich_ids]
require(
    not missing,
    f"Final observation IDs missing from rich cohort: {missing[:5]}",
)

rich = (
    rich.set_index("observation_id", drop=False)
    .loc[final_ids]
    .reset_index(drop=True)
)

normalized_units = set(
    rich["normalized_unit"]
    .str.strip()
    .str.lower()
    .unique()
)

require(
    normalized_units == {"mg/l"},
    f"Unexpected normalized MIC units: {sorted(normalized_units)}",
)

# Use the standard public notation employed in the manuscript.
rich["normalized_unit"] = "mg/L"

for column in (
    "observation_id",
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
):
    require(
        index[column].equals(rich[column]),
        f"Index/rich mismatch in {column}.",
    )

target_difference = (
    pd.to_numeric(
        index["mic_target_log2_mg_per_l"],
        errors="raise",
    )
    - pd.to_numeric(
        rich["mic_target_log2_mg_per_l"],
        errors="raise",
    )
).abs().max()

require(
    target_difference <= 1e-12,
    "Index/rich MIC targets do not agree.",
)

for column in (
    "is_exact_observation",
    "is_censored_observation",
):
    require(
        boolean(index[column], f"index.{column}").equals(
            boolean(rich[column], f"rich.{column}")
        ),
        f"Index/rich mismatch in {column}.",
    )

source_ids_by_observation = []
wanted_source_ids = set()

for row_number, (count, value) in enumerate(
    zip(
        rich["source_record_count"],
        rich["source_record_ids"],
    ),
    start=1,
):
    source_ids = [
        part.strip()
        for part in value.split("|")
        if part.strip()
    ]

    require(
        len(source_ids) == int(float(count)),
        f"Source-record count mismatch at selected row {row_number}.",
    )

    source_ids_by_observation.append(source_ids)
    wanted_source_ids.update(source_ids)

print(
    f"Recovering {len(wanted_source_ids):,} "
    "source-record annotations."
)

raw_lookup = {}

for chunk in pd.read_csv(
    RAW_PATH,
    sep="\t",
    dtype=str,
    na_filter=False,
    usecols=RAW_COLUMNS,
    chunksize=250_000,
    low_memory=False,
):
    selected = chunk.loc[
        chunk["id"].isin(wanted_source_ids),
        RAW_COLUMNS,
    ]

    for values in selected.itertuples(index=False, name=None):
        record_id = values[0]
        annotations = tuple(values[1:])

        require(
            record_id not in raw_lookup
            or raw_lookup[record_id] == annotations,
            f"Conflicting raw records for BV-BRC ID {record_id}.",
        )

        raw_lookup[record_id] = annotations

missing = sorted(wanted_source_ids - set(raw_lookup))
require(
    not missing,
    f"Source IDs missing from raw snapshot: {missing[:5]}",
)

enriched_columns = {
    "source_measurement_units": 0,
    "source_resistant_phenotypes": 1,
    "source_modification_dates": 2,
    "source_evidence": 3,
}

for column, position in enriched_columns.items():
    rich[column] = [
        "|".join(
            raw_lookup[record_id][position]
            for record_id in source_ids
        )
        for source_ids in source_ids_by_observation
    ]

shared = {
    "observation_id",
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
    "mic_target_log2_mg_per_l",
    "is_exact_observation",
    "is_censored_observation",
}

output = index.copy()

for column in RICH_COLUMNS:
    if column not in shared:
        output[column] = rich[column]

for column in enriched_columns:
    output[column] = rich[column]

require(
    not output["observation_id"].duplicated().any(),
    "Duplicate observation IDs in output.",
)

require(
    not output.duplicated(
        [
            "species_code",
            "genome_id",
            "normalized_antibiotic",
        ]
    ).any(),
    "Duplicate species-genome-antibiotic keys in output.",
)

require(
    output["genome_id"].nunique() == 21_394,
    "Unexpected total genome count.",
)

require(
    output["normalized_antibiotic"].nunique() == 19,
    "Unexpected total antibiotic count.",
)

exact = boolean(
    output["is_exact_observation"],
    "is_exact_observation",
)
censored = boolean(
    output["is_censored_observation"],
    "is_censored_observation",
)

require(
    (exact ^ censored).all(),
    "Each observation must be exactly one of exact or censored.",
)

require(
    (int(exact.sum()), int(censored.sum()))
    == (59_968, 108_395),
    "Unexpected exact/censored totals.",
)

species = output["species_code"].str.strip().str.lower()
require(
    set(species) == set(EXPECTED),
    "Unexpected species codes.",
)

for code, expected in EXPECTED.items():
    subset = output.loc[species == code]

    observed = (
        len(subset),
        subset["genome_id"].nunique(),
        subset["normalized_antibiotic"].nunique(),
        int(exact.loc[subset.index].sum()),
        int(censored.loc[subset.index].sum()),
    )

    require(
        observed == expected,
        f"Summary mismatch for {code}: "
        f"{observed} != {expected}",
    )

essential = [
    "observation_id",
    "species_code",
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
    "mic_target_point_mg_per_l",
    "mic_target_log2_mg_per_l",
    "normalized_unit",
    "source_record_count",
    "source_record_ids",
]

empty = {
    column: int(
        output[column].str.strip().eq("").sum()
    )
    for column in essential
}

require(
    not any(empty.values()),
    f"Empty essential fields: {empty}",
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
temporary = OUTPUT_PATH.with_name(
    OUTPUT_PATH.name + ".tmp"
)

if temporary.exists():
    temporary.unlink()

print(f"Writing benchmark: {OUTPUT_PATH}")

output.to_csv(
    temporary,
    sep="\t",
    index=False,
    encoding="utf-8",
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL,
    compression={
        "method": "gzip",
        "compresslevel": 9,
        "mtime": 0,
    },
)

temporary.replace(OUTPUT_PATH)

hasher = hashlib.sha256()

with OUTPUT_PATH.open("rb") as handle:
    for block in iter(
        lambda: handle.read(1024 * 1024),
        b"",
    ):
        hasher.update(block)

digest = hasher.hexdigest()

checksum_path = OUTPUT_PATH.with_name(
    OUTPUT_PATH.name + ".sha256"
)
checksum_path.write_text(
    f"{digest}  {OUTPUT_PATH.name}\n",
    encoding="utf-8",
)

print("Benchmark validation: PASS")
print(f"Rows: {len(output):,}")
print(f"Columns: {len(output.columns)}")
print(f"Genomes: {output['genome_id'].nunique():,}")
print(
    "Antibiotics: "
    f"{output['normalized_antibiotic'].nunique()}"
)
print(
    "Output size: "
    f"{OUTPUT_PATH.stat().st_size / 2**20:.2f} MiB"
)
print(f"SHA256: {digest}")