#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_ROOT = Path(
    "data/raw/genome_metadata"
)

METADATA_GLOB = (
    "bvbrc_shortlist_genome_metadata_*.tsv"
)

MANIFEST_PATH = Path(
    "metadata/shortlist/"
    "shortlist_candidate_genome_manifest.tsv"
)

CONFLICT_PATH = Path(
    "metadata/shortlist/"
    "shortlist_candidate_genome_conflicts.tsv"
)

AUDIT_ROOT = Path(
    "metadata/audits"
)

RESULT_ROOT = Path(
    "results/tables"
)

NUMERIC_FIELDS = [
    "checkm_completeness",
    "checkm_contamination",
    "coarse_consistency",
    "fine_consistency",
    "contigs",
    "contig_n50",
    "contig_l50",
    "genome_length",
    "gc_content",
]

AVAILABILITY_FIELDS = [
    "species",
    "genus",
    "taxon_id",
    "taxon_lineage_ids",
    "taxon_lineage_names",
    "genome_status",
    "genome_quality",
    "genome_quality_flags",
    "checkm_completeness",
    "checkm_contamination",
    "coarse_consistency",
    "fine_consistency",
    "contigs",
    "contig_n50",
    "contig_l50",
    "genome_length",
    "gc_content",
    "assembly_accession",
    "genbank_accessions",
    "bioproject_accession",
    "biosample_accession",
    "sra_accession",
    "strain",
    "serovar",
    "mlst",
    "collection_date",
    "collection_year",
    "isolation_country",
    "geographic_location",
    "isolation_source",
    "host_name",
    "host_scientific_name",
    "host_common_name",
    "sequencing_platform",
    "sequencing_status",
]


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


def normalized_text(
    series: pd.Series,
) -> pd.Series:
    return (
        clean_text(series)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.casefold()
    )


def lineage_has_exact_species(
    raw_lineage: str,
    expected_species: str,
) -> bool:
    raw_text = str(
        raw_lineage
    ).strip()

    expected = str(
        expected_species
    ).strip().casefold()

    if not raw_text or not expected:
        return False

    try:
        parsed = json.loads(
            raw_text
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return False

    if not isinstance(
        parsed,
        list,
    ):
        return False

    normalized_names = {
        str(name)
        .strip()
        .casefold()
        for name in parsed
        if str(name).strip()
    }

    return expected in normalized_names


def value_count_table(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    work = frame[
        [
            "provisional_species",
            column,
        ]
    ].copy()

    work[column] = clean_text(
        work[column]
    ).replace(
        {
            "": "<blank>",
        }
    )

    result = (
        work.groupby(
            [
                "provisional_species",
                column,
            ],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "size": "genomes",
            }
        )
    )

    totals = (
        result.groupby(
            "provisional_species"
        )["genomes"]
        .transform("sum")
    )

    result["fraction"] = (
        result["genomes"]
        / totals
    )

    return result.sort_values(
        [
            "provisional_species",
            "genomes",
            column,
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_numeric_quantiles(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for species, group in frame.groupby(
        "provisional_species",
        sort=True,
    ):
        for field in NUMERIC_FIELDS:
            numeric_column = (
                f"{field}_numeric"
            )

            values = (
                group[numeric_column]
                .dropna()
                .astype(float)
            )

            row: dict[str, object] = {
                "provisional_species":
                    species,
                "metric":
                    field,
                "total_genomes":
                    len(group),
                "present_values":
                    len(values),
                "missing_values":
                    len(group) - len(values),
            }

            if values.empty:
                row.update(
                    {
                        "minimum": np.nan,
                        "q01": np.nan,
                        "q05": np.nan,
                        "q25": np.nan,
                        "median": np.nan,
                        "q75": np.nan,
                        "q95": np.nan,
                        "q99": np.nan,
                        "maximum": np.nan,
                        "mean": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "minimum":
                            values.min(),
                        "q01":
                            values.quantile(0.01),
                        "q05":
                            values.quantile(0.05),
                        "q25":
                            values.quantile(0.25),
                        "median":
                            values.median(),
                        "q75":
                            values.quantile(0.75),
                        "q95":
                            values.quantile(0.95),
                        "q99":
                            values.quantile(0.99),
                        "maximum":
                            values.max(),
                        "mean":
                            values.mean(),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_availability(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, object]
    ] = []

    for species, group in frame.groupby(
        "provisional_species",
        sort=True,
    ):
        total = len(group)

        for field in AVAILABILITY_FIELDS:
            present = int(
                clean_text(
                    group[field]
                ).ne("").sum()
            )

            rows.append(
                {
                    "provisional_species":
                        species,
                    "field":
                        field,
                    "total_genomes":
                        total,
                    "present":
                        present,
                    "missing":
                        total - present,
                    "present_fraction":
                        (
                            present / total
                            if total
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(rows)


def build_threshold_sensitivity(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    exact_or_lineage = (
        frame["taxonomy_exact_species"]
        | frame[
            "taxonomy_lineage_contains_species"
        ]
    )

    exact_lineage_or_genus = (
        exact_or_lineage
        | frame[
            "taxonomy_genus_only"
        ]
    )

    completeness = frame[
        "checkm_completeness_numeric"
    ]

    contamination = frame[
        "checkm_contamination_numeric"
    ]

    contigs = frame[
        "contigs_numeric"
    ]

    n50 = frame[
        "contig_n50_numeric"
    ]

    checkm_90_5 = (
        completeness.ge(90)
        & contamination.le(5)
    )

    checkm_95_5 = (
        completeness.ge(95)
        & contamination.le(5)
    )

    structural_90 = (
        frame["genome_quality_good"]
        & checkm_90_5
        & contigs.le(500)
        & n50.ge(20_000)
    )

    structural_95 = (
        frame["genome_quality_good"]
        & checkm_95_5
        & contigs.le(500)
        & n50.ge(20_000)
    )

    scenarios = {
        "all_requested":
            pd.Series(
                True,
                index=frame.index,
            ),
        "taxonomy_exact_or_lineage":
            exact_or_lineage,
        "taxonomy_exact_lineage_or_genus_only":
            exact_lineage_or_genus,
        "genome_quality_good":
            frame["genome_quality_good"],
        "checkm_completeness90_contamination5":
            checkm_90_5,
        "checkm_completeness95_contamination5":
            checkm_95_5,
        "structural_qc90":
            structural_90,
        "structural_qc95":
            structural_95,
        "structural_qc95_and_exact_or_lineage":
            (
                structural_95
                & exact_or_lineage
            ),
        "structural_qc95_and_taxonomy_plus_genus":
            (
                structural_95
                & exact_lineage_or_genus
            ),
        "structural_qc95_taxonomy_plus_genus_no_name_conflict":
            (
                structural_95
                & exact_lineage_or_genus
                & ~frame[
                    "source_name_conflict"
                ]
            ),
    }

    rows: list[
        dict[str, object]
    ] = []

    for scenario, mask in scenarios.items():
        mask = (
            mask
            .fillna(False)
            .astype(bool)
        )

        for species, group in frame.groupby(
            "provisional_species",
            sort=True,
        ):
            selected = int(
                mask.loc[
                    group.index
                ].sum()
            )

            total = len(group)

            rows.append(
                {
                    "scenario":
                        scenario,
                    "provisional_species":
                        species,
                    "total_genomes":
                        total,
                    "passing_genomes":
                        selected,
                    "excluded_genomes":
                        total - selected,
                    "passing_fraction":
                        (
                            selected / total
                            if total
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print(
        "===== AUDIT SHORTLIST TAXONOMY "
        "AND ASSEMBLY QC ====="
    )

    metadata_paths = sorted(
        METADATA_ROOT.glob(
            METADATA_GLOB
        )
    )

    if len(metadata_paths) != 1:
        raise RuntimeError(
            "Expected exactly one downloaded "
            "shortlist metadata table; found "
            f"{len(metadata_paths)}: "
            f"{metadata_paths}"
        )

    metadata_path = metadata_paths[0]

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    conflicts = pd.read_csv(
        CONFLICT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    if metadata[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Downloaded metadata contains "
            "duplicate genome IDs."
        )

    if manifest[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Shortlist manifest contains "
            "duplicate genome IDs."
        )

    metadata_ids = set(
        metadata["genome_id"]
    )

    manifest_ids = set(
        manifest["genome_id"]
    )

    if metadata_ids != manifest_ids:
        raise RuntimeError(
            "Metadata and manifest genome-ID "
            "sets are not identical."
        )

    merged = manifest.merge(
        metadata,
        on="genome_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_manifest",
            "_metadata",
        ),
    )

    conflict_ids = set(
        conflicts["genome_id"]
    )

    merged[
        "source_name_conflict"
    ] = merged[
        "genome_id"
    ].isin(
        conflict_ids
    )

    provisional_normalized = (
        normalized_text(
            merged[
                "provisional_species"
            ]
        )
    )

    metadata_species_normalized = (
        normalized_text(
            merged["species"]
        )
    )

    metadata_genus_normalized = (
        normalized_text(
            merged["genus"]
        )
    )

    lineage_normalized = (
        normalized_text(
            merged[
                "taxon_lineage_names"
            ]
        )
    )

    expected_genus = (
        provisional_normalized
        .str.split()
        .str[0]
        .fillna("")
    )

    exact_species = (
        metadata_species_normalized
        .eq(
            provisional_normalized
        )
    )

    lineage_exact_membership = pd.Series(
        [
            lineage_has_exact_species(
                raw_lineage,
                provisional,
            )
            for raw_lineage, provisional
            in zip(
                merged[
                    "taxon_lineage_names"
                ],
                provisional_normalized,
                strict=True,
            )
        ],
        index=merged.index,
        dtype=bool,
    )

    genus_match = (
        metadata_genus_normalized
        .eq(expected_genus)
    )

    species_blank = (
        metadata_species_normalized
        .eq("")
    )

    lineage_contains_species = (
        species_blank
        & lineage_exact_membership
    )

    explicit_species_conflict = (
        ~species_blank
        & ~exact_species
    )

    genus_only = (
        species_blank
        & ~lineage_contains_species
        & genus_match
    )

    merged[
        "taxonomy_exact_species"
    ] = exact_species

    merged[
        "taxonomy_lineage_contains_species"
    ] = lineage_contains_species

    merged[
        "taxonomy_genus_match"
    ] = genus_match

    merged[
        "taxonomy_species_blank"
    ] = species_blank

    merged[
        "taxonomy_genus_only"
    ] = genus_only

    merged[
        "taxonomy_support_category"
    ] = np.select(
        [
            exact_species,
            lineage_contains_species,
            (
                explicit_species_conflict
                & genus_match
            ),
            (
                explicit_species_conflict
                & ~genus_match
            ),
            genus_only,
            (
                species_blank
                & ~genus_match
            ),
        ],
        [
            "exact_species",
            "lineage_species_species_blank",
            "species_conflict_same_genus",
            "species_conflict_other_genus",
            "genus_only_species_blank",
            "species_blank_genus_mismatch",
        ],
        default=(
            "unclassified_taxonomy_state"
        ),
    )

    for field in NUMERIC_FIELDS:
        merged[
            f"{field}_numeric"
        ] = pd.to_numeric(
            merged[field],
            errors="coerce",
        )

    merged[
        "genome_quality_good"
    ] = (
        normalized_text(
            merged[
                "genome_quality"
            ]
        )
        .eq("good")
    )

    merged[
        "genome_status_wgs"
    ] = (
        normalized_text(
            merged[
                "genome_status"
            ]
        )
        .eq("wgs")
    )

    taxonomy_summary = (
        merged.groupby(
            [
                "provisional_species",
                "taxonomy_support_category",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "genomes",
            }
        )
    )

    taxonomy_totals = (
        taxonomy_summary.groupby(
            "provisional_species"
        )["genomes"]
        .transform("sum")
    )

    taxonomy_summary[
        "fraction"
    ] = (
        taxonomy_summary[
            "genomes"
        ]
        / taxonomy_totals
    )

    taxonomy_summary = (
        taxonomy_summary.sort_values(
            [
                "provisional_species",
                "genomes",
                "taxonomy_support_category",
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

    metadata_species_values = (
        value_count_table(
            merged,
            "species",
        )
    )

    taxon_id_values = (
        value_count_table(
            merged,
            "taxon_id",
        )
    )

    genome_quality_values = (
        value_count_table(
            merged,
            "genome_quality",
        )
    )

    genome_status_values = (
        value_count_table(
            merged,
            "genome_status",
        )
    )

    genome_quality_flag_values = (
        value_count_table(
            merged,
            "genome_quality_flags",
        )
    )

    numeric_quantiles = (
        build_numeric_quantiles(
            merged
        )
    )

    availability = (
        build_availability(
            merged
        )
    )

    threshold_sensitivity = (
        build_threshold_sensitivity(
            merged
        )
    )

    taxonomy_attention = merged.loc[
        ~merged[
            "taxonomy_support_category"
        ].isin(
            [
                "exact_species",
                "lineage_species_species_blank",
                "genus_only_species_blank",
            ]
        )
        |
        merged[
            "source_name_conflict"
        ]
    ].copy()

    AUDIT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        AUDIT_ROOT
        / "shortlist_genome_taxonomy_qc_audit.tsv"
    )

    taxonomy_attention_path = (
        AUDIT_ROOT
        / "shortlist_taxonomy_attention_genomes.tsv"
    )

    outputs = {
        audit_path:
            merged,
        taxonomy_attention_path:
            taxonomy_attention,
        (
            RESULT_ROOT
            / "shortlist_taxonomy_support_summary.tsv"
        ):
            taxonomy_summary,
        (
            RESULT_ROOT
            / "shortlist_metadata_species_values.tsv"
        ):
            metadata_species_values,
        (
            RESULT_ROOT
            / "shortlist_taxon_id_values.tsv"
        ):
            taxon_id_values,
        (
            RESULT_ROOT
            / "shortlist_genome_quality_values.tsv"
        ):
            genome_quality_values,
        (
            RESULT_ROOT
            / "shortlist_genome_status_values.tsv"
        ):
            genome_status_values,
        (
            RESULT_ROOT
            / "shortlist_genome_quality_flag_values.tsv"
        ):
            genome_quality_flag_values,
        (
            RESULT_ROOT
            / "shortlist_qc_numeric_quantiles.tsv"
        ):
            numeric_quantiles,
        (
            RESULT_ROOT
            / "shortlist_metadata_field_availability.tsv"
        ):
            availability,
        (
            RESULT_ROOT
            / "shortlist_qc_threshold_sensitivity.tsv"
        ):
            threshold_sensitivity,
    }

    for path, table in outputs.items():
        table.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )

    checksum_path = (
        AUDIT_ROOT
        / "script07_outputs_sha256.txt"
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
        "Metadata genomes:",
        f"{len(metadata):,}",
    )

    print(
        "Merged audit genomes:",
        f"{len(merged):,}",
    )

    print(
        "Taxonomy/name attention genomes:",
        f"{len(taxonomy_attention):,}",
    )

    print()
    print(
        "===== TAXONOMY SUPPORT ====="
    )

    print(
        taxonomy_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== QC THRESHOLD "
        "SENSITIVITY ====="
    )

    print(
        threshold_sensitivity.to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: SHORTLIST TAXONOMY "
        "AND QC AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
