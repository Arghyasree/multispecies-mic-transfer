#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


FULL_COHORT_PATH = Path(
    "data/processed/modelling/"
    "multispecies_single_structure_eligible_mic_cohort.tsv"
)

MAIN_PANEL_PATH = Path(
    "data/processed/modelling/"
    "multispecies_finalized_panel_mic_cohort.tsv"
)

QC_MANIFEST_PATH = Path(
    "metadata/qc/"
    "shortlist_baseline_metadata_qc_manifest.tsv"
)

OUTPUT_ROOT = Path(
    "metadata/genomes"
)

TABLE_ROOT = Path(
    "results/tables"
)

FASTA_ROOT = Path(
    "genomes/raw/bvbrc_fasta"
)

EXPECTED_FULL_ROWS = 177_850
EXPECTED_MAIN_ROWS = 176_571
EXPECTED_MODELLING_GENOMES = 23_632

EXPECTED_FULL_EXACT = 65_258
EXPECTED_MAIN_EXACT = 64_558

EXPECTED_QC_ROWS = 29_567
EXPECTED_QC_PASS = 28_908
EXPECTED_QC_FAIL = 659
EXPECTED_SPECIES = 5

EXPECTED_ERTAPENEM_ROWS = 1_279
EXPECTED_ERTAPENEM_EXACT = 700


def read_tsv(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    for column in frame.columns:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return frame


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns from {label}: "
            + "|".join(missing)
        )


def parse_bool(
    series: pd.Series,
    column: str,
) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    parsed = (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .map(mapping)
    )

    if parsed.isna().any():
        bad = sorted(
            set(
                series.loc[
                    parsed.isna()
                ]
            )
        )

        raise RuntimeError(
            f"Cannot parse Boolean column "
            f"{column}: {bad}"
        )

    return parsed.astype(bool)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
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
        "===== SCRIPT 46 BUILD MODELLING "
        "GENOME ACQUISITION MANIFEST ====="
    )

    for path in [
        FULL_COHORT_PATH,
        MAIN_PANEL_PATH,
        QC_MANIFEST_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    full = read_tsv(
        FULL_COHORT_PATH
    )

    main_panel = read_tsv(
        MAIN_PANEL_PATH
    )

    qc = read_tsv(
        QC_MANIFEST_PATH
    )

    require_columns(
        full,
        [
            "genome_id",
            "provisional_species",
            "normalized_antibiotic",
            "reduced_sign",
            "observation_id",
            "pair_id",
        ],
        "full modelling cohort",
    )

    require_columns(
        main_panel,
        [
            "genome_id",
            "provisional_species",
            "normalized_antibiotic",
            "reduced_sign",
            "observation_id",
            "pair_id",
        ],
        "main-panel cohort",
    )

    require_columns(
        qc,
        [
            "genome_id",
            "provisional_species",
            "representative_genome_name",
            "genome_name",
            "species",
            "genus",
            "taxon_id",
            "taxon_lineage_names",
            "genome_status",
            "genome_quality",
            "checkm_completeness_numeric",
            "checkm_contamination_numeric",
            "contigs_numeric",
            "contig_n50_numeric",
            "genome_length_numeric",
            "gc_content_numeric",
            "assembly_accession",
            "genbank_accessions",
            "bioproject_accession",
            "biosample_accession",
            "sra_accession",
            "strain",
            "serovar",
            "mlst",
            "taxonomy_support_category",
            "passes_taxonomy_rule",
            "passes_source_identity_rule",
            "passes_genome_quality_rule",
            "passes_completeness_rule",
            "passes_contamination_rule",
            "passes_contig_count_rule",
            "passes_n50_rule",
            "passes_baseline_metadata_qc",
            "baseline_failure_reasons",
            "robust_genome_length_outlier",
            "robust_gc_content_outlier",
            "robust_length_or_gc_outlier",
        ],
        "baseline genome-QC manifest",
    )

    if len(full) != EXPECTED_FULL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_ROWS:,} "
            f"full-cohort rows; found "
            f"{len(full):,}."
        )

    if len(main_panel) != EXPECTED_MAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MAIN_ROWS:,} "
            f"main-panel rows; found "
            f"{len(main_panel):,}."
        )

    if len(qc) != EXPECTED_QC_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_QC_ROWS:,} "
            f"QC rows; found {len(qc):,}."
        )

    if qc["genome_id"].eq("").any():
        raise RuntimeError(
            "Blank genome ID in QC manifest."
        )

    if qc["genome_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate genome ID in QC manifest."
        )

    qc_pass = parse_bool(
        qc[
            "passes_baseline_metadata_qc"
        ],
        "passes_baseline_metadata_qc",
    )

    if int(qc_pass.sum()) != EXPECTED_QC_PASS:
        raise RuntimeError(
            "Expected 28,908 baseline-QC "
            "passing genomes."
        )

    if int((~qc_pass).sum()) != EXPECTED_QC_FAIL:
        raise RuntimeError(
            "Expected 659 baseline-QC "
            "failing genomes."
        )

    full_genomes = set(
        full["genome_id"]
    )

    main_genomes = set(
        main_panel["genome_id"]
    )

    if len(full_genomes) != (
        EXPECTED_MODELLING_GENOMES
    ):
        raise RuntimeError(
            "Unexpected full-cohort genome count."
        )

    if len(main_genomes) != (
        EXPECTED_MODELLING_GENOMES
    ):
        raise RuntimeError(
            "Unexpected main-panel genome count."
        )

    if full_genomes != main_genomes:
        raise RuntimeError(
            "Full and main-panel genome sets "
            "are not identical."
        )

    if full[
        "provisional_species"
    ].nunique() != EXPECTED_SPECIES:
        raise RuntimeError(
            "Expected five modelling species."
        )

    species_per_genome = (
        full.groupby(
            "genome_id",
            sort=False,
        )[
            "provisional_species"
        ]
        .nunique()
    )

    if species_per_genome.gt(1).any():
        bad = list(
            species_per_genome.loc[
                species_per_genome.gt(1)
            ].index[:20]
        )

        raise RuntimeError(
            "A modelling genome belongs to "
            "multiple provisional species: "
            + "|".join(bad)
        )

    if int(
        full[
            "reduced_sign"
        ].eq("=").sum()
    ) != EXPECTED_FULL_EXACT:
        raise RuntimeError(
            "Unexpected full-cohort exact count."
        )

    if int(
        main_panel[
            "reduced_sign"
        ].eq("=").sum()
    ) != EXPECTED_MAIN_EXACT:
        raise RuntimeError(
            "Unexpected main-panel exact count."
        )

    ertapenem = full.loc[
        full[
            "normalized_antibiotic"
        ].eq("ertapenem")
    ]

    if len(ertapenem) != EXPECTED_ERTAPENEM_ROWS:
        raise RuntimeError(
            "Expected 1,279 ertapenem rows."
        )

    if int(
        ertapenem[
            "reduced_sign"
        ].eq("=").sum()
    ) != EXPECTED_ERTAPENEM_EXACT:
        raise RuntimeError(
            "Expected 700 exact ertapenem rows."
        )

    full_stats_source = full.assign(
        _exact=full[
            "reduced_sign"
        ].eq("="),
        _censored=full[
            "reduced_sign"
        ].ne("="),
        _left=full[
            "reduced_sign"
        ].isin(
            ["<", "<="]
        ),
        _right=full[
            "reduced_sign"
        ].isin(
            [">", ">="]
        ),
    )

    genome_stats = (
        full_stats_source.groupby(
            "genome_id",
            sort=True,
        )
        .agg(
            provisional_species=(
                "provisional_species",
                "first",
            ),
            full_cohort_observations=(
                "observation_id",
                "size",
            ),
            full_cohort_antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
            full_cohort_exact_observations=(
                "_exact",
                "sum",
            ),
            full_cohort_censored_observations=(
                "_censored",
                "sum",
            ),
            full_cohort_left_censored_observations=(
                "_left",
                "sum",
            ),
            full_cohort_right_censored_observations=(
                "_right",
                "sum",
            ),
        )
        .reset_index()
    )

    main_stats_source = main_panel.assign(
        _exact=main_panel[
            "reduced_sign"
        ].eq("="),
        _censored=main_panel[
            "reduced_sign"
        ].ne("="),
    )

    main_stats = (
        main_stats_source.groupby(
            "genome_id",
            sort=True,
        )
        .agg(
            main_panel_observations=(
                "observation_id",
                "size",
            ),
            main_panel_antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
            main_panel_exact_observations=(
                "_exact",
                "sum",
            ),
            main_panel_censored_observations=(
                "_censored",
                "sum",
            ),
        )
        .reset_index()
    )

    ertapenem_stats = (
        ertapenem.assign(
            _exact=ertapenem[
                "reduced_sign"
            ].eq("=")
        )
        .groupby(
            "genome_id",
            sort=True,
        )
        .agg(
            ertapenem_observations=(
                "observation_id",
                "size",
            ),
            ertapenem_exact_observations=(
                "_exact",
                "sum",
            ),
        )
        .reset_index()
    )

    qc_for_merge = qc.rename(
        columns={
            "provisional_species":
                "qc_provisional_species",
        }
    ).copy()

    manifest = genome_stats.merge(
        main_stats,
        on="genome_id",
        how="left",
        validate="one_to_one",
    )

    manifest = manifest.merge(
        ertapenem_stats,
        on="genome_id",
        how="left",
        validate="one_to_one",
    )

    manifest[
        [
            "ertapenem_observations",
            "ertapenem_exact_observations",
        ]
    ] = manifest[
        [
            "ertapenem_observations",
            "ertapenem_exact_observations",
        ]
    ].fillna("0")

    manifest = manifest.merge(
        qc_for_merge,
        on="genome_id",
        how="left",
        validate="one_to_one",
        indicator="_metadata_merge_status",
    )

    missing_metadata = manifest[
        "_metadata_merge_status"
    ].ne("both")

    if missing_metadata.any():
        bad = list(
            manifest.loc[
                missing_metadata,
                "genome_id",
            ][:20]
        )

        raise RuntimeError(
            "Modelling genomes missing from "
            "the frozen QC manifest: "
            + "|".join(bad)
        )

    merged_qc_pass = parse_bool(
        manifest[
            "passes_baseline_metadata_qc"
        ],
        "merged passes_baseline_metadata_qc",
    )

    if not merged_qc_pass.all():
        bad = list(
            manifest.loc[
                ~merged_qc_pass,
                "genome_id",
            ][:20]
        )

        raise RuntimeError(
            "Modelling genomes failing frozen "
            "baseline QC: "
            + "|".join(bad)
        )

    species_mismatch = manifest[
        "provisional_species"
    ].ne(
        manifest[
            "qc_provisional_species"
        ]
    )

    if species_mismatch.any():
        bad = list(
            manifest.loc[
                species_mismatch,
                "genome_id",
            ][:20]
        )

        raise RuntimeError(
            "Species assignment differs between "
            "modelling and QC manifests: "
            + "|".join(bad)
        )

    robust_outlier = parse_bool(
        manifest[
            "robust_length_or_gc_outlier"
        ],
        "robust_length_or_gc_outlier",
    )

    manifest[
        "baseline_qc_pass_confirmed"
    ] = True

    manifest[
        "robust_outlier_policy"
    ] = robust_outlier.map(
        {
            True:
                "retained_audit_only",
            False:
                "not_flagged",
        }
    )

    manifest[
        "download_source"
    ] = "BV-BRC"

    manifest[
        "download_identifier_type"
    ] = "genome_id"

    manifest[
        "download_identifier"
    ] = manifest[
        "genome_id"
    ]

    manifest[
        "download_endpoint_probe_status"
    ] = "not_yet_probed"

    manifest[
        "download_status"
    ] = "pending"

    manifest[
        "download_attempts"
    ] = 0

    manifest[
        "local_fasta_filename"
    ] = (
        manifest[
            "genome_id"
        ]
        + ".fna"
    )

    manifest[
        "local_fasta_path"
    ] = (
        FASTA_ROOT.as_posix()
        + "/"
        + manifest[
            "local_fasta_filename"
        ]
    )

    manifest[
        "fasta_sha256"
    ] = ""

    manifest[
        "fasta_size_bytes"
    ] = ""

    manifest[
        "fasta_sequence_count"
    ] = ""

    manifest[
        "sequence_download_validation_status"
    ] = "pending"

    manifest[
        "sequence_taxonomy_verification_status"
    ] = "pending"

    manifest[
        "sequence_qc_status"
    ] = "pending"

    manifest = manifest.drop(
        columns=[
            "_metadata_merge_status",
        ]
    )

    manifest = manifest.sort_values(
        [
            "provisional_species",
            "genome_id",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    manifest.insert(
        0,
        "acquisition_order",
        range(
            1,
            len(manifest) + 1,
        ),
    )

    front_columns = [
        "acquisition_order",
        "genome_id",
        "provisional_species",
        "qc_provisional_species",
        "representative_genome_name",
        "genome_name",
        "species",
        "genus",
        "taxon_id",
        "taxonomy_support_category",
        "assembly_accession",
        "genbank_accessions",
        "bioproject_accession",
        "biosample_accession",
        "sra_accession",
        "strain",
        "serovar",
        "mlst",
        "full_cohort_observations",
        "full_cohort_antibiotics",
        "full_cohort_exact_observations",
        "full_cohort_censored_observations",
        "full_cohort_left_censored_observations",
        "full_cohort_right_censored_observations",
        "main_panel_observations",
        "main_panel_antibiotics",
        "main_panel_exact_observations",
        "main_panel_censored_observations",
        "ertapenem_observations",
        "ertapenem_exact_observations",
        "passes_baseline_metadata_qc",
        "baseline_qc_pass_confirmed",
        "robust_length_or_gc_outlier",
        "robust_outlier_policy",
        "download_source",
        "download_identifier_type",
        "download_identifier",
        "download_endpoint_probe_status",
        "download_status",
        "download_attempts",
        "local_fasta_filename",
        "local_fasta_path",
        "fasta_sha256",
        "fasta_size_bytes",
        "fasta_sequence_count",
        "sequence_download_validation_status",
        "sequence_taxonomy_verification_status",
        "sequence_qc_status",
    ]

    remaining_columns = [
        column
        for column in manifest.columns
        if column not in front_columns
    ]

    manifest = manifest[
        front_columns
        + remaining_columns
    ]

    if len(manifest) != (
        EXPECTED_MODELLING_GENOMES
    ):
        raise RuntimeError(
            "Unexpected acquisition-manifest "
            "row count."
        )

    if manifest[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate genome ID in acquisition "
            "manifest."
        )

    if not manifest[
        "download_status"
    ].eq("pending").all():
        raise RuntimeError(
            "A download was marked complete "
            "prematurely."
        )

    if not manifest[
        "sequence_taxonomy_verification_status"
    ].eq("pending").all():
        raise RuntimeError(
            "A sequence taxonomy decision was "
            "made prematurely."
        )

    species_antibiotic_counts = (
        full.groupby(
            "provisional_species",
            sort=True,
        )[
            "normalized_antibiotic"
        ]
        .nunique()
        .rename(
            "full_cohort_antibiotics"
        )
    )

    species_summary = (
        manifest.assign(
            _assembly_present=manifest[
                "assembly_accession"
            ].ne(""),
            _robust_outlier=robust_outlier.values,
        )
        .groupby(
            "provisional_species",
            sort=True,
        )
        .agg(
            genomes=(
                "genome_id",
                "size",
            ),
            full_cohort_observations=(
                "full_cohort_observations",
                lambda values:
                    int(
                        pd.to_numeric(
                            values,
                            errors="raise",
                        ).sum()
                    ),
            ),
            main_panel_observations=(
                "main_panel_observations",
                lambda values:
                    int(
                        pd.to_numeric(
                            values,
                            errors="raise",
                        ).sum()
                    ),
            ),
            full_cohort_exact_observations=(
                "full_cohort_exact_observations",
                lambda values:
                    int(
                        pd.to_numeric(
                            values,
                            errors="raise",
                        ).sum()
                    ),
            ),
            full_cohort_censored_observations=(
                "full_cohort_censored_observations",
                lambda values:
                    int(
                        pd.to_numeric(
                            values,
                            errors="raise",
                        ).sum()
                    ),
            ),
            genomes_with_assembly_accession=(
                "_assembly_present",
                "sum",
            ),
            robust_length_or_gc_outliers=(
                "_robust_outlier",
                "sum",
            ),
            baseline_qc_pass=(
                "baseline_qc_pass_confirmed",
                "sum",
            ),
            pending_fasta_downloads=(
                "download_status",
                lambda values:
                    int(
                        values.eq(
                            "pending"
                        ).sum()
                    ),
            ),
        )
        .join(
            species_antibiotic_counts
        )
        .reset_index()
    )

    if len(species_summary) != EXPECTED_SPECIES:
        raise RuntimeError(
            "Expected five species-summary rows."
        )

    exceptions = pd.DataFrame(
        columns=[
            "genome_id",
            "provisional_species",
            "exception_type",
            "exception_details",
        ]
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FASTA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        OUTPUT_ROOT
        / "modelling_genome_acquisition_manifest.tsv"
    )

    ids_path = (
        OUTPUT_ROOT
        / "modelling_genome_ids_for_download.txt"
    )

    summary_path = (
        TABLE_ROOT
        / "modelling_genome_acquisition_species_summary.tsv"
    )

    exceptions_path = (
        OUTPUT_ROOT
        / "modelling_genome_acquisition_exceptions.tsv"
    )

    manifest.to_csv(
        manifest_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    ids_path.write_text(
        "\n".join(
            manifest[
                "genome_id"
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    species_summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    exceptions.to_csv(
        exceptions_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        manifest_path,
        ids_path,
        summary_path,
        exceptions_path,
    ]

    checksum_path = (
        OUTPUT_ROOT
        / "script46_outputs_sha256.txt"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            output_paths,
            key=lambda item:
                item.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.as_posix()}\n"
            )

    print(
        "Frozen QC metadata rows:",
        f"{len(qc):,}",
    )

    print(
        "Frozen baseline-QC pass:",
        f"{int(qc_pass.sum()):,}",
    )

    print(
        "Frozen baseline-QC fail:",
        f"{int((~qc_pass).sum()):,}",
    )

    print()
    print(
        "Modelling genomes:",
        f"{len(manifest):,}",
    )

    print(
        "Modelling genomes found in QC manifest:",
        f"{len(manifest):,}",
    )

    print(
        "Modelling genomes missing metadata:",
        "0",
    )

    print(
        "Modelling genomes failing baseline QC:",
        "0",
    )

    print(
        "Species-assignment mismatches:",
        "0",
    )

    print(
        "Acquisition exceptions:",
        len(exceptions),
    )

    print()
    print(
        "FASTA downloads pending:",
        f"{manifest['download_status'].eq('pending').sum():,}",
    )

    print(
        "FASTA downloads completed:",
        "0",
    )

    print(
        "Endpoint probes completed:",
        "0",
    )

    print(
        "Sequence taxonomy verifications "
        "completed:",
        "0",
    )

    print()
    print(
        "===== ACQUISITION SPECIES SUMMARY ====="
    )

    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Frozen Script 45 cohort modified:",
        "NO",
    )

    print(
        "Frozen baseline QC manifest modified:",
        "NO",
    )

    print(
        "Genome FASTA files downloaded:",
        "NO",
    )

    print(
        "Baseline QC criteria changed:",
        "NO",
    )

    print(
        "Robust outliers excluded:",
        "NO",
    )

    print()
    print(
        "STATUS: MODELLING GENOME "
        "ACQUISITION MANIFEST BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
