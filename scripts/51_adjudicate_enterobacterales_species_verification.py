#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


KLEBORATE_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_kleborate_species_manifest.tsv"
)

EXCEPTION_AUDIT_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_kleborate_exception_metadata_mic_audit.tsv"
)

ADJUDICATION_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_taxonomy_adjudication_manifest.tsv"
)

EXCLUSION_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_target_species_exclusion_manifest.tsv"
)

RETAINED_IDS_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_retained_target_species_genome_ids.txt"
)

SUMMARY_PATH = Path(
    "results/tables/taxonomy/"
    "enterobacterales_taxonomy_adjudication_summary.tsv"
)

OUTPUT_SHA_PATH = Path(
    "metadata/taxonomy/"
    "script51_outputs_sha256.txt"
)

EXPECTED_TOTAL = 21_478
EXPECTED_RETAINED = 21_394
EXPECTED_EXCLUDED = 84

EXPECTED_SPECIES_TOTALS = {
    "Escherichia coli": 6_687,
    "Klebsiella pneumoniae": 5_672,
    "Salmonella enterica": 9_119,
}

EXPECTED_RETAINED_TOTALS = {
    "Escherichia coli": 6_673,
    "Klebsiella pneumoniae": 5_602,
    "Salmonella enterica": 9_119,
}

EXPECTED_EXCLUDED_TOTALS = {
    "Escherichia coli": 14,
    "Klebsiella pneumoniae": 70,
    "Salmonella enterica": 0,
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    print(
        "===== SCRIPT 51 ENTEROBACTERALES "
        "TAXONOMY ADJUDICATION ====="
    )

    for path in [
        KLEBORATE_PATH,
        EXCEPTION_AUDIT_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    calls = read_tsv(
        KLEBORATE_PATH
    )

    exception_audit = read_tsv(
        EXCEPTION_AUDIT_PATH
    )

    if len(calls) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL:,} Kleborate rows; "
            f"found {len(calls):,}."
        )

    if calls["genome_id"].nunique() != EXPECTED_TOTAL:
        raise RuntimeError(
            "Duplicate genome IDs in Kleborate manifest."
        )

    if len(exception_audit) != EXPECTED_EXCLUDED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED} exception-audit "
            f"rows; found {len(exception_audit)}."
        )

    if exception_audit[
        "genome_id"
    ].nunique() != EXPECTED_EXCLUDED:
        raise RuntimeError(
            "Duplicate genome IDs in exception audit."
        )

    allowed_statuses = {
        "concordant_strong",
        "discordant_strong",
    }

    if not calls[
        "taxonomy_concordance_status"
    ].isin(allowed_statuses).all():
        raise RuntimeError(
            "Unexpected Kleborate concordance status."
        )

    calls[
        "taxonomy_adjudication_action"
    ] = calls[
        "taxonomy_concordance_status"
    ].map(
        {
            "concordant_strong":
                "retain_target_species",
            "discordant_strong":
                "exclude_strong_species_discordance",
        }
    )

    calls[
        "passes_enterobacterales_target_species_qc"
    ] = calls[
        "taxonomy_adjudication_action"
    ].eq(
        "retain_target_species"
    )

    calls[
        "taxonomy_exclusion_reason"
    ] = ""

    discordant = calls[
        "taxonomy_concordance_status"
    ].eq(
        "discordant_strong"
    )

    calls.loc[
        discordant,
        "taxonomy_exclusion_reason",
    ] = (
        "strong_kleborate_species_call_"
        + calls.loc[
            discordant,
            "kleborate_species",
        ]
        .str.strip()
        .str.casefold()
        .str.replace(
            r"[^a-z0-9]+",
            "_",
            regex=True,
        )
        .str.strip("_")
    )

    mic_columns = [
        "genome_id",
        "eligible_mic_rows",
        "eligible_antibiotics",
    ]

    missing_mic_columns = (
        set(mic_columns)
        - set(exception_audit.columns)
    )

    if missing_mic_columns:
        raise RuntimeError(
            "Missing exception MIC-audit columns: "
            + ", ".join(
                sorted(missing_mic_columns)
            )
        )

    calls = calls.merge(
        exception_audit[
            mic_columns
        ],
        on="genome_id",
        how="left",
        validate="one_to_one",
    )

    calls[
        "eligible_mic_rows"
    ] = pd.to_numeric(
        calls[
            "eligible_mic_rows"
        ],
        errors="coerce",
    ).fillna(0).astype("int64")

    calls[
        "eligible_antibiotics"
    ] = pd.to_numeric(
        calls[
            "eligible_antibiotics"
        ],
        errors="coerce",
    ).fillna(0).astype("int64")

    calls[
        "acquisition_order_numeric"
    ] = pd.to_numeric(
        calls[
            "acquisition_order"
        ],
        errors="raise",
    ).astype("int64")

    calls = calls.sort_values(
        "acquisition_order_numeric",
        kind="mergesort",
    ).reset_index(drop=True)

    retained = calls.loc[
        calls[
            "passes_enterobacterales_target_species_qc"
        ]
    ].copy()

    excluded = calls.loc[
        ~calls[
            "passes_enterobacterales_target_species_qc"
        ]
    ].copy()

    if len(retained) != EXPECTED_RETAINED:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED:,} retained "
            f"genomes; found {len(retained):,}."
        )

    if len(excluded) != EXPECTED_EXCLUDED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED} excluded "
            f"genomes; found {len(excluded)}."
        )

    if int(
        excluded[
            "eligible_mic_rows"
        ].sum()
    ) != 918:
        raise RuntimeError(
            "Expected 918 eligible MIC rows among "
            "excluded genomes."
        )

    observed_total_counts = (
        calls[
            "provisional_species"
        ]
        .value_counts()
        .to_dict()
    )

    observed_retained_counts = (
        retained[
            "provisional_species"
        ]
        .value_counts()
        .to_dict()
    )

    observed_excluded_counts = {
        species: int(
            excluded[
                "provisional_species"
            ].eq(species).sum()
        )
        for species in EXPECTED_SPECIES_TOTALS
    }

    if observed_total_counts != EXPECTED_SPECIES_TOTALS:
        raise RuntimeError(
            "Unexpected original species counts."
        )

    if (
        observed_retained_counts
        != EXPECTED_RETAINED_TOTALS
    ):
        raise RuntimeError(
            "Unexpected retained species counts."
        )

    if (
        observed_excluded_counts
        != EXPECTED_EXCLUDED_TOTALS
    ):
        raise RuntimeError(
            "Unexpected excluded species counts."
        )

    output_columns = [
        "acquisition_order",
        "genome_id",
        "provisional_species",
        "expected_kleborate_species",
        "kleborate_species",
        "kleborate_species_match",
        "taxonomy_concordance_status",
        "taxonomy_adjudication_action",
        "passes_enterobacterales_target_species_qc",
        "taxonomy_exclusion_reason",
        "eligible_mic_rows",
        "eligible_antibiotics",
        "chunk_order",
        "local_fasta_path",
        "fasta_sha256",
    ]

    calls[
        output_columns
    ].to_csv(
        ADJUDICATION_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    excluded[
        output_columns
    ].to_csv(
        EXCLUSION_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    RETAINED_IDS_PATH.write_text(
        "\n".join(
            retained[
                "genome_id"
            ].tolist()
        )
        + "\n",
        encoding="utf-8",
    )

    summary_rows = []

    for species in EXPECTED_SPECIES_TOTALS:
        species_calls = calls.loc[
            calls[
                "provisional_species"
            ].eq(species)
        ]

        species_retained = species_calls.loc[
            species_calls[
                "passes_enterobacterales_target_species_qc"
            ]
        ]

        species_excluded = species_calls.loc[
            ~species_calls[
                "passes_enterobacterales_target_species_qc"
            ]
        ]

        summary_rows.append(
            {
                "provisional_species": species,
                "input_genomes":
                    len(species_calls),
                "retained_genomes":
                    len(species_retained),
                "excluded_genomes":
                    len(species_excluded),
                "retention_rate":
                    len(species_retained)
                    / len(species_calls),
                "excluded_eligible_mic_rows":
                    int(
                        species_excluded[
                            "eligible_mic_rows"
                        ].sum()
                    ),
                "excluded_unique_called_species":
                    species_excluded[
                        "kleborate_species"
                    ].nunique(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        ADJUDICATION_PATH,
        EXCLUSION_PATH,
        RETAINED_IDS_PATH,
        SUMMARY_PATH,
    ]

    with OUTPUT_SHA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for output_path in sorted(
            output_paths,
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(output_path)}  "
                f"{output_path}\n"
            )

    print(
        "Input Enterobacterales genomes:",
        f"{len(calls):,}",
    )

    print(
        "Retained target-species genomes:",
        f"{len(retained):,}",
    )

    print(
        "Excluded strong-discordant genomes:",
        f"{len(excluded):,}",
    )

    print(
        "Excluded eligible MIC rows:",
        f"{excluded['eligible_mic_rows'].sum():,}",
    )

    print()
    print(
        "===== ADJUDICATION SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Production FASTA files deleted:",
        "0",
    )

    print(
        "Original modelling cohorts modified:",
        "NO",
    )

    print(
        "Final five-species cohort constructed:",
        "NO",
    )

    print()
    print(
        "STATUS: ENTEROBACTERALES TAXONOMY "
        "ADJUDICATION COMPLETE"
    )


if __name__ == "__main__":
    main()
