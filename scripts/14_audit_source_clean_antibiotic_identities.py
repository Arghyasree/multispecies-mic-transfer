#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/mic/"
    "postqc_quantitative_candidates_source_clean.tsv"
)

RESULT_ROOT = Path(
    "results/tables"
)

AUDIT_ROOT = Path(
    "metadata/mic_audit"
)

EXPECTED_ROWS = 356_165

KNOWN_SPACE_COMBINATIONS = {
    "cefepime taniborbactam",
    "trimethoprim sulfonamide",
}

GENERIC_OR_UNSPECIFIED = {
    "beta-lactam",
    "beta lactam",
    "sulfa",
    "sulfonamide",
    "sulfonamides",
    "cephalosporin",
    "cephalosporins",
    "carbapenem",
    "carbapenems",
    "fluoroquinolone",
    "fluoroquinolones",
    "aminoglycoside",
    "aminoglycosides",
    "antibiotic",
    "antibiotics",
    "unknown",
    "other",
}


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
            for value in values
            if str(value).strip()
        }
    )

    return "|".join(items)


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


def classify_identity(
    identity: str,
) -> str:
    value = identity.strip().casefold()

    if not value:
        return "blank_identity"

    if any(
        marker in value
        for marker in (
            "Â",
            "Ã",
            "â",
            "�",
        )
    ):
        return "encoding_problem"

    if (
        "/" in value
        or "+" in value
        or " plus " in value
    ):
        return "explicit_combination"

    if value in KNOWN_SPACE_COMBINATIONS:
        return "known_space_combination"

    if value in GENERIC_OR_UNSPECIFIED:
        return "generic_or_unspecified"

    return "single_named_agent"


def main() -> None:
    print(
        "===== AUDIT SOURCE-CLEAN "
        "ANTIBIOTIC IDENTITIES ====="
    )

    frame = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "id",
        "genome_id",
        "provisional_species",
        "antibiotic",
        "normalized_antibiotic",
        "normalized_sign",
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} rows; "
            f"found {len(frame):,}."
        )

    if frame["id"].duplicated().any():
        raise RuntimeError(
            "Input contains duplicate record IDs."
        )

    for column in required:
        frame[column] = clean(
            frame[column]
        )

    if frame[
        "normalized_antibiotic"
    ].eq("").any():
        raise RuntimeError(
            "Blank normalized antibiotic "
            "identities were found."
        )

    alias_audit = (
        frame.groupby(
            [
                "antibiotic",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            rows=(
                "id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            species_count=(
                "provisional_species",
                "nunique",
            ),
            species=(
                "provisional_species",
                join_unique,
            ),
        )
    )

    alias_audit[
        "alias_changed"
    ] = (
        alias_audit[
            "antibiotic"
        ].str.casefold()
        !=
        alias_audit[
            "normalized_antibiotic"
        ].str.casefold()
    )

    alias_audit[
        "raw_encoding_flag"
    ] = (
        alias_audit[
            "antibiotic"
        ].str.contains(
            r"[ÂÃâ�]",
            regex=True,
            na=False,
        )
    )

    alias_audit = alias_audit.sort_values(
        [
            "rows",
            "normalized_antibiotic",
            "antibiotic",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    identity_audit = (
        frame.groupby(
            "normalized_antibiotic",
            as_index=False,
        )
        .agg(
            rows=(
                "id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            species_count=(
                "provisional_species",
                "nunique",
            ),
            species=(
                "provisional_species",
                join_unique,
            ),
            raw_alias_count=(
                "antibiotic",
                "nunique",
            ),
            raw_aliases=(
                "antibiotic",
                join_unique,
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
    )

    identity_audit[
        "identity_structure"
    ] = identity_audit[
        "normalized_antibiotic"
    ].map(
        classify_identity
    )

    identity_audit[
        "contains_slash"
    ] = identity_audit[
        "normalized_antibiotic"
    ].str.contains(
        "/",
        regex=False,
    )

    identity_audit[
        "contains_plus"
    ] = identity_audit[
        "normalized_antibiotic"
    ].str.contains(
        "+",
        regex=False,
    )

    identity_audit[
        "needs_manual_review"
    ] = (
        identity_audit[
            "identity_structure"
        ].ne(
            "single_named_agent"
        )
        |
        identity_audit[
            "raw_alias_count"
        ].gt(1)
    )

    identity_audit = identity_audit.sort_values(
        [
            "identity_structure",
            "rows",
            "normalized_antibiotic",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    species_identity = (
        frame.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            rows=(
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
    )

    structure_summary = (
        identity_audit.groupby(
            "identity_structure",
            as_index=False,
        )
        .agg(
            normalized_identities=(
                "normalized_antibiotic",
                "size",
            ),
            rows=(
                "rows",
                "sum",
            ),
            unique_genome_identity_pairs=(
                "unique_genomes",
                "sum",
            ),
        )
        .sort_values(
            "rows",
            ascending=False,
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

    identity_path = (
        RESULT_ROOT
        / "postqc_source_clean_antibiotic_identity_audit.tsv"
    )

    species_path = (
        RESULT_ROOT
        / "postqc_source_clean_species_antibiotic_identity_audit.tsv"
    )

    structure_path = (
        RESULT_ROOT
        / "postqc_source_clean_antibiotic_structure_summary.tsv"
    )

    alias_path = (
        AUDIT_ROOT
        / "postqc_source_clean_antibiotic_alias_audit.tsv"
    )

    identity_audit.to_csv(
        identity_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    species_identity.to_csv(
        species_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    structure_summary.to_csv(
        structure_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    alias_audit.to_csv(
        alias_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    outputs = [
        identity_path,
        species_path,
        structure_path,
        alias_path,
    ]

    checksum_path = (
        AUDIT_ROOT
        / "script14_outputs_sha256.txt"
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

    review = identity_audit.loc[
        identity_audit[
            "needs_manual_review"
        ]
    ]

    changed_aliases = alias_audit.loc[
        alias_audit[
            "alias_changed"
        ]
        |
        alias_audit[
            "raw_encoding_flag"
        ]
    ]

    print(
        "Source-clean records:",
        f"{len(frame):,}",
    )

    print(
        "Represented genomes:",
        f"{frame['genome_id'].nunique():,}",
    )

    print(
        "Raw antibiotic labels:",
        f"{frame['antibiotic'].nunique():,}",
    )

    print(
        "Normalized antibiotic identities:",
        f"{frame['normalized_antibiotic'].nunique():,}",
    )

    print()
    print(
        "===== IDENTITY-STRUCTURE SUMMARY ====="
    )

    print(
        structure_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== IDENTITIES REQUIRING REVIEW ====="
    )

    if review.empty:
        print("None")
    else:
        print(
            review.to_string(
                index=False
            )
        )

    print()
    print(
        "===== CHANGED OR ENCODING-AFFECTED "
        "ALIASES ====="
    )

    if changed_aliases.empty:
        print("None")
    else:
        print(
            changed_aliases.to_string(
                index=False
            )
        )

    print()
    print(
        "STATUS: ANTIBIOTIC IDENTITY "
        "AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
