#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/mic/"
    "postqc_quantitative_candidates_source_clean.tsv"
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

EXPECTED_INPUT_ROWS = 356_165
EXPECTED_COMBINATION_ROWS = 35_324
EXPECTED_INVALID_ROWS = 9
EXPECTED_PASSING_ROWS = 320_832
EXPECTED_PASSING_IDENTITIES = 83

COMBINATION_IDENTITIES = {
    "amoxicillin/clavulanic acid",
    "trimethoprim/sulfamethoxazole",
    "piperacillin/tazobactam",
    "cefotaxime/clavulanic acid",
    "ceftazidime/clavulanic acid",
    "ceftolozane/tazobactam",
    "ceftazidime/avibactam",
    "ampicillin/sulbactam",
    "imipenem/relebactam",
    "cefoperazone/sulbactam",
    "ticarcillin/clavulanic acid",
    "cefepime taniborbactam",
    "trimethoprim sulfonamide",
}

INVALID_IDENTITIES = {
    "publicatiion",
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


def main() -> None:
    print(
        "===== BUILD MONOTHERAPY MIC "
        "CANDIDATES ====="
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
        "normalized_antibiotic",
        "normalized_sign",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing)
        )

    if len(frame) != EXPECTED_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_ROWS:,} rows; "
            f"found {len(frame):,}."
        )

    if frame["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate record IDs found."
        )

    for column in required:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    identity_key = (
        frame["normalized_antibiotic"]
        .str.casefold()
    )

    combination_mask = identity_key.isin(
        COMBINATION_IDENTITIES
    )

    invalid_mask = identity_key.isin(
        INVALID_IDENTITIES
    )

    if (
        combination_mask
        & invalid_mask
    ).any():
        raise RuntimeError(
            "Identity classified as both "
            "combination and invalid."
        )

    frame[
        "antibiotic_identity_class"
    ] = "monotherapy"

    frame.loc[
        combination_mask,
        "antibiotic_identity_class",
    ] = "combination"

    frame.loc[
        invalid_mask,
        "antibiotic_identity_class",
    ] = "invalid_non_drug_label"

    frame[
        "identity_exclusion_reason"
    ] = ""

    frame.loc[
        combination_mask,
        "identity_exclusion_reason",
    ] = "combination_drug_identity"

    frame.loc[
        invalid_mask,
        "identity_exclusion_reason",
    ] = "invalid_non_drug_label"

    frame[
        "passes_monotherapy_identity_qc"
    ] = frame[
        "identity_exclusion_reason"
    ].eq("")

    exclusions = frame.loc[
        ~frame[
            "passes_monotherapy_identity_qc"
        ]
    ].copy()

    passing = frame.loc[
        frame[
            "passes_monotherapy_identity_qc"
        ]
    ].copy()

    combination_rows = int(
        combination_mask.sum()
    )

    invalid_rows = int(
        invalid_mask.sum()
    )

    if combination_rows != EXPECTED_COMBINATION_ROWS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_COMBINATION_ROWS:,} "
            "combination rows; found "
            f"{combination_rows:,}."
        )

    if invalid_rows != EXPECTED_INVALID_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_INVALID_ROWS:,} "
            "invalid rows; found "
            f"{invalid_rows:,}."
        )

    if len(passing) != EXPECTED_PASSING_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PASSING_ROWS:,} "
            "passing rows; found "
            f"{len(passing):,}."
        )

    passing_identities = passing[
        "normalized_antibiotic"
    ].nunique()

    if (
        passing_identities
        != EXPECTED_PASSING_IDENTITIES
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_PASSING_IDENTITIES} "
            "monotherapy identities; found "
            f"{passing_identities}."
        )

    if passing[
        "normalized_antibiotic"
    ].str.casefold().isin(
        COMBINATION_IDENTITIES
        | INVALID_IDENTITIES
    ).any():
        raise RuntimeError(
            "Excluded identity remains in "
            "passing cohort."
        )

    flow = pd.DataFrame(
        [
            {
                "stage":
                    "source_value_clean_input",
                "records":
                    len(frame),
            },
            {
                "stage":
                    "combination_identity_excluded",
                "records":
                    combination_rows,
            },
            {
                "stage":
                    "invalid_non_drug_label_excluded",
                "records":
                    invalid_rows,
            },
            {
                "stage":
                    "monotherapy_identity_qc_passing",
                "records":
                    len(passing),
            },
        ]
    )

    exclusion_summary = (
        exclusions.groupby(
            [
                "antibiotic_identity_class",
                "normalized_antibiotic",
                "identity_exclusion_reason",
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
            species_count=(
                "provisional_species",
                "nunique",
            ),
        )
        .sort_values(
            "records",
            ascending=False,
        )
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
            monotherapy_antibiotics=(
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

    audited_path = (
        OUTPUT_ROOT
        / "postqc_source_clean_identity_audited.tsv"
    )

    passing_path = (
        OUTPUT_ROOT
        / "postqc_monotherapy_quantitative_candidates.tsv"
    )

    exclusions_path = (
        AUDIT_ROOT
        / "postqc_antibiotic_identity_exclusions.tsv"
    )

    flow_path = (
        RESULT_ROOT
        / "postqc_monotherapy_identity_qc_flow.tsv"
    )

    exclusion_summary_path = (
        RESULT_ROOT
        / "postqc_antibiotic_identity_exclusion_summary.tsv"
    )

    species_summary_path = (
        RESULT_ROOT
        / "postqc_monotherapy_species_summary.tsv"
    )

    antibiotic_summary_path = (
        RESULT_ROOT
        / "postqc_monotherapy_species_antibiotic_summary.tsv"
    )

    outputs = {
        audited_path: frame,
        passing_path: passing,
        exclusions_path: exclusions,
        flow_path: flow,
        exclusion_summary_path:
            exclusion_summary,
        species_summary_path:
            species_summary,
        antibiotic_summary_path:
            antibiotic_summary,
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
        / "script15_outputs_sha256.txt"
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
        "Input records:",
        f"{len(frame):,}",
    )

    print(
        "Combination exclusions:",
        f"{combination_rows:,}",
    )

    print(
        "Invalid-label exclusions:",
        f"{invalid_rows:,}",
    )

    print(
        "Monotherapy records:",
        f"{len(passing):,}",
    )

    print(
        "Monotherapy identities:",
        f"{passing_identities:,}",
    )

    print()
    print("===== QC FLOW =====")
    print(
        flow.to_string(
            index=False
        )
    )

    print()
    print("===== SPECIES SUMMARY =====")
    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: MONOTHERAPY MIC "
        "CANDIDATES COMPLETE"
    )


if __name__ == "__main__":
    main()
