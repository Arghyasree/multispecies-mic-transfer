#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pandas as pd


INPUT_ROOT = Path("data/raw/amr")
INPUT_GLOB = "bvbrc_primary_laboratory_amr_*.tsv"

OUTPUT_ROOT = Path("results/tables")
METADATA_ROOT = Path("metadata/profiling")

NUMERIC_PATTERN = re.compile(
    r"[+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)

ANTIBIOTIC_ALIASES = {
    "amoxicillin clavulanat":
        "amoxicillin/clavulanic acid",
    "amoxicillin clavulanate":
        "amoxicillin/clavulanic acid",
    "amoxicillin clavulanic acid":
        "amoxicillin/clavulanic acid",
    "amoxicillin/clavulanate":
        "amoxicillin/clavulanic acid",
    "piperacillin tazobactam":
        "piperacillin/tazobactam",
    "trimethoprim sulfamethoxazole":
        "trimethoprim/sulfamethoxazole",
    "trimethoprim-sulfamethoxazole":
        "trimethoprim/sulfamethoxazole",
    "sulfamethoxazole/trimethoprim":
        "trimethoprim/sulfamethoxazole",
    "co-trimoxazole":
        "trimethoprim/sulfamethoxazole",
    "quinupristin dalfopristin":
        "quinupristin/dalfopristin",
    "ampicillin sulbactam":
        "ampicillin/sulbactam",
    "cefoperazone sulbactam":
        "cefoperazone/sulbactam",
    "ceftazidime avibactam":
        "ceftazidime/avibactam",
    "ceftolozane tazobactam":
        "ceftolozane/tazobactam",
    "meropenem vaborbactam":
        "meropenem/vaborbactam",
    "imipenem relebactam":
        "imipenem/relebactam",
    "tetracyklin":
        "tetracycline",
    "tigecyklin":
        "tigecycline",
}

BROAD_CLASS_LABELS = {
    "aminoglycoside",
    "beta-lactam",
    "carbapenem",
    "cephalosporin",
    "fluoroquinolone",
    "macrolide",
    "penicillin class",
    "tetracyclines",
}


def provisional_species(
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

    invalid_second_tokens = {
        "sp.",
        "spp.",
        "strain",
        "bacterium",
    }

    if (
        not first
        or not second
        or second.lower()
        in invalid_second_tokens
    ):
        return ""

    return f"{first} {second}"


def normalize_antibiotic(
    value: str,
) -> str:
    text = value.strip().lower()

    text = (
        text
        .replace("_", " ")
        .replace("–", "-")
        .replace("—", "-")
    )

    text = re.sub(
        r"\s*/\s*",
        "/",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return ANTIBIOTIC_ALIASES.get(
        text,
        text,
    )


def join_unique(
    series: pd.Series,
) -> str:
    values = {
        str(value)
        for value in series
        if str(value)
    }

    return "|".join(
        sorted(values)
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
    input_paths = sorted(
        INPUT_ROOT.glob(
            INPUT_GLOB
        )
    )

    if len(input_paths) != 1:
        raise ValueError(
            "Expected exactly one primary AMR "
            f"input; found {len(input_paths)}: "
            f"{input_paths}"
        )

    input_path = input_paths[0]

    print(
        "===== PROVISIONAL SPECIES–DRUG "
        "OVERLAP ====="
    )
    print("Input:", input_path)

    required_columns = [
        "genome_id",
        "genome_name",
        "taxon_id",
        "antibiotic",
        "measurement",
        "measurement_sign",
        "measurement_value",
        "measurement_unit",
        "laboratory_typing_method",
    ]

    frame = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=required_columns,
        low_memory=False,
    )

    for column in required_columns:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    normalized_sign = (
        frame["measurement_sign"]
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

    normalized_unit = (
        frame["measurement_unit"]
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
        .replace(
            {
                "mcg/ml": "ug/ml",
                "microgram/ml": "ug/ml",
                "milligram/liter": "mg/l",
                "milligram/litre": "mg/l",
            }
        )
    )

    scalar_numeric = (
        frame["measurement_value"]
        .str.fullmatch(
            NUMERIC_PATTERN,
            na=False,
        )
    )

    numeric_value = pd.to_numeric(
        frame["measurement_value"].where(
            scalar_numeric
        ),
        errors="coerce",
    )

    paired_value = (
        frame["measurement_value"]
        .str.contains(
            "/",
            regex=False,
        )
        |
        frame["measurement"]
        .str.contains(
            "/",
            regex=False,
        )
    )

    supported_sign = (
        normalized_sign.isin(
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

    supported_unit = (
        normalized_unit.isin(
            {
                "mg/l",
                "ug/ml",
            }
        )
    )

    candidate = (
        scalar_numeric
        & numeric_value.gt(0)
        & supported_unit
        & supported_sign
        & ~paired_value
    )

    work = frame.loc[
        candidate
    ].copy()

    work["positive_mic"] = (
        numeric_value.loc[candidate]
    )

    work["uncensored"] = (
        normalized_sign.loc[candidate]
        .isin(
            {
                "",
                "=",
            }
        )
        .astype(int)
    )

    work["censored"] = (
        normalized_sign.loc[candidate]
        .isin(
            {
                "<",
                "<=",
                ">",
                ">=",
            }
        )
        .astype(int)
    )

    work[
        "disk_diffusion_labeled"
    ] = (
        work[
            "laboratory_typing_method"
        ]
        .str.lower()
        .str.contains(
            r"disk|disc|kirby",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    work[
        "provisional_species"
    ] = work[
        "genome_name"
    ].map(
        provisional_species
    )

    work[
        "normalized_antibiotic"
    ] = work[
        "antibiotic"
    ].map(
        normalize_antibiotic
    )

    work = work.loc[
        work[
            "provisional_species"
        ].ne("")
        &
        work[
            "normalized_antibiotic"
        ].ne("")
    ].copy()

    print(
        "Positive quantitative candidates:",
        f"{len(work):,}",
    )

    coverage = (
        work.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            raw_antibiotic_variants=(
                "antibiotic",
                join_unique,
            ),
            candidate_rows=(
                "genome_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            uncensored_rows=(
                "uncensored",
                "sum",
            ),
            censored_rows=(
                "censored",
                "sum",
            ),
            disk_diffusion_labeled_rows=(
                "disk_diffusion_labeled",
                "sum",
            ),
            distinct_positive_mic_values=(
                "positive_mic",
                "nunique",
            ),
            minimum_positive_mic=(
                "positive_mic",
                "min",
            ),
            maximum_positive_mic=(
                "positive_mic",
                "max",
            ),
            taxon_ids=(
                "taxon_id",
                join_unique,
            ),
        )
    )

    coverage[
        "censoring_fraction"
    ] = (
        coverage["censored_rows"]
        / coverage["candidate_rows"]
    )

    coverage[
        "is_combination"
    ] = (
        coverage[
            "normalized_antibiotic"
        ]
        .str.contains(
            "/",
            regex=False,
        )
    )

    coverage[
        "is_class_label"
    ] = (
        coverage[
            "normalized_antibiotic"
        ]
        .isin(
            BROAD_CLASS_LABELS
        )
    )

    eligible_single_agent = (
        ~coverage["is_combination"]
        & ~coverage["is_class_label"]
    )

    coverage[
        "eligible_g200_u100"
    ] = (
        eligible_single_agent
        & coverage[
            "unique_genomes"
        ].ge(200)
        & coverage[
            "uncensored_rows"
        ].ge(100)
    )

    coverage[
        "eligible_g500_u200"
    ] = (
        eligible_single_agent
        & coverage[
            "unique_genomes"
        ].ge(500)
        & coverage[
            "uncensored_rows"
        ].ge(200)
    )

    coverage = coverage.sort_values(
        [
            "eligible_g500_u200",
            "unique_genomes",
            "uncensored_rows",
            "provisional_species",
            "normalized_antibiotic",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    species_base = (
        work.groupby(
            "provisional_species",
            as_index=False,
        )
        .agg(
            candidate_rows=(
                "genome_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
        )
    )

    summary_rows = []

    for species, group in coverage.groupby(
        "provisional_species",
        sort=False,
    ):
        strict = group.loc[
            group[
                "eligible_g500_u200"
            ]
        ]

        lenient = group.loc[
            group[
                "eligible_g200_u100"
            ]
        ]

        taxon_tokens = (
            group["taxon_ids"]
            .str.split("|")
            .explode()
        )

        summary_rows.append(
            {
                "provisional_species":
                    species,
                "normalized_antibiotics":
                    group[
                        "normalized_antibiotic"
                    ].nunique(),
                "eligible_antibiotics_g200_u100":
                    len(lenient),
                "eligible_antibiotics_g500_u200":
                    len(strict),
                "eligible_rows_g500_u200":
                    int(
                        strict[
                            "candidate_rows"
                        ].sum()
                    ),
                "eligible_names_g500_u200":
                    "|".join(
                        sorted(
                            strict[
                                "normalized_antibiotic"
                            ]
                        )
                    ),
                "taxon_ids":
                    join_unique(
                        taxon_tokens
                    ),
            }
        )

    species_summary = (
        species_base.merge(
            pd.DataFrame(
                summary_rows
            ),
            on="provisional_species",
            how="left",
        )
        .sort_values(
            [
                "eligible_antibiotics_g500_u200",
                "eligible_rows_g500_u200",
                "unique_genomes",
                "provisional_species",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    strict_drug_sets = {
        species: set(
            group.loc[
                group[
                    "eligible_g500_u200"
                ],
                "normalized_antibiotic",
            ]
        )
        for species, group
        in coverage.groupby(
            "provisional_species"
        )
    }

    active_species = sorted(
        species
        for species, drugs
        in strict_drug_sets.items()
        if drugs
    )

    overlap_rows = []

    for source_species in active_species:
        for target_species in active_species:
            if (
                source_species
                == target_species
            ):
                continue

            source_drugs = (
                strict_drug_sets[
                    source_species
                ]
            )

            target_drugs = (
                strict_drug_sets[
                    target_species
                ]
            )

            shared = (
                source_drugs
                & target_drugs
            )

            union = (
                source_drugs
                | target_drugs
            )

            overlap_rows.append(
                {
                    "source_species":
                        source_species,
                    "target_species":
                        target_species,
                    "source_eligible_drugs":
                        len(source_drugs),
                    "target_eligible_drugs":
                        len(target_drugs),
                    "shared_eligible_drugs":
                        len(shared),
                    "jaccard":
                        (
                            len(shared)
                            / len(union)
                            if union
                            else math.nan
                        ),
                    "source_only_drugs":
                        len(
                            source_drugs
                            - target_drugs
                        ),
                    "target_only_drugs":
                        len(
                            target_drugs
                            - source_drugs
                        ),
                    "shared_drug_names":
                        "|".join(
                            sorted(shared)
                        ),
                    "target_only_drug_names":
                        "|".join(
                            sorted(
                                target_drugs
                                - source_drugs
                            )
                        ),
                }
            )

    overlap = (
        pd.DataFrame(
            overlap_rows
        )
        .sort_values(
            [
                "shared_eligible_drugs",
                "jaccard",
                "target_only_drugs",
                "source_species",
                "target_species",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    alias_audit = (
        work.groupby(
            [
                "antibiotic",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "rows",
                "antibiotic":
                    "raw_antibiotic",
            }
        )
        .sort_values(
            [
                "normalized_antibiotic",
                "rows",
                "raw_antibiotic",
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

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        (
            OUTPUT_ROOT
            / "bvbrc_provisional_species_antibiotic_coverage.tsv"
        ): coverage,
        (
            OUTPUT_ROOT
            / "bvbrc_provisional_species_summary.tsv"
        ): species_summary,
        (
            OUTPUT_ROOT
            / "bvbrc_provisional_species_pairwise_overlap_g500_u200.tsv"
        ): overlap,
        (
            OUTPUT_ROOT
            / "bvbrc_antibiotic_normalization_audit.tsv"
        ): alias_audit,
    }

    for path, table in outputs.items():
        table.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )

    manifest_path = (
        METADATA_ROOT
        / "script03_provisional_overlap_outputs_sha256.txt"
    )

    with manifest_path.open(
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

    print()
    print(
        "===== TOP PROVISIONAL "
        "SPECIES ====="
    )

    print(
        species_summary
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        "===== TOP DIRECTIONAL "
        "OVERLAPS ====="
    )

    print(
        overlap
        .head(30)
        .to_string(index=False)
    )

    print()
    print(
        "STATUS: PROVISIONAL SPECIES-DRUG "
        "OVERLAP COMPLETE"
    )


if __name__ == "__main__":
    main()
