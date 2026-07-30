#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


COVERAGE_PATH = Path(
    "results/tables/"
    "final_mic_postreconciliation_"
    "species_antibiotic_coverage.tsv"
)

GLOBAL_PATH = Path(
    "results/tables/"
    "final_mic_postreconciliation_"
    "global_antibiotic_coverage.tsv"
)

FINAL_MIC_PATH = Path(
    "data/processed/mic/"
    "multispecies_monotherapy_"
    "quantitative_mic_reconciled.tsv"
)

OUTPUT_ROOT = Path(
    "metadata/antibiotics"
)

RESULT_ROOT = Path(
    "results/tables"
)

EXPECTED_ELIGIBLE_CELLS = 72
EXPECTED_ELIGIBLE_DRUGS = 34

EXPECTED_SUPPORT_DISTRIBUTION = {
    1: 13,
    2: 10,
    3: 7,
    4: 2,
    5: 2,
}

EXPECTED_AT_LEAST_TWO = 21
EXPECTED_AT_LEAST_THREE = 11
EXPECTED_AT_LEAST_FOUR = 4
EXPECTED_ALL_FIVE = 2

SPECIES_ORDER = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Salmonella enterica",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
]

PRIMARY_TRIAD = {
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Salmonella enterica",
}

EXPECTED_TRIAD_DRUGS = {
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "meropenem",
    "tetracycline",
}

EXPECTED_AB_PANEL = {
    "ceftazidime",
    "imipenem",
    "levofloxacin",
    "meropenem",
}

EXPECTED_PA_PANEL = {
    "amikacin",
    "ceftazidime",
    "ciprofloxacin",
    "levofloxacin",
    "meropenem",
    "tobramycin",
}


def clean(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_bool(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    parsed = (
        clean(series)
        .str.casefold()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if parsed.isna().any():
        bad = sorted(
            set(
                clean(series.loc[
                    parsed.isna()
                ])
            )
        )

        raise RuntimeError(
            f"Cannot parse {column_name}: {bad}"
        )

    return parsed.astype(bool)


def split_tokens(
    values: pd.Series,
    pattern: str,
) -> list[str]:
    result: set[str] = set()

    for value in values:
        for token in re.split(
            pattern,
            str(value),
        ):
            token = token.strip()

            if (
                not token
                or token == "<blank>"
            ):
                continue

            result.add(token)

    return sorted(result)


def join_pipe(
    values: list[str],
) -> str:
    return "|".join(values)


def ordered_species(
    values: pd.Series,
) -> str:
    observed = set(
        clean(values)
    )

    unexpected = (
        observed - set(SPECIES_ORDER)
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected species names: "
            f"{sorted(unexpected)}"
        )

    return "|".join(
        species
        for species in SPECIES_ORDER
        if species in observed
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
        "===== BUILD COVERAGE-ELIGIBLE "
        "ANTIBIOTIC IDENTITY MANIFEST ====="
    )

    coverage = pd.read_csv(
        COVERAGE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    global_coverage = pd.read_csv(
        GLOBAL_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    mic = pd.read_csv(
        FINAL_MIC_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "normalized_antibiotic",
            "normalized_unit",
            "source_antibiotic_labels",
            "source_pmids",
        ],
        low_memory=False,
    )

    for table in [
        coverage,
        global_coverage,
        mic,
    ]:
        for column in table.columns:
            table[column] = clean(
                table[column]
            )

    required_coverage = {
        "provisional_species",
        "normalized_antibiotic",
        "observations",
        "unique_genomes",
        "exact_observations",
        "left_censored_observations",
        "right_censored_observations",
        "unique_pmids",
        "pmids",
        "meets_g500_exact200",
    }

    missing = sorted(
        required_coverage
        - set(coverage.columns)
    )

    if missing:
        raise RuntimeError(
            "Coverage table is missing: "
            + ", ".join(missing)
        )

    coverage[
        "eligible"
    ] = parse_bool(
        coverage[
            "meets_g500_exact200"
        ],
        "meets_g500_exact200",
    )

    eligible = coverage.loc[
        coverage["eligible"]
    ].copy()

    if len(eligible) != EXPECTED_ELIGIBLE_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_CELLS} "
            f"eligible cells; found {len(eligible)}."
        )

    if (
        eligible[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_ELIGIBLE_DRUGS
    ):
        raise RuntimeError(
            "Unexpected number of eligible drugs."
        )

    numeric_columns = [
        "observations",
        "unique_genomes",
        "exact_observations",
        "left_censored_observations",
        "right_censored_observations",
    ]

    for column in numeric_columns:
        eligible[column] = pd.to_numeric(
            eligible[column],
            errors="raise",
        )

    eligible_species = (
        eligible.groupby(
            "normalized_antibiotic"
        )[
            "provisional_species"
        ]
        .apply(set)
        .to_dict()
    )

    support = (
        eligible.groupby(
            "normalized_antibiotic"
        )[
            "provisional_species"
        ]
        .nunique()
    )

    observed_distribution = (
        support.value_counts()
        .sort_index()
        .to_dict()
    )

    if (
        observed_distribution
        != EXPECTED_SUPPORT_DISTRIBUTION
    ):
        raise RuntimeError(
            "Unexpected support distribution: "
            f"{observed_distribution}"
        )

    support_checks = {
        "at_least_two":
            int(support.ge(2).sum()),
        "at_least_three":
            int(support.ge(3).sum()),
        "at_least_four":
            int(support.ge(4).sum()),
        "all_five":
            int(support.eq(5).sum()),
    }

    expected_support_checks = {
        "at_least_two":
            EXPECTED_AT_LEAST_TWO,
        "at_least_three":
            EXPECTED_AT_LEAST_THREE,
        "at_least_four":
            EXPECTED_AT_LEAST_FOUR,
        "all_five":
            EXPECTED_ALL_FIVE,
    }

    if support_checks != expected_support_checks:
        raise RuntimeError(
            "Unexpected cross-species support: "
            f"{support_checks}"
        )

    triad_drugs = {
        antibiotic
        for antibiotic, species
        in eligible_species.items()
        if PRIMARY_TRIAD.issubset(species)
    }

    if triad_drugs != EXPECTED_TRIAD_DRUGS:
        raise RuntimeError(
            "Unexpected complete primary-triad "
            f"panel: {sorted(triad_drugs)}"
        )

    ab_panel = set(
        eligible.loc[
            eligible[
                "provisional_species"
            ].eq(
                "Acinetobacter baumannii"
            ),
            "normalized_antibiotic",
        ]
    )

    pa_panel = set(
        eligible.loc[
            eligible[
                "provisional_species"
            ].eq(
                "Pseudomonas aeruginosa"
            ),
            "normalized_antibiotic",
        ]
    )

    if ab_panel != EXPECTED_AB_PANEL:
        raise RuntimeError(
            "Unexpected A. baumannii panel."
        )

    if pa_panel != EXPECTED_PA_PANEL:
        raise RuntimeError(
            "Unexpected P. aeruginosa panel."
        )

    aggregated = (
        eligible.groupby(
            "normalized_antibiotic",
            as_index=False,
        )
        .agg(
            eligible_species_count=(
                "provisional_species",
                "nunique",
            ),
            eligible_species=(
                "provisional_species",
                ordered_species,
            ),
            eligible_species_drug_cells=(
                "provisional_species",
                "size",
            ),
            eligible_observations=(
                "observations",
                "sum",
            ),
            eligible_exact_observations=(
                "exact_observations",
                "sum",
            ),
            eligible_left_censored_observations=(
                "left_censored_observations",
                "sum",
            ),
            eligible_right_censored_observations=(
                "right_censored_observations",
                "sum",
            ),
            eligible_pmids=(
                "pmids",
                lambda values:
                    join_pipe(
                        split_tokens(
                            values,
                            r"[|;]",
                        )
                    ),
            ),
        )
    )

    aggregated[
        "eligible_pmid_count"
    ] = aggregated[
        "eligible_pmids"
    ].map(
        lambda value:
            0 if not value else len(
                value.split("|")
            )
    )

    global_required = {
        "normalized_antibiotic",
        "species_count",
        "species",
        "total_observations",
        "total_exact_observations",
        "total_left_censored_observations",
        "total_right_censored_observations",
    }

    missing = sorted(
        global_required
        - set(global_coverage.columns)
    )

    if missing:
        raise RuntimeError(
            "Global coverage table is missing: "
            + ", ".join(missing)
        )

    global_subset = global_coverage[
        [
            "normalized_antibiotic",
            "species_count",
            "species",
            "total_observations",
            "total_exact_observations",
            "total_left_censored_observations",
            "total_right_censored_observations",
        ]
    ].rename(
        columns={
            "species_count":
                "observed_species_count",
            "species":
                "observed_species",
            "total_observations":
                "global_observations",
            "total_exact_observations":
                "global_exact_observations",
            "total_left_censored_observations":
                "global_left_censored_observations",
            "total_right_censored_observations":
                "global_right_censored_observations",
        }
    )

    manifest = aggregated.merge(
        global_subset,
        on="normalized_antibiotic",
        how="left",
        validate="one_to_one",
    )

    raw_provenance = (
        mic.loc[
            mic[
                "normalized_antibiotic"
            ].isin(
                manifest[
                    "normalized_antibiotic"
                ]
            )
        ]
        .groupby(
            "normalized_antibiotic",
            as_index=False,
        )
        .agg(
            raw_antibiotic_labels=(
                "source_antibiotic_labels",
                lambda values:
                    join_pipe(
                        split_tokens(
                            values,
                            r"[|]",
                        )
                    ),
            ),
            normalized_units=(
                "normalized_unit",
                lambda values:
                    join_pipe(
                        split_tokens(
                            values,
                            r"[|]",
                        )
                    ),
            ),
            all_source_pmids=(
                "source_pmids",
                lambda values:
                    join_pipe(
                        split_tokens(
                            values,
                            r"[|;]",
                        )
                    ),
            ),
        )
    )

    raw_provenance[
        "all_source_pmid_count"
    ] = raw_provenance[
        "all_source_pmids"
    ].map(
        lambda value:
            0 if not value else len(
                value.split("|")
            )
    )

    manifest = manifest.merge(
        raw_provenance,
        on="normalized_antibiotic",
        how="left",
        validate="one_to_one",
    )

    manifest[
        "primary_triad_complete"
    ] = manifest[
        "normalized_antibiotic"
    ].isin(
        EXPECTED_TRIAD_DRUGS
    )

    manifest[
        "primary_loao_candidate"
    ] = manifest[
        "eligible_species_count"
    ].ge(3)

    manifest[
        "extended_loao_candidate"
    ] = manifest[
        "eligible_species_count"
    ].ge(2)

    manifest[
        "eligible_in_all_five_species"
    ] = manifest[
        "eligible_species_count"
    ].eq(5)

    manifest[
        "ab_hard_shift_candidate"
    ] = manifest[
        "normalized_antibiotic"
    ].isin(
        EXPECTED_AB_PANEL
    )

    manifest[
        "pa_hard_shift_candidate"
    ] = manifest[
        "normalized_antibiotic"
    ].isin(
        EXPECTED_PA_PANEL
    )

    manifest[
        "structure_query_name"
    ] = manifest[
        "normalized_antibiotic"
    ]

    manifest[
        "chemical_identity_status"
    ] = "unresolved"

    manifest[
        "preferred_parent_compound_name"
    ] = ""

    manifest[
        "structure_source"
    ] = ""

    manifest[
        "structure_source_compound_id"
    ] = ""

    manifest[
        "canonical_smiles"
    ] = ""

    manifest[
        "isomeric_smiles"
    ] = ""

    manifest[
        "standard_inchi"
    ] = ""

    manifest[
        "inchikey"
    ] = ""

    manifest[
        "molecular_formula"
    ] = ""

    manifest[
        "molecular_weight"
    ] = ""

    manifest[
        "salt_form_policy"
    ] = "unresolved"

    manifest[
        "stereochemistry_policy"
    ] = "unresolved"

    manifest[
        "molecular_representation_status"
    ] = "unresolved"

    manifest[
        "chemical_identity_notes"
    ] = ""

    manifest = manifest.sort_values(
        [
            "eligible_species_count",
            "eligible_observations",
            "normalized_antibiotic",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if len(manifest) != EXPECTED_ELIGIBLE_DRUGS:
        raise RuntimeError(
            "Final identity manifest does not "
            "contain 34 rows."
        )

    if manifest[
        "normalized_antibiotic"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate antibiotic identities "
            "in the manifest."
        )

    if manifest[
        "normalized_units"
    ].eq("").any():
        raise RuntimeError(
            "An eligible antibiotic lacks a "
            "normalized unit."
        )

    support_summary = (
        manifest.groupby(
            "eligible_species_count",
            as_index=False,
        )
        .agg(
            antibiotics=(
                "normalized_antibiotic",
                "size",
            ),
            antibiotic_names=(
                "normalized_antibiotic",
                lambda values:
                    "|".join(
                        sorted(values)
                    ),
            ),
            eligible_observations=(
                "eligible_observations",
                "sum",
            ),
            eligible_exact_observations=(
                "eligible_exact_observations",
                "sum",
            ),
        )
        .sort_values(
            "eligible_species_count",
            ascending=False,
        )
    )

    panel_summary = pd.DataFrame(
        [
            {
                "panel":
                    "primary_triad_complete",
                "antibiotics":
                    int(
                        manifest[
                            "primary_triad_complete"
                        ].sum()
                    ),
                "antibiotic_names":
                    "|".join(
                        sorted(
                            manifest.loc[
                                manifest[
                                    "primary_triad_complete"
                                ],
                                "normalized_antibiotic",
                            ]
                        )
                    ),
            },
            {
                "panel":
                    "primary_loao_candidate",
                "antibiotics":
                    int(
                        manifest[
                            "primary_loao_candidate"
                        ].sum()
                    ),
                "antibiotic_names":
                    "|".join(
                        sorted(
                            manifest.loc[
                                manifest[
                                    "primary_loao_candidate"
                                ],
                                "normalized_antibiotic",
                            ]
                        )
                    ),
            },
            {
                "panel":
                    "extended_loao_candidate",
                "antibiotics":
                    int(
                        manifest[
                            "extended_loao_candidate"
                        ].sum()
                    ),
                "antibiotic_names":
                    "|".join(
                        sorted(
                            manifest.loc[
                                manifest[
                                    "extended_loao_candidate"
                                ],
                                "normalized_antibiotic",
                            ]
                        )
                    ),
            },
            {
                "panel":
                    "ab_hard_shift_candidate",
                "antibiotics":
                    int(
                        manifest[
                            "ab_hard_shift_candidate"
                        ].sum()
                    ),
                "antibiotic_names":
                    "|".join(
                        sorted(
                            manifest.loc[
                                manifest[
                                    "ab_hard_shift_candidate"
                                ],
                                "normalized_antibiotic",
                            ]
                        )
                    ),
            },
            {
                "panel":
                    "pa_hard_shift_candidate",
                "antibiotics":
                    int(
                        manifest[
                            "pa_hard_shift_candidate"
                        ].sum()
                    ),
                "antibiotic_names":
                    "|".join(
                        sorted(
                            manifest.loc[
                                manifest[
                                    "pa_hard_shift_candidate"
                                ],
                                "normalized_antibiotic",
                            ]
                        )
                    ),
            },
        ]
    )

    expected_panel_counts = {
        "primary_triad_complete": 7,
        "primary_loao_candidate": 11,
        "extended_loao_candidate": 21,
        "ab_hard_shift_candidate": 4,
        "pa_hard_shift_candidate": 6,
    }

    observed_panel_counts = dict(
        zip(
            panel_summary["panel"],
            panel_summary["antibiotics"],
        )
    )

    if (
        observed_panel_counts
        != expected_panel_counts
    ):
        raise RuntimeError(
            "Unexpected panel counts: "
            f"{observed_panel_counts}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        OUTPUT_ROOT
        / "coverage_eligible_antibiotic_"
        "chemical_identity_manifest.tsv"
    )

    support_path = (
        RESULT_ROOT
        / "coverage_eligible_antibiotic_"
        "support_distribution.tsv"
    )

    panel_path = (
        RESULT_ROOT
        / "coverage_eligible_antibiotic_"
        "benchmark_panels.tsv"
    )

    outputs = {
        manifest_path:
            manifest,
        support_path:
            support_summary,
        panel_path:
            panel_summary,
    }

    for path, table in outputs.items():
        table.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
        )

    checksum_path = (
        OUTPUT_ROOT
        / "script28_outputs_sha256.txt"
    )

    with checksum_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            outputs,
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.as_posix()}\n"
            )

    print(
        "Coverage-eligible antibiotics:",
        f"{len(manifest):,}",
    )

    print(
        "Primary-triad panel:",
        f"{manifest['primary_triad_complete'].sum():,}",
    )

    print(
        "Primary LOAO panel:",
        f"{manifest['primary_loao_candidate'].sum():,}",
    )

    print(
        "Extended LOAO panel:",
        f"{manifest['extended_loao_candidate'].sum():,}",
    )

    print()
    print(
        "===== SUPPORT DISTRIBUTION ====="
    )

    print(
        support_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== BENCHMARK PANELS ====="
    )

    print(
        panel_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== IDENTITY MANIFEST ====="
    )

    print(
        manifest[
            [
                "normalized_antibiotic",
                "eligible_species_count",
                "eligible_species",
                "eligible_observations",
                "eligible_exact_observations",
                "raw_antibiotic_labels",
                "normalized_units",
                "chemical_identity_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "STATUS: COVERAGE-ELIGIBLE ANTIBIOTIC "
        "IDENTITY MANIFEST COMPLETE"
    )


if __name__ == "__main__":
    main()
