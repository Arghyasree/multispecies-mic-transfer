#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MIC_PATH = Path(
    "data/processed/mic/"
    "multispecies_monotherapy_quantitative_mic_reconciled.tsv"
)

CHEMICAL_STATE_PATH = Path(
    "metadata/antibiotics/"
    "chemical_entity_adjudication_state_after_batch2.tsv"
)

OUTPUT_DATA_ROOT = Path(
    "data/processed/modelling"
)

OUTPUT_METADATA_ROOT = Path(
    "metadata/modelling"
)

OUTPUT_TABLE_ROOT = Path(
    "results/tables"
)

EXPECTED_MIC_ROWS = 285_797
EXPECTED_MIC_GENOMES = 24_892
EXPECTED_MIC_ANTIBIOTICS = 83
EXPECTED_MIC_SPECIES = 5

EXPECTED_SIGN_COUNTS = {
    "=": 111_091,
    "<=": 88_936,
    ">": 68_988,
    ">=": 14_464,
    "<": 2_318,
}

EXPECTED_CHEMICAL_IDENTITIES = 34
EXPECTED_ELIGIBLE_CHEMICAL_IDENTITIES = 20
EXPECTED_INELIGIBLE_CHEMICAL_IDENTITIES = 2
EXPECTED_PENDING_CHEMICAL_IDENTITIES = 12

EXPECTED_ELIGIBLE_CELLS = 72
EXPECTED_ELIGIBLE_CELL_ANTIBIOTICS = 34
EXPECTED_ELIGIBLE_CELL_OBSERVATIONS = 244_074
EXPECTED_ELIGIBLE_CELL_EXACT = 102_370

EXPECTED_SINGLE_STRUCTURE_OBSERVATIONS = 177_850
EXPECTED_SINGLE_STRUCTURE_EXACT = 65_258
EXPECTED_SINGLE_STRUCTURE_ANTIBIOTICS = 20

EXPECTED_PANEL_UNION_ANTIBIOTICS = 19
EXPECTED_NONPANEL_ELIGIBLE_IDENTITIES = {
    "ertapenem",
}

PANEL_COLUMNS = [
    "primary_triad_complete",
    "primary_loao_candidate",
    "extended_loao_candidate",
    "ab_hard_shift_candidate",
    "pa_hard_shift_candidate",
]

POINT_TARGET_FACTORS = {
    "=": 1.0,
    "<=": 1.0,
    ">=": 1.0,
    "<": 0.5,
    ">": 2.0,
}

POINT_TARGET_RULES = {
    "=": "exact_value",
    "<=": "inclusive_left_threshold_used",
    ">=": "inclusive_right_threshold_used",
    "<": "strict_left_threshold_divided_by_2",
    ">": "strict_right_threshold_multiplied_by_2",
}

CENSORING_DIRECTIONS = {
    "=": "exact",
    "<=": "left",
    "<": "left",
    ">=": "right",
    ">": "right",
}

CENSORING_STRICTNESS = {
    "=": "exact",
    "<=": "inclusive",
    ">=": "inclusive",
    "<": "strict",
    ">": "strict",
}


def read_table(
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


def parse_flag_value(
    value: Any,
    *,
    column: str,
) -> bool:
    text = str(value).strip().casefold()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "",
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    raise RuntimeError(
        f"Cannot parse Boolean value in "
        f"{column}: {value!r}"
    )


def classify_eligibility(
    values: pd.Series,
) -> pd.Series:
    folded = (
        values
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    classes = pd.Series(
        "pending",
        index=values.index,
        dtype="object",
    )

    ineligible = folded.str.contains(
        "ineligible",
        regex=False,
    )

    eligible = (
        folded.str.contains(
            "eligible",
            regex=False,
        )
        & ~ineligible
    )

    classes.loc[
        eligible
    ] = "eligible"

    classes.loc[
        ineligible
    ] = "ineligible"

    return classes


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns from {label}: "
            + ", ".join(missing)
        )


def main() -> None:
    print(
        "===== SCRIPT 45 BUILD MOLECULARLY "
        "ELIGIBLE MODELLING COHORT ====="
    )

    require_paths = [
        MIC_PATH,
        CHEMICAL_STATE_PATH,
    ]

    for path in require_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    mic = read_table(
        MIC_PATH
    )

    chemical = read_table(
        CHEMICAL_STATE_PATH
    )

    require_columns(
        mic,
        [
            "provisional_species",
            "genome_id",
            "normalized_antibiotic",
            "reduced_constraint_type",
            "reduced_sign",
            "reduced_mic_value",
            "intersection_lower",
            "intersection_upper",
            "intersection_lower_closed",
            "intersection_upper_closed",
            "intersection_notation",
            "pair_id",
            "observation_id",
            "constraint_origin",
        ],
        "reconciled MIC table",
    )

    require_columns(
        chemical,
        [
            "normalized_antibiotic",
            "eligible_species_count",
            "eligible_species",
            "eligible_observations",
            "eligible_exact_observations",
            "authoritative_identity_status",
            "final_entity_class",
            "final_preferred_parent_compound_name",
            "final_structure_source",
            "final_structure_source_compound_id",
            "final_isomeric_smiles",
            "final_standard_inchi",
            "final_inchikey",
            "final_salt_form_policy",
            "final_stereochemistry_policy",
            "molecular_representation_decision",
            "molecular_benchmark_eligibility",
            "exclusion_reason",
            "phenotype_retention_policy",
            *PANEL_COLUMNS,
        ],
        "chemical adjudication state",
    )

    if len(mic) != EXPECTED_MIC_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MIC_ROWS:,} MIC "
            f"rows; found {len(mic):,}."
        )

    if (
        mic["genome_id"].nunique()
        != EXPECTED_MIC_GENOMES
    ):
        raise RuntimeError(
            "Unexpected reconciled MIC genome "
            "count."
        )

    if (
        mic[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_MIC_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Unexpected reconciled MIC "
            "antibiotic count."
        )

    if (
        mic[
            "provisional_species"
        ].nunique()
        != EXPECTED_MIC_SPECIES
    ):
        raise RuntimeError(
            "Unexpected reconciled MIC species "
            "count."
        )

    if len(chemical) != (
        EXPECTED_CHEMICAL_IDENTITIES
    ):
        raise RuntimeError(
            "Unexpected chemical-state row "
            "count."
        )

    if chemical[
        "normalized_antibiotic"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate antibiotic identity in "
            "chemical state."
        )

    observed_sign_counts = (
        mic[
            "reduced_sign"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        observed_sign_counts
        != EXPECTED_SIGN_COUNTS
    ):
        raise RuntimeError(
            "Unexpected reconciled sign counts: "
            f"{observed_sign_counts}"
        )

    eligibility_class = (
        classify_eligibility(
            chemical[
                "molecular_benchmark_eligibility"
            ]
        )
    )

    chemical = chemical.copy()

    chemical[
        "derived_eligibility_class"
    ] = eligibility_class

    observed_eligibility_counts = (
        chemical[
            "derived_eligibility_class"
        ]
        .value_counts()
        .to_dict()
    )

    expected_eligibility_counts = {
        "eligible":
            EXPECTED_ELIGIBLE_CHEMICAL_IDENTITIES,
        "ineligible":
            EXPECTED_INELIGIBLE_CHEMICAL_IDENTITIES,
        "pending":
            EXPECTED_PENDING_CHEMICAL_IDENTITIES,
    }

    if (
        observed_eligibility_counts
        != expected_eligibility_counts
    ):
        raise RuntimeError(
            "Unexpected chemical eligibility "
            f"counts: {observed_eligibility_counts}"
        )

    eligible_chemical = chemical.loc[
        chemical[
            "derived_eligibility_class"
        ].eq("eligible")
    ].copy()

    for panel_column in PANEL_COLUMNS:
        eligible_chemical[
            panel_column
        ] = [
            parse_flag_value(
                value,
                column=panel_column,
            )
            for value in eligible_chemical[
                panel_column
            ]
        ]

    eligible_chemical[
        "main_finalized_panel_member"
    ] = eligible_chemical[
        PANEL_COLUMNS
    ].any(
        axis=1
    )

    required_final_fields = [
        "final_preferred_parent_compound_name",
        "final_structure_source",
        "final_structure_source_compound_id",
        "final_isomeric_smiles",
        "final_standard_inchi",
        "final_inchikey",
        "molecular_representation_decision",
    ]

    for column in required_final_fields:
        if eligible_chemical[
            column
        ].eq("").any():
            bad = list(
                eligible_chemical.loc[
                    eligible_chemical[
                        column
                    ].eq(""),
                    "normalized_antibiotic",
                ]
            )

            raise RuntimeError(
                f"Blank finalized field {column} "
                f"for: {bad}"
            )

    disconnected_final_structures = (
        eligible_chemical[
            "final_isomeric_smiles"
        ].str.contains(
            ".",
            regex=False,
        )
    )

    if disconnected_final_structures.any():
        bad = list(
            eligible_chemical.loc[
                disconnected_final_structures,
                "normalized_antibiotic",
            ]
        )

        raise RuntimeError(
            "Eligible finalized structures "
            "contain disconnected components: "
            + "|".join(bad)
        )

    if (
        eligible_chemical[
            "final_inchikey"
        ].nunique()
        != EXPECTED_SINGLE_STRUCTURE_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Final eligible InChIKeys are not "
            "unique."
        )

    chemical_observation_total = int(
        pd.to_numeric(
            eligible_chemical[
                "eligible_observations"
            ],
            errors="raise",
        ).sum()
    )

    chemical_exact_total = int(
        pd.to_numeric(
            eligible_chemical[
                "eligible_exact_observations"
            ],
            errors="raise",
        ).sum()
    )

    if chemical_observation_total != (
        EXPECTED_SINGLE_STRUCTURE_OBSERVATIONS
    ):
        raise RuntimeError(
            "Chemical-state eligible observation "
            "total does not equal 177,850."
        )

    if chemical_exact_total != (
        EXPECTED_SINGLE_STRUCTURE_EXACT
    ):
        raise RuntimeError(
            "Chemical-state eligible exact total "
            "does not equal 65,258."
        )

    # --------------------------------------------------
    # Independently reconstruct G500/E200 cells
    # --------------------------------------------------

    cell_summary = (
        mic.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            observations=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            exact_observations=(
                "reduced_sign",
                lambda values:
                    int(
                        values.eq("=").sum()
                    ),
            ),
            left_censored_observations=(
                "reduced_sign",
                lambda values:
                    int(
                        values.isin(
                            ["<", "<="]
                        ).sum()
                    ),
            ),
            right_censored_observations=(
                "reduced_sign",
                lambda values:
                    int(
                        values.isin(
                            [">", ">="]
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    cell_summary[
        "passes_minimum_500_unique_genomes"
    ] = (
        cell_summary[
            "unique_genomes"
        ] >= 500
    )

    cell_summary[
        "passes_minimum_200_exact_observations"
    ] = (
        cell_summary[
            "exact_observations"
        ] >= 200
    )

    cell_summary[
        "coverage_eligible_g500_e200"
    ] = (
        cell_summary[
            "passes_minimum_500_unique_genomes"
        ]
        & cell_summary[
            "passes_minimum_200_exact_observations"
        ]
    )

    eligible_cells = cell_summary.loc[
        cell_summary[
            "coverage_eligible_g500_e200"
        ]
    ].copy()

    if len(eligible_cells) != (
        EXPECTED_ELIGIBLE_CELLS
    ):
        raise RuntimeError(
            f"Expected 72 eligible cells; found "
            f"{len(eligible_cells)}."
        )

    if (
        eligible_cells[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_ELIGIBLE_CELL_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Expected 34 antibiotics among "
            "eligible cells."
        )

    if int(
        eligible_cells[
            "observations"
        ].sum()
    ) != EXPECTED_ELIGIBLE_CELL_OBSERVATIONS:
        raise RuntimeError(
            "Eligible-cell observations do not "
            "sum to 244,074."
        )

    if int(
        eligible_cells[
            "exact_observations"
        ].sum()
    ) != EXPECTED_ELIGIBLE_CELL_EXACT:
        raise RuntimeError(
            "Eligible-cell exact observations "
            "do not sum to 102,370."
        )

    eligible_cell_keys = (
        eligible_cells[
            [
                "provisional_species",
                "normalized_antibiotic",
            ]
        ]
        .drop_duplicates()
    )

    coverage_eligible_mic = mic.merge(
        eligible_cell_keys,
        on=[
            "provisional_species",
            "normalized_antibiotic",
        ],
        how="inner",
        validate="many_to_one",
    )

    if len(
        coverage_eligible_mic
    ) != EXPECTED_ELIGIBLE_CELL_OBSERVATIONS:
        raise RuntimeError(
            "Reconstructed coverage-eligible MIC "
            "row total is incorrect."
        )

    if int(
        coverage_eligible_mic[
            "reduced_sign"
        ].eq("=").sum()
    ) != EXPECTED_ELIGIBLE_CELL_EXACT:
        raise RuntimeError(
            "Reconstructed coverage-eligible "
            "exact total is incorrect."
        )

    # --------------------------------------------------
    # Intersect coverage eligibility with finalized
    # single-structure chemical eligibility
    # --------------------------------------------------

    manifest_columns = [
        "normalized_antibiotic",
        "eligible_species_count",
        "eligible_species",
        "eligible_observations",
        "eligible_exact_observations",
        "authoritative_identity_status",
        "final_entity_class",
        "final_preferred_parent_compound_name",
        "final_structure_source",
        "final_structure_source_compound_id",
        "final_isomeric_smiles",
        "final_standard_inchi",
        "final_inchikey",
        "final_salt_form_policy",
        "final_stereochemistry_policy",
        "molecular_representation_decision",
        "molecular_benchmark_eligibility",
        "phenotype_retention_policy",
        *PANEL_COLUMNS,
        "main_finalized_panel_member",
    ]

    manifest = eligible_chemical[
        manifest_columns
    ].copy()

    manifest = manifest.rename(
        columns={
            "eligible_observations":
                "chemical_state_eligible_observations",
            "eligible_exact_observations":
                "chemical_state_eligible_exact_observations",
        }
    )

    cohort = coverage_eligible_mic.merge(
        manifest,
        on="normalized_antibiotic",
        how="inner",
        validate="many_to_one",
    )

    if len(cohort) != (
        EXPECTED_SINGLE_STRUCTURE_OBSERVATIONS
    ):
        raise RuntimeError(
            "Expected 177,850 molecularly "
            f"eligible observations; found "
            f"{len(cohort):,}."
        )

    if int(
        cohort[
            "reduced_sign"
        ].eq("=").sum()
    ) != EXPECTED_SINGLE_STRUCTURE_EXACT:
        raise RuntimeError(
            "Expected 65,258 exact observations "
            "in molecularly eligible cohort."
        )

    if (
        cohort[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_SINGLE_STRUCTURE_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Expected 20 antibiotics in the "
            "molecularly eligible cohort."
        )

    if cohort[
        "observation_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate observation ID in the "
            "molecularly eligible cohort."
        )

    if cohort[
        "pair_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate pair ID in the "
            "molecularly eligible cohort."
        )

    pair_duplicates = cohort.duplicated(
        subset=[
            "provisional_species",
            "genome_id",
            "normalized_antibiotic",
        ],
        keep=False,
    )

    if pair_duplicates.any():
        raise RuntimeError(
            "Duplicate genome–antibiotic pair in "
            "molecularly eligible cohort."
        )

    mic_values = pd.to_numeric(
        cohort[
            "reduced_mic_value"
        ],
        errors="raise",
    )

    if (
        ~np.isfinite(
            mic_values
        )
    ).any():
        raise RuntimeError(
            "Non-finite reduced MIC values."
        )

    if mic_values.le(0).any():
        raise RuntimeError(
            "Non-positive reduced MIC values."
        )

    unsupported_signs = sorted(
        set(
            cohort[
                "reduced_sign"
            ]
        )
        - set(
            POINT_TARGET_FACTORS
        )
    )

    if unsupported_signs:
        raise RuntimeError(
            "Unsupported MIC signs: "
            + "|".join(
                unsupported_signs
            )
        )

    factors = cohort[
        "reduced_sign"
    ].map(
        POINT_TARGET_FACTORS
    ).astype(float)

    cohort[
        "mic_target_point_mg_per_l"
    ] = mic_values * factors

    cohort[
        "mic_target_log2_mg_per_l"
    ] = np.log2(
        cohort[
            "mic_target_point_mg_per_l"
        ].astype(float)
    )

    cohort[
        "mic_target_substitution_rule"
    ] = cohort[
        "reduced_sign"
    ].map(
        POINT_TARGET_RULES
    )

    cohort[
        "censoring_direction"
    ] = cohort[
        "reduced_sign"
    ].map(
        CENSORING_DIRECTIONS
    )

    cohort[
        "censoring_strictness"
    ] = cohort[
        "reduced_sign"
    ].map(
        CENSORING_STRICTNESS
    )

    cohort[
        "is_exact_observation"
    ] = cohort[
        "reduced_sign"
    ].eq("=")

    cohort[
        "is_censored_observation"
    ] = ~cohort[
        "is_exact_observation"
    ]

    cohort[
        "point_target_version"
    ] = (
        "v1_strict_bounds_one_dilution_"
        "inclusive_bounds_at_threshold"
    )

    if (
        ~np.isfinite(
            cohort[
                "mic_target_point_mg_per_l"
            ].astype(float)
        )
    ).any():
        raise RuntimeError(
            "Non-finite point MIC targets."
        )

    if (
        ~np.isfinite(
            cohort[
                "mic_target_log2_mg_per_l"
            ].astype(float)
        )
    ).any():
        raise RuntimeError(
            "Non-finite log2 MIC targets."
        )

    if cohort[
        "mic_target_point_mg_per_l"
    ].astype(float).le(0).any():
        raise RuntimeError(
            "Non-positive point MIC targets."
        )

    nonpanel_eligible = set(
        manifest.loc[
            ~manifest[
                "main_finalized_panel_member"
            ],
            "normalized_antibiotic",
        ]
    )

    if (
        nonpanel_eligible
        != EXPECTED_NONPANEL_ELIGIBLE_IDENTITIES
    ):
        raise RuntimeError(
            "Unexpected finalized eligible "
            "antibiotics outside the main panel "
            f"union: {sorted(nonpanel_eligible)}"
        )

    main_panel_cohort = cohort.loc[
        cohort[
            "main_finalized_panel_member"
        ]
    ].copy()

    if (
        main_panel_cohort[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_PANEL_UNION_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Expected 19 antibiotics in the "
            "finalized main-panel union."
        )

    # Deterministic ordering.
    sort_columns = [
        "provisional_species",
        "normalized_antibiotic",
        "genome_id",
        "observation_id",
    ]

    cohort = cohort.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    main_panel_cohort = (
        main_panel_cohort.sort_values(
            sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    manifest = manifest.sort_values(
        "normalized_antibiotic",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    eligible_cells = (
        eligible_cells.sort_values(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Summaries
    # --------------------------------------------------

    species_antibiotic_summary = (
        cohort.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            observations=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            exact_observations=(
                "is_exact_observation",
                "sum",
            ),
            censored_observations=(
                "is_censored_observation",
                "sum",
            ),
            left_censored_observations=(
                "censoring_direction",
                lambda values:
                    int(
                        values.eq("left").sum()
                    ),
            ),
            right_censored_observations=(
                "censoring_direction",
                lambda values:
                    int(
                        values.eq("right").sum()
                    ),
            ),
            minimum_point_target_log2=(
                "mic_target_log2_mg_per_l",
                "min",
            ),
            maximum_point_target_log2=(
                "mic_target_log2_mg_per_l",
                "max",
            ),
            primary_triad_complete=(
                "primary_triad_complete",
                "first",
            ),
            primary_loao_candidate=(
                "primary_loao_candidate",
                "first",
            ),
            extended_loao_candidate=(
                "extended_loao_candidate",
                "first",
            ),
            ab_hard_shift_candidate=(
                "ab_hard_shift_candidate",
                "first",
            ),
            pa_hard_shift_candidate=(
                "pa_hard_shift_candidate",
                "first",
            ),
            main_finalized_panel_member=(
                "main_finalized_panel_member",
                "first",
            ),
        )
        .reset_index()
    )

    panel_summary_rows: list[
        dict[str, Any]
    ] = []

    panel_definitions = [
        (
            "primary_triad_complete",
            "primary_triad_complete",
        ),
        (
            "primary_loao_candidate",
            "primary_loao_candidate",
        ),
        (
            "extended_loao_candidate",
            "extended_loao_candidate",
        ),
        (
            "ab_hard_shift_candidate",
            "ab_hard_shift_candidate",
        ),
        (
            "pa_hard_shift_candidate",
            "pa_hard_shift_candidate",
        ),
        (
            "main_finalized_panel_union",
            "main_finalized_panel_member",
        ),
    ]

    for panel_order, (
        panel_name,
        flag_column,
    ) in enumerate(
        panel_definitions,
        start=1,
    ):
        subset = cohort.loc[
            cohort[
                flag_column
            ]
        ]

        antibiotics = sorted(
            subset[
                "normalized_antibiotic"
            ].unique()
        )

        panel_summary_rows.append(
            {
                "panel_order":
                    panel_order,
                "panel_name":
                    panel_name,
                "eligible_antibiotics":
                    len(antibiotics),
                "eligible_antibiotic_names":
                    "|".join(
                        antibiotics
                    ),
                "observations":
                    len(subset),
                "exact_observations":
                    int(
                        subset[
                            "is_exact_observation"
                        ].sum()
                    ),
                "censored_observations":
                    int(
                        subset[
                            "is_censored_observation"
                        ].sum()
                    ),
                "unique_genomes":
                    subset[
                        "genome_id"
                    ].nunique(),
                "species":
                    subset[
                        "provisional_species"
                    ].nunique(),
            }
        )

    panel_summary = pd.DataFrame(
        panel_summary_rows
    )

    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------

    OUTPUT_DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_path = (
        OUTPUT_DATA_ROOT
        / "multispecies_single_structure_"
        "eligible_mic_cohort.tsv"
    )

    main_panel_path = (
        OUTPUT_DATA_ROOT
        / "multispecies_finalized_panel_"
        "mic_cohort.tsv"
    )

    manifest_path = (
        OUTPUT_METADATA_ROOT
        / "finalized_single_structure_"
        "antibiotic_manifest.tsv"
    )

    eligible_cells_path = (
        OUTPUT_METADATA_ROOT
        / "eligible_species_antibiotic_"
        "cells_g500_e200.tsv"
    )

    species_antibiotic_summary_path = (
        OUTPUT_TABLE_ROOT
        / "modelling_cohort_species_"
        "antibiotic_summary.tsv"
    )

    panel_summary_path = (
        OUTPUT_TABLE_ROOT
        / "modelling_panel_summary.tsv"
    )

    cohort.to_csv(
        cohort_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    main_panel_cohort.to_csv(
        main_panel_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    manifest.to_csv(
        manifest_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    eligible_cells.to_csv(
        eligible_cells_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    species_antibiotic_summary.to_csv(
        species_antibiotic_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    panel_summary.to_csv(
        panel_summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        cohort_path,
        main_panel_path,
        manifest_path,
        eligible_cells_path,
        species_antibiotic_summary_path,
        panel_summary_path,
    ]

    checksum_path = (
        OUTPUT_METADATA_ROOT
        / "script45_outputs_sha256.txt"
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

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print(
        "Reconciled MIC observations:",
        f"{len(mic):,}",
    )

    print(
        "Reconstructed G500/E200 cells:",
        len(eligible_cells),
    )

    print(
        "G500/E200 antibiotics:",
        eligible_cells[
            "normalized_antibiotic"
        ].nunique(),
    )

    print(
        "G500/E200 observations:",
        f"{len(coverage_eligible_mic):,}",
    )

    print(
        "G500/E200 exact observations:",
        f"{int(coverage_eligible_mic['reduced_sign'].eq('=').sum()):,}",
    )

    print()
    print(
        "Finalized single-structure antibiotics:",
        len(manifest),
    )

    print(
        "Finalized single-structure observations:",
        f"{len(cohort):,}",
    )

    print(
        "Finalized single-structure exact "
        "observations:",
        f"{int(cohort['is_exact_observation'].sum()):,}",
    )

    print(
        "Finalized single-structure censored "
        "observations:",
        f"{int(cohort['is_censored_observation'].sum()):,}",
    )

    print(
        "Main finalized-panel antibiotics:",
        main_panel_cohort[
            "normalized_antibiotic"
        ].nunique(),
    )

    print(
        "Main finalized-panel observations:",
        f"{len(main_panel_cohort):,}",
    )

    print(
        "Eligible nonpanel antibiotics:",
        "|".join(
            sorted(
                nonpanel_eligible
            )
        ),
    )

    print()
    print(
        "===== FINALIZED ANTIBIOTIC "
        "MANIFEST ====="
    )

    manifest_display_columns = [
        "normalized_antibiotic",
        "final_preferred_parent_compound_name",
        "final_structure_source",
        "final_structure_source_compound_id",
        "final_inchikey",
        "primary_triad_complete",
        "primary_loao_candidate",
        "extended_loao_candidate",
        "ab_hard_shift_candidate",
        "pa_hard_shift_candidate",
        "main_finalized_panel_member",
    ]

    print(
        manifest[
            manifest_display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== MODELLING PANEL SUMMARY ====="
    )

    print(
        panel_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== POINT-TARGET RULE COUNTS ====="
    )

    point_rule_summary = (
        cohort[
            [
                "reduced_sign",
                "mic_target_substitution_rule",
            ]
        ]
        .value_counts()
        .rename(
            "observations"
        )
        .reset_index()
    )

    print(
        point_rule_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Original censoring signs preserved:",
        "YES",
    )

    print(
        "Original censoring intervals preserved:",
        "YES",
    )

    print(
        "Point targets added without replacing "
        "original fields:",
        "YES",
    )

    print(
        "Frozen reconciled MIC table modified:",
        "NO",
    )

    print(
        "Frozen chemical state modified:",
        "NO",
    )

    print(
        "Deferred DailyMed adjudication resumed:",
        "NO",
    )

    print(
        "Genome FASTA files downloaded:",
        "NO",
    )

    print()
    print(
        "STATUS: MOLECULARLY ELIGIBLE "
        "MODELLING COHORT BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
