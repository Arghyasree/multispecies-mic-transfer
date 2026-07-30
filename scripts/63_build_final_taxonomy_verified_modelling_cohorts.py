#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


FULL_INPUT_PATH = Path(
    "data/processed/modelling/"
    "multispecies_single_structure_eligible_mic_cohort.tsv"
)

PANEL_INPUT_PATH = Path(
    "data/processed/modelling/"
    "multispecies_finalized_panel_mic_cohort.tsv"
)

ANTIBIOTIC_MANIFEST_PATH = Path(
    "metadata/modelling/"
    "finalized_single_structure_antibiotic_manifest.tsv"
)

ENTEROBACTERALES_IDS_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_retained_target_species_genome_ids.txt"
)

NON_ENTEROBACTERALES_IDS_PATH = Path(
    "metadata/taxonomy/"
    "non_enterobacterales_retained_target_species_genome_ids.txt"
)

FULL_OUTPUT_PATH = Path(
    "data/processed/modelling/"
    "multispecies_taxonomy_verified_"
    "single_structure_mic_cohort.tsv"
)

PANEL_OUTPUT_PATH = Path(
    "data/processed/modelling/"
    "multispecies_taxonomy_verified_"
    "finalized_panel_mic_cohort.tsv"
)

FULL_EXCLUSION_ROWS_PATH = Path(
    "data/processed/modelling/audits/"
    "multispecies_taxonomy_excluded_"
    "single_structure_mic_observations.tsv"
)

PANEL_EXCLUSION_ROWS_PATH = Path(
    "data/processed/modelling/audits/"
    "multispecies_taxonomy_excluded_"
    "finalized_panel_mic_observations.tsv"
)

RETAINED_GENOME_MANIFEST_PATH = Path(
    "metadata/taxonomy/"
    "final_five_species_retained_genome_manifest.tsv"
)

RETAINED_GENOME_IDS_PATH = Path(
    "metadata/taxonomy/"
    "final_five_species_retained_genome_ids.txt"
)

EXCLUDED_GENOME_MANIFEST_PATH = Path(
    "metadata/taxonomy/"
    "final_five_species_excluded_genome_manifest.tsv"
)

SPECIES_SUMMARY_PATH = Path(
    "results/tables/modelling/"
    "final_taxonomy_verified_species_summary.tsv"
)

ANTIBIOTIC_SUMMARY_PATH = Path(
    "results/tables/modelling/"
    "final_taxonomy_verified_antibiotic_summary.tsv"
)

SPECIES_ANTIBIOTIC_SUMMARY_PATH = Path(
    "results/tables/modelling/"
    "final_taxonomy_verified_species_antibiotic_summary.tsv"
)

DECISION_PATH = Path(
    "metadata/taxonomy/"
    "final_five_species_taxonomy_filter_decision_2026-07-24.txt"
)

OUTPUT_SHA_PATH = Path(
    "metadata/taxonomy/"
    "script63_outputs_sha256.txt"
)


EXPECTED_FULL_INPUT_ROWS = 177_850
EXPECTED_PANEL_INPUT_ROWS = 176_571
EXPECTED_INPUT_GENOMES = 23_632

EXPECTED_ENTEROBACTERALES_GENOMES = 21_394
EXPECTED_NON_ENTEROBACTERALES_GENOMES = 2_142
EXPECTED_RETAINED_GENOMES = 23_536

EXPECTED_EXCLUDED_GENOMES = 96
EXPECTED_EXCLUDED_FULL_ROWS = 960
EXPECTED_FINAL_FULL_ROWS = 176_890

EXPECTED_FULL_ANTIBIOTICS = 20
EXPECTED_PANEL_ANTIBIOTICS = 19

EXPECTED_SPECIES_GENOMES = {
    "Acinetobacter baumannii": 1_169,
    "Escherichia coli": 6_673,
    "Klebsiella pneumoniae": 5_602,
    "Pseudomonas aeruginosa": 973,
    "Salmonella enterica": 9_119,
}

EXPECTED_FULL_SPECIES_ROWS = {
    "Acinetobacter baumannii": 3_329,
    "Escherichia coli": 70_151,
    "Klebsiella pneumoniae": 50_299,
    "Pseudomonas aeruginosa": 3_928,
    "Salmonella enterica": 49_183,
}

ENTEROBACTERALES_SPECIES = {
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Salmonella enterica",
}

NON_ENTEROBACTERALES_SPECIES = {
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
}

SORT_COLUMNS = [
    "provisional_species",
    "genome_id",
    "normalized_antibiotic",
    "observation_id",
]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def read_id_file(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    if len(values) != len(set(values)):
        raise RuntimeError(
            f"Duplicate IDs in {path}."
        )

    return values


def parse_bool(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    invalid = normalized.loc[
        ~normalized.isin(mapping)
    ].unique()

    if len(invalid) > 0:
        raise RuntimeError(
            f"{column_name}: invalid Boolean values: "
            f"{invalid[:10].tolist()}"
        )

    return normalized.map(
        mapping
    ).astype(bool)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def stable_sort(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    available = [
        column
        for column in SORT_COLUMNS
        if column in frame.columns
    ]

    return (
        frame.sort_values(
            available,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_species_summary(
    frame: pd.DataFrame,
    cohort_scope: str,
) -> pd.DataFrame:
    working = frame.copy()

    working["_exact"] = parse_bool(
        working[
            "is_exact_observation"
        ],
        "is_exact_observation",
    )

    working["_censored"] = parse_bool(
        working[
            "is_censored_observation"
        ],
        "is_censored_observation",
    )

    if not (
        working["_exact"]
        ^ working["_censored"]
    ).all():
        raise RuntimeError(
            f"{cohort_scope}: exact/censored "
            "indicators are not mutually exclusive "
            "and collectively exhaustive."
        )

    summary = (
        working.groupby(
            "provisional_species",
            sort=True,
        )
        .agg(
            mic_rows=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            unique_antibiotics=(
                "normalized_antibiotic",
                "nunique",
            ),
            exact_observations=(
                "_exact",
                "sum",
            ),
            censored_observations=(
                "_censored",
                "sum",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "cohort_scope",
        cohort_scope,
    )

    summary[
        "censored_fraction"
    ] = (
        summary[
            "censored_observations"
        ]
        / summary["mic_rows"]
    )

    return summary


def build_antibiotic_summary(
    frame: pd.DataFrame,
    cohort_scope: str,
) -> pd.DataFrame:
    working = frame.copy()

    working["_exact"] = parse_bool(
        working[
            "is_exact_observation"
        ],
        "is_exact_observation",
    )

    working["_censored"] = parse_bool(
        working[
            "is_censored_observation"
        ],
        "is_censored_observation",
    )

    summary = (
        working.groupby(
            "normalized_antibiotic",
            sort=True,
        )
        .agg(
            mic_rows=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            represented_species=(
                "provisional_species",
                "nunique",
            ),
            exact_observations=(
                "_exact",
                "sum",
            ),
            censored_observations=(
                "_censored",
                "sum",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "cohort_scope",
        cohort_scope,
    )

    summary[
        "censored_fraction"
    ] = (
        summary[
            "censored_observations"
        ]
        / summary["mic_rows"]
    )

    return summary


def build_species_antibiotic_summary(
    frame: pd.DataFrame,
    cohort_scope: str,
) -> pd.DataFrame:
    working = frame.copy()

    working["_exact"] = parse_bool(
        working[
            "is_exact_observation"
        ],
        "is_exact_observation",
    )

    working["_censored"] = parse_bool(
        working[
            "is_censored_observation"
        ],
        "is_censored_observation",
    )

    summary = (
        working.groupby(
            [
                "provisional_species",
                "normalized_antibiotic",
            ],
            sort=True,
        )
        .agg(
            mic_rows=(
                "observation_id",
                "size",
            ),
            unique_genomes=(
                "genome_id",
                "nunique",
            ),
            exact_observations=(
                "_exact",
                "sum",
            ),
            censored_observations=(
                "_censored",
                "sum",
            ),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "cohort_scope",
        cohort_scope,
    )

    summary[
        "censored_fraction"
    ] = (
        summary[
            "censored_observations"
        ]
        / summary["mic_rows"]
    )

    return summary


def main() -> None:
    for path in [
        FULL_INPUT_PATH,
        PANEL_INPUT_PATH,
        ANTIBIOTIC_MANIFEST_PATH,
        ENTEROBACTERALES_IDS_PATH,
        NON_ENTEROBACTERALES_IDS_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    full = read_tsv(
        FULL_INPUT_PATH
    )

    panel = read_tsv(
        PANEL_INPUT_PATH
    )

    antibiotic_manifest = read_tsv(
        ANTIBIOTIC_MANIFEST_PATH
    )

    required_columns = {
        "provisional_species",
        "genome_id",
        "normalized_antibiotic",
        "observation_id",
        "main_finalized_panel_member",
        "is_exact_observation",
        "is_censored_observation",
    }

    for label, frame in [
        ("full", full),
        ("panel", panel),
    ]:
        missing = (
            required_columns
            - set(frame.columns)
        )

        if missing:
            raise RuntimeError(
                f"{label} cohort is missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

    if list(full.columns) != list(
        panel.columns
    ):
        raise RuntimeError(
            "Full and panel cohort schemas differ."
        )

    if len(full) != EXPECTED_FULL_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_INPUT_ROWS:,} "
            f"full input rows; found {len(full):,}."
        )

    if len(panel) != EXPECTED_PANEL_INPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PANEL_INPUT_ROWS:,} "
            f"panel input rows; found {len(panel):,}."
        )

    if (
        full["genome_id"].nunique()
        != EXPECTED_INPUT_GENOMES
    ):
        raise RuntimeError(
            "Unexpected full-cohort input genome count."
        )

    if (
        panel["genome_id"].nunique()
        != EXPECTED_INPUT_GENOMES
    ):
        raise RuntimeError(
            "Unexpected panel input genome count."
        )

    if full[
        [
            "genome_id",
            "normalized_antibiotic",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "The full cohort contains duplicate "
            "genome–antibiotic pairs."
        )

    if panel[
        [
            "genome_id",
            "normalized_antibiotic",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "The panel cohort contains duplicate "
            "genome–antibiotic pairs."
        )

    if full[
        "observation_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate observation IDs in the full cohort."
        )

    if panel[
        "observation_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate observation IDs in the panel cohort."
        )

    if (
        full[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_FULL_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Unexpected full-cohort antibiotic count."
        )

    if (
        panel[
            "normalized_antibiotic"
        ].nunique()
        != EXPECTED_PANEL_ANTIBIOTICS
    ):
        raise RuntimeError(
            "Unexpected panel antibiotic count."
        )

    full_panel_mask = parse_bool(
        full[
            "main_finalized_panel_member"
        ],
        "full.main_finalized_panel_member",
    )

    panel_member_mask = parse_bool(
        panel[
            "main_finalized_panel_member"
        ],
        "panel.main_finalized_panel_member",
    )

    if not panel_member_mask.all():
        raise RuntimeError(
            "The finalized panel cohort contains a "
            "non-panel antibiotic row."
        )

    expected_panel_observation_ids = set(
        full.loc[
            full_panel_mask,
            "observation_id",
        ]
    )

    observed_panel_observation_ids = set(
        panel["observation_id"]
    )

    if (
        expected_panel_observation_ids
        != observed_panel_observation_ids
    ):
        raise RuntimeError(
            "The finalized panel cohort does not exactly "
            "match the panel-member subset of the full cohort."
        )

    if len(antibiotic_manifest) != 20:
        raise RuntimeError(
            "Expected 20 antibiotic-manifest rows."
        )

    if (
        antibiotic_manifest[
            "normalized_antibiotic"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate antibiotics in the manifest."
        )

    manifest_panel_mask = parse_bool(
        antibiotic_manifest[
            "main_finalized_panel_member"
        ],
        "manifest.main_finalized_panel_member",
    )

    if manifest_panel_mask.sum() != 19:
        raise RuntimeError(
            "Expected 19 main-panel manifest members."
        )

    enterobacterales_ids = set(
        read_id_file(
            ENTEROBACTERALES_IDS_PATH
        )
    )

    non_enterobacterales_ids = set(
        read_id_file(
            NON_ENTEROBACTERALES_IDS_PATH
        )
    )

    if (
        len(enterobacterales_ids)
        != EXPECTED_ENTEROBACTERALES_GENOMES
    ):
        raise RuntimeError(
            "Unexpected retained Enterobacterales count."
        )

    if (
        len(non_enterobacterales_ids)
        != EXPECTED_NON_ENTEROBACTERALES_GENOMES
    ):
        raise RuntimeError(
            "Unexpected retained non-Enterobacterales count."
        )

    if (
        enterobacterales_ids
        & non_enterobacterales_ids
    ):
        raise RuntimeError(
            "Enterobacterales and non-Enterobacterales "
            "retained sets overlap."
        )

    retained_ids = (
        enterobacterales_ids
        | non_enterobacterales_ids
    )

    if len(retained_ids) != EXPECTED_RETAINED_GENOMES:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED_GENOMES:,} "
            f"retained genomes; found "
            f"{len(retained_ids):,}."
        )

    input_ids = set(
        full["genome_id"]
    )

    if not retained_ids.issubset(
        input_ids
    ):
        raise RuntimeError(
            "Some retained genome IDs are absent "
            "from the full input cohort."
        )

    excluded_ids = (
        input_ids
        - retained_ids
    )

    if len(excluded_ids) != EXPECTED_EXCLUDED_GENOMES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED_GENOMES} "
            f"excluded genomes; found "
            f"{len(excluded_ids)}."
        )

    genome_species_counts = (
        full.groupby(
            "genome_id",
            sort=False,
        )[
            "provisional_species"
        ].nunique()
    )

    if not genome_species_counts.eq(1).all():
        raise RuntimeError(
            "A genome is associated with multiple "
            "provisional species."
        )

    genome_species = (
        full[
            [
                "genome_id",
                "provisional_species",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "genome_id"
        )[
            "provisional_species"
        ]
    )

    observed_entero_species = {
        genome_species.loc[
            genome_id
        ]
        for genome_id in (
            enterobacterales_ids
        )
    }

    observed_non_entero_species = {
        genome_species.loc[
            genome_id
        ]
        for genome_id in (
            non_enterobacterales_ids
        )
    }

    if (
        observed_entero_species
        != ENTEROBACTERALES_SPECIES
    ):
        raise RuntimeError(
            "Unexpected species in the retained "
            "Enterobacterales set."
        )

    if (
        observed_non_entero_species
        != NON_ENTEROBACTERALES_SPECIES
    ):
        raise RuntimeError(
            "Unexpected species in the retained "
            "non-Enterobacterales set."
        )

    # These four partitions must be constructed
    # unconditionally after both retained-species
    # membership checks have passed.
    full_retained = full.loc[
        full[
            "genome_id"
        ].isin(
            retained_ids
        )
    ].copy()

    panel_retained = panel.loc[
        panel[
            "genome_id"
        ].isin(
            retained_ids
        )
    ].copy()

    full_excluded = full.loc[
        full[
            "genome_id"
        ].isin(
            excluded_ids
        )
    ].copy()

    panel_excluded = panel.loc[
        panel[
            "genome_id"
        ].isin(
            excluded_ids
        )
    ].copy()

    if len(full_retained) != EXPECTED_FINAL_FULL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_FULL_ROWS:,} "
            f"retained full-cohort rows; found "
            f"{len(full_retained):,}."
        )

    if len(full_excluded) != EXPECTED_EXCLUDED_FULL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED_FULL_ROWS} "
            f"excluded full-cohort rows; found "
            f"{len(full_excluded)}."
        )

    if (
        len(full_retained)
        + len(full_excluded)
        != len(full)
    ):
        raise RuntimeError(
            "Retained and excluded full rows do not "
            "reconstruct the input cohort."
        )

    if (
        len(panel_retained)
        + len(panel_excluded)
        != len(panel)
    ):
        raise RuntimeError(
            "Retained and excluded panel rows do not "
            "reconstruct the input panel."
        )

    if (
        full_retained[
            "genome_id"
        ].nunique()
        != EXPECTED_RETAINED_GENOMES
    ):
        raise RuntimeError(
            "Unexpected final full-cohort genome count."
        )

    if (
        panel_retained[
            "genome_id"
        ].nunique()
        != EXPECTED_RETAINED_GENOMES
    ):
        raise RuntimeError(
            "Unexpected final panel genome count."
        )

    observed_species_genomes = (
        full_retained.groupby(
            "provisional_species",
            sort=True,
        )[
            "genome_id"
        ].nunique()
        .to_dict()
    )

    if (
        observed_species_genomes
        != EXPECTED_SPECIES_GENOMES
    ):
        raise RuntimeError(
            "Unexpected final species genome counts: "
            f"{observed_species_genomes}"
        )

    observed_species_rows = (
        full_retained[
            "provisional_species"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        observed_species_rows
        != EXPECTED_FULL_SPECIES_ROWS
    ):
        raise RuntimeError(
            "Unexpected final full-cohort species rows: "
            f"{observed_species_rows}"
        )

    if (
        full_retained[
            "normalized_antibiotic"
        ].nunique()
        != 20
    ):
        raise RuntimeError(
            "The final full cohort does not retain "
            "all 20 molecularly eligible antibiotics."
        )

    if (
        panel_retained[
            "normalized_antibiotic"
        ].nunique()
        != 19
    ):
        raise RuntimeError(
            "The final main panel does not contain "
            "exactly 19 antibiotics."
        )

    final_panel_expected_ids = set(
        full_retained.loc[
            parse_bool(
                full_retained[
                    "main_finalized_panel_member"
                ],
                "retained.main_finalized_panel_member",
            ),
            "observation_id",
        ]
    )

    if final_panel_expected_ids != set(
        panel_retained[
            "observation_id"
        ]
    ):
        raise RuntimeError(
            "The final panel is not the exact panel-member "
            "subset of the final full cohort."
        )

    full_retained = stable_sort(
        full_retained
    )

    panel_retained = stable_sort(
        panel_retained
    )

    full_excluded = stable_sort(
        full_excluded
    )

    panel_excluded = stable_sort(
        panel_excluded
    )

    retained_manifest_rows = []

    for genome_id in sorted(
        retained_ids
    ):
        species = genome_species.loc[
            genome_id
        ]

        if genome_id in (
            enterobacterales_ids
        ):
            branch = (
                "enterobacterales_kleborate"
            )

            source_path = (
                ENTEROBACTERALES_IDS_PATH
            )
        else:
            branch = (
                "non_enterobacterales_mash_fastani"
            )

            source_path = (
                NON_ENTEROBACTERALES_IDS_PATH
            )

        retained_manifest_rows.append(
            {
                "genome_id": genome_id,
                "provisional_species":
                    species,
                "taxonomy_verification_branch":
                    branch,
                "retained_id_source":
                    str(source_path),
                "retain_for_final_modelling":
                    True,
            }
        )

    retained_manifest = pd.DataFrame(
        retained_manifest_rows
    ).sort_values(
        [
            "provisional_species",
            "genome_id",
        ],
        kind="mergesort",
    )

    full_excluded_counts = (
        full_excluded.groupby(
            [
                "genome_id",
                "provisional_species",
            ],
            sort=True,
        )
        .size()
        .rename(
            "excluded_full_cohort_mic_rows"
        )
        .reset_index()
    )

    panel_excluded_counts = (
        panel_excluded.groupby(
            "genome_id",
            sort=True,
        )
        .size()
        .rename(
            "excluded_main_panel_mic_rows"
        )
    )

    excluded_manifest = (
        full_excluded_counts.copy()
    )

    excluded_manifest[
        "excluded_main_panel_mic_rows"
    ] = (
        excluded_manifest[
            "genome_id"
        ].map(
            panel_excluded_counts
        ).fillna(0).astype("int64")
    )

    excluded_manifest[
        "taxonomy_verification_branch"
    ] = excluded_manifest[
        "provisional_species"
    ].map(
        lambda species: (
            "enterobacterales_kleborate"
            if species
            in ENTEROBACTERALES_SPECIES
            else
            "non_enterobacterales_mash_fastani"
        )
    )

    excluded_manifest[
        "exclusion_basis"
    ] = (
        "failed_target_species_sequence_verification"
    )

    excluded_manifest = (
        excluded_manifest[
            [
                "genome_id",
                "provisional_species",
                "taxonomy_verification_branch",
                "exclusion_basis",
                "excluded_full_cohort_mic_rows",
                "excluded_main_panel_mic_rows",
            ]
        ]
        .sort_values(
            [
                "provisional_species",
                "genome_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(excluded_manifest) != 96:
        raise RuntimeError(
            "Final excluded-genome manifest "
            "does not contain 96 genomes."
        )

    species_summary = pd.concat(
        [
            build_species_summary(
                full_retained,
                "full_20_antibiotic",
            ),
            build_species_summary(
                panel_retained,
                "main_19_antibiotic",
            ),
        ],
        ignore_index=True,
    )

    antibiotic_summary = pd.concat(
        [
            build_antibiotic_summary(
                full_retained,
                "full_20_antibiotic",
            ),
            build_antibiotic_summary(
                panel_retained,
                "main_19_antibiotic",
            ),
        ],
        ignore_index=True,
    )

    species_antibiotic_summary = pd.concat(
        [
            build_species_antibiotic_summary(
                full_retained,
                "full_20_antibiotic",
            ),
            build_species_antibiotic_summary(
                panel_retained,
                "main_19_antibiotic",
            ),
        ],
        ignore_index=True,
    )

    output_paths = [
        FULL_OUTPUT_PATH,
        PANEL_OUTPUT_PATH,
        FULL_EXCLUSION_ROWS_PATH,
        PANEL_EXCLUSION_ROWS_PATH,
        RETAINED_GENOME_MANIFEST_PATH,
        RETAINED_GENOME_IDS_PATH,
        EXCLUDED_GENOME_MANIFEST_PATH,
        SPECIES_SUMMARY_PATH,
        ANTIBIOTIC_SUMMARY_PATH,
        SPECIES_ANTIBIOTIC_SUMMARY_PATH,
        DECISION_PATH,
    ]

    for path in output_paths:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    full_retained.to_csv(
        FULL_OUTPUT_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    panel_retained.to_csv(
        PANEL_OUTPUT_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    full_excluded.to_csv(
        FULL_EXCLUSION_ROWS_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    panel_excluded.to_csv(
        PANEL_EXCLUSION_ROWS_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    retained_manifest.to_csv(
        RETAINED_GENOME_MANIFEST_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    RETAINED_GENOME_IDS_PATH.write_text(
        "\n".join(
            retained_manifest[
                "genome_id"
            ].tolist()
        )
        + "\n",
        encoding="utf-8",
    )

    excluded_manifest.to_csv(
        EXCLUDED_GENOME_MANIFEST_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    species_summary.to_csv(
        SPECIES_SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    antibiotic_summary.to_csv(
        ANTIBIOTIC_SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    species_antibiotic_summary.to_csv(
        SPECIES_ANTIBIOTIC_SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    excluded_species_counts = (
        excluded_manifest[
            "provisional_species"
        ].value_counts().sort_index()
    )

    decision_text = f"""Stage: Final five-species taxonomy-filtered modelling-cohort construction

Input molecularly eligible cohort:
- genomes: {full['genome_id'].nunique():,}
- observations: {len(full):,}
- antibiotics: {full['normalized_antibiotic'].nunique():,}

Input main finalized panel:
- genomes: {panel['genome_id'].nunique():,}
- observations: {len(panel):,}
- antibiotics: {panel['normalized_antibiotic'].nunique():,}

Sequence-taxonomy retained genome sets:
- Enterobacterales retained genomes: {len(enterobacterales_ids):,}
- non-Enterobacterales retained genomes: {len(non_enterobacterales_ids):,}
- combined retained genomes: {len(retained_ids):,}

Sequence-taxonomy exclusions:
- excluded genomes: {len(excluded_ids):,}
- excluded full-cohort observations: {len(full_excluded):,}
- excluded main-panel observations: {len(panel_excluded):,}

Final full molecular cohort:
- genomes: {full_retained['genome_id'].nunique():,}
- observations: {len(full_retained):,}
- antibiotics: {full_retained['normalized_antibiotic'].nunique():,}

Final main modelling panel:
- genomes: {panel_retained['genome_id'].nunique():,}
- observations: {len(panel_retained):,}
- antibiotics: {panel_retained['normalized_antibiotic'].nunique():,}

Final retained genomes by species:
{retained_manifest['provisional_species'].value_counts().sort_index().to_string()}

Excluded genomes by provisional species:
{excluded_species_counts.to_string()}

Policy:
- the original pre-taxonomy modelling cohorts remain unchanged;
- taxonomy-filtered cohorts are written to new output files;
- excluded MIC observations remain preserved in audit files;
- no production FASTA was deleted or modified;
- the final 19-antibiotic panel is the exact main-panel subset of the final 20-antibiotic cohort;
- all subsequent representation, split and machine-learning experiments must use the taxonomy-verified outputs created by Script 63.
"""

    DECISION_PATH.write_text(
        decision_text,
        encoding="utf-8",
    )

    with OUTPUT_SHA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            output_paths,
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  {path}\n"
            )

    print(
        "===== SCRIPT 63 FINAL TAXONOMY-VERIFIED "
        "MODELLING COHORTS ====="
    )

    print(
        "Input full-cohort rows:",
        f"{len(full):,}",
    )

    print(
        "Final full-cohort rows:",
        f"{len(full_retained):,}",
    )

    print(
        "Excluded full-cohort rows:",
        f"{len(full_excluded):,}",
    )

    print()
    print(
        "Input main-panel rows:",
        f"{len(panel):,}",
    )

    print(
        "Final main-panel rows:",
        f"{len(panel_retained):,}",
    )

    print(
        "Excluded main-panel rows:",
        f"{len(panel_excluded):,}",
    )

    print()
    print(
        "Final retained genomes:",
        f"{len(retained_ids):,}",
    )

    print(
        "Final excluded genomes:",
        f"{len(excluded_ids):,}",
    )

    print()
    print(
        "===== FINAL SPECIES SUMMARY ====="
    )

    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Final full-cohort antibiotics:",
        full_retained[
            "normalized_antibiotic"
        ].nunique(),
    )

    print(
        "Final main-panel antibiotics:",
        panel_retained[
            "normalized_antibiotic"
        ].nunique(),
    )

    print()
    print(
        "Original input cohorts modified:",
        "NO",
    )

    print(
        "Production FASTAs modified:",
        "NO",
    )

    print()
    print(
        "STATUS: FINAL TAXONOMY-VERIFIED "
        "MODELLING COHORTS COMPLETE"
    )


if __name__ == "__main__":
    main()
