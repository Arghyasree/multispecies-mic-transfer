#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

FREEZE137 = (
    PROJECT
    / "metadata/config_selection/"
      "script137_successful_run_core_sha256.txt"
)

AUDIT_TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "amrfinder_full_annotation_audit_v1"
)

TOKEN_PREVALENCE_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_token_prevalence_by_species_v1.tsv"
)

TOKEN_BREADTH_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_token_species_breadth_v1.tsv"
)

CATEGORY_SUMMARY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_candidate_category_summary_v1.tsv"
)

TYPE_SUMMARY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_type_summary_v1.tsv"
)

SUBTYPE_SUMMARY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_subtype_summary_v1.tsv"
)

SCOPE_SUMMARY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_scope_summary_v1.tsv"
)

METHOD_SUMMARY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_method_summary_v1.tsv"
)

MULTIPLICITY_PATH = (
    AUDIT_TABLE_ROOT
    / "nested_loso_amrfinder_per_genome_token_multiplicity_v1.tsv"
)

OUTPUT_META_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "amr_vocabulary_design_audit_v1"
)

OUTPUT_TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "amr_vocabulary_design_audit_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_META_ROOT
    / "script145_outputs_sha256.txt"
)

SPECIES_DENOMINATORS = {
    "kp": 5_602,
    "ec": 6_673,
    "se": 9_119,
}

OUTER_DEVELOPMENT_PAIRS = {
    "ec": ("kp", "se"),
    "se": ("kp", "ec"),
    "kp": ("ec", "se"),
}

CATEGORIES = [
    "nonpoint_amr_candidate",
    "point_mutation_candidate",
    "other_nonpoint_call",
]

MIN_TOTAL_GENOMES = [
    1,
    2,
    5,
    10,
    20,
    50,
    100,
]

MAX_POOLED_PREVALENCE = [
    1.00,
    0.999,
    0.99,
    0.95,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def project_path(path_text: str) -> Path:
    path = Path(path_text.strip())

    if path.is_absolute():
        return path

    return PROJECT / path


def verify_sha_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)

    verified: list[Path] = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed SHA manifest line "
                f"{line_number}: {path}"
            )

        expected, path_text = parts
        candidate = project_path(path_text)

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        observed = sha256_file(candidate)

        if observed != expected:
            raise RuntimeError(
                f"SHA mismatch: {candidate}"
            )

        verified.append(candidate)

    if not verified:
        raise RuntimeError(
            f"Empty SHA manifest: {path}"
        )

    return verified


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def write_tsv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def coerce_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            )

    return result


def make_pair_inventory(
    prevalence: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for outer_target, development_species in (
        OUTER_DEVELOPMENT_PAIRS.items()
    ):
        species_a, species_b = (
            development_species
        )

        pair_denominator = (
            SPECIES_DENOMINATORS[species_a]
            + SPECIES_DENOMINATORS[species_b]
        )

        for category in CATEGORIES:
            subset = prevalence.loc[
                prevalence[
                    "candidate_category"
                ].eq(category)
                & prevalence[
                    "species_code"
                ].isin(
                    development_species
                )
            ].copy()

            if subset.empty:
                continue

            pivot_genomes = (
                subset.pivot_table(
                    index="Element symbol",
                    columns="species_code",
                    values="genomes",
                    aggfunc="sum",
                    fill_value=0,
                )
            )

            pivot_rows = (
                subset.pivot_table(
                    index="Element symbol",
                    columns="species_code",
                    values="result_rows",
                    aggfunc="sum",
                    fill_value=0,
                )
            )

            for species_code in (
                development_species
            ):
                if species_code not in (
                    pivot_genomes.columns
                ):
                    pivot_genomes[
                        species_code
                    ] = 0

                if species_code not in (
                    pivot_rows.columns
                ):
                    pivot_rows[
                        species_code
                    ] = 0

            pivot_genomes = (
                pivot_genomes.loc[
                    :,
                    list(development_species),
                ]
            )

            pivot_rows = (
                pivot_rows.loc[
                    :,
                    list(development_species),
                ]
            )

            for token in sorted(
                pivot_genomes.index.astype(
                    str
                ).tolist()
            ):
                genomes_a = int(
                    pivot_genomes.loc[
                        token,
                        species_a,
                    ]
                )

                genomes_b = int(
                    pivot_genomes.loc[
                        token,
                        species_b,
                    ]
                )

                rows_a = int(
                    pivot_rows.loc[
                        token,
                        species_a,
                    ]
                )

                rows_b = int(
                    pivot_rows.loc[
                        token,
                        species_b,
                    ]
                )

                total_genomes = (
                    genomes_a + genomes_b
                )

                pooled_prevalence = (
                    total_genomes
                    / pair_denominator
                )

                records.append(
                    {
                        "outer_target_code":
                            outer_target,
                        "development_species_a":
                            species_a,
                        "development_species_b":
                            species_b,
                        "candidate_category":
                            category,
                        "element_symbol":
                            token,
                        (
                            f"{species_a}_"
                            "genomes"
                        ):
                            genomes_a,
                        (
                            f"{species_b}_"
                            "genomes"
                        ):
                            genomes_b,
                        (
                            f"{species_a}_"
                            "result_rows"
                        ):
                            rows_a,
                        (
                            f"{species_b}_"
                            "result_rows"
                        ):
                            rows_b,
                        "development_total_genomes":
                            total_genomes,
                        "development_total_result_rows":
                            rows_a + rows_b,
                        "development_pooled_prevalence":
                            pooled_prevalence,
                        "present_in_both_development_species":
                            (
                                genomes_a > 0
                                and genomes_b > 0
                            ),
                        "development_species_breadth":
                            int(genomes_a > 0)
                            + int(genomes_b > 0),
                    }
                )

    result = pd.DataFrame(records)

    if result.empty:
        raise RuntimeError(
            "Pairwise token inventory is empty."
        )

    return result


def make_threshold_grid(
    pair_inventory: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (
        outer_target,
        category,
    ), group in pair_inventory.groupby(
        [
            "outer_target_code",
            "candidate_category",
        ],
        dropna=False,
    ):
        for (
            minimum_genomes,
            maximum_prevalence,
        ) in product(
            MIN_TOTAL_GENOMES,
            MAX_POOLED_PREVALENCE,
        ):
            retained = group.loc[
                (
                    group[
                        "development_total_genomes"
                    ]
                    >= minimum_genomes
                )
                & (
                    group[
                        "development_pooled_prevalence"
                    ]
                    <= maximum_prevalence
                )
            ]

            records.append(
                {
                    "outer_target_code":
                        outer_target,
                    "candidate_category":
                        category,
                    "minimum_development_total_genomes":
                        minimum_genomes,
                    "maximum_development_pooled_prevalence":
                        maximum_prevalence,
                    "retained_tokens":
                        len(retained),
                    "retained_tokens_present_in_both_development_species":
                        int(
                            retained[
                                "present_in_both_development_species"
                            ].astype(bool).sum()
                        ),
                    "retained_single_species_tokens":
                        int(
                            (
                                ~retained[
                                    "present_in_both_development_species"
                                ].astype(bool)
                            ).sum()
                        ),
                    "minimum_observed_prevalence":
                        (
                            float(
                                retained[
                                    "development_pooled_prevalence"
                                ].min()
                            )
                            if not retained.empty
                            else np.nan
                        ),
                    "maximum_observed_prevalence":
                        (
                            float(
                                retained[
                                    "development_pooled_prevalence"
                                ].max()
                            )
                            if not retained.empty
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(records)


def make_pair_summary(
    pair_inventory: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (
        outer_target,
        category,
    ), group in pair_inventory.groupby(
        [
            "outer_target_code",
            "candidate_category",
        ],
        dropna=False,
    ):
        records.append(
            {
                "outer_target_code":
                    outer_target,
                "candidate_category":
                    category,
                "union_tokens":
                    len(group),
                "tokens_present_in_both_development_species":
                    int(
                        group[
                            "present_in_both_development_species"
                        ].astype(bool).sum()
                    ),
                "tokens_present_in_one_development_species":
                    int(
                        (
                            ~group[
                                "present_in_both_development_species"
                            ].astype(bool)
                        ).sum()
                    ),
                "singleton_tokens":
                    int(
                        (
                            group[
                                "development_total_genomes"
                            ]
                            == 1
                        ).sum()
                    ),
                "tokens_at_least_5_genomes":
                    int(
                        (
                            group[
                                "development_total_genomes"
                            ]
                            >= 5
                        ).sum()
                    ),
                "tokens_at_least_10_genomes":
                    int(
                        (
                            group[
                                "development_total_genomes"
                            ]
                            >= 10
                        ).sum()
                    ),
                "tokens_at_least_20_genomes":
                    int(
                        (
                            group[
                                "development_total_genomes"
                            ]
                            >= 20
                        ).sum()
                    ),
                "tokens_prevalence_above_99_percent":
                    int(
                        (
                            group[
                                "development_pooled_prevalence"
                            ]
                            > 0.99
                        ).sum()
                    ),
                "maximum_pooled_prevalence":
                    float(
                        group[
                            "development_pooled_prevalence"
                        ].max()
                    ),
            }
        )

    return pd.DataFrame(records)


def make_top_tokens(
    pair_inventory: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    frames = []

    for (
        outer_target,
        category,
    ), group in pair_inventory.groupby(
        [
            "outer_target_code",
            "candidate_category",
        ],
        dropna=False,
    ):
        top = (
            group.sort_values(
                [
                    "development_total_genomes",
                    "development_total_result_rows",
                    "element_symbol",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .head(top_n)
            .copy()
        )

        top[
            "within_category_rank"
        ] = np.arange(
            1,
            len(top) + 1,
            dtype=int,
        )

        frames.append(top)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def make_cross_tabs(
    type_summary: pd.DataFrame,
    subtype_summary: pd.DataFrame,
    scope_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    specifications = [
        (
            "Type",
            type_summary,
            "type",
        ),
        (
            "Subtype",
            subtype_summary,
            "subtype",
        ),
        (
            "Scope",
            scope_summary,
            "scope",
        ),
        (
            "Method",
            method_summary,
            "method",
        ),
    ]

    for field_name, frame, value_column in (
        specifications
    ):
        for row in frame.itertuples(
            index=False
        ):
            records.append(
                {
                    "field":
                        field_name,
                    "species_code":
                        str(
                            getattr(
                                row,
                                "species_code",
                            )
                        ),
                    "value":
                        str(
                            getattr(
                                row,
                                value_column,
                            )
                        ),
                    "result_rows":
                        int(
                            getattr(
                                row,
                                "result_rows",
                            )
                        ),
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    verified_freeze_files = (
        verify_sha_manifest(
            FREEZE137
        )
    )

    input_paths = [
        TOKEN_PREVALENCE_PATH,
        TOKEN_BREADTH_PATH,
        CATEGORY_SUMMARY_PATH,
        TYPE_SUMMARY_PATH,
        SUBTYPE_SUMMARY_PATH,
        SCOPE_SUMMARY_PATH,
        METHOD_SUMMARY_PATH,
        MULTIPLICITY_PATH,
    ]

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    prevalence = coerce_numeric(
        read_tsv(
            TOKEN_PREVALENCE_PATH
        ),
        [
            "result_rows",
            "genomes",
            "species_genomes",
            "genome_prevalence",
        ],
    )

    breadth = coerce_numeric(
        read_tsv(
            TOKEN_BREADTH_PATH
        ),
        [
            "species_count",
            "total_genomes",
            "total_result_rows",
        ],
    )

    category_summary = coerce_numeric(
        read_tsv(
            CATEGORY_SUMMARY_PATH
        ),
        [
            "result_rows",
        ],
    )

    type_summary = coerce_numeric(
        read_tsv(
            TYPE_SUMMARY_PATH
        ),
        [
            "result_rows",
        ],
    )

    subtype_summary = coerce_numeric(
        read_tsv(
            SUBTYPE_SUMMARY_PATH
        ),
        [
            "result_rows",
        ],
    )

    scope_summary = coerce_numeric(
        read_tsv(
            SCOPE_SUMMARY_PATH
        ),
        [
            "result_rows",
        ],
    )

    method_summary = coerce_numeric(
        read_tsv(
            METHOD_SUMMARY_PATH
        ),
        [
            "result_rows",
        ],
    )

    multiplicity = coerce_numeric(
        read_tsv(
            MULTIPLICITY_PATH
        ),
        [
            "genome_token_pairs",
            "pairs_with_multiple_calls",
            "maximum_call_multiplicity",
        ],
    )

    expected_prevalence_columns = {
        "candidate_category",
        "species_code",
        "Element symbol",
        "result_rows",
        "genomes",
        "species_genomes",
        "genome_prevalence",
    }

    missing_columns = sorted(
        expected_prevalence_columns
        .difference(
            prevalence.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing prevalence columns: "
            + "|".join(missing_columns)
        )

    observed_species = sorted(
        prevalence[
            "species_code"
        ].unique().tolist()
    )

    if observed_species != [
        "ec",
        "kp",
        "se",
    ]:
        raise RuntimeError(
            "Unexpected prevalence species: "
            f"{observed_species}"
        )

    observed_categories = sorted(
        prevalence[
            "candidate_category"
        ].unique().tolist()
    )

    if observed_categories != sorted(
        CATEGORIES
    ):
        raise RuntimeError(
            "Unexpected candidate categories: "
            f"{observed_categories}"
        )

    pair_inventory = (
        make_pair_inventory(
            prevalence
        )
    )

    threshold_grid = (
        make_threshold_grid(
            pair_inventory
        )
    )

    pair_summary = (
        make_pair_summary(
            pair_inventory
        )
    )

    top_tokens = (
        make_top_tokens(
            pair_inventory
        )
    )

    cross_tabs = make_cross_tabs(
        type_summary,
        subtype_summary,
        scope_summary,
        method_summary,
    )

    global_breadth_summary = (
        breadth.groupby(
            [
                "candidate_category",
                "species_count",
            ],
            dropna=False,
        )
        .agg(
            tokens=(
                "Element symbol",
                "size",
            ),
            total_genomes_across_tokens=(
                "total_genomes",
                "sum",
            ),
        )
        .reset_index()
    )

    protocol = pd.DataFrame(
        [
            {
                "item":
                    "objective",
                "value":
                    (
                        "labels-free audit for "
                        "pair-specific AMR vocabulary "
                        "design"
                    ),
            },
            {
                "item":
                    "outer_development_pairs",
                "value":
                    (
                        "ec:kp+se|se:kp+ec|"
                        "kp:ec+se"
                    ),
            },
            {
                "item":
                    "candidate_categories",
                "value":
                    "|".join(CATEGORIES),
            },
            {
                "item":
                    "minimum_genome_thresholds_audited",
                "value":
                    "|".join(
                        str(value)
                        for value
                        in MIN_TOTAL_GENOMES
                    ),
            },
            {
                "item":
                    "maximum_prevalence_thresholds_audited",
                "value":
                    "|".join(
                        str(value)
                        for value
                        in MAX_POOLED_PREVALENCE
                    ),
            },
            {
                "item":
                    "vocabulary_selection",
                "value":
                    (
                        "none; thresholds remain "
                        "unselected"
                    ),
            },
            {
                "item":
                    "mic_labels_used",
                "value":
                    "none",
            },
            {
                "item":
                    "models_trained",
                "value":
                    "none",
            },
        ]
    )

    output_paths = {
        "pair_inventory":
            OUTPUT_META_ROOT
            / "nested_loso_pairwise_amr_token_inventory_v1.tsv",
        "threshold_grid":
            OUTPUT_TABLE_ROOT
            / "nested_loso_pairwise_amr_threshold_grid_v1.tsv",
        "pair_summary":
            OUTPUT_TABLE_ROOT
            / "nested_loso_pairwise_amr_token_summary_v1.tsv",
        "top_tokens":
            OUTPUT_TABLE_ROOT
            / "nested_loso_pairwise_amr_top_tokens_v1.tsv",
        "cross_tabs":
            OUTPUT_TABLE_ROOT
            / "nested_loso_amrfinder_type_subtype_scope_method_v1.tsv",
        "global_breadth":
            OUTPUT_TABLE_ROOT
            / "nested_loso_amr_global_species_breadth_summary_v1.tsv",
        "category_summary":
            OUTPUT_TABLE_ROOT
            / "nested_loso_amr_candidate_category_summary_copy_v1.tsv",
        "multiplicity":
            OUTPUT_TABLE_ROOT
            / "nested_loso_amr_token_multiplicity_summary_copy_v1.tsv",
        "protocol":
            OUTPUT_META_ROOT
            / "nested_loso_amr_vocabulary_design_audit_protocol_v1.tsv",
    }

    for path in output_paths.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    write_tsv(
        pair_inventory,
        output_paths[
            "pair_inventory"
        ],
    )

    write_tsv(
        threshold_grid,
        output_paths[
            "threshold_grid"
        ],
    )

    write_tsv(
        pair_summary,
        output_paths[
            "pair_summary"
        ],
    )

    write_tsv(
        top_tokens,
        output_paths[
            "top_tokens"
        ],
    )

    write_tsv(
        cross_tabs,
        output_paths[
            "cross_tabs"
        ],
    )

    write_tsv(
        global_breadth_summary,
        output_paths[
            "global_breadth"
        ],
    )

    write_tsv(
        category_summary,
        output_paths[
            "category_summary"
        ],
    )

    write_tsv(
        multiplicity,
        output_paths[
            "multiplicity"
        ],
    )

    write_tsv(
        protocol,
        output_paths[
            "protocol"
        ],
    )

    input_manifest = pd.DataFrame(
        [
            {
                "file_path":
                    str(
                        path.relative_to(
                            PROJECT
                        )
                    ),
                "file_size_bytes":
                    path.stat().st_size,
                "sha256":
                    sha256_file(path),
            }
            for path in sorted(
                [
                    Path(__file__).resolve(),
                    FREEZE137,
                    *input_paths,
                ],
                key=lambda value:
                    value.as_posix(),
            )
        ]
    )

    input_manifest_path = (
        OUTPUT_META_ROOT
        / "script145_input_manifest.tsv"
    )

    write_tsv(
        input_manifest,
        input_manifest_path,
    )

    manifest_paths = [
        *output_paths.values(),
        input_manifest_path,
    ]

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            manifest_paths,
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(PROJECT)}\n"
            )

    verify_sha_manifest(
        OUTPUT_MANIFEST
    )

    print(
        "===== SCRIPT 145 AMR VOCABULARY "
        "DESIGN AUDIT ====="
    )

    print()
    print(
        "===== TYPE / SUBTYPE / SCOPE / METHOD ====="
    )

    print(
        cross_tabs.to_string(
            index=False
        )
    )

    print()
    print(
        "===== GLOBAL TOKEN SPECIES BREADTH ====="
    )

    print(
        global_breadth_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== PAIRWISE TOKEN SUMMARY ====="
    )

    print(
        pair_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== TOKEN MULTIPLICITY SUMMARY ====="
    )

    print(
        multiplicity.to_string(
            index=False
        )
    )

    print()
    print(
        "===== THRESHOLD SNAPSHOT: "
        "MIN 5, MAX 0.99 ====="
    )

    snapshot = threshold_grid.loc[
        (
            threshold_grid[
                "minimum_development_total_genomes"
            ]
            == 5
        )
        & (
            np.isclose(
                threshold_grid[
                    "maximum_development_pooled_prevalence"
                ].astype(float),
                0.99,
            )
        )
    ].copy()

    print(
        snapshot.to_string(
            index=False
        )
    )

    print()
    print(
        "Verified Script 137 freeze files:",
        len(
            verified_freeze_files
        ),
    )

    print(
        "Pairwise inventory rows:",
        len(pair_inventory),
    )

    print(
        "Threshold-grid rows:",
        len(threshold_grid),
    )

    print(
        "Vocabulary selected: NO"
    )

    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 145 AMR VOCABULARY "
        "DESIGN AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
