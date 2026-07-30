#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT145_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "amr_vocabulary_design_audit_v1/"
      "script145_outputs_sha256.txt"
)

PAIR_INVENTORY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "amr_vocabulary_design_audit_v1/"
      "nested_loso_pairwise_amr_token_inventory_v1.tsv"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "common_cross_species_amr_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "common_cross_species_amr_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT
    / "script146_outputs_sha256.txt"
)

OUTER_DEVELOPMENT_PAIRS = {
    "ec": ("kp", "se"),
    "se": ("kp", "ec"),
    "kp": ("ec", "se"),
}

INCLUDED_CATEGORIES = [
    "nonpoint_amr_candidate",
    "point_mutation_candidate",
]

CATEGORY_TO_BLOCK = {
    "nonpoint_amr_candidate":
        "acquired_or_nonpoint_amr",
    "point_mutation_candidate":
        "shared_point_mutation",
}

CATEGORY_TO_PREFIX = {
    "nonpoint_amr_candidate":
        "amr",
    "point_mutation_candidate":
        "mutation",
}

MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES = 5
MAXIMUM_DEVELOPMENT_POOLED_PREVALENCE = 0.99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
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
                f"Malformed SHA line "
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


def main() -> None:
    verified_script145_outputs = (
        verify_sha_manifest(
            SCRIPT145_MANIFEST
        )
    )

    inventory = read_tsv(
        PAIR_INVENTORY_PATH
    )

    required_columns = {
        "outer_target_code",
        "development_species_a",
        "development_species_b",
        "candidate_category",
        "element_symbol",
        "development_total_genomes",
        "development_total_result_rows",
        "development_pooled_prevalence",
        "present_in_both_development_species",
        "development_species_breadth",
    }

    missing_columns = sorted(
        required_columns.difference(
            inventory.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Missing pair-inventory columns: "
            + "|".join(missing_columns)
        )

    numeric_columns = [
        "development_total_genomes",
        "development_total_result_rows",
        "development_pooled_prevalence",
        "development_species_breadth",
    ]

    for column in numeric_columns:
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="raise",
        )

    inventory[
        "present_in_both_development_species"
    ] = (
        inventory[
            "present_in_both_development_species"
        ]
        .astype(str)
        .str.casefold()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if inventory[
        "present_in_both_development_species"
    ].isna().any():
        raise RuntimeError(
            "Could not parse commonality flag."
        )

    vocabulary_frames: list[
        pd.DataFrame
    ] = []

    rejected_frames: list[
        pd.DataFrame
    ] = []

    registry_records: list[
        dict[str, object]
    ] = []

    output_paths: list[Path] = []

    for outer_target, development_pair in (
        OUTER_DEVELOPMENT_PAIRS.items()
    ):
        species_a, species_b = (
            development_pair
        )

        subset = inventory.loc[
            inventory[
                "outer_target_code"
            ].eq(outer_target)
        ].copy()

        if subset.empty:
            raise RuntimeError(
                f"No inventory rows for "
                f"outer target {outer_target}."
            )

        observed_pairs = {
            (
                str(row[
                    "development_species_a"
                ]),
                str(row[
                    "development_species_b"
                ]),
            )
            for _, row in subset.iterrows()
        }

        if observed_pairs != {
            development_pair
        }:
            raise RuntimeError(
                f"Development-pair mismatch "
                f"for {outer_target}: "
                f"{observed_pairs}"
            )

        genomes_a_column = (
            f"{species_a}_genomes"
        )

        genomes_b_column = (
            f"{species_b}_genomes"
        )

        result_rows_a_column = (
            f"{species_a}_result_rows"
        )

        result_rows_b_column = (
            f"{species_b}_result_rows"
        )

        dynamic_columns = {
            genomes_a_column,
            genomes_b_column,
            result_rows_a_column,
            result_rows_b_column,
        }

        missing_dynamic = sorted(
            dynamic_columns.difference(
                subset.columns
            )
        )

        if missing_dynamic:
            raise RuntimeError(
                f"Missing dynamic columns for "
                f"{outer_target}: "
                + "|".join(missing_dynamic)
            )

        for column in dynamic_columns:
            subset[column] = pd.to_numeric(
                subset[column],
                errors="raise",
            )

        subset[
            "included_category"
        ] = subset[
            "candidate_category"
        ].isin(
            INCLUDED_CATEGORIES
        )

        subset[
            "meets_commonality_rule"
        ] = subset[
            "present_in_both_development_species"
        ].astype(bool)

        subset[
            "meets_species_a_minimum"
        ] = (
            subset[
                genomes_a_column
            ]
            >= (
                MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES
            )
        )

        subset[
            "meets_species_b_minimum"
        ] = (
            subset[
                genomes_b_column
            ]
            >= (
                MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES
            )
        )

        subset[
            "meets_maximum_prevalence"
        ] = (
            subset[
                "development_pooled_prevalence"
            ]
            <= (
                MAXIMUM_DEVELOPMENT_POOLED_PREVALENCE
            )
        )

        subset[
            "selected_for_vocabulary"
        ] = (
            subset[
                "included_category"
            ]
            & subset[
                "meets_commonality_rule"
            ]
            & subset[
                "meets_species_a_minimum"
            ]
            & subset[
                "meets_species_b_minimum"
            ]
            & subset[
                "meets_maximum_prevalence"
            ]
        )

        selected = subset.loc[
            subset[
                "selected_for_vocabulary"
            ]
        ].copy()

        if selected.empty:
            raise RuntimeError(
                f"No selected AMR tokens for "
                f"outer target {outer_target}."
            )

        selected[
            "feature_block"
        ] = selected[
            "candidate_category"
        ].map(
            CATEGORY_TO_BLOCK
        )

        selected[
            "feature_name"
        ] = (
            selected[
                "candidate_category"
            ].map(
                CATEGORY_TO_PREFIX
            )
            + "::"
            + selected[
                "element_symbol"
            ].astype(str)
        )

        if selected[
            "feature_name"
        ].duplicated().any():
            duplicated = selected.loc[
                selected[
                    "feature_name"
                ].duplicated(
                    keep=False
                ),
                "feature_name",
            ].tolist()

            raise RuntimeError(
                "Duplicate AMR feature names: "
                + "|".join(
                    duplicated[:20]
                )
            )

        category_order = {
            category: index
            for index, category
            in enumerate(
                INCLUDED_CATEGORIES
            )
        }

        selected[
            "_category_order"
        ] = selected[
            "candidate_category"
        ].map(
            category_order
        )

        selected = (
            selected.sort_values(
                [
                    "_category_order",
                    "feature_name",
                ]
            )
            .reset_index(drop=True)
        )

        selected[
            "feature_column"
        ] = range(
            len(selected)
        )

        selected[
            "outer_target_code"
        ] = outer_target

        selected[
            "development_species_a"
        ] = species_a

        selected[
            "development_species_b"
        ] = species_b

        selected[
            "minimum_genomes_per_development_species"
        ] = (
            MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES
        )

        selected[
            "maximum_development_pooled_prevalence"
        ] = (
            MAXIMUM_DEVELOPMENT_POOLED_PREVALENCE
        )

        vocabulary_columns = [
            "outer_target_code",
            "development_species_a",
            "development_species_b",
            "feature_column",
            "feature_name",
            "feature_block",
            "candidate_category",
            "element_symbol",
            genomes_a_column,
            genomes_b_column,
            result_rows_a_column,
            result_rows_b_column,
            "development_total_genomes",
            "development_total_result_rows",
            "development_pooled_prevalence",
            "minimum_genomes_per_development_species",
            "maximum_development_pooled_prevalence",
        ]

        vocabulary = selected[
            vocabulary_columns
        ].copy()

        vocabulary_path = (
            OUTPUT_ROOT
            / (
                f"outer_{outer_target}_"
                "common_cross_species_amr_"
                "vocabulary_v1.tsv"
            )
        )

        write_tsv(
            vocabulary,
            vocabulary_path,
        )

        output_paths.append(
            vocabulary_path
        )

        vocabulary_frames.append(
            vocabulary
        )

        rejected = subset.loc[
            ~subset[
                "selected_for_vocabulary"
            ]
        ].copy()

        def rejection_reason(
            row: pd.Series,
        ) -> str:
            reasons: list[str] = []

            if not bool(
                row[
                    "included_category"
                ]
            ):
                reasons.append(
                    "excluded_category"
                )

            if not bool(
                row[
                    "meets_commonality_rule"
                ]
            ):
                reasons.append(
                    "not_present_in_both_"
                    "development_species"
                )

            if not bool(
                row[
                    "meets_species_a_minimum"
                ]
            ):
                reasons.append(
                    f"{species_a}_genomes_below_"
                    f"{MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES}"
                )

            if not bool(
                row[
                    "meets_species_b_minimum"
                ]
            ):
                reasons.append(
                    f"{species_b}_genomes_below_"
                    f"{MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES}"
                )

            if not bool(
                row[
                    "meets_maximum_prevalence"
                ]
            ):
                reasons.append(
                    "pooled_prevalence_above_"
                    f"{MAXIMUM_DEVELOPMENT_POOLED_PREVALENCE}"
                )

            return "|".join(reasons)

        rejected[
            "rejection_reason"
        ] = rejected.apply(
            rejection_reason,
            axis=1,
        )

        rejected_path = (
            TABLE_ROOT
            / (
                f"outer_{outer_target}_"
                "common_cross_species_amr_"
                "rejected_tokens_v1.tsv"
            )
        )

        write_tsv(
            rejected,
            rejected_path,
        )

        output_paths.append(
            rejected_path
        )

        rejected_frames.append(
            rejected
        )

        block_counts = (
            vocabulary.groupby(
                "feature_block"
            )
            .size()
            .to_dict()
        )

        registry_records.append(
            {
                "outer_target_code":
                    outer_target,
                "development_species_a":
                    species_a,
                "development_species_b":
                    species_b,
                "total_features":
                    len(vocabulary),
                "acquired_or_nonpoint_amr_features":
                    int(
                        block_counts.get(
                            "acquired_or_nonpoint_amr",
                            0,
                        )
                    ),
                "shared_point_mutation_features":
                    int(
                        block_counts.get(
                            "shared_point_mutation",
                            0,
                        )
                    ),
                "minimum_genomes_per_development_species":
                    (
                        MINIMUM_GENOMES_PER_DEVELOPMENT_SPECIES
                    ),
                "maximum_development_pooled_prevalence":
                    (
                        MAXIMUM_DEVELOPMENT_POOLED_PREVALENCE
                    ),
                "target_species_used_for_vocabulary":
                    "NO",
                "mic_labels_used":
                    "NO",
                "matrix_generated":
                    "NO",
                "models_trained":
                    "NO",
            }
        )

    combined_vocabulary = pd.concat(
        vocabulary_frames,
        ignore_index=True,
    )

    all_rejected = pd.concat(
        rejected_frames,
        ignore_index=True,
    )

    registry = pd.DataFrame(
        registry_records
    )

    protocol = pd.DataFrame(
        [
            {
                "item":
                    "representation_id",
                "value":
                    "common_cross_species_AMR",
            },
            {
                "item":
                    "vocabulary_scope",
                "value":
                    (
                        "separate vocabulary for "
                        "each outer target, derived "
                        "only from its two development "
                        "species"
                    ),
            },
            {
                "item":
                    "included_amrfinder_categories",
                "value":
                    (
                        "nonpoint_amr_candidate|"
                        "point_mutation_candidate"
                    ),
            },
            {
                "item":
                    "excluded_amrfinder_categories",
                "value":
                    (
                        "other_nonpoint_call "
                        "(STRESS/BIOCIDE)"
                    ),
            },
            {
                "item":
                    "commonality_rule",
                "value":
                    (
                        "exact Element symbol must "
                        "occur in both development "
                        "species"
                    ),
            },
            {
                "item":
                    "minimum_frequency_rule",
                "value":
                    (
                        "at least 5 genomes in each "
                        "development species"
                    ),
            },
            {
                "item":
                    "maximum_frequency_rule",
                "value":
                    (
                        "development-pair pooled "
                        "prevalence at most 0.99"
                    ),
            },
            {
                "item":
                    "feature_value",
                "value":
                    (
                        "binary genome-level presence; "
                        "copy multiplicity ignored"
                    ),
            },
            {
                "item":
                    "mutation_policy",
                "value":
                    (
                        "only exact point-mutation "
                        "symbols shared by both "
                        "development species enter "
                        "the transferable common-AMR "
                        "matrix; all species-specific "
                        "raw mutation calls remain "
                        "preserved in Script 137"
                    ),
            },
            {
                "item":
                    "rationale_for_shared_mutations",
                "value":
                    (
                        "species-prefixed mutation "
                        "dimensions would be untrained "
                        "when transferring from one "
                        "development species to the "
                        "other"
                    ),
            },
            {
                "item":
                    "outer_target_annotation_policy",
                "value":
                    (
                        "outer-target raw annotations "
                        "may later be projected into "
                        "the frozen vocabulary but "
                        "cannot influence feature "
                        "selection"
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
                    "matrices_generated",
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

    combined_vocabulary_path = (
        OUTPUT_ROOT
        / "nested_loso_common_cross_species_"
          "amr_vocabulary_registry_v1.tsv"
    )

    registry_path = (
        OUTPUT_ROOT
        / "nested_loso_common_cross_species_"
          "amr_outer_target_registry_v1.tsv"
    )

    protocol_path = (
        OUTPUT_ROOT
        / "nested_loso_common_cross_species_"
          "amr_protocol_v1.tsv"
    )

    all_rejected_path = (
        TABLE_ROOT
        / "nested_loso_common_cross_species_"
          "amr_all_rejected_tokens_v1.tsv"
    )

    write_tsv(
        combined_vocabulary,
        combined_vocabulary_path,
    )

    write_tsv(
        registry,
        registry_path,
    )

    write_tsv(
        protocol,
        protocol_path,
    )

    write_tsv(
        all_rejected,
        all_rejected_path,
    )

    output_paths.extend(
        [
            combined_vocabulary_path,
            registry_path,
            protocol_path,
            all_rejected_path,
        ]
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
                    SCRIPT145_MANIFEST,
                    PAIR_INVENTORY_PATH,
                ],
                key=lambda value:
                    value.as_posix(),
            )
        ]
    )

    input_manifest_path = (
        OUTPUT_ROOT
        / "script146_input_manifest.tsv"
    )

    write_tsv(
        input_manifest,
        input_manifest_path,
    )

    output_paths.append(
        input_manifest_path
    )

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            set(output_paths),
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
        "===== SCRIPT 146 COMMON-AMR "
        "VOCABULARY PREREGISTRATION ====="
    )

    print(
        registry.to_string(
            index=False
        )
    )

    print()
    print(
        "Total outer-target vocabularies:",
        len(registry),
    )

    print(
        "Total registry feature rows:",
        len(combined_vocabulary),
    )

    print(
        "Verified Script 145 outputs:",
        len(
            verified_script145_outputs
        ),
    )

    print(
        "Matrices generated: NO"
    )

    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 146 COMMON-AMR "
        "VOCABULARIES PREREGISTERED"
    )


if __name__ == "__main__":
    main()
