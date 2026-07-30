#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
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

SCRIPT146 = (
    PROJECT
    / "scripts/"
      "146_preregister_nested_loso_common_amr_vocabularies.py"
)

SCRIPT146_OUTPUT_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "common_cross_species_amr_v1/"
      "script146_outputs_sha256.txt"
)

FREEZE146 = (
    PROJECT
    / "metadata/config_selection/"
      "script146_successful_preregistration_core_sha256.txt"
)

FREEZE137 = (
    PROJECT
    / "metadata/config_selection/"
      "script137_successful_run_core_sha256.txt"
)

RAW_INVENTORY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "amrfinder_full_annotation_audit_v1/"
      "nested_loso_amrfinder_raw_output_inventory_v1.tsv"
)

KMER_ROWS_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_features/canonical_kmer_v1/"
      "nested_loso_all_species_kmer_feature_rows_v1.tsv"
)

VOCABULARY_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "common_cross_species_amr_v1"
)

FEATURE_ROOT = (
    PROJECT
    / "features/genome_representation/nested_loso_v1/"
      "common_cross_species_amr"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "common_cross_species_amr_matrix_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "common_cross_species_amr_matrix_v1"
)

OUTPUT_MANIFEST = (
    METADATA_ROOT
    / "script147_outputs_sha256.txt"
)

EXPECTED_GENOMES = 21_394

EXPECTED_SPECIES_COUNTS = {
    "kp": 5_602,
    "ec": 6_673,
    "se": 9_119,
}

EXPECTED_FEATURE_COUNTS = {
    "ec": 92,
    "se": 108,
    "kp": 91,
}

EXPECTED_COLUMNS = [
    "Protein id",
    "Contig id",
    "Start",
    "Stop",
    "Strand",
    "Element symbol",
    "Element name",
    "Scope",
    "Type",
    "Subtype",
    "Class",
    "Subclass",
    "Method",
    "Target length",
    "Reference sequence length",
    "% Coverage of reference",
    "% Identity to reference",
    "Alignment length",
    "Closest reference accession",
    "Closest reference name",
    "HMM accession",
    "HMM description",
]

POINT_SUBTYPES = {
    "POINT",
    "POINT_DISRUPT",
}


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


def write_sha_manifest(
    paths: list[Path],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            set(paths),
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(PROJECT)}\n"
            )


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


def parse_integral_value(
    value: object,
    field_name: str,
) -> int:
    try:
        numeric = float(
            str(value).strip()
        )
    except ValueError as error:
        raise RuntimeError(
            f"Could not parse integral value "
            f"for {field_name}: {value!r}"
        ) from error

    if not np.isfinite(numeric):
        raise RuntimeError(
            f"Non-finite value for "
            f"{field_name}: {value!r}"
        )

    rounded = round(numeric)

    if not np.isclose(
        numeric,
        rounded,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            f"Non-integral value for "
            f"{field_name}: {value!r}"
        )

    return int(rounded)


def detect_column(
    frame: pd.DataFrame,
    aliases: list[str],
    field_name: str,
) -> str:
    lookup = {
        str(column).casefold():
            str(column)
        for column in frame.columns
    }

    matches = [
        lookup[alias.casefold()]
        for alias in aliases
        if alias.casefold() in lookup
    ]

    matches = list(dict.fromkeys(matches))

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely detect "
            f"{field_name}; matches={matches}; "
            f"columns={frame.columns.tolist()}"
        )

    return matches[0]


def category_for_row(
    subtype: str,
    type_value: str,
) -> str:
    subtype_norm = str(
        subtype
    ).strip().upper()

    type_norm = str(
        type_value
    ).strip().upper()

    if subtype_norm in POINT_SUBTYPES:
        return "point_mutation_candidate"

    if type_norm == "AMR":
        return "nonpoint_amr_candidate"

    return "other_nonpoint_call"


def save_npy_atomic(
    array: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("wb") as handle:
        np.save(
            handle,
            array,
            allow_pickle=False,
        )

    temporary.replace(path)


def freeze_script146() -> list[Path]:
    for path in [
        SCRIPT146,
        SCRIPT146_OUTPUT_MANIFEST,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_paths = verify_sha_manifest(
        SCRIPT146_OUTPUT_MANIFEST
    )

    freeze_paths = [
        SCRIPT146,
        SCRIPT146_OUTPUT_MANIFEST,
        *output_paths,
    ]

    write_sha_manifest(
        freeze_paths,
        FREEZE146,
    )

    verify_sha_manifest(
        FREEZE146
    )

    return sorted(
        set(freeze_paths),
        key=lambda value:
            value.as_posix(),
    )


def main() -> None:
    verified_script137_freeze = (
        verify_sha_manifest(
            FREEZE137
        )
    )

    frozen_script146_paths = (
        freeze_script146()
    )

    for path in [
        RAW_INVENTORY_PATH,
        KMER_ROWS_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_inventory = read_tsv(
        RAW_INVENTORY_PATH
    )

    required_inventory_columns = {
        "species_code",
        "genome_id",
        "raw_output_path",
        "result_rows",
        "sha256",
    }

    missing_inventory_columns = sorted(
        required_inventory_columns.difference(
            raw_inventory.columns
        )
    )

    if missing_inventory_columns:
        raise RuntimeError(
            "Missing raw-inventory columns: "
            + "|".join(
                missing_inventory_columns
            )
        )

    if len(raw_inventory) != EXPECTED_GENOMES:
        raise RuntimeError(
            "Raw inventory row-count mismatch: "
            f"{len(raw_inventory)}"
        )

    if raw_inventory[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate genome IDs in raw inventory."
        )

    observed_species_counts = (
        raw_inventory[
            "species_code"
        ]
        .value_counts()
        .to_dict()
    )

    if observed_species_counts != (
        EXPECTED_SPECIES_COUNTS
    ):
        raise RuntimeError(
            "Raw-inventory species counts "
            f"mismatch: {observed_species_counts}"
        )

    kmer_rows = read_tsv(
        KMER_ROWS_PATH
    )

    if len(kmer_rows) != EXPECTED_GENOMES:
        raise RuntimeError(
            "K-mer row-registry count mismatch: "
            f"{len(kmer_rows)}"
        )

    genome_column = detect_column(
        kmer_rows,
        [
            "genome_id",
            "genome",
        ],
        "genome ID",
    )

    row_column = detect_column(
        kmer_rows,
        [
            "genome_feature_row",
            "feature_row",
            "row_index",
            "matrix_row",
        ],
        "genome feature row",
    )

    species_column = detect_column(
        kmer_rows,
        [
            "species_code",
            "development_species_code",
        ],
        "species code",
    )

    kmer_rows[row_column] = pd.to_numeric(
        kmer_rows[row_column],
        errors="raise",
    ).astype(np.int64)

    expected_rows = np.arange(
        EXPECTED_GENOMES,
        dtype=np.int64,
    )

    observed_rows = (
        kmer_rows[row_column]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        observed_rows,
        expected_rows,
    ):
        raise RuntimeError(
            "K-mer row registry is not "
            "contiguous 0..21393 in file order."
        )

    if kmer_rows[
        genome_column
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate genome IDs in K-mer "
            "row registry."
        )

    row_species_counts = (
        kmer_rows[
            species_column
        ]
        .value_counts()
        .to_dict()
    )

    if row_species_counts != (
        EXPECTED_SPECIES_COUNTS
    ):
        raise RuntimeError(
            "K-mer row-registry species counts "
            f"mismatch: {row_species_counts}"
        )

    inventory_by_genome = (
        raw_inventory.set_index(
            "genome_id"
        )
    )

    if not inventory_by_genome.index.is_unique:
        raise RuntimeError(
            "Duplicate genome IDs after "
            "indexing raw inventory."
        )

    row_genomes = (
        kmer_rows[
            genome_column
        ]
        .astype(str)
        .tolist()
    )

    missing_inventory_genomes = sorted(
        set(row_genomes).difference(
            inventory_by_genome.index
        )
    )

    extra_inventory_genomes = sorted(
        set(
            inventory_by_genome.index
        ).difference(row_genomes)
    )

    if (
        missing_inventory_genomes
        or extra_inventory_genomes
    ):
        raise RuntimeError(
            "Raw inventory and K-mer rows "
            "do not contain the same genomes."
        )

    row_species = (
        kmer_rows[
            species_column
        ]
        .astype(str)
        .to_numpy()
    )

    inventory_species_aligned = (
        inventory_by_genome.loc[
            row_genomes,
            "species_code",
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        row_species,
        inventory_species_aligned,
    ):
        raise RuntimeError(
            "Species mismatch between raw "
            "inventory and K-mer row registry."
        )

    vocabulary_by_outer: dict[
        str,
        pd.DataFrame,
    ] = {}

    lookup_by_outer: dict[
        str,
        dict[tuple[str, str], int],
    ] = {}

    for outer_target, expected_features in (
        EXPECTED_FEATURE_COUNTS.items()
    ):
        vocabulary_path = (
            VOCABULARY_ROOT
            / (
                f"outer_{outer_target}_"
                "common_cross_species_amr_"
                "vocabulary_v1.tsv"
            )
        )

        vocabulary = read_tsv(
            vocabulary_path
        )

        if len(vocabulary) != expected_features:
            raise RuntimeError(
                f"Outer {outer_target} vocabulary "
                f"count mismatch: "
                f"{len(vocabulary)} != "
                f"{expected_features}"
            )

        required_vocabulary_columns = {
            "feature_column",
            "feature_name",
            "candidate_category",
            "element_symbol",
            "development_species_a",
            "development_species_b",
            "minimum_genomes_per_development_species",
            "maximum_development_pooled_prevalence",
        }

        missing_vocabulary_columns = sorted(
            required_vocabulary_columns
            .difference(
                vocabulary.columns
            )
        )

        if missing_vocabulary_columns:
            raise RuntimeError(
                f"Outer {outer_target} vocabulary "
                "missing columns: "
                + "|".join(
                    missing_vocabulary_columns
                )
            )

        vocabulary[
            "feature_column"
        ] = pd.to_numeric(
            vocabulary[
                "feature_column"
            ],
            errors="raise",
        ).astype(np.int64)

        if not np.array_equal(
            vocabulary[
                "feature_column"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                expected_features,
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                f"Outer {outer_target} feature "
                "columns are not contiguous."
            )

        if vocabulary[
            "feature_name"
        ].duplicated().any():
            raise RuntimeError(
                f"Outer {outer_target} duplicate "
                "feature names."
            )

        key_tuples = list(
            zip(
                vocabulary[
                    "candidate_category"
                ].astype(str),
                vocabulary[
                    "element_symbol"
                ].astype(str),
            )
        )

        if len(set(key_tuples)) != len(
            key_tuples
        ):
            raise RuntimeError(
                f"Outer {outer_target} duplicate "
                "category/symbol keys."
            )

        lookup_by_outer[
            outer_target
        ] = {
            key: int(column)
            for key, column
            in zip(
                key_tuples,
                vocabulary[
                    "feature_column"
                ].tolist(),
            )
        }

        vocabulary_by_outer[
            outer_target
        ] = vocabulary

    matrices = {
        outer_target:
            np.zeros(
                (
                    EXPECTED_GENOMES,
                    expected_features,
                ),
                dtype=np.uint8,
            )
        for outer_target, expected_features
        in EXPECTED_FEATURE_COUNTS.items()
    }

    raw_rows_read = 0
    raw_files_verified = 0

    for row_index, genome_id in enumerate(
        row_genomes
    ):
        inventory_row = (
            inventory_by_genome.loc[
                genome_id
            ]
        )

        raw_path = project_path(
            str(
                inventory_row[
                    "raw_output_path"
                ]
            )
        )

        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        expected_raw_sha = str(
            inventory_row["sha256"]
        )

        observed_raw_sha = sha256_file(
            raw_path
        )

        if observed_raw_sha != expected_raw_sha:
            raise RuntimeError(
                f"Raw-output SHA mismatch: "
                f"{raw_path}"
            )

        raw_files_verified += 1

        raw_frame = pd.read_csv(
            raw_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

        if raw_frame.columns.tolist() != (
            EXPECTED_COLUMNS
        ):
            raise RuntimeError(
                "AMRFinder schema mismatch: "
                f"{raw_path}"
            )

        expected_result_rows = int(
            inventory_row[
                "result_rows"
            ]
        )

        if len(raw_frame) != expected_result_rows:
            raise RuntimeError(
                f"Raw row-count mismatch: "
                f"{raw_path}"
            )

        raw_rows_read += len(raw_frame)

        if raw_frame.empty:
            continue

        tokens_for_genome: set[
            tuple[str, str]
        ] = set()

        for record in raw_frame[
            [
                "Element symbol",
                "Type",
                "Subtype",
            ]
        ].itertuples(
            index=False,
            name=None,
        ):
            element_symbol, type_value, subtype = (
                record
            )

            element_symbol = str(
                element_symbol
            ).strip()

            if not element_symbol:
                raise RuntimeError(
                    f"Blank Element symbol: "
                    f"{raw_path}"
                )

            category = category_for_row(
                subtype,
                type_value,
            )

            tokens_for_genome.add(
                (
                    category,
                    element_symbol,
                )
            )

        for outer_target, lookup in (
            lookup_by_outer.items()
        ):
            columns = [
                lookup[token]
                for token in tokens_for_genome
                if token in lookup
            ]

            if columns:
                matrices[
                    outer_target
                ][
                    row_index,
                    np.asarray(
                        columns,
                        dtype=np.int64,
                    ),
                ] = 1

    if raw_files_verified != EXPECTED_GENOMES:
        raise RuntimeError(
            "Not all raw files were verified."
        )

    if raw_rows_read != 180_739:
        raise RuntimeError(
            "Total AMRFinder rows read "
            f"mismatch: {raw_rows_read}"
        )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_registry_records: list[
        dict[str, object]
    ] = []

    species_audit_records: list[
        dict[str, object]
    ] = []

    prevalence_records: list[
        dict[str, object]
    ] = []

    output_paths: list[Path] = []

    for outer_target in [
        "ec",
        "se",
        "kp",
    ]:
        matrix = matrices[
            outer_target
        ]

        vocabulary = vocabulary_by_outer[
            outer_target
        ].copy()

        expected_shape = (
            EXPECTED_GENOMES,
            EXPECTED_FEATURE_COUNTS[
                outer_target
            ],
        )

        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"Outer {outer_target} matrix "
                f"shape mismatch: {matrix.shape}"
            )

        unique_values = set(
            np.unique(matrix).tolist()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise RuntimeError(
                f"Outer {outer_target} matrix "
                f"is not binary: {unique_values}"
            )

        matrix_path = (
            FEATURE_ROOT
            / (
                f"outer_{outer_target}_"
                "common_cross_species_amr_"
                "binary_v1.npy"
            )
        )

        save_npy_atomic(
            matrix,
            matrix_path,
        )

        reloaded = np.load(
            matrix_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if (
            reloaded.shape != matrix.shape
            or reloaded.dtype != np.uint8
        ):
            raise RuntimeError(
                f"Saved matrix validation "
                f"failed: {matrix_path}"
            )

        matrix_registry_records.append(
            {
                "outer_target_code":
                    outer_target,
                "matrix_path":
                    str(
                        matrix_path.relative_to(
                            PROJECT
                        )
                    ),
                "rows":
                    matrix.shape[0],
                "columns":
                    matrix.shape[1],
                "dtype":
                    str(matrix.dtype),
                "nonzero_values":
                    int(
                        np.count_nonzero(
                            matrix
                        )
                    ),
                "rows_with_any_feature":
                    int(
                        (
                            matrix.sum(
                                axis=1
                            )
                            > 0
                        ).sum()
                    ),
                "active_feature_columns":
                    int(
                        (
                            matrix.sum(
                                axis=0
                            )
                            > 0
                        ).sum()
                    ),
                "sha256":
                    sha256_file(
                        matrix_path
                    ),
            }
        )

        output_paths.append(
            matrix_path
        )

        for species_code in [
            "kp",
            "ec",
            "se",
        ]:
            mask = (
                row_species
                == species_code
            )

            submatrix = matrix[
                mask
            ]

            row_sums = submatrix.sum(
                axis=1
            )

            species_audit_records.append(
                {
                    "outer_target_code":
                        outer_target,
                    "species_code":
                        species_code,
                    "genomes":
                        int(mask.sum()),
                    "rows_with_any_feature":
                        int(
                            (
                                row_sums > 0
                            ).sum()
                        ),
                    "rows_with_no_features":
                        int(
                            (
                                row_sums == 0
                            ).sum()
                        ),
                    "mean_features_per_genome":
                        float(
                            row_sums.mean()
                        ),
                    "median_features_per_genome":
                        float(
                            np.median(
                                row_sums
                            )
                        ),
                    "maximum_features_per_genome":
                        int(
                            row_sums.max()
                        ),
                    "active_feature_columns":
                        int(
                            (
                                submatrix.sum(
                                    axis=0
                                )
                                > 0
                            ).sum()
                        ),
                }
            )

        for vocabulary_row in (
            vocabulary.itertuples(
                index=False
            )
        ):
            feature_column = int(
                vocabulary_row.feature_column
            )

            record: dict[
                str,
                object
            ] = {
                "outer_target_code":
                    outer_target,
                "feature_column":
                    feature_column,
                "feature_name":
                    str(
                        vocabulary_row.feature_name
                    ),
                "candidate_category":
                    str(
                        vocabulary_row.candidate_category
                    ),
                "element_symbol":
                    str(
                        vocabulary_row.element_symbol
                    ),
            }

            for species_code in [
                "kp",
                "ec",
                "se",
            ]:
                mask = (
                    row_species
                    == species_code
                )

                present_genomes = int(
                    matrix[
                        mask,
                        feature_column,
                    ].sum()
                )

                record[
                    f"{species_code}_genomes"
                ] = present_genomes

                record[
                    f"{species_code}_prevalence"
                ] = (
                    present_genomes
                    / EXPECTED_SPECIES_COUNTS[
                        species_code
                    ]
                )

            development_a = str(
                vocabulary_row
                .development_species_a
            )

            development_b = str(
                vocabulary_row
                .development_species_b
            )

            registered_a_column = (
                f"{development_a}_genomes"
            )

            registered_b_column = (
                f"{development_b}_genomes"
            )

            registered_a = (
                parse_integral_value(
                    getattr(
                        vocabulary_row,
                        registered_a_column,
                    ),
                    (
                        f"outer_{outer_target}:"
                        f"{record['feature_name']}:"
                        f"{registered_a_column}"
                    ),
                )
            )

            registered_b = (
                parse_integral_value(
                    getattr(
                        vocabulary_row,
                        registered_b_column,
                    ),
                    (
                        f"outer_{outer_target}:"
                        f"{record['feature_name']}:"
                        f"{registered_b_column}"
                    ),
                )
            )

            if (
                record[
                    registered_a_column
                ]
                != registered_a
                or record[
                    registered_b_column
                ]
                != registered_b
            ):
                raise RuntimeError(
                    "Matrix prevalence does not "
                    "reproduce preregistered "
                    "development counts: "
                    f"outer={outer_target}, "
                    f"feature="
                    f"{record['feature_name']}"
                )

            if (
                registered_a < 5
                or registered_b < 5
            ):
                raise RuntimeError(
                    "Selected vocabulary feature "
                    "violates minimum frequency."
                )

            prevalence_records.append(
                record
            )

        if (
            matrix.sum(axis=0) == 0
        ).any():
            raise RuntimeError(
                f"Outer {outer_target} contains "
                "globally zero feature columns."
            )

    row_registry = pd.DataFrame(
        {
            "genome_feature_row":
                expected_rows,
            "genome_id":
                row_genomes,
            "species_code":
                row_species,
        }
    )

    matrix_registry = pd.DataFrame(
        matrix_registry_records
    )

    species_audit = pd.DataFrame(
        species_audit_records
    )

    prevalence_audit = pd.DataFrame(
        prevalence_records
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
                    "row_alignment",
                "value":
                    (
                        "exact Script 136 canonical "
                        "k-mer genome row order"
                    ),
            },
            {
                "item":
                    "matrix_scope",
                "value":
                    (
                        "one frozen vocabulary and "
                        "one 21394-row matrix per "
                        "outer target"
                    ),
            },
            {
                "item":
                    "feature_value",
                "value":
                    (
                        "binary exact Element-symbol "
                        "presence per genome"
                    ),
            },
            {
                "item":
                    "copy_multiplicity",
                "value":
                    "collapsed to presence",
            },
            {
                "item":
                    "development_vocabulary_rule",
                "value":
                    (
                        "token present in at least "
                        "5 genomes in each development "
                        "species; pooled prevalence "
                        "at most 0.99"
                    ),
            },
            {
                "item":
                    "included_categories",
                "value":
                    (
                        "nonpoint_amr_candidate|"
                        "point_mutation_candidate"
                    ),
            },
            {
                "item":
                    "excluded_categories",
                "value":
                    "other_nonpoint_call",
            },
            {
                "item":
                    "outer_target_projection",
                "value":
                    (
                        "outer-target raw annotations "
                        "projected into frozen "
                        "development-derived vocabulary"
                    ),
            },
            {
                "item":
                    "outer_target_vocabulary_influence",
                "value":
                    "none",
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

    row_registry_path = (
        METADATA_ROOT
        / "nested_loso_common_cross_species_"
          "amr_feature_rows_v1.tsv"
    )

    matrix_registry_path = (
        METADATA_ROOT
        / "nested_loso_common_cross_species_"
          "amr_matrix_registry_v1.tsv"
    )

    protocol_path = (
        METADATA_ROOT
        / "nested_loso_common_cross_species_"
          "amr_matrix_protocol_v1.tsv"
    )

    species_audit_path = (
        TABLE_ROOT
        / "nested_loso_common_cross_species_"
          "amr_species_matrix_audit_v1.tsv"
    )

    prevalence_audit_path = (
        TABLE_ROOT
        / "nested_loso_common_cross_species_"
          "amr_feature_prevalence_audit_v1.tsv"
    )

    write_tsv(
        row_registry,
        row_registry_path,
    )

    write_tsv(
        matrix_registry,
        matrix_registry_path,
    )

    write_tsv(
        protocol,
        protocol_path,
    )

    write_tsv(
        species_audit,
        species_audit_path,
    )

    write_tsv(
        prevalence_audit,
        prevalence_audit_path,
    )

    output_paths.extend(
        [
            row_registry_path,
            matrix_registry_path,
            protocol_path,
            species_audit_path,
            prevalence_audit_path,
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
                    FREEZE137,
                    FREEZE146,
                    RAW_INVENTORY_PATH,
                    KMER_ROWS_PATH,
                    *[
                        VOCABULARY_ROOT
                        / (
                            f"outer_{outer_target}_"
                            "common_cross_species_amr_"
                            "vocabulary_v1.tsv"
                        )
                        for outer_target in [
                            "ec",
                            "se",
                            "kp",
                        ]
                    ],
                ],
                key=lambda value:
                    value.as_posix(),
            )
        ]
    )

    input_manifest_path = (
        METADATA_ROOT
        / "script147_input_manifest.tsv"
    )

    write_tsv(
        input_manifest,
        input_manifest_path,
    )

    output_paths.append(
        input_manifest_path
    )

    write_sha_manifest(
        output_paths,
        OUTPUT_MANIFEST,
    )

    verify_sha_manifest(
        OUTPUT_MANIFEST
    )

    print(
        "===== SCRIPT 147 COMMON-AMR "
        "MATRIX GENERATION ====="
    )

    print()
    print(
        "===== MATRIX REGISTRY ====="
    )

    print(
        matrix_registry.to_string(
            index=False
        )
    )

    print()
    print(
        "===== SPECIES MATRIX AUDIT ====="
    )

    print(
        species_audit.to_string(
            index=False
        )
    )

    print()
    print(
        "Raw files SHA-verified:",
        raw_files_verified,
    )

    print(
        "Raw AMRFinder rows read:",
        raw_rows_read,
    )

    print(
        "Feature rows:",
        len(row_registry),
    )

    print(
        "Script 146 frozen files:",
        len(
            frozen_script146_paths
        ),
    )

    print(
        "Script 137 frozen files verified:",
        len(
            verified_script137_freeze
        ),
    )

    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 147 COMMON-AMR "
        "MATRICES COMPLETE"
    )


if __name__ == "__main__":
    main()
