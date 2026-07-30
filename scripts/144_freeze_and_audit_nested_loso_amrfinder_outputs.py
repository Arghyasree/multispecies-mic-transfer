#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from collections import Counter
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

SCRIPT137 = (
    PROJECT
    / "scripts/"
      "137_run_nested_loso_amrfinder_full_annotation.py"
)

STATUS137 = (
    PROJECT
    / "results/logs/config_selection/nested_loso_v1/"
      "script137_amrfinder_full.exit_status.txt"
)

STDOUT137 = (
    PROJECT
    / "results/logs/config_selection/nested_loso_v1/"
      "script137_amrfinder_full.stdout.log"
)

STDERR137 = (
    PROJECT
    / "results/logs/config_selection/nested_loso_v1/"
      "script137_amrfinder_full.stderr.log"
)

RAW_ROOT = (
    PROJECT
    / "results/amr/nested_loso_v1/full_v1"
)

AUDIT_META_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "amrfinder_full_annotation_audit_v1"
)

AUDIT_TABLE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "amrfinder_full_annotation_audit_v1"
)

FREEZE137 = (
    PROJECT
    / "metadata/config_selection/"
      "script137_successful_run_core_sha256.txt"
)

OUTPUT_MANIFEST144 = (
    AUDIT_META_ROOT
    / "script144_outputs_sha256.txt"
)

EXPECTED_COUNTS = {
    "kp": 5_602,
    "ec": 6_673,
    "se": 9_119,
}

SPECIES_NAMES = {
    "kp": "Klebsiella pneumoniae",
    "ec": "Escherichia coli",
    "se": "Salmonella enterica",
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


def verify_sha_manifest(
    manifest_path: Path,
) -> list[Path]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    verified: list[Path] = []

    for line_number, line in enumerate(
        manifest_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed SHA line "
                f"{line_number}: {manifest_path}"
            )

        expected, path_text = parts
        path = project_path(path_text)

        if not path.is_file():
            raise FileNotFoundError(path)

        observed = sha256_file(path)

        if observed != expected:
            raise RuntimeError(
                f"SHA mismatch: {path}"
            )

        verified.append(path)

    if not verified:
        raise RuntimeError(
            f"Empty SHA manifest: {manifest_path}"
        )

    return verified


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


def discover_script137_manifest() -> Path:
    candidates = sorted(
        {
            path.resolve()
            for root in [
                PROJECT / "metadata",
                PROJECT / "results",
            ]
            if root.is_dir()
            for path in root.rglob(
                "script137_outputs_sha256.txt"
            )
        },
        key=lambda path:
            path.as_posix(),
    )

    if len(candidates) != 1:
        rendered = "\n".join(
            str(path)
            for path in candidates
        )

        raise RuntimeError(
            "Expected exactly one "
            "script137_outputs_sha256.txt; "
            f"observed {len(candidates)}.\n"
            f"{rendered}"
        )

    return candidates[0]


def genome_id_from_path(path: Path) -> str:
    suffix = ".amrfinder.tsv"

    if not path.name.endswith(suffix):
        raise RuntimeError(
            f"Unexpected AMRFinder filename: "
            f"{path.name}"
        )

    genome_id = path.name[
        :-len(suffix)
    ]

    if not genome_id:
        raise RuntimeError(
            f"Blank genome ID in {path}"
        )

    return genome_id


def normalized(value: object) -> str:
    return str(value).strip()


def category_for_row(row: dict[str, str]) -> str:
    subtype = normalized(
        row.get("Subtype", "")
    ).upper()

    type_value = normalized(
        row.get("Type", "")
    ).upper()

    if subtype in POINT_SUBTYPES:
        return "point_mutation_candidate"

    if type_value == "AMR":
        return "nonpoint_amr_candidate"

    return "other_nonpoint_call"


def main() -> None:
    for path in [
        SCRIPT137,
        STATUS137,
        STDOUT137,
        STDERR137,
        RAW_ROOT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    recorded_status = (
        STATUS137.read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if recorded_status != "0":
        raise RuntimeError(
            "Script 137 recorded exit status "
            f"is {recorded_status!r}, not '0'."
        )

    script137_manifest = (
        discover_script137_manifest()
    )

    verified_script137_outputs = (
        verify_sha_manifest(
            script137_manifest
        )
    )

    raw_files_by_species: dict[
        str,
        list[Path],
    ] = {}

    all_genome_ids: set[str] = set()

    for species_code, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        species_dir = (
            RAW_ROOT / species_code
        )

        if not species_dir.is_dir():
            raise FileNotFoundError(
                species_dir
            )

        paths = sorted(
            species_dir.glob(
                "*.amrfinder.tsv"
            ),
            key=lambda path:
                path.name,
        )

        if len(paths) != expected_count:
            raise RuntimeError(
                f"{species_code} raw-output count "
                f"mismatch: {len(paths)} != "
                f"{expected_count}"
            )

        genome_ids = [
            genome_id_from_path(path)
            for path in paths
        ]

        if len(set(genome_ids)) != len(
            genome_ids
        ):
            raise RuntimeError(
                f"Duplicate genome filenames "
                f"within {species_code}."
            )

        overlap = (
            all_genome_ids
            .intersection(genome_ids)
        )

        if overlap:
            raise RuntimeError(
                "Genome IDs occur in multiple "
                "species directories: "
                + "|".join(
                    sorted(overlap)[:20]
                )
            )

        all_genome_ids.update(
            genome_ids
        )

        raw_files_by_species[
            species_code
        ] = paths

    if len(all_genome_ids) != sum(
        EXPECTED_COUNTS.values()
    ):
        raise RuntimeError(
            "Unexpected global unique-genome "
            "count."
        )

    inventory_records: list[
        dict[str, object]
    ] = []

    call_records: list[
        dict[str, object]
    ] = []

    type_counter = Counter()
    subtype_counter = Counter()
    scope_counter = Counter()
    method_counter = Counter()
    class_counter = Counter()
    category_counter = Counter()

    exact_duplicate_rows_total = 0
    missing_element_symbol_rows = 0
    empty_result_files = 0

    for species_code in [
        "kp",
        "ec",
        "se",
    ]:
        for path in (
            raw_files_by_species[
                species_code
            ]
        ):
            genome_id = (
                genome_id_from_path(path)
            )

            frame = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                low_memory=False,
            )

            observed_columns = (
                frame.columns.tolist()
            )

            if observed_columns != EXPECTED_COLUMNS:
                raise RuntimeError(
                    "AMRFinder schema mismatch: "
                    f"{path}\n"
                    f"Observed: "
                    f"{observed_columns}"
                )

            exact_duplicates = int(
                frame.duplicated().sum()
            )

            exact_duplicate_rows_total += (
                exact_duplicates
            )

            if frame.empty:
                empty_result_files += 1

            inventory_records.append(
                {
                    "species_code":
                        species_code,
                    "species":
                        SPECIES_NAMES[
                            species_code
                        ],
                    "genome_id":
                        genome_id,
                    "raw_output_path":
                        str(
                            path.relative_to(
                                PROJECT
                            )
                        ),
                    "result_rows":
                        len(frame),
                    "exact_duplicate_rows":
                        exact_duplicates,
                    "file_size_bytes":
                        path.stat().st_size,
                    "sha256":
                        sha256_file(path),
                }
            )

            for record in frame.to_dict(
                orient="records"
            ):
                clean = {
                    column:
                        normalized(
                            record[column]
                        )
                    for column
                    in EXPECTED_COLUMNS
                }

                category = (
                    category_for_row(clean)
                )

                element_symbol = clean[
                    "Element symbol"
                ]

                if not element_symbol:
                    missing_element_symbol_rows += 1

                type_counter[
                    (
                        species_code,
                        clean["Type"],
                    )
                ] += 1

                subtype_counter[
                    (
                        species_code,
                        clean["Subtype"],
                    )
                ] += 1

                scope_counter[
                    (
                        species_code,
                        clean["Scope"],
                    )
                ] += 1

                method_counter[
                    (
                        species_code,
                        clean["Method"],
                    )
                ] += 1

                class_counter[
                    (
                        species_code,
                        clean["Class"],
                        clean["Subclass"],
                    )
                ] += 1

                category_counter[
                    (
                        species_code,
                        category,
                    )
                ] += 1

                call_records.append(
                    {
                        "species_code":
                            species_code,
                        "species":
                            SPECIES_NAMES[
                                species_code
                            ],
                        "genome_id":
                            genome_id,
                        "candidate_category":
                            category,
                        **clean,
                    }
                )

    inventory = pd.DataFrame(
        inventory_records
    )

    calls = pd.DataFrame(
        call_records
    )

    if len(inventory) != 21_394:
        raise RuntimeError(
            "Inventory genome count mismatch: "
            f"{len(inventory)}"
        )

    if len(calls) != 180_739:
        raise RuntimeError(
            "Total result-row mismatch: "
            f"{len(calls)} != 180739"
        )

    species_summary = (
        inventory.groupby(
            [
                "species_code",
                "species",
            ],
            dropna=False,
        )
        .agg(
            genomes=(
                "genome_id",
                "nunique",
            ),
            result_rows=(
                "result_rows",
                "sum",
            ),
            genomes_with_zero_calls=(
                "result_rows",
                lambda values:
                    int(
                        (
                            pd.to_numeric(
                                values,
                                errors="raise",
                            )
                            == 0
                        ).sum()
                    ),
            ),
            median_result_rows=(
                "result_rows",
                "median",
            ),
            maximum_result_rows=(
                "result_rows",
                "max",
            ),
            exact_duplicate_rows=(
                "exact_duplicate_rows",
                "sum",
            ),
            total_output_bytes=(
                "file_size_bytes",
                "sum",
            ),
        )
        .reset_index()
    )

    type_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "type":
                    key[1],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                type_counter.items()
            )
        ]
    )

    subtype_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "subtype":
                    key[1],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                subtype_counter.items()
            )
        ]
    )

    scope_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "scope":
                    key[1],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                scope_counter.items()
            )
        ]
    )

    method_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "method":
                    key[1],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                method_counter.items()
            )
        ]
    )

    class_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "class":
                    key[1],
                "subclass":
                    key[2],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                class_counter.items()
            )
        ]
    )

    category_summary = pd.DataFrame(
        [
            {
                "species_code":
                    key[0],
                "candidate_category":
                    key[1],
                "result_rows":
                    count,
            }
            for key, count
            in sorted(
                category_counter.items()
            )
        ]
    )

    calls[
        "element_symbol_nonblank"
    ] = calls[
        "Element symbol"
    ].astype(str).str.len().gt(0)

    token_prevalence = (
        calls.loc[
            calls[
                "element_symbol_nonblank"
            ]
        ]
        .groupby(
            [
                "candidate_category",
                "species_code",
                "Element symbol",
            ],
            dropna=False,
        )
        .agg(
            result_rows=(
                "genome_id",
                "size",
            ),
            genomes=(
                "genome_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    species_denominators = {
        code: count
        for code, count
        in EXPECTED_COUNTS.items()
    }

    token_prevalence[
        "species_genomes"
    ] = token_prevalence[
        "species_code"
    ].map(
        species_denominators
    )

    token_prevalence[
        "genome_prevalence"
    ] = (
        token_prevalence[
            "genomes"
        ]
        / token_prevalence[
            "species_genomes"
        ]
    )

    token_species_breadth = (
        token_prevalence.groupby(
            [
                "candidate_category",
                "Element symbol",
            ],
            dropna=False,
        )
        .agg(
            species_count=(
                "species_code",
                "nunique",
            ),
            species_codes=(
                "species_code",
                lambda values:
                    "|".join(
                        sorted(
                            set(
                                str(value)
                                for value
                                in values
                            )
                        )
                    ),
            ),
            total_genomes=(
                "genomes",
                "sum",
            ),
            total_result_rows=(
                "result_rows",
                "sum",
            ),
        )
        .reset_index()
    )

    per_genome_token = (
        calls.loc[
            calls[
                "element_symbol_nonblank"
            ],
            [
                "species_code",
                "genome_id",
                "candidate_category",
                "Element symbol",
            ],
        ]
        .groupby(
            [
                "species_code",
                "genome_id",
                "candidate_category",
                "Element symbol",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="call_multiplicity"
        )
    )

    multiplicity_summary = (
        per_genome_token.groupby(
            [
                "species_code",
                "candidate_category",
            ],
            dropna=False,
        )
        .agg(
            genome_token_pairs=(
                "call_multiplicity",
                "size",
            ),
            pairs_with_multiple_calls=(
                "call_multiplicity",
                lambda values:
                    int(
                        (
                            pd.to_numeric(
                                values,
                                errors="raise",
                            )
                            > 1
                        ).sum()
                    ),
            ),
            maximum_call_multiplicity=(
                "call_multiplicity",
                "max",
            ),
        )
        .reset_index()
    )

    protocol = pd.DataFrame(
        [
            {
                "item":
                    "audit_scope",
                "value":
                    (
                        "all raw AMRFinderPlus outputs "
                        "from Script 137"
                    ),
            },
            {
                "item":
                    "software_output_schema_columns",
                "value":
                    len(EXPECTED_COLUMNS),
            },
            {
                "item":
                    "point_mutation_candidate_rule",
                "value":
                    (
                        "Subtype exactly POINT or "
                        "POINT_DISRUPT"
                    ),
            },
            {
                "item":
                    "nonpoint_amr_candidate_rule",
                "value":
                    (
                        "Type exactly AMR and Subtype "
                        "not POINT/POINT_DISRUPT"
                    ),
            },
            {
                "item":
                    "other_nonpoint_rule",
                "value":
                    (
                        "all remaining nonpoint calls; "
                        "audited but not automatically "
                        "included in AMR matrices"
                    ),
            },
            {
                "item":
                    "vocabulary_selection",
                "value":
                    (
                        "none; this script performs "
                        "schema/category/prevalence "
                        "audit only"
                    ),
            },
            {
                "item":
                    "outer_target_information",
                "value":
                    (
                        "not used for token selection "
                        "or filtering"
                    ),
            },
            {
                "item":
                    "models_trained",
                "value":
                    "none",
            },
        ]
    )

    summary_records = [
        {
            "metric":
                "raw_genomes",
            "value":
                len(inventory),
        },
        {
            "metric":
                "raw_result_rows",
            "value":
                len(calls),
        },
        {
            "metric":
                "empty_result_files",
            "value":
                empty_result_files,
        },
        {
            "metric":
                "exact_duplicate_rows",
            "value":
                exact_duplicate_rows_total,
        },
        {
            "metric":
                "missing_element_symbol_rows",
            "value":
                missing_element_symbol_rows,
        },
        {
            "metric":
                "unique_element_symbols",
            "value":
                calls.loc[
                    calls[
                        "element_symbol_nonblank"
                    ],
                    "Element symbol",
                ].nunique(),
        },
        {
            "metric":
                "verified_script137_manifest_files",
            "value":
                len(
                    verified_script137_outputs
                ),
        },
    ]

    audit_summary = pd.DataFrame(
        summary_records
    )

    output_paths = {
        "raw_output_inventory":
            AUDIT_META_ROOT
            / "nested_loso_amrfinder_raw_output_inventory_v1.tsv",
        "species_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_species_summary_v1.tsv",
        "type_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_type_summary_v1.tsv",
        "subtype_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_subtype_summary_v1.tsv",
        "scope_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_scope_summary_v1.tsv",
        "method_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_method_summary_v1.tsv",
        "class_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_class_subclass_summary_v1.tsv",
        "category_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_candidate_category_summary_v1.tsv",
        "token_prevalence":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_token_prevalence_by_species_v1.tsv",
        "token_species_breadth":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_token_species_breadth_v1.tsv",
        "multiplicity_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_per_genome_token_multiplicity_v1.tsv",
        "protocol":
            AUDIT_META_ROOT
            / "nested_loso_amrfinder_annotation_audit_protocol_v1.tsv",
        "audit_summary":
            AUDIT_TABLE_ROOT
            / "nested_loso_amrfinder_annotation_audit_summary_v1.tsv",
    }

    for path in output_paths.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    write_tsv(
        inventory,
        output_paths[
            "raw_output_inventory"
        ],
    )

    write_tsv(
        species_summary,
        output_paths[
            "species_summary"
        ],
    )

    write_tsv(
        type_summary,
        output_paths[
            "type_summary"
        ],
    )

    write_tsv(
        subtype_summary,
        output_paths[
            "subtype_summary"
        ],
    )

    write_tsv(
        scope_summary,
        output_paths[
            "scope_summary"
        ],
    )

    write_tsv(
        method_summary,
        output_paths[
            "method_summary"
        ],
    )

    write_tsv(
        class_summary,
        output_paths[
            "class_summary"
        ],
    )

    write_tsv(
        category_summary,
        output_paths[
            "category_summary"
        ],
    )

    write_tsv(
        token_prevalence,
        output_paths[
            "token_prevalence"
        ],
    )

    write_tsv(
        token_species_breadth,
        output_paths[
            "token_species_breadth"
        ],
    )

    write_tsv(
        multiplicity_summary,
        output_paths[
            "multiplicity_summary"
        ],
    )

    write_tsv(
        protocol,
        output_paths[
            "protocol"
        ],
    )

    write_tsv(
        audit_summary,
        output_paths[
            "audit_summary"
        ],
    )

    manifest_paths = [
        *output_paths.values(),
    ]

    with OUTPUT_MANIFEST144.open(
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
        OUTPUT_MANIFEST144
    )

    freeze_paths = [
        SCRIPT137,
        STATUS137,
        STDOUT137,
        STDERR137,
        script137_manifest,
        Path(__file__).resolve(),
        OUTPUT_MANIFEST144,
        *manifest_paths,
    ]

    FREEZE137.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FREEZE137.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            set(freeze_paths),
            key=lambda value:
                value.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.relative_to(PROJECT)}\n"
            )

    verify_sha_manifest(
        FREEZE137
    )

    print(
        "===== SCRIPT 144 AMRFINDER AUDIT ====="
    )

    print(
        species_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== CANDIDATE CATEGORY SUMMARY ====="
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Raw genomes:",
        len(inventory),
    )

    print(
        "Raw result rows:",
        len(calls),
    )

    print(
        "Empty result files:",
        empty_result_files,
    )

    print(
        "Exact duplicate rows:",
        exact_duplicate_rows_total,
    )

    print(
        "Missing Element symbol rows:",
        missing_element_symbol_rows,
    )

    print(
        "Unique Element symbols:",
        int(
            calls.loc[
                calls[
                    "element_symbol_nonblank"
                ],
                "Element symbol",
            ].nunique()
        ),
    )

    print(
        "Script 137 output-manifest files:",
        len(
            verified_script137_outputs
        ),
    )

    print(
        "Script 137 frozen core files:",
        len(
            set(freeze_paths)
        ),
    )

    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 144 AMRFINDER "
        "FREEZE AND AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
