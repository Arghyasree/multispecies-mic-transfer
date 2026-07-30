#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_ROOT = Path(
    "references/taxonomy/ncbi_type_material"
)

REPORTS = {
    "Acinetobacter": (
        REFERENCE_ROOT
        / "metadata/acinetobacter/ncbi_dataset/data/"
        "assembly_data_report.jsonl"
    ),
    "Pseudomonas": (
        REFERENCE_ROOT
        / "metadata/pseudomonas/ncbi_dataset/data/"
        "assembly_data_report.jsonl"
    ),
}

EXPECTED_REPORT_ROWS = {
    "Acinetobacter": 374,
    "Pseudomonas": 1_168,
}

TARGET_SPECIES = {
    "Acinetobacter":
        "Acinetobacter baumannii",
    "Pseudomonas":
        "Pseudomonas aeruginosa",
}

INVENTORY_PATH = Path(
    "metadata/taxonomy/"
    "ncbi_type_material_assembly_inventory.tsv"
)

ELIGIBLE_PATH = Path(
    "metadata/taxonomy/"
    "ncbi_type_material_reference_eligible.tsv"
)

REJECTED_PATH = Path(
    "metadata/taxonomy/"
    "ncbi_type_material_reference_rejected.tsv"
)

SELECTED_PATH = Path(
    "metadata/taxonomy/"
    "ncbi_type_material_selected_reference_manifest.tsv"
)

SUMMARY_PATH = Path(
    "results/tables/taxonomy/"
    "ncbi_type_material_reference_selection_summary.tsv"
)

SELECTED_ROOT = (
    REFERENCE_ROOT
    / "selected"
)

COMBINED_ACCESSIONS_PATH = (
    SELECTED_ROOT
    / "selected_type_reference_accessions.txt"
)

GENUS_ACCESSION_PATHS = {
    "Acinetobacter": (
        SELECTED_ROOT
        / "acinetobacter_selected_type_reference_accessions.txt"
    ),
    "Pseudomonas": (
        SELECTED_ROOT
        / "pseudomonas_selected_type_reference_accessions.txt"
    ),
}

OUTPUT_SHA_PATH = Path(
    "metadata/taxonomy/"
    "script52_outputs_sha256.txt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def nested(
    record: dict[str, Any],
    *keys: str,
    default: Any = "",
) -> Any:
    current: Any = record

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def first_present(
    record: dict[str, Any],
    paths: list[tuple[str, ...]],
    default: Any = "",
) -> Any:
    for path in paths:
        value = nested(
            record,
            *path,
            default="",
        )

        if value not in (
            "",
            None,
            [],
            {},
        ):
            return value

    return default


def recursive_values(
    value: Any,
    target_key: str,
) -> list[str]:
    collected: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key.casefold()
                == target_key.casefold()
            ):
                if isinstance(
                    child,
                    (str, int, float, bool),
                ):
                    collected.append(
                        str(child)
                    )

            collected.extend(
                recursive_values(
                    child,
                    target_key,
                )
            )

    elif isinstance(value, list):
        for child in value:
            collected.extend(
                recursive_values(
                    child,
                    target_key,
                )
            )

    return collected


def unique_join(values: list[str]) -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip()
        }
    )

    return "|".join(cleaned)


def extract_accessions(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
    )

    accessions = re.findall(
        r"\bGC[AF]_\d+\.\d+\b",
        serialized,
    )

    return unique_join(accessions)


def numeric(
    value: Any,
) -> float:
    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return math.nan

    if not math.isfinite(result):
        return math.nan

    return result


def integer(
    value: Any,
) -> float:
    result = numeric(value)

    if math.isnan(result):
        return math.nan

    return float(int(result))


def enum_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def source_rank(value: str) -> int:
    normalized = value.upper()

    if "REFSEQ" in normalized:
        return 0

    if "GENBANK" in normalized:
        return 1

    return 2


def refseq_category_rank(
    value: str,
) -> int:
    normalized = value.casefold()

    if "reference genome" in normalized:
        return 0

    if "representative genome" in normalized:
        return 1

    return 2


def assembly_level_rank(
    value: str,
) -> int:
    normalized = value.casefold()

    order = {
        "complete genome": 0,
        "chromosome": 1,
        "scaffold": 2,
        "contig": 3,
    }

    return order.get(
        normalized,
        4,
    )


def ani_status_rank(
    value: str,
) -> int:
    normalized = value.upper()

    if normalized == "OK" or normalized.endswith("_OK"):
        return 0

    if "INCONCLUSIVE" in normalized:
        return 1

    if not normalized:
        return 2

    return 3


def current_status_ok(
    assembly_status: str,
) -> bool:
    normalized = assembly_status.upper()

    forbidden = [
        "SUPPRESSED",
        "RETIRED",
        "PREVIOUS",
    ]

    return not any(
        token in normalized
        for token in forbidden
    )


def named_species(
    genus: str,
    organism_name: str,
) -> bool:
    normalized = (
        organism_name
        .strip()
        .replace("[", "")
        .replace("]", "")
    )

    if not normalized.casefold().startswith(
        genus.casefold() + " "
    ):
        return False

    tokens = normalized.split()

    if len(tokens) < 2:
        return False

    second = (
        tokens[1]
        .strip(".,;:()")
        .casefold()
    )

    excluded_second_tokens = {
        "sp",
        "spp",
        "bacterium",
        "phage",
        "virus",
    }

    if second in excluded_second_tokens:
        return False

    excluded_phrases = [
        "uncultured",
        "unclassified",
        "environmental sample",
    ]

    lowered = normalized.casefold()

    return not any(
        phrase in lowered
        for phrase in excluded_phrases
    )



def normalize_taxon_name(
    value: Any,
) -> str:
    normalized = (
        str(value)
        .strip()
        .replace("[", "")
        .replace("]", "")
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def canonical_species_label(
    genus: str,
    organism_name: str,
    submitted_species: str,
) -> str:
    excluded_species_tokens = {
        "sp",
        "spp",
        "bacterium",
        "phage",
        "virus",
    }

    for candidate in [
        submitted_species,
        organism_name,
    ]:
        normalized = normalize_taxon_name(
            candidate
        )

        tokens = normalized.split()

        if len(tokens) < 2:
            continue

        if (
            tokens[0].casefold()
            != genus.casefold()
        ):
            continue

        species_token = tokens[1].strip(
            ".,;:()"
        )

        if (
            species_token.casefold()
            in excluded_species_tokens
        ):
            continue

        lowered = normalized.casefold()

        if any(
            phrase in lowered
            for phrase in [
                "uncultured",
                "unclassified",
                "environmental sample",
            ]
        ):
            continue

        return (
            f"{genus} {species_token}"
        )

    return ""

def flatten_record(
    genus: str,
    record: dict[str, Any],
    report_order: int,
) -> dict[str, Any]:
    assembly_info = record.get(
        "assemblyInfo",
        {},
    )

    assembly_stats = record.get(
        "assemblyStats",
        {},
    )

    organism = record.get(
        "organism",
        {},
    )

    checkm = record.get(
        "checkmInfo",
        {},
    )

    ani = record.get(
        "averageNucleotideIdentity",
        {},
    )

    type_material = record.get(
        "typeMaterial",
        {},
    )

    accession = str(
        record.get(
            "accession",
            "",
        )
    ).strip()

    current_accession = str(
        record.get(
            "currentAccession",
            "",
        )
    ).strip()

    organism_name = str(
        organism.get(
            "organismName",
            "",
        )
    ).strip()

    taxon_id = str(
        organism.get(
            "taxId",
            "",
        )
    ).strip()

    assembly_status = enum_text(
        first_present(
            record,
            [
                (
                    "assemblyInfo",
                    "assemblyStatus",
                ),
                (
                    "assemblyStatus",
                ),
            ],
        )
    )

    ani_check_status = enum_text(
        first_present(
            record,
            [
                (
                    "averageNucleotideIdentity",
                    "taxonomyCheckStatus",
                ),
                (
                    "averageNucleotideIdentity",
                    "taxonomy_check_status",
                ),
            ],
        )
    )

    ani_best_match_status = enum_text(
        first_present(
            record,
            [
                (
                    "averageNucleotideIdentity",
                    "bestMatchStatus",
                ),
                (
                    "averageNucleotideIdentity",
                    "best_match_status",
                ),
            ],
        )
    )

    best_ani_match = first_present(
        record,
        [
            (
                "averageNucleotideIdentity",
                "bestAniMatch",
            ),
            (
                "averageNucleotideIdentity",
                "best_ani_match",
            ),
        ],
        default={},
    )

    if not isinstance(
        best_ani_match,
        dict,
    ):
        best_ani_match = {}

    submitted_species = str(
        first_present(
            record,
            [
                (
                    "averageNucleotideIdentity",
                    "submittedSpecies",
                ),
                (
                    "averageNucleotideIdentity",
                    "submitted_species",
                ),
            ],
        )
    ).strip()

    canonical_species = (
        canonical_species_label(
            genus,
            organism_name,
            submitted_species,
        )
    )

    species_level_name_exact = (
        normalize_taxon_name(
            organism_name
        ).casefold()
        == canonical_species.casefold()
    )

    type_labels = unique_join(
        recursive_values(
            type_material,
            "typeLabel",
        )
    )

    type_display_text = unique_join(
        recursive_values(
            type_material,
            "typeDisplayText",
        )
    )

    total_length = integer(
        first_present(
            record,
            [
                (
                    "assemblyStats",
                    "totalSequenceLength",
                ),
                (
                    "assemblyStats",
                    "total_sequence_length",
                ),
            ],
        )
    )

    contigs = integer(
        first_present(
            record,
            [
                (
                    "assemblyStats",
                    "numberOfContigs",
                ),
                (
                    "assemblyStats",
                    "number_of_contigs",
                ),
            ],
        )
    )

    contig_n50 = integer(
        first_present(
            record,
            [
                (
                    "assemblyStats",
                    "contigN50",
                ),
                (
                    "assemblyStats",
                    "contig_n50",
                ),
            ],
        )
    )

    scaffold_n50 = integer(
        first_present(
            record,
            [
                (
                    "assemblyStats",
                    "scaffoldN50",
                ),
                (
                    "assemblyStats",
                    "scaffold_n50",
                ),
            ],
        )
    )

    completeness = numeric(
        first_present(
            record,
            [
                (
                    "checkmInfo",
                    "completeness",
                ),
                (
                    "checkmInfo",
                    "checkmMarkerSet",
                    "completeness",
                ),
            ],
        )
    )

    contamination = numeric(
        first_present(
            record,
            [
                (
                    "checkmInfo",
                    "contamination",
                ),
                (
                    "checkmInfo",
                    "checkmMarkerSet",
                    "contamination",
                ),
            ],
        )
    )

    best_ani = numeric(
        best_ani_match.get(
            "ani",
            "",
        )
    )

    best_ani_assembly_coverage = numeric(
        first_present(
            best_ani_match,
            [
                (
                    "assemblyCoverage",
                ),
                (
                    "assembly_coverage",
                ),
            ],
        )
    )

    best_ani_type_coverage = numeric(
        first_present(
            best_ani_match,
            [
                (
                    "typeAssemblyCoverage",
                ),
                (
                    "type_assembly_coverage",
                ),
            ],
        )
    )

    ani_failed = (
        "FAILED"
        in ani_check_status.upper()
    )

    has_named_species = named_species(
        genus,
        organism_name,
    )

    is_current = current_status_ok(
        assembly_status
    )

    has_type_evidence = bool(
        type_labels
        or type_display_text
        or type_material
    )

    rejection_reasons: list[str] = []

    if not accession:
        rejection_reasons.append(
            "missing_accession"
        )

    if not taxon_id:
        rejection_reasons.append(
            "missing_taxon_id"
        )

    if not has_named_species:
        rejection_reasons.append(
            "not_named_species"
        )

    if not canonical_species:
        rejection_reasons.append(
            "missing_canonical_species_label"
        )

    if not has_type_evidence:
        rejection_reasons.append(
            "missing_type_material_evidence"
        )

    if not is_current:
        rejection_reasons.append(
            "noncurrent_assembly_status"
        )

    if ani_failed:
        rejection_reasons.append(
            "ncbi_ani_taxonomy_check_failed"
        )

    if (
        math.isnan(total_length)
        or total_length <= 0
    ):
        rejection_reasons.append(
            "missing_or_invalid_total_length"
        )

    selection_eligible = (
        len(rejection_reasons) == 0
    )

    return {
        "source_genus": genus,
        "report_order": report_order,
        "accession": accession,
        "current_accession":
            current_accession,
        "paired_accessions":
            extract_accessions(
                record.get(
                    "pairedAccession",
                    {},
                )
            ),
        "source_database": enum_text(
            record.get(
                "sourceDatabase",
                "",
            )
        ),
        "taxon_id": taxon_id,
        "organism_name": organism_name,
        "canonical_species_label":
            canonical_species,
        "species_level_name_exact":
            species_level_name_exact,
        "ani_submitted_species":
            submitted_species,
        "strain": str(
            nested(
                organism,
                "infraspecificNames",
                "strain",
                default="",
            )
        ).strip(),
        "assembly_name": str(
            assembly_info.get(
                "assemblyName",
                "",
            )
        ).strip(),
        "assembly_level": enum_text(
            assembly_info.get(
                "assemblyLevel",
                "",
            )
        ),
        "assembly_status":
            assembly_status,
        "refseq_category": enum_text(
            assembly_info.get(
                "refseqCategory",
                "",
            )
        ),
        "release_date": str(
            assembly_info.get(
                "releaseDate",
                "",
            )
        ).strip(),
        "submitter": str(
            assembly_info.get(
                "submitter",
                "",
            )
        ).strip(),
        "type_material_label":
            type_labels,
        "type_material_display_text":
            type_display_text,
        "total_sequence_length":
            total_length,
        "number_of_contigs":
            contigs,
        "contig_n50": contig_n50,
        "scaffold_n50": scaffold_n50,
        "checkm_completeness":
            completeness,
        "checkm_contamination":
            contamination,
        "ani_check_status":
            ani_check_status,
        "ani_best_match_status":
            ani_best_match_status,
        "ani_best_match_assembly": str(
            best_ani_match.get(
                "assembly",
                "",
            )
        ).strip(),
        "ani_best_match_organism": str(
            best_ani_match.get(
                "organismName",
                best_ani_match.get(
                    "organism",
                    "",
                ),
            )
        ).strip(),
        "ani_best_match":
            best_ani,
        "ani_best_match_assembly_coverage":
            best_ani_assembly_coverage,
        "ani_best_match_type_coverage":
            best_ani_type_coverage,
        "has_named_species":
            has_named_species,
        "has_type_material_evidence":
            has_type_evidence,
        "current_assembly_status":
            is_current,
        "ncbi_ani_check_failed":
            ani_failed,
        "selection_eligible":
            selection_eligible,
        "selection_rejection_reason":
            "|".join(
                rejection_reasons
            ),
    }


def write_sha_manifest(
    paths: list[Path],
) -> None:
    lines = [
        f"{sha256_file(path)}  {path}"
        for path in sorted(
            paths,
            key=lambda item:
                item.as_posix(),
        )
    ]

    OUTPUT_SHA_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SELECTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[
        dict[str, Any]
    ] = []

    for genus, report_path in REPORTS.items():
        if not report_path.is_file():
            raise FileNotFoundError(
                report_path
            )

        with report_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            genus_records = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]

        expected = EXPECTED_REPORT_ROWS[
            genus
        ]

        if len(genus_records) != expected:
            raise RuntimeError(
                f"Expected {expected:,} {genus} "
                f"records; found "
                f"{len(genus_records):,}."
            )

        for report_order, record in enumerate(
            genus_records,
            start=1,
        ):
            rows.append(
                flatten_record(
                    genus,
                    record,
                    report_order,
                )
            )

    inventory = pd.DataFrame(
        rows
    )

    if len(inventory) != 1_542:
        raise RuntimeError(
            f"Expected 1,542 total assembly "
            f"records; found {len(inventory):,}."
        )

    if inventory[
        "accession"
    ].duplicated().any():
        duplicated = inventory.loc[
            inventory[
                "accession"
            ].duplicated(
                keep=False
            ),
            "accession",
        ].tolist()

        raise RuntimeError(
            "Duplicate assembly accessions in "
            "combined reports: "
            + ", ".join(
                duplicated[:10]
            )
        )

    eligible = inventory.loc[
        inventory[
            "selection_eligible"
        ]
    ].copy()

    rejected = inventory.loc[
        ~inventory[
            "selection_eligible"
        ]
    ].copy()

    eligible[
        "_species_level_name_rank"
    ] = (
        ~eligible[
            "species_level_name_exact"
        ].astype(bool)
    ).astype(int)

    eligible[
        "_source_rank"
    ] = eligible[
        "source_database"
    ].map(
        source_rank
    )

    eligible[
        "_refseq_category_rank"
    ] = eligible[
        "refseq_category"
    ].map(
        refseq_category_rank
    )

    eligible[
        "_assembly_level_rank"
    ] = eligible[
        "assembly_level"
    ].map(
        assembly_level_rank
    )

    eligible[
        "_ani_status_rank"
    ] = eligible[
        "ani_check_status"
    ].map(
        ani_status_rank
    )

    eligible[
        "_completeness_rank"
    ] = (
        pd.to_numeric(
            eligible[
                "checkm_completeness"
            ],
            errors="coerce",
        )
        .fillna(-1.0)
        .mul(-1.0)
    )

    eligible[
        "_contamination_rank"
    ] = pd.to_numeric(
        eligible[
            "checkm_contamination"
        ],
        errors="coerce",
    ).fillna(1_000_000.0)

    eligible[
        "_contig_count_rank"
    ] = pd.to_numeric(
        eligible[
            "number_of_contigs"
        ],
        errors="coerce",
    ).fillna(1_000_000_000.0)

    eligible[
        "_contig_n50_rank"
    ] = (
        pd.to_numeric(
            eligible[
                "contig_n50"
            ],
            errors="coerce",
        )
        .fillna(-1.0)
        .mul(-1.0)
    )

    eligible[
        "_total_length_rank"
    ] = (
        pd.to_numeric(
            eligible[
                "total_sequence_length"
            ],
            errors="coerce",
        )
        .fillna(-1.0)
        .mul(-1.0)
    )

    ranking_columns = [
        "source_genus",
        "canonical_species_label",
        "_species_level_name_rank",
        "_ani_status_rank",
        "_source_rank",
        "_refseq_category_rank",
        "_assembly_level_rank",
        "_completeness_rank",
        "_contamination_rank",
        "_contig_count_rank",
        "_contig_n50_rank",
        "_total_length_rank",
        "accession",
    ]

    eligible = eligible.sort_values(
        ranking_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    eligible[
        "selection_rank_within_species"
    ] = (
        eligible.groupby(
            [
                "source_genus",
                "canonical_species_label",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    eligible[
        "candidate_assemblies_for_species"
    ] = (
        eligible.groupby(
            [
                "source_genus",
                "canonical_species_label",
            ]
        )[
            "accession"
        ]
        .transform("size")
    )

    selected = eligible.loc[
        eligible[
            "selection_rank_within_species"
        ].eq(1)
    ].copy()

    selected[
        "selection_reason"
    ] = (
        "preferred_current_nonfailed_"
        "type_assembly_per_canonical_species"
    )

    if selected[
        "accession"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate selected reference accession."
        )

    if selected[
        [
            "source_genus",
            "canonical_species_label",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "More than one selected assembly "
            "for a canonical species."
        )

    target_rows = []

    for genus, target_species in (
        TARGET_SPECIES.items()
    ):
        matches = selected.loc[
            selected[
                "canonical_species_label"
            ].eq(target_species)
        ].copy()

        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one selected "
                f"reference for {target_species}; "
                f"found {len(matches)}."
            )

        target_rows.append(
            matches
        )

    target_reference_table = pd.concat(
        target_rows,
        ignore_index=True,
    )

    drop_ranking_columns = [
        column
        for column in eligible.columns
        if column.startswith("_")
    ]

    eligible_output = eligible.drop(
        columns=drop_ranking_columns
    )

    selected_output = selected.drop(
        columns=drop_ranking_columns
    )

    inventory.to_csv(
        INVENTORY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    eligible_output.to_csv(
        ELIGIBLE_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    rejected.to_csv(
        REJECTED_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    selected_output.to_csv(
        SELECTED_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    for genus, path in (
        GENUS_ACCESSION_PATHS.items()
    ):
        genus_accessions = (
            selected_output.loc[
                selected_output[
                    "source_genus"
                ].eq(genus),
                "accession",
            ]
            .sort_values(
                kind="mergesort"
            )
            .tolist()
        )

        path.write_text(
            "\n".join(
                genus_accessions
            )
            + "\n",
            encoding="utf-8",
        )

    combined_accessions = (
        selected_output[
            "accession"
        ]
        .sort_values(
            kind="mergesort"
        )
        .tolist()
    )

    COMBINED_ACCESSIONS_PATH.write_text(
        "\n".join(
            combined_accessions
        )
        + "\n",
        encoding="utf-8",
    )

    summary_rows = []

    for genus in REPORTS:
        genus_inventory = inventory.loc[
            inventory[
                "source_genus"
            ].eq(genus)
        ]

        genus_eligible = eligible_output.loc[
            eligible_output[
                "source_genus"
            ].eq(genus)
        ]

        genus_rejected = rejected.loc[
            rejected[
                "source_genus"
            ].eq(genus)
        ]

        genus_selected = selected_output.loc[
            selected_output[
                "source_genus"
            ].eq(genus)
        ]

        target_species = TARGET_SPECIES[
            genus
        ]

        target_present = bool(
            genus_selected[
                "canonical_species_label"
            ].eq(
                target_species
            ).any()
        )

        summary_rows.append(
            {
                "source_genus": genus,
                "input_assembly_records":
                    len(genus_inventory),
                "unique_accessions":
                    genus_inventory[
                        "accession"
                    ].nunique(),
                "unique_taxon_ids":
                    genus_inventory[
                        "taxon_id"
                    ].replace(
                        "",
                        pd.NA,
                    ).nunique(),
                "unique_canonical_species":
                    genus_inventory[
                        "canonical_species_label"
                    ].replace(
                        "",
                        pd.NA,
                    ).nunique(),
                "named_species_records":
                    int(
                        genus_inventory[
                            "has_named_species"
                        ].sum()
                    ),
                "ncbi_ani_failed_records":
                    int(
                        genus_inventory[
                            "ncbi_ani_check_failed"
                        ].sum()
                    ),
                "selection_eligible_records":
                    len(genus_eligible),
                "rejected_records":
                    len(genus_rejected),
                "selected_type_references":
                    len(genus_selected),
                "selected_refseq_references":
                    int(
                        genus_selected[
                            "source_database"
                        ]
                        .str.upper()
                        .str.contains(
                            "REFSEQ"
                        )
                        .sum()
                    ),
                "selected_complete_genomes":
                    int(
                        genus_selected[
                            "assembly_level"
                        ]
                        .str.casefold()
                        .eq(
                            "complete genome"
                        )
                        .sum()
                    ),
                "target_species":
                    target_species,
                "target_reference_present":
                    target_present,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        INVENTORY_PATH,
        ELIGIBLE_PATH,
        REJECTED_PATH,
        SELECTED_PATH,
        SUMMARY_PATH,
        COMBINED_ACCESSIONS_PATH,
        *GENUS_ACCESSION_PATHS.values(),
    ]

    write_sha_manifest(
        output_paths
    )

    print(
        "===== SCRIPT 52 TYPE-MATERIAL "
        "REFERENCE SELECTION ====="
    )

    print(
        "Input assembly records:",
        f"{len(inventory):,}",
    )

    print(
        "Eligible assembly records:",
        f"{len(eligible_output):,}",
    )

    print(
        "Rejected assembly records:",
        f"{len(rejected):,}",
    )

    print(
        "Selected nonredundant references:",
        f"{len(selected_output):,}",
    )

    print()
    print(
        "===== REFERENCE-SELECTION SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "===== SELECTED TARGET REFERENCES ====="
    )

    print(
        target_reference_table[
            [
                "source_genus",
                "accession",
                "canonical_species_label",
                "organism_name",
                "taxon_id",
                "source_database",
                "assembly_level",
                "refseq_category",
                "ani_check_status",
                "number_of_contigs",
                "contig_n50",
                "checkm_completeness",
                "checkm_contamination",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Reference genome sequences downloaded:",
        "NO",
    )

    print(
        "Production genome FASTAs modified:",
        "NO",
    )

    print()
    print(
        "STATUS: NONREDUNDANT TYPE-MATERIAL "
        "REFERENCE SELECTION COMPLETE"
    )


if __name__ == "__main__":
    main()
