#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.bv-brc.org/api/genome/"

ID_PATH = Path(
    "metadata/shortlist/"
    "shortlist_candidate_genome_ids.txt"
)

MANIFEST_PATH = Path(
    "metadata/shortlist/"
    "shortlist_candidate_genome_manifest.tsv"
)

DOWNLOAD_DATE = date.today().isoformat()

BATCH_SIZE = 200

BATCH_ROOT = Path(
    "data/raw/genome_metadata/"
    f"bvbrc_shortlist_{DOWNLOAD_DATE}_batches"
)

OUTPUT_PATH = Path(
    "data/raw/genome_metadata/"
    f"bvbrc_shortlist_genome_metadata_"
    f"{DOWNLOAD_DATE}.tsv"
)

SUMMARY_PATH = Path(
    "results/tables/"
    f"bvbrc_shortlist_genome_metadata_"
    f"summary_{DOWNLOAD_DATE}.tsv"
)

MISSING_PATH = Path(
    "metadata/shortlist/"
    f"bvbrc_shortlist_metadata_missing_ids_"
    f"{DOWNLOAD_DATE}.txt"
)

UNEXPECTED_PATH = Path(
    "metadata/shortlist/"
    f"bvbrc_shortlist_metadata_unexpected_ids_"
    f"{DOWNLOAD_DATE}.txt"
)

DUPLICATE_PATH = Path(
    "metadata/shortlist/"
    f"bvbrc_shortlist_metadata_duplicate_ids_"
    f"{DOWNLOAD_DATE}.txt"
)

SOURCE_PATH = Path(
    "metadata/downloads/"
    f"bvbrc_shortlist_genome_metadata_"
    f"{DOWNLOAD_DATE}_source.txt"
)

CHECKSUM_PATH = Path(
    "metadata/downloads/"
    f"bvbrc_shortlist_genome_metadata_"
    f"{DOWNLOAD_DATE}_outputs_sha256.txt"
)

FIELDS = [
    "genome_id",
    "genome_name",
    "species",
    "genus",
    "taxon_id",
    "taxon_lineage_ids",
    "taxon_lineage_names",
    "genome_status",
    "genome_quality",
    "genome_quality_flags",
    "checkm_completeness",
    "checkm_contamination",
    "coarse_consistency",
    "fine_consistency",
    "contigs",
    "contig_n50",
    "contig_l50",
    "genome_length",
    "gc_content",
    "assembly_accession",
    "genbank_accessions",
    "bioproject_accession",
    "biosample_accession",
    "sra_accession",
    "strain",
    "serovar",
    "mlst",
    "collection_date",
    "collection_year",
    "isolation_country",
    "geographic_location",
    "isolation_source",
    "host_name",
    "host_scientific_name",
    "host_common_name",
    "sequencing_platform",
    "sequencing_status",
    "date_inserted",
    "date_modified",
    "owner",
    "public",
]


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


def create_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=2.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
        ],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "multispecies-mic-transfer/1.0"
            ),
        }
    )

    return session


def load_genome_ids() -> list[str]:
    if not ID_PATH.exists():
        raise FileNotFoundError(
            f"Missing genome-ID manifest: {ID_PATH}"
        )

    genome_ids = [
        line.strip()
        for line in ID_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not genome_ids:
        raise ValueError(
            "Genome-ID manifest is empty."
        )

    if len(genome_ids) != len(set(genome_ids)):
        raise ValueError(
            "Genome-ID manifest contains duplicates."
        )

    return genome_ids


def split_batches(
    genome_ids: list[str],
) -> list[list[str]]:
    return [
        genome_ids[start:start + BATCH_SIZE]
        for start in range(
            0,
            len(genome_ids),
            BATCH_SIZE,
        )
    ]


def batch_path(
    batch_number: int,
) -> Path:
    return (
        BATCH_ROOT
        / f"batch_{batch_number:04d}.json"
    )


def build_url(
    genome_ids: list[str],
) -> str:
    id_expression = ",".join(
        genome_ids
    )

    selected_fields = ",".join(
        FIELDS
    )

    query = (
        f"in(genome_id,({id_expression}))"
        f"&select({selected_fields})"
        "&sort(+genome_id)"
        f"&limit({BATCH_SIZE})"
    )

    return f"{BASE_URL}?{query}"


def validate_batch_payload(
    payload: dict[str, Any],
    expected_ids: list[str],
) -> list[dict[str, Any]]:
    stored_ids = payload.get(
        "requested_ids"
    )

    if stored_ids != expected_ids:
        raise ValueError(
            "Existing batch was created for a "
            "different genome-ID set."
        )

    records = payload.get(
        "records"
    )

    if not isinstance(records, list):
        raise ValueError(
            "Existing batch has no valid "
            "records list."
        )

    requested_set = set(
        expected_ids
    )

    observed_ids = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "A batch record is not a JSON object."
            )

        genome_id = str(
            record.get(
                "genome_id",
                "",
            )
        ).strip()

        if not genome_id:
            raise ValueError(
                "A batch record has no genome_id."
            )

        if genome_id not in requested_set:
            raise ValueError(
                "Batch returned an unexpected "
                f"genome ID: {genome_id}"
            )

        observed_ids.append(
            genome_id
        )

    if len(observed_ids) != len(
        set(observed_ids)
    ):
        raise ValueError(
            "A batch contains duplicate genome IDs."
        )

    return records


def load_or_download_batch(
    session: requests.Session,
    batch_number: int,
    genome_ids: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    path = batch_path(
        batch_number
    )

    if path.exists():
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        records = validate_batch_payload(
            payload,
            genome_ids,
        )

        return records, True

    url = build_url(
        genome_ids
    )

    response = session.get(
        url,
        timeout=(30, 300),
    )

    response.raise_for_status()

    records = response.json()

    if not isinstance(records, list):
        raise ValueError(
            "BV-BRC returned a non-list JSON response."
        )

    payload = {
        "batch_number": batch_number,
        "downloaded_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "requested_ids": genome_ids,
        "content_range": (
            response.headers.get(
                "Content-Range"
            )
        ),
        "record_count": len(records),
        "records": records,
    }

    validate_batch_payload(
        payload,
        genome_ids,
    )

    temporary_path = Path(
        str(path) + ".part"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return records, False


def tsv_value(
    value: Any,
) -> Any:
    if value is None:
        return ""

    if isinstance(
        value,
        (list, dict),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return value


def write_lines(
    path: Path,
    values: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        (
            "\n".join(values) + "\n"
            if values
            else ""
        ),
        encoding="utf-8",
    )


def build_summary(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    available = metadata[
        [
            "genome_id",
            "species",
            "taxon_lineage_names",
            "genome_quality",
            "genome_status",
            "checkm_completeness",
            "checkm_contamination",
            "contig_n50",
            "assembly_accession",
        ]
    ].copy()

    merged = manifest.merge(
        available,
        on="genome_id",
        how="left",
        indicator=True,
    )

    rows = []

    for species, group in merged.groupby(
        "provisional_species",
        sort=True,
    ):
        metadata_found = group[
            "_merge"
        ].eq("both")

        exact_species = (
            group["species"]
            .fillna("")
            .eq(species)
        )

        lineage_match = (
            group[
                "taxon_lineage_names"
            ]
            .fillna("")
            .str.contains(
                species,
                regex=False,
            )
        )

        rows.append(
            {
                "provisional_species": species,
                "requested_genomes": len(group),
                "metadata_returned": int(
                    metadata_found.sum()
                ),
                "metadata_missing": int(
                    (~metadata_found).sum()
                ),
                "species_exact_match": int(
                    exact_species.sum()
                ),
                "lineage_contains_species": int(
                    lineage_match.sum()
                ),
                "genome_quality_good": int(
                    group[
                        "genome_quality"
                    ]
                    .fillna("")
                    .str.lower()
                    .eq("good")
                    .sum()
                ),
                "genome_status_wgs": int(
                    group[
                        "genome_status"
                    ]
                    .fillna("")
                    .str.upper()
                    .eq("WGS")
                    .sum()
                ),
                "checkm_completeness_present": int(
                    group[
                        "checkm_completeness"
                    ]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "checkm_contamination_present": int(
                    group[
                        "checkm_contamination"
                    ]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "contig_n50_present": int(
                    group[
                        "contig_n50"
                    ]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
                "assembly_accession_present": int(
                    group[
                        "assembly_accession"
                    ]
                    .fillna("")
                    .ne("")
                    .sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    print(
        "===== DOWNLOAD BV-BRC SHORTLIST "
        "GENOME METADATA ====="
    )

    genome_ids = load_genome_ids()

    batches = split_batches(
        genome_ids
    )

    print(
        "Requested genome IDs:",
        f"{len(genome_ids):,}",
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Number of batches:",
        len(batches),
    )

    BATCH_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = create_session()

    records_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    duplicate_ids: list[str] = []

    resumed_batches = 0
    downloaded_batches = 0

    started = time.monotonic()

    for batch_number, requested_ids in enumerate(
        batches,
        start=1,
    ):
        records, resumed = (
            load_or_download_batch(
                session,
                batch_number,
                requested_ids,
            )
        )

        if resumed:
            resumed_batches += 1
        else:
            downloaded_batches += 1

        for record in records:
            genome_id = str(
                record["genome_id"]
            ).strip()

            if genome_id in records_by_id:
                duplicate_ids.append(
                    genome_id
                )
            else:
                records_by_id[
                    genome_id
                ] = record

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or batch_number == len(batches)
        ):
            print(
                "Processed batch",
                f"{batch_number}/{len(batches)};",
                "metadata records:",
                f"{len(records_by_id):,}",
                flush=True,
            )

        if not resumed:
            time.sleep(0.1)

    requested_set = set(
        genome_ids
    )

    returned_set = set(
        records_by_id
    )

    missing_ids = sorted(
        requested_set - returned_set
    )

    unexpected_ids = sorted(
        returned_set - requested_set
    )

    duplicate_ids = sorted(
        set(duplicate_ids)
    )

    rows = []

    for genome_id in genome_ids:
        record = records_by_id.get(
            genome_id
        )

        if record is None:
            continue

        row = {
            field: tsv_value(
                record.get(
                    field,
                    "",
                )
            )
            for field in FIELDS
        }

        rows.append(
            row
        )

    metadata = pd.DataFrame(
        rows,
        columns=FIELDS,
    )

    metadata.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    summary = build_summary(
        metadata
    )

    summary.to_csv(
        SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    write_lines(
        MISSING_PATH,
        missing_ids,
    )

    write_lines(
        UNEXPECTED_PATH,
        unexpected_ids,
    )

    write_lines(
        DUPLICATE_PATH,
        duplicate_ids,
    )

    output_checksum = sha256_file(
        OUTPUT_PATH
    )

    source_lines = [
        (
            "dataset=BV-BRC shortlist "
            "genome metadata"
        ),
        "collection=genome",
        f"download_date={DOWNLOAD_DATE}",
        f"requested_genomes={len(genome_ids)}",
        f"returned_genomes={len(metadata)}",
        f"missing_genomes={len(missing_ids)}",
        f"unexpected_genomes={len(unexpected_ids)}",
        f"duplicate_genomes={len(duplicate_ids)}",
        f"batch_size={BATCH_SIZE}",
        f"number_of_batches={len(batches)}",
        f"resumed_batches={resumed_batches}",
        f"downloaded_batches={downloaded_batches}",
        f"selected_fields={','.join(FIELDS)}",
        f"input_id_path={ID_PATH.as_posix()}",
        (
            "input_id_sha256="
            f"{sha256_file(ID_PATH)}"
        ),
        f"output_path={OUTPUT_PATH.as_posix()}",
        f"output_sha256={output_checksum}",
        (
            "note=no genome sequences or FASTA "
            "files were downloaded"
        ),
    ]

    SOURCE_PATH.write_text(
        "\n".join(
            source_lines
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_targets = [
        OUTPUT_PATH,
        SUMMARY_PATH,
        MISSING_PATH,
        UNEXPECTED_PATH,
        DUPLICATE_PATH,
        SOURCE_PATH,
    ]

    with CHECKSUM_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in sorted(
            checksum_targets,
            key=lambda item: item.as_posix(),
        ):
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.as_posix()}\n"
            )

    elapsed = (
        time.monotonic()
        - started
    )

    print()
    print(
        "===== DOWNLOAD SUMMARY ====="
    )

    print(
        "Returned genome metadata:",
        f"{len(metadata):,}",
    )

    print(
        "Missing genome IDs:",
        f"{len(missing_ids):,}",
    )

    print(
        "Unexpected genome IDs:",
        f"{len(unexpected_ids):,}",
    )

    print(
        "Duplicate genome IDs:",
        f"{len(duplicate_ids):,}",
    )

    print(
        "Downloaded batches:",
        downloaded_batches,
    )

    print(
        "Resumed batches:",
        resumed_batches,
    )

    print(
        "Elapsed seconds:",
        f"{elapsed:,.1f}",
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    if (
        missing_ids
        or unexpected_ids
        or duplicate_ids
    ):
        print()
        print(
            "STATUS: SHORTLIST GENOME "
            "METADATA DOWNLOAD INCOMPLETE"
        )

        raise SystemExit(2)

    print()
    print(
        "STATUS: SHORTLIST GENOME "
        "METADATA DOWNLOAD COMPLETE"
    )


if __name__ == "__main__":
    main()
