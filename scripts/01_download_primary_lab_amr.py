#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = (
    "https://www.bv-brc.org/api/genome_amr/"
)

EVIDENCE_VALUE = "Laboratory Method"

FIELDS = [
    "id",
    "genome_id",
    "genome_name",
    "taxon_id",
    "antibiotic",
    "evidence",
    "resistant_phenotype",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "measurement_unit",
    "laboratory_typing_method",
    "laboratory_typing_method_version",
    "laboratory_typing_platform",
    "vendor",
    "testing_standard",
    "testing_standard_year",
    "pmid",
    "date_inserted",
    "date_modified",
]

DOWNLOAD_DATE = date.today().isoformat()

OUTPUT_PATH = Path(
    "data/raw/amr/"
    f"bvbrc_primary_laboratory_amr_{DOWNLOAD_DATE}.tsv"
)

PART_PATH = Path(
    str(OUTPUT_PATH) + ".part"
)

SOURCE_RECORD_PATH = Path(
    "metadata/downloads/"
    f"bvbrc_primary_laboratory_amr_"
    f"{DOWNLOAD_DATE}_source.txt"
)

CHECKSUM_PATH = Path(
    "metadata/downloads/"
    f"bvbrc_primary_laboratory_amr_"
    f"{DOWNLOAD_DATE}_sha256.txt"
)

PROGRESS_INTERVAL_BYTES = (
    100 * 1024 * 1024
)

CHUNK_SIZE = (
    1024 * 1024
)


def create_session() -> requests.Session:
    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
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
        max_retries=retry_policy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "multispecies-mic-transfer/1.0"
            ),
        }
    )

    return session


def parse_total_records(
    content_range: str,
) -> int:
    match = re.search(
        r"/(\d+)\s*$",
        content_range,
    )

    if match is None:
        raise ValueError(
            "Could not parse total record count "
            f"from Content-Range: {content_range!r}"
        )

    return int(match.group(1))


def probe_record_count(
    session: requests.Session,
) -> tuple[int, str]:
    probe_query = (
        "eq(evidence,Laboratory%20Method)"
        "&sort(+id)"
        "&limit(1)"
    )

    probe_url = (
        f"{BASE_URL}?{probe_query}"
    )

    response = session.get(
        probe_url,
        headers={
            "Accept": "application/json",
        },
        timeout=(30, 180),
    )

    print(
        "Probe HTTP status:",
        response.status_code,
    )

    response.raise_for_status()

    content_range = response.headers.get(
        "Content-Range",
        "",
    )

    print(
        "Probe Content-Range:",
        content_range,
    )

    records = response.json()

    if len(records) != 1:
        raise ValueError(
            "Expected one probe record; "
            f"received {len(records)}."
        )

    if (
        records[0].get("evidence")
        != EVIDENCE_VALUE
    ):
        raise ValueError(
            "Probe returned an unexpected "
            "evidence value."
        )

    total_records = parse_total_records(
        content_range
    )

    return total_records, probe_url


def build_download_url() -> str:
    selected_fields = ",".join(FIELDS)

    query = (
        "eq(evidence,Laboratory%20Method)"
        f"&select({selected_fields})"
        "&sort(+id)"
        "&limit(2000000)"
        "&http_download=true"
    )

    return f"{BASE_URL}?{query}"


def inspect_header(
    path: Path,
) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="strict",
    ) as handle:
        header_line = handle.readline()

    if not header_line:
        raise ValueError(
            "Downloaded TSV file is empty."
        )

    return header_line.rstrip(
        "\r\n"
    ).split("\t")


def count_lines(
    path: Path,
) -> int:
    line_count = 0

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                16 * 1024 * 1024
            ),
            b"",
        ):
            line_count += block.count(b"\n")

    return line_count


def main() -> None:
    print(
        "===== BV-BRC PRIMARY LABORATORY "
        "AMR DOWNLOAD ====="
    )

    print(
        "Download time:",
        datetime.now(
            timezone.utc
        ).isoformat(),
    )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Final output already exists: "
            f"{OUTPUT_PATH}"
        )

    if PART_PATH.exists():
        raise FileExistsError(
            "Partial file already exists. "
            "Inspect or remove it before rerunning: "
            f"{PART_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_RECORD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = create_session()

    total_records, probe_url = (
        probe_record_count(session)
    )

    print(
        "Expected laboratory records:",
        f"{total_records:,}",
    )

    download_url = build_download_url()

    print()
    print(
        "Beginning HTTPS TSV download."
    )

    started = time.monotonic()

    response = session.get(
        download_url,
        headers={
            "Accept": "text/tsv",
        },
        stream=True,
        timeout=(30, 1800),
    )

    print(
        "Download HTTP status:",
        response.status_code,
    )

    print(
        "Download Content-Type:",
        response.headers.get(
            "Content-Type",
        ),
    )

    print(
        "Download Content-Range:",
        response.headers.get(
            "Content-Range",
        ),
    )

    print(
        "Download Content-Length:",
        response.headers.get(
            "Content-Length",
        ),
    )

    response.raise_for_status()

    digest = hashlib.sha256()

    downloaded_bytes = 0
    next_progress = (
        PROGRESS_INTERVAL_BYTES
    )

    try:
        with PART_PATH.open("wb") as handle:
            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):
                if not chunk:
                    continue

                handle.write(chunk)
                digest.update(chunk)

                downloaded_bytes += len(chunk)

                if (
                    downloaded_bytes
                    >= next_progress
                ):
                    elapsed = (
                        time.monotonic()
                        - started
                    )

                    speed_mib = (
                        downloaded_bytes
                        / 1024
                        / 1024
                        / elapsed
                    )

                    print(
                        "Downloaded:",
                        f"{downloaded_bytes / 1024 / 1024:,.1f}",
                        "MiB;",
                        "average speed:",
                        f"{speed_mib:,.2f}",
                        "MiB/s",
                        flush=True,
                    )

                    next_progress += (
                        PROGRESS_INTERVAL_BYTES
                    )

    except Exception:
        print(
            "Download interrupted. "
            f"Partial file retained at {PART_PATH}.",
            file=sys.stderr,
        )

        raise

    if downloaded_bytes == 0:
        raise ValueError(
            "The server returned an empty file."
        )

    PART_PATH.replace(
        OUTPUT_PATH
    )

    observed_header = inspect_header(
        OUTPUT_PATH
    )

    if observed_header != FIELDS:
        raise ValueError(
            "Unexpected TSV header.\n"
            f"Expected: {FIELDS}\n"
            f"Observed: {observed_header}"
        )

    total_lines = count_lines(
        OUTPUT_PATH
    )

    data_rows = max(
        total_lines - 1,
        0,
    )

    print()
    print(
        "Downloaded bytes:",
        f"{downloaded_bytes:,}",
    )

    print(
        "TSV lines:",
        f"{total_lines:,}",
    )

    print(
        "TSV data rows:",
        f"{data_rows:,}",
    )

    if data_rows != total_records:
        raise ValueError(
            "Downloaded row count does not match "
            "the API probe count: "
            f"expected {total_records:,}; "
            f"observed {data_rows:,}."
        )

    checksum = digest.hexdigest()

    CHECKSUM_PATH.write_text(
        f"{checksum}  "
        f"{OUTPUT_PATH.as_posix()}\n",
        encoding="utf-8",
    )

    source_lines = [
        (
            "dataset=BV-BRC genome_amr "
            "primary laboratory records"
        ),
        (
            "collection=genome_amr"
        ),
        (
            "evidence_filter="
            "Laboratory Method"
        ),
        (
            f"expected_records={total_records}"
        ),
        (
            f"observed_records={data_rows}"
        ),
        (
            f"selected_fields="
            f"{','.join(FIELDS)}"
        ),
        (
            f"download_date={DOWNLOAD_DATE}"
        ),
        (
            f"probe_url={probe_url}"
        ),
        (
            f"download_url={download_url}"
        ),
        (
            f"local_path="
            f"{OUTPUT_PATH.as_posix()}"
        ),
        (
            f"sha256={checksum}"
        ),
        (
            "legacy_missing_evidence_records="
            "not included in this primary file"
        ),
        (
            "computational_method_records="
            "excluded"
        ),
    ]

    SOURCE_RECORD_PATH.write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )

    elapsed = (
        time.monotonic()
        - started
    )

    print(
        "Elapsed seconds:",
        f"{elapsed:,.1f}",
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print(
        "Checksum:",
        CHECKSUM_PATH,
    )

    print(
        "Source record:",
        SOURCE_RECORD_PATH,
    )

    print()
    print(
        "STATUS: PRIMARY LABORATORY "
        "AMR DOWNLOAD COMPLETE"
    )


if __name__ == "__main__":
    main()
