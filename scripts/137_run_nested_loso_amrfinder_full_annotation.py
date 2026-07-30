#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


AMRFINDER = Path(
    os.environ.get(
        "AMRFINDER_EXECUTABLE",
        shutil.which("amrfinder") or "amrfinder",
    )
)

DATABASE_LINK = Path(
    "resources/amrfinderplus_database/latest"
)

EXPECTED_SOFTWARE_VERSION = "4.2.7"
EXPECTED_DATABASE_VERSION = "2026-05-15.1"

SOURCE_MANIFEST = Path(
    "metadata/config_selection/nested_loso_v1/"
    "feature_asset_audit_v1/"
    "species_configuration_genome_manifest.tsv"
)

PROBE_SHA_MANIFEST = Path(
    "metadata/config_selection/nested_loso_v1/"
    "amrfinder_probe_v1/"
    "amrfinder_probe_outputs_sha256.txt"
)

SCRIPT136_FROZEN_MANIFEST = Path(
    "metadata/config_selection/"
    "script136_successful_run_core_sha256.txt"
)

DATABASE_VERSION_RECORD = Path(
    "metadata/config_selection/nested_loso_v1/"
    "amrfinder_database_v1/"
    "amrfinder_database_version.txt"
)

ORGANISM_RECORD = Path(
    "metadata/config_selection/nested_loso_v1/"
    "amrfinder_database_v1/"
    "amrfinder_supported_organisms.txt"
)

METADATA_ROOT = Path(
    "metadata/config_selection/nested_loso_v1/"
    "amrfinder_full_v1"
)

RESULT_ROOT = Path(
    "results/amr/nested_loso_v1/full_v1"
)

LOG_ROOT = Path(
    "results/logs/config_selection/nested_loso_v1/"
    "amrfinder_full_v1"
)

CHECKPOINT_PATH = (
    METADATA_ROOT /
    "amrfinder_full_checkpoint.tsv"
)

FINAL_STATUS_PATH = (
    METADATA_ROOT /
    "amrfinder_full_run_status.tsv"
)

SUMMARY_PATH = (
    METADATA_ROOT /
    "amrfinder_full_run_summary.tsv"
)

PROTOCOL_PATH = (
    METADATA_ROOT /
    "amrfinder_full_protocol.tsv"
)

INPUT_MANIFEST_PATH = (
    METADATA_ROOT /
    "script137_input_manifest.tsv"
)

DECISION_PATH = (
    METADATA_ROOT /
    "amrfinder_full_decision_2026-07-27.txt"
)

OUTPUT_SHA_PATH = (
    METADATA_ROOT /
    "script137_outputs_sha256.txt"
)

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

ORGANISM_MAP = {
    "kp": "Klebsiella_pneumoniae",
    "ec": "Escherichia",
    "se": "Salmonella",
}

EXPECTED_SPECIES_COUNTS = {
    "kp": 5_602,
    "ec": 6_673,
    "se": 9_119,
}

EXPECTED_TOTAL = sum(
    EXPECTED_SPECIES_COUNTS.values()
)

CHECKPOINT_EVERY = 50


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resume-safe AMRFinderPlus nucleotide "
            "annotation for all nested-LOSO genomes."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help=(
            "Number of concurrent AMRFinder jobs. "
            "Default: 14."
        ),
    )

    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=2,
        help=(
            "AMRFinder --threads value for each job. "
            "Default: 2."
        ),
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_sha_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    if not lines:
        raise RuntimeError(
            f"Empty SHA manifest: {path}"
        )

    for line in lines:
        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed SHA line in {path}: "
                f"{line!r}"
            )

        expected, file_text = parts
        file_path = Path(
            file_text.strip()
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                file_path
            )

        observed = sha256_file(
            file_path
        )

        if observed != expected:
            raise RuntimeError(
                f"SHA mismatch: {file_path}"
            )


def write_tsv_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(
            handle.name
        )

        frame.to_csv(
            handle,
            sep="\t",
            index=False,
            lineterminator="\n",
        )

    temporary_path.replace(path)


def normalize_genome_id(
    value: object,
) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def inspect_output(
    path: Path,
) -> tuple[bool, int, str]:
    if not path.is_file():
        return (
            False,
            -1,
            "output_missing",
        )

    if path.stat().st_size == 0:
        return (
            False,
            -1,
            "output_empty",
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            header = handle.readline().rstrip(
                "\r\n"
            )

            observed_columns = header.split(
                "\t"
            )

            if observed_columns != EXPECTED_COLUMNS:
                return (
                    False,
                    -1,
                    (
                        "schema_mismatch:"
                        + "|".join(
                            observed_columns
                        )
                    ),
                )

            row_count = sum(
                1
                for line in handle
                if line.rstrip("\r\n") != ""
            )

    except Exception as error:
        return (
            False,
            -1,
            (
                f"read_error:"
                f"{type(error).__name__}:"
                f"{error}"
            ),
        )

    return (
        True,
        row_count,
        "",
    )


def resolve_database() -> Path:
    if not DATABASE_LINK.exists():
        raise FileNotFoundError(
            DATABASE_LINK
        )

    database = DATABASE_LINK.resolve()

    if not database.is_dir():
        raise RuntimeError(
            "Resolved AMRFinder database is "
            f"not a directory: {database}"
        )

    command = [
        str(AMRFINDER),
        "--database",
        str(database),
        "--database_version",
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "AMRFinder database-version check "
            f"failed:\n{completed.stdout}"
        )

    if (
        f"Database version: "
        f"{EXPECTED_DATABASE_VERSION}"
        not in completed.stdout
    ):
        raise RuntimeError(
            "Unexpected AMRFinder database "
            f"version:\n{completed.stdout}"
        )

    return database


def verify_software() -> None:
    if not AMRFINDER.is_file():
        raise FileNotFoundError(
            AMRFINDER
        )

    completed = subprocess.run(
        [
            str(AMRFINDER),
            "--version",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "AMRFinder version check failed:\n"
            + completed.stdout
        )

    if (
        EXPECTED_SOFTWARE_VERSION
        not in completed.stdout
    ):
        raise RuntimeError(
            "Unexpected AMRFinder software "
            f"version:\n{completed.stdout}"
        )


def load_source_manifest() -> pd.DataFrame:
    frame = pd.read_csv(
        SOURCE_MANIFEST,
        sep="\t",
        dtype={
            "species_code": str,
            "species": str,
            "genome_id": str,
            "fasta_status": str,
            "fasta_path": str,
        },
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "species_code",
        "species",
        "genome_id",
        "fasta_status",
        "fasta_path",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Source manifest missing columns: "
            + "|".join(
                sorted(missing)
            )
        )

    frame["species_code"] = (
        frame["species_code"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    frame["genome_id"] = (
        frame["genome_id"].map(
            normalize_genome_id
        )
    )

    frame["fasta_path"] = (
        frame["fasta_path"]
        .astype(str)
        .str.strip()
    )

    if frame["genome_id"].duplicated().any():
        duplicates = frame.loc[
            frame["genome_id"].duplicated(
                keep=False
            ),
            "genome_id",
        ].nunique()

        raise RuntimeError(
            "Duplicate genome IDs in source "
            f"manifest: {duplicates}"
        )

    if not frame["fasta_status"].eq(
        "present"
    ).all():
        raise RuntimeError(
            "At least one FASTA is not marked "
            "present."
        )

    observed_counts = (
        frame.groupby(
            "species_code"
        )["genome_id"]
        .nunique()
        .to_dict()
    )

    if (
        observed_counts
        != EXPECTED_SPECIES_COUNTS
    ):
        raise RuntimeError(
            "Species-count mismatch. "
            f"Expected {EXPECTED_SPECIES_COUNTS}; "
            f"observed {observed_counts}."
        )

    if len(frame) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Total genome-count mismatch: "
            f"{len(frame)}"
        )

    species_order = {
        "kp": 0,
        "ec": 1,
        "se": 2,
    }

    frame["_species_order"] = (
        frame["species_code"].map(
            species_order
        )
    )

    if frame["_species_order"].isna().any():
        raise RuntimeError(
            "Unexpected species code."
        )

    frame = (
        frame.sort_values(
            [
                "_species_order",
                "genome_id",
            ]
        )
        .drop(
            columns="_species_order"
        )
        .reset_index(drop=True)
    )

    frame.insert(
        0,
        "annotation_row",
        range(len(frame)),
    )

    frame["amrfinder_organism"] = (
        frame["species_code"].map(
            ORGANISM_MAP
        )
    )

    frame["output_path"] = [
        str(
            RESULT_ROOT
            / species_code
            / f"{genome_id}.amrfinder.tsv"
        )
        for species_code, genome_id
        in zip(
            frame["species_code"],
            frame["genome_id"],
        )
    ]

    frame["stderr_path"] = [
        str(
            LOG_ROOT
            / species_code
            / f"{genome_id}.stderr.log"
        )
        for species_code, genome_id
        in zip(
            frame["species_code"],
            frame["genome_id"],
        )
    ]

    return frame


def run_one(
    record: dict[str, Any],
    database: Path,
    threads_per_job: int,
) -> dict[str, Any]:
    annotation_row = int(
        record["annotation_row"]
    )

    species_code = str(
        record["species_code"]
    )

    species = str(
        record["species"]
    )

    genome_id = str(
        record["genome_id"]
    )

    organism = str(
        record["amrfinder_organism"]
    )

    fasta_path = Path(
        str(record["fasta_path"])
    )

    output_path = Path(
        str(record["output_path"])
    )

    stderr_path = Path(
        str(record["stderr_path"])
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stderr_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_valid, existing_rows, _ = (
        inspect_output(
            output_path
        )
    )

    if existing_valid:
        return {
            "annotation_row":
                annotation_row,
            "species_code":
                species_code,
            "species":
                species,
            "genome_id":
                genome_id,
            "amrfinder_organism":
                organism,
            "fasta_path":
                str(fasta_path),
            "output_path":
                str(output_path),
            "stderr_path":
                str(stderr_path),
            "run_state":
                "reused_valid_output",
            "exit_status":
                0,
            "output_valid":
                True,
            "result_rows":
                existing_rows,
            "output_size_bytes":
                output_path.stat().st_size,
            "output_sha256":
                sha256_file(output_path),
            "elapsed_seconds":
                0.0,
            "started_utc":
                "",
            "completed_utc":
                utc_now(),
            "error_message":
                "",
        }

    if not fasta_path.is_file():
        return {
            "annotation_row":
                annotation_row,
            "species_code":
                species_code,
            "species":
                species,
            "genome_id":
                genome_id,
            "amrfinder_organism":
                organism,
            "fasta_path":
                str(fasta_path),
            "output_path":
                str(output_path),
            "stderr_path":
                str(stderr_path),
            "run_state":
                "failed",
            "exit_status":
                2,
            "output_valid":
                False,
            "result_rows":
                -1,
            "output_size_bytes":
                -1,
            "output_sha256":
                "",
            "elapsed_seconds":
                0.0,
            "started_utc":
                "",
            "completed_utc":
                utc_now(),
            "error_message":
                "fasta_missing",
        }

    worker_token = (
        f"{os.getpid()}_"
        f"{threading.get_ident()}"
    )

    temporary_output = (
        output_path.parent
        / (
            f".{output_path.name}."
            f"{worker_token}.tmp"
        )
    )

    temporary_stderr = (
        stderr_path.parent
        / (
            f".{stderr_path.name}."
            f"{worker_token}.tmp"
        )
    )

    for path in [
        temporary_output,
        temporary_stderr,
    ]:
        if path.exists():
            path.unlink()

    started_utc = utc_now()
    started = time.perf_counter()

    command = [
        str(AMRFINDER),
        "--database",
        str(database),
        "--nucleotide",
        str(fasta_path),
        "--organism",
        organism,
        "--threads",
        str(threads_per_job),
        "--output",
        str(temporary_output),
    ]

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(
        threads_per_job
    )

    with temporary_stderr.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            "COMMAND: "
            + " ".join(command)
            + "\n"
        )

        log_handle.write(
            f"STARTED_UTC: {started_utc}\n"
        )

        log_handle.flush()

        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )

        elapsed_seconds = (
            time.perf_counter()
            - started
        )

        completed_utc = utc_now()

        log_handle.write(
            f"\nCOMPLETED_UTC: "
            f"{completed_utc}\n"
        )

        log_handle.write(
            f"ELAPSED_SECONDS: "
            f"{elapsed_seconds:.6f}\n"
        )

        log_handle.write(
            f"EXIT_STATUS: "
            f"{completed.returncode}\n"
        )

    temporary_stderr.replace(
        stderr_path
    )

    valid = False
    result_rows = -1
    validation_error = ""

    if completed.returncode == 0:
        (
            valid,
            result_rows,
            validation_error,
        ) = inspect_output(
            temporary_output
        )

    if (
        completed.returncode == 0
        and valid
    ):
        temporary_output.replace(
            output_path
        )

        return {
            "annotation_row":
                annotation_row,
            "species_code":
                species_code,
            "species":
                species,
            "genome_id":
                genome_id,
            "amrfinder_organism":
                organism,
            "fasta_path":
                str(fasta_path),
            "output_path":
                str(output_path),
            "stderr_path":
                str(stderr_path),
            "run_state":
                "generated",
            "exit_status":
                0,
            "output_valid":
                True,
            "result_rows":
                result_rows,
            "output_size_bytes":
                output_path.stat().st_size,
            "output_sha256":
                sha256_file(output_path),
            "elapsed_seconds":
                elapsed_seconds,
            "started_utc":
                started_utc,
            "completed_utc":
                completed_utc,
            "error_message":
                "",
        }

    if temporary_output.exists():
        temporary_output.unlink()

    error_message = (
        f"amrfinder_exit_status="
        f"{completed.returncode}"
    )

    if validation_error:
        error_message += (
            f";{validation_error}"
        )

    return {
        "annotation_row":
            annotation_row,
        "species_code":
            species_code,
        "species":
            species,
        "genome_id":
            genome_id,
        "amrfinder_organism":
            organism,
        "fasta_path":
            str(fasta_path),
        "output_path":
            str(output_path),
        "stderr_path":
            str(stderr_path),
        "run_state":
            "failed",
        "exit_status":
            int(completed.returncode),
        "output_valid":
            False,
        "result_rows":
            -1,
        "output_size_bytes":
            -1,
        "output_sha256":
            "",
        "elapsed_seconds":
            elapsed_seconds,
        "started_utc":
            started_utc,
        "completed_utc":
            completed_utc,
        "error_message":
            error_message,
    }


def main() -> None:
    arguments = parse_arguments()

    if arguments.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    if arguments.threads_per_job < 1:
        raise ValueError(
            "--threads-per-job must be "
            "at least 1."
        )

    if (
        arguments.workers
        * arguments.threads_per_job
        > 32
    ):
        raise RuntimeError(
            "Requested worker/thread product "
            "exceeds the audited 32 CPU threads."
        )

    started = time.perf_counter()

    for path in [
        SOURCE_MANIFEST,
        PROBE_SHA_MANIFEST,
        SCRIPT136_FROZEN_MANIFEST,
        DATABASE_VERSION_RECORD,
        ORGANISM_RECORD,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    verify_sha_manifest(
        PROBE_SHA_MANIFEST
    )

    verify_sha_manifest(
        SCRIPT136_FROZEN_MANIFEST
    )

    verify_software()

    database = resolve_database()

    source = load_source_manifest()

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== SCRIPT 137 FULL "
        "ANNOTATION PLAN ====="
    )

    print(
        source.groupby(
            [
                "species_code",
                "species",
                "amrfinder_organism",
            ]
        )
        .agg(
            genomes=(
                "genome_id",
                "nunique",
            )
        )
        .reset_index()
        .to_string(index=False)
    )

    print()
    print(
        "Database:",
        database,
    )

    print(
        "Concurrent jobs:",
        arguments.workers,
    )

    print(
        "Threads per job:",
        arguments.threads_per_job,
    )

    print(
        "Maximum requested AMRFinder "
        "threads:",
        (
            arguments.workers
            * arguments.threads_per_job
        ),
    )

    print()
    print(
        "===== RUNNING / RESUMING "
        "AMRFINDER ANNOTATION =====",
        flush=True,
    )

    records = []
    completed_count = 0
    generated_count = 0
    reused_count = 0
    failed_count = 0

    futures = {}

    with ThreadPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        for record in source.to_dict(
            orient="records"
        ):
            future = executor.submit(
                run_one,
                record,
                database,
                arguments.threads_per_job,
            )

            futures[future] = (
                record["annotation_row"],
                record["genome_id"],
            )

        for future in as_completed(
            futures
        ):
            annotation_row, genome_id = (
                futures[future]
            )

            try:
                result = future.result()
            except Exception as error:
                result = {
                    "annotation_row":
                        int(annotation_row),
                    "species_code":
                        "",
                    "species":
                        "",
                    "genome_id":
                        str(genome_id),
                    "amrfinder_organism":
                        "",
                    "fasta_path":
                        "",
                    "output_path":
                        "",
                    "stderr_path":
                        "",
                    "run_state":
                        "worker_exception",
                    "exit_status":
                        99,
                    "output_valid":
                        False,
                    "result_rows":
                        -1,
                    "output_size_bytes":
                        -1,
                    "output_sha256":
                        "",
                    "elapsed_seconds":
                        0.0,
                    "started_utc":
                        "",
                    "completed_utc":
                        utc_now(),
                    "error_message":
                        (
                            f"{type(error).__name__}:"
                            f"{error}"
                        ),
                }

            records.append(result)
            completed_count += 1

            if (
                result["run_state"]
                == "generated"
            ):
                generated_count += 1
            elif (
                result["run_state"]
                == "reused_valid_output"
            ):
                reused_count += 1
            else:
                failed_count += 1

            if (
                completed_count
                % CHECKPOINT_EVERY
                == 0
                or completed_count
                == EXPECTED_TOTAL
            ):
                checkpoint = (
                    pd.DataFrame(records)
                    .sort_values(
                        "annotation_row"
                    )
                    .reset_index(drop=True)
                )

                write_tsv_atomic(
                    checkpoint,
                    CHECKPOINT_PATH,
                )

                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    "Completed:",
                    f"{completed_count}/"
                    f"{EXPECTED_TOTAL}",
                    f"generated={generated_count}",
                    f"reused={reused_count}",
                    f"failed={failed_count}",
                    f"elapsed="
                    f"{elapsed / 3600:.2f} h",
                    flush=True,
                )

    status = (
        pd.DataFrame(records)
        .sort_values(
            "annotation_row"
        )
        .reset_index(drop=True)
    )

    if len(status) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Final status row count mismatch: "
            f"{len(status)}"
        )

    if status[
        "annotation_row"
    ].tolist() != list(
        range(EXPECTED_TOTAL)
    ):
        raise RuntimeError(
            "Final status annotation-row order "
            "is incomplete or duplicated."
        )

    if status[
        "genome_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate genome IDs in final "
            "status table."
        )

    write_tsv_atomic(
        status,
        FINAL_STATUS_PATH,
    )

    failed = status.loc[
        (
            ~status["exit_status"].eq(0)
        )
        | (
            ~status["output_valid"].eq(True)
        )
    ].copy()

    if not failed.empty:
        print()
        print(
            "===== FAILED ANNOTATIONS ====="
        )

        print(
            failed[
                [
                    "annotation_row",
                    "species_code",
                    "genome_id",
                    "exit_status",
                    "error_message",
                    "stderr_path",
                ]
            ]
            .head(100)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"{len(failed)} AMRFinder "
            "annotations failed. Re-run Script "
            "137 to resume only missing/invalid "
            "outputs."
        )

    summary = (
        status.groupby(
            [
                "species_code",
                "species",
                "amrfinder_organism",
            ],
            sort=True,
        )
        .agg(
            genomes=(
                "genome_id",
                "nunique",
            ),
            generated=(
                "run_state",
                lambda values:
                    int(
                        values.eq(
                            "generated"
                        ).sum()
                    ),
            ),
            reused=(
                "run_state",
                lambda values:
                    int(
                        values.eq(
                            "reused_valid_output"
                        ).sum()
                    ),
            ),
            total_result_rows=(
                "result_rows",
                "sum",
            ),
            median_elapsed_seconds=(
                "elapsed_seconds",
                "median",
            ),
            maximum_elapsed_seconds=(
                "elapsed_seconds",
                "max",
            ),
            total_output_bytes=(
                "output_size_bytes",
                "sum",
            ),
        )
        .reset_index()
    )

    write_tsv_atomic(
        summary,
        SUMMARY_PATH,
    )

    protocol = pd.DataFrame(
        [
            {
                "item":
                    "annotation_scope",
                "value":
                    (
                        "all 21,394 genomes in "
                        "the nested-LOSO "
                        "configuration panels"
                    ),
            },
            {
                "item":
                    "annotation_mode",
                "value":
                    (
                        "AMRFinderPlus nucleotide "
                        "search with species-specific "
                        "--organism mutation models"
                    ),
            },
            {
                "item":
                    "software_version",
                "value":
                    EXPECTED_SOFTWARE_VERSION,
            },
            {
                "item":
                    "database_version",
                "value":
                    EXPECTED_DATABASE_VERSION,
            },
            {
                "item":
                    "database_resolved_path",
                "value":
                    str(database),
            },
            {
                "item":
                    "organism_mapping",
                "value":
                    "|".join(
                        (
                            f"{species}:"
                            f"{organism}"
                        )
                        for species, organism
                        in ORGANISM_MAP.items()
                    ),
            },
            {
                "item":
                    "plus_option",
                "value":
                    "not enabled",
            },
            {
                "item":
                    "workers",
                "value":
                    arguments.workers,
            },
            {
                "item":
                    "threads_per_job",
                "value":
                    arguments.threads_per_job,
            },
            {
                "item":
                    "resume_policy",
                "value":
                    (
                        "reuse an existing output "
                        "only when its 22-column "
                        "schema is exact and readable"
                    ),
            },
            {
                "item":
                    "output_policy",
                "value":
                    (
                        "one raw AMRFinder TSV per "
                        "genome; atomic temporary-file "
                        "replacement"
                    ),
            },
            {
                "item":
                    "target_label_policy",
                "value":
                    (
                        "no MIC labels used during "
                        "genome annotation"
                    ),
            },
            {
                "item":
                    "models_trained",
                "value":
                    "none",
            },
            {
                "item":
                    "software_runtime",
                "value":
                    (
                        f"python="
                        f"{platform.python_version()};"
                        f"pandas={pd.__version__}"
                    ),
            },
        ]
    )

    write_tsv_atomic(
        protocol,
        PROTOCOL_PATH,
    )

    input_paths = {
        AMRFINDER,
        SOURCE_MANIFEST,
        PROBE_SHA_MANIFEST,
        SCRIPT136_FROZEN_MANIFEST,
        DATABASE_VERSION_RECORD,
        ORGANISM_RECORD,
    }

    input_manifest = pd.DataFrame(
        [
            {
                "file_path":
                    str(path),
                "file_size_bytes":
                    path.stat().st_size,
                "sha256":
                    sha256_file(path),
            }
            for path in sorted(
                input_paths,
                key=lambda value:
                    value.as_posix(),
            )
        ]
    )

    write_tsv_atomic(
        input_manifest,
        INPUT_MANIFEST_PATH,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    DECISION_PATH.write_text(
        "\n".join(
            [
                (
                    "Nested-LOSO full "
                    "AMRFinderPlus annotation"
                ),
                "",
                (
                    f"Genomes: "
                    f"{EXPECTED_TOTAL}"
                ),
                (
                    "Software version: "
                    f"{EXPECTED_SOFTWARE_VERSION}"
                ),
                (
                    "Database version: "
                    f"{EXPECTED_DATABASE_VERSION}"
                ),
                (
                    "Concurrent jobs: "
                    f"{arguments.workers}"
                ),
                (
                    "Threads per job: "
                    f"{arguments.threads_per_job}"
                ),
                (
                    "Total AMRFinder call rows: "
                    f"{int(status['result_rows'].sum())}"
                ),
                (
                    "Elapsed hours: "
                    f"{elapsed_seconds / 3600:.3f}"
                ),
                "Models trained: none",
                "",
                (
                    "STATUS: SCRIPT 137 "
                    "THREE-SPECIES AMRFINDER "
                    "ANNOTATION COMPLETE."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_paths = {
        FINAL_STATUS_PATH,
        SUMMARY_PATH,
        PROTOCOL_PATH,
        INPUT_MANIFEST_PATH,
        DECISION_PATH,
        *[
            Path(path)
            for path in status[
                "output_path"
            ]
        ],
    }

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
                f"{sha256_file(path)}  "
                f"{path}\n"
            )

    verify_sha_manifest(
        OUTPUT_SHA_PATH
    )

    print()
    print(
        "===== SCRIPT 137 FINAL SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Total genomes:",
        len(status),
    )

    print(
        "Total AMRFinder result rows:",
        int(
            status[
                "result_rows"
            ].sum()
        ),
    )

    print(
        "Elapsed hours:",
        f"{elapsed_seconds / 3600:.3f}",
    )

    print(
        "Models trained: NO"
    )

    print()
    print(
        "STATUS: SCRIPT 137 "
        "THREE-SPECIES AMRFINDER "
        "ANNOTATION COMPLETE"
    )


if __name__ == "__main__":
    main()
