#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_PATH = Path(
    "metadata/genomes/"
    "modelling_genome_acquisition_manifest.tsv"
)

KLEBORATE_EXECUTABLE = Path(
    os.environ.get(
        "KLEBORATE_EXECUTABLE",
        shutil.which("kleborate") or "kleborate",
    )
)

CHUNK_ROOT = Path(
    "results/taxonomy/"
    "kleborate_enterobacterales_chunks"
)

CHUNK_LOG_ROOT = Path(
    "results/logs/taxonomy/"
    "kleborate_enterobacterales_chunks"
)

OUTPUT_MANIFEST_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_kleborate_species_manifest.tsv"
)

EXCEPTION_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_kleborate_species_exceptions.tsv"
)

CHUNK_MANIFEST_PATH = Path(
    "metadata/taxonomy/"
    "enterobacterales_kleborate_chunk_manifest.tsv"
)

SUMMARY_PATH = Path(
    "results/tables/taxonomy/"
    "enterobacterales_kleborate_species_summary.tsv"
)

OUTPUT_SHA_PATH = Path(
    "metadata/taxonomy/"
    "script50_outputs_sha256.txt"
)

EXPECTED_SPECIES = {
    "Escherichia coli":
        "Escherichia coli / Shigella",
    "Klebsiella pneumoniae":
        "Klebsiella pneumoniae",
    "Salmonella enterica":
        "Salmonella enterica",
}

EXPECTED_TOTAL_GENOMES = 21_478


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def write_sha_manifest(
    paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        f"{sha256_file(path)}  {path}"
        for path in sorted(
            paths,
            key=lambda item: str(item),
        )
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_existing_chunk(
    output_path: Path,
    expected_ids: list[str],
) -> pd.DataFrame | None:
    if not output_path.is_file():
        return None

    try:
        frame = pd.read_csv(
            output_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except Exception:
        return None

    required = {
        "strain",
        "species",
        "species_match",
    }

    if not required.issubset(frame.columns):
        return None

    observed_ids = frame["strain"].tolist()

    if observed_ids != expected_ids:
        return None

    return frame


def run_chunk(
    chunk_order: int,
    chunk: pd.DataFrame,
) -> dict[str, Any]:
    chunk_name = f"chunk_{chunk_order:04d}"

    output_dir = CHUNK_ROOT / chunk_name

    output_path = (
        output_dir
        / "enterobacterales__species_output.txt"
    )

    stdout_path = (
        CHUNK_LOG_ROOT
        / f"{chunk_name}.stdout.log"
    )

    stderr_path = (
        CHUNK_LOG_ROOT
        / f"{chunk_name}.stderr.log"
    )

    expected_ids = (
        chunk["genome_id"]
        .astype(str)
        .tolist()
    )

    existing = parse_existing_chunk(
        output_path,
        expected_ids,
    )

    if existing is not None:
        return {
            "chunk_order": chunk_order,
            "chunk_name": chunk_name,
            "genomes": len(chunk),
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "execution_status":
                "validated_existing_chunk",
            "output_path": str(output_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "output_sha256":
                sha256_file(output_path),
            "frame": existing,
        }

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stdout_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(KLEBORATE_EXECUTABLE),
        "-a",
        *chunk["local_fasta_path"].tolist(),
        "-o",
        str(output_dir),
        "-m",
        "enterobacterales__species",
        "--trim_headers",
    ]

    start = time.monotonic()

    with (
        stdout_path.open(
            "w",
            encoding="utf-8",
        ) as stdout_handle,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle,
    ):
        execution_environment = os.environ.copy()

        kleborate_bin = str(
            KLEBORATE_EXECUTABLE.parent
        )

        execution_environment["PATH"] = (
            kleborate_bin
            + os.pathsep
            + execution_environment.get(
                "PATH",
                "",
            )
        )

        completed = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=execution_environment,
        )

    elapsed = time.monotonic() - start

    if completed.returncode != 0:
        raise RuntimeError(
            f"{chunk_name} failed with return code "
            f"{completed.returncode}. See {stderr_path}."
        )

    parsed = parse_existing_chunk(
        output_path,
        expected_ids,
    )

    if parsed is None:
        raise RuntimeError(
            f"{chunk_name} produced an invalid or "
            "incomplete output table."
        )

    return {
        "chunk_order": chunk_order,
        "chunk_name": chunk_name,
        "genomes": len(chunk),
        "return_code": completed.returncode,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "execution_status":
            "completed_new_chunk",
        "output_path": str(output_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_sha256":
            sha256_file(output_path),
        "frame": parsed,
    }


def classify_call(
    provisional_species: str,
    kleborate_species: str,
    species_match: str,
) -> str:
    expected = EXPECTED_SPECIES[
        provisional_species
    ]

    observed = str(
        kleborate_species
    ).strip()

    match_strength = str(
        species_match
    ).strip().casefold()

    if not observed or observed.casefold() == "unknown":
        return "unknown"

    if observed == expected:
        if match_strength == "strong":
            return "concordant_strong"

        if match_strength == "weak":
            return "concordant_weak"

        return "concordant_unspecified_strength"

    if match_strength == "strong":
        return "discordant_strong"

    if match_strength == "weak":
        return "discordant_weak"

    return "discordant_unspecified_strength"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    if args.chunk_size < 1:
        raise ValueError(
            "--chunk-size must be at least 1."
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    if not KLEBORATE_EXECUTABLE.is_file():
        raise FileNotFoundError(
            KLEBORATE_EXECUTABLE
        )

    CHUNK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHUNK_LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_manifest_columns = {
        "acquisition_order",
        "genome_id",
        "provisional_species",
        "local_fasta_path",
        "fasta_sha256",
    }

    missing_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing acquisition-manifest columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    manifest[
        "acquisition_order_numeric"
    ] = pd.to_numeric(
        manifest["acquisition_order"],
        errors="raise",
    ).astype("int64")

    cohort = (
        manifest.loc[
            manifest[
                "provisional_species"
            ].isin(
                EXPECTED_SPECIES
            )
        ]
        .sort_values(
            "acquisition_order_numeric",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(cohort) != EXPECTED_TOTAL_GENOMES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_GENOMES:,} "
            "Enterobacterales genomes; found "
            f"{len(cohort):,}."
        )

    if cohort["genome_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate Enterobacterales genome IDs."
        )

    missing_fastas = [
        path_text
        for path_text in cohort[
            "local_fasta_path"
        ]
        if not Path(path_text).is_file()
    ]

    if missing_fastas:
        raise FileNotFoundError(
            f"Missing {len(missing_fastas):,} "
            "Enterobacterales FASTA files."
        )

    chunks: list[
        tuple[int, pd.DataFrame]
    ] = []

    total_chunks = math.ceil(
        len(cohort) / args.chunk_size
    )

    for chunk_index, start in enumerate(
        range(
            0,
            len(cohort),
            args.chunk_size,
        ),
        start=1,
    ):
        chunk = cohort.iloc[
            start:
            start + args.chunk_size
        ].copy()

        chunks.append(
            (
                chunk_index,
                chunk,
            )
        )

    print(
        "===== SCRIPT 50 ENTEROBACTERALES "
        "KLEBORATE SPECIES VERIFICATION =====",
        flush=True,
    )

    print(
        "Enterobacterales genomes:",
        f"{len(cohort):,}",
        flush=True,
    )

    print(
        "Species represented:",
        cohort[
            "provisional_species"
        ].nunique(),
        flush=True,
    )

    print(
        "Chunk size:",
        args.chunk_size,
        flush=True,
    )

    print(
        "Chunks:",
        total_chunks,
        flush=True,
    )

    print(
        "Concurrent workers:",
        args.workers,
        flush=True,
    )

    chunk_results: list[
        dict[str, Any]
    ] = []

    completed_genomes = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                run_chunk,
                chunk_order,
                chunk,
            ): chunk_order
            for chunk_order, chunk in chunks
        }

        for completed_count, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            try:
                result = future.result()
            except Exception:
                for pending_future in futures:
                    pending_future.cancel()

                raise

            chunk_results.append(result)

            completed_genomes += int(
                result["genomes"]
            )

            print(
                "Completed chunks:",
                f"{completed_count:,}/{total_chunks:,};",
                "processed genomes:",
                f"{completed_genomes:,}/{len(cohort):,};",
                "latest:",
                result["chunk_name"],
                result["execution_status"],
                flush=True,
            )

    chunk_results.sort(
        key=lambda item: int(
            item["chunk_order"]
        )
    )

    combined_frames = []

    for result in chunk_results:
        frame = result["frame"].copy()

        frame.insert(
            0,
            "chunk_order",
            int(result["chunk_order"]),
        )

        combined_frames.append(frame)

    calls = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    if len(calls) != len(cohort):
        raise RuntimeError(
            "Combined Kleborate row count does "
            "not match the Enterobacterales cohort."
        )

    if calls["strain"].duplicated().any():
        raise RuntimeError(
            "Duplicate strain IDs in combined "
            "Kleborate output."
        )

    calls = calls.rename(
        columns={
            "strain": "genome_id",
            "species": "kleborate_species",
            "species_match":
                "kleborate_species_match",
        }
    )

    combined = cohort.merge(
        calls,
        on="genome_id",
        how="left",
        validate="one_to_one",
    )

    if combined[
        "kleborate_species"
    ].isna().any():
        raise RuntimeError(
            "Missing Kleborate results after merge."
        )

    combined[
        "expected_kleborate_species"
    ] = combined[
        "provisional_species"
    ].map(
        EXPECTED_SPECIES
    )

    combined[
        "taxonomy_concordance_status"
    ] = [
        classify_call(
            provisional_species,
            kleborate_species,
            species_match,
        )
        for (
            provisional_species,
            kleborate_species,
            species_match,
        ) in zip(
            combined[
                "provisional_species"
            ],
            combined[
                "kleborate_species"
            ],
            combined[
                "kleborate_species_match"
            ],
            strict=True,
        )
    ]

    combined[
        "passes_kleborate_strong_concordance"
    ] = combined[
        "taxonomy_concordance_status"
    ].eq(
        "concordant_strong"
    )

    combined = combined.sort_values(
        "acquisition_order_numeric",
        kind="mergesort",
    ).reset_index(drop=True)

    output_columns = [
        "acquisition_order",
        "genome_id",
        "provisional_species",
        "expected_kleborate_species",
        "kleborate_species",
        "kleborate_species_match",
        "taxonomy_concordance_status",
        "passes_kleborate_strong_concordance",
        "chunk_order",
        "local_fasta_path",
        "fasta_sha256",
    ]

    combined[
        output_columns
    ].to_csv(
        OUTPUT_MANIFEST_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    exceptions = combined.loc[
        ~combined[
            "passes_kleborate_strong_concordance"
        ],
        output_columns,
    ].copy()

    exceptions.to_csv(
        EXCEPTION_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    status_counts = (
        combined.groupby(
            [
                "provisional_species",
                "taxonomy_concordance_status",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "genomes",
            }
        )
    )

    species_totals = (
        combined.groupby(
            "provisional_species",
            as_index=False,
        )
        .agg(
            total_genomes=(
                "genome_id",
                "size",
            ),
            strong_concordant_genomes=(
                "passes_kleborate_strong_concordance",
                "sum",
            ),
        )
    )

    species_totals[
        "strong_concordance_rate"
    ] = (
        species_totals[
            "strong_concordant_genomes"
        ]
        / species_totals[
            "total_genomes"
        ]
    )

    summary = status_counts.merge(
        species_totals,
        on="provisional_species",
        how="left",
        validate="many_to_one",
    )

    summary.to_csv(
        SUMMARY_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    chunk_manifest = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "frame"
            }
            for result in chunk_results
        ]
    )

    chunk_manifest.to_csv(
        CHUNK_MANIFEST_PATH,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    write_sha_manifest(
        [
            OUTPUT_MANIFEST_PATH,
            EXCEPTION_PATH,
            CHUNK_MANIFEST_PATH,
            SUMMARY_PATH,
        ],
        OUTPUT_SHA_PATH,
    )

    print()
    print(
        "===== FINAL TAXONOMY STATUS COUNTS ====="
    )

    print(
        status_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Strong concordant genomes:",
        f"{combined['passes_kleborate_strong_concordance'].sum():,}",
    )

    print(
        "Exception genomes:",
        f"{len(exceptions):,}",
    )

    print(
        "Combined manifest rows:",
        f"{len(combined):,}",
    )

    print(
        "Production FASTA files modified:",
        "NO",
    )

    print(
        "Genomes automatically excluded:",
        "0",
    )

    print()
    print(
        "STATUS: ENTEROBACTERALES KLEBORATE "
        "SPECIES VERIFICATION COMPLETE"
    )


if __name__ == "__main__":
    main()
