#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()

SCRIPT48_CORE = Path(
    "metadata/genomes/"
    "script48_successful_run_core_sha256.txt"
)

FASTA_ROOT = Path(
    "genomes/raw/bvbrc_fasta"
)

OUTPUT_ROOT = Path(
    "metadata/taxonomy"
)

TABLE_ROOT = Path(
    "results/tables/taxonomy"
)

EXPECTED_FASTAS = 23_632


TOOL_VERSION_ARGUMENTS: dict[str, list[str]] = {
    "kleborate": ["--version"],
    "mash": ["--version"],
    "sourmash": ["--version"],
    "fastANI": ["--version"],
    "skani": ["--version"],
    "kraken2": ["--version"],
    "kraken2-inspect": ["--version"],
    "centrifuge": ["--version"],
    "gtdbtk": ["--version"],
    "checkm": ["--version"],
    "checkm2": ["--version"],
    "mlst": ["--version"],
    "SeqSero2_package.py": ["-h"],
    "seqsero2": ["-h"],
    "sistr": ["--version"],
    "poppunk": ["--version"],
    "poppunk_visualise": ["--version"],
    "blastn": ["-version"],
    "makeblastdb": ["-version"],
    "minimap2": ["--version"],
    "mmseqs": ["version"],
    "diamond": ["version"],
    "prodigal": ["-v"],
    "datasets": ["version"],
    "ncbi-genome-download": ["--version"],
    "amrfinder": ["--version"],
    "abricate": ["--version"],
    "kma": ["-v"],
}


PACKAGE_KEYWORDS = (
    "kleborate",
    "mash",
    "sourmash",
    "fastani",
    "skani",
    "kraken",
    "centrifuge",
    "gtdb",
    "checkm",
    "mlst",
    "seqsero",
    "sistr",
    "poppunk",
    "blast",
    "minimap",
    "mmseqs",
    "diamond",
    "prodigal",
    "amrfinder",
    "abricate",
    "kma",
)


DATABASE_ENVIRONMENT_VARIABLES = [
    "GTDBTK_DATA_PATH",
    "KRAKEN2_DEFAULT_DB",
    "CHECKM_DATA_PATH",
    "CHECKM2DB",
    "CHECKM2_DB",
    "KLEBORATE_DATA_DIR",
    "SOURMASH_DB",
    "CENTRIFUGE_INDEXES",
    "BLASTDB",
]


DATABASE_DIRECTORY_KEYWORDS = (
    "gtdb",
    "kraken",
    "centrifuge",
    "checkm",
    "kleborate",
    "seqsero",
    "sistr",
    "sourmash",
    "mash",
    "refseq",
    "taxonomy",
    "taxdump",
    "species",
    "reference",
)


DATABASE_FILE_MARKERS = (
    "hash.k2d",
    "opts.k2d",
    "taxo.k2d",
    "nodes.dmp",
    "names.dmp",
    "merged.dmp",
    "delnodes.dmp",
)


DATABASE_FILE_SUFFIXES = (
    ".msh",
    ".dmnd",
    ".mmi",
    ".sbt.zip",
    ".sig",
    ".sketch",
)


PRUNE_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "pkgs",
    "cache",
    "tmp",
    "temp",
}


PRUNE_PATHS = {
    (PROJECT_ROOT / "genomes/raw/bvbrc_fasta").resolve(),
    (PROJECT_ROOT / "genomes/probes").resolve(),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def run_command(
    command: list[str],
    timeout_seconds: int = 15,
) -> tuple[int, str, str]:
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        return (
            process.returncode,
            process.stdout,
            process.stderr,
        )

    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout
            if isinstance(error.stdout, str)
            else ""
        )

        stderr = (
            error.stderr
            if isinstance(error.stderr, str)
            else ""
        )

        return (
            124,
            stdout,
            stderr + "\nCOMMAND TIMED OUT",
        )

    except Exception as error:
        return (
            125,
            "",
            f"{type(error).__name__}: {error}",
        )


def compact_output(
    stdout: str,
    stderr: str,
    maximum_lines: int = 4,
) -> str:
    combined = []

    for text in [stdout, stderr]:
        for line in text.splitlines():
            line = line.strip()

            if line:
                combined.append(line)

    return " | ".join(
        combined[:maximum_lines]
    )[:1000]


def discover_conda_environments() -> list[Path]:
    environments: list[Path] = []

    conda = shutil.which("conda")

    if conda:
        return_code, stdout, _ = run_command(
            [
                conda,
                "env",
                "list",
                "--json",
            ],
            timeout_seconds=30,
        )

        if return_code == 0:
            try:
                payload = json.loads(stdout)

                for value in payload.get(
                    "envs",
                    [],
                ):
                    path = Path(value).resolve()

                    if path.is_dir():
                        environments.append(path)

            except json.JSONDecodeError:
                pass

    fallback_roots = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]

    for root in fallback_roots:
        if root.is_dir():
            environments.append(
                root.resolve()
            )

            env_root = root / "envs"

            if env_root.is_dir():
                for path in env_root.iterdir():
                    if path.is_dir():
                        environments.append(
                            path.resolve()
                        )

    environments.append(
        Path(sys.prefix).resolve()
    )

    return sorted(
        set(environments),
        key=lambda path: path.as_posix(),
    )


def environment_name(path: Path) -> str:
    if path.parent.name == "envs":
        return path.name

    if path.name in {
        "miniconda3",
        "anaconda3",
    }:
        return "base"

    return path.name


def inventory_tools(
    environments: list[Path],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    seen_paths: set[
        tuple[str, str]
    ] = set()

    for environment in environments:
        bin_directory = (
            environment / "bin"
        )

        for tool, version_arguments in (
            TOOL_VERSION_ARGUMENTS.items()
        ):
            executable = (
                bin_directory / tool
            )

            if not (
                executable.is_file()
                and os.access(
                    executable,
                    os.X_OK,
                )
            ):
                continue

            key = (
                environment.as_posix(),
                executable.resolve().as_posix(),
            )

            if key in seen_paths:
                continue

            seen_paths.add(key)

            return_code, stdout, stderr = run_command(
                [
                    executable.as_posix(),
                    *version_arguments,
                ]
            )

            records.append(
                {
                    "environment":
                        environment_name(
                            environment
                        ),
                    "environment_path":
                        environment.as_posix(),
                    "tool":
                        tool,
                    "executable_path":
                        executable.resolve().as_posix(),
                    "version_command":
                        " ".join(
                            [
                                tool,
                                *version_arguments,
                            ]
                        ),
                    "version_return_code":
                        return_code,
                    "version_output":
                        compact_output(
                            stdout,
                            stderr,
                        ),
                }
            )

    for tool, version_arguments in (
        TOOL_VERSION_ARGUMENTS.items()
    ):
        discovered = shutil.which(tool)

        if discovered is None:
            continue

        executable = Path(
            discovered
        ).resolve()

        if any(
            record[
                "executable_path"
            ] == executable.as_posix()
            for record in records
        ):
            continue

        return_code, stdout, stderr = run_command(
            [
                executable.as_posix(),
                *version_arguments,
            ]
        )

        records.append(
            {
                "environment":
                    "current_PATH",
                "environment_path":
                    "",
                "tool":
                    tool,
                "executable_path":
                    executable.as_posix(),
                "version_command":
                    " ".join(
                        [
                            tool,
                            *version_arguments,
                        ]
                    ),
                "version_return_code":
                    return_code,
                "version_output":
                    compact_output(
                        stdout,
                        stderr,
                    ),
            }
        )

    columns = [
        "environment",
        "environment_path",
        "tool",
        "executable_path",
        "version_command",
        "version_return_code",
        "version_output",
    ]

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "tool",
                "environment",
                "executable_path",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


def inventory_conda_packages(
    environments: list[Path],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for environment in environments:
        metadata_root = (
            environment / "conda-meta"
        )

        if not metadata_root.is_dir():
            continue

        for path in metadata_root.glob(
            "*.json"
        ):
            try:
                payload = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue

            package_name = str(
                payload.get(
                    "name",
                    "",
                )
            )

            lowered = package_name.casefold()

            if not any(
                keyword in lowered
                for keyword in PACKAGE_KEYWORDS
            ):
                continue

            records.append(
                {
                    "environment":
                        environment_name(
                            environment
                        ),
                    "environment_path":
                        environment.as_posix(),
                    "package_name":
                        package_name,
                    "version":
                        str(
                            payload.get(
                                "version",
                                "",
                            )
                        ),
                    "build":
                        str(
                            payload.get(
                                "build",
                                "",
                            )
                        ),
                    "channel":
                        str(
                            payload.get(
                                "channel",
                                "",
                            )
                        ),
                    "metadata_path":
                        path.as_posix(),
                }
            )

    columns = [
        "environment",
        "environment_path",
        "package_name",
        "version",
        "build",
        "channel",
        "metadata_path",
    ]

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(records)
        .drop_duplicates()
        .sort_values(
            [
                "package_name",
                "environment",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


def path_is_pruned(path: Path) -> bool:
    resolved = path.resolve()

    for prune_path in PRUNE_PATHS:
        if (
            resolved == prune_path
            or prune_path
            in resolved.parents
        ):
            return True

    return False


def candidate_database_roots(
    environments: list[Path],
) -> list[Path]:
    roots: set[Path] = set()

    for variable in (
        DATABASE_ENVIRONMENT_VARIABLES
    ):
        value = os.environ.get(
            variable,
            "",
        ).strip()

        if value:
            path = Path(
                value
            ).expanduser()

            if path.exists():
                roots.add(
                    path.resolve()
                )

    named_candidates = [
        PROJECT_ROOT / "database",
        PROJECT_ROOT / "databases",
        PROJECT_ROOT / "db",
        PROJECT_ROOT / "reference",
        PROJECT_ROOT / "references",
        PROJECT_ROOT / "refdata",
        Path.home() / "database",
        Path.home() / "databases",
        Path.home() / "db",
        Path.home() / "reference",
        Path.home() / "references",
        Path.home() / "refs",
        Path.home() / "refdata",
        Path.home() / "taxonomy",
    ]

    for path in named_candidates:
        if path.exists():
            roots.add(
                path.resolve()
            )

    for environment in environments:
        share = environment / "share"

        if share.is_dir():
            roots.add(
                share.resolve()
            )

    for broad_root in [
        Path.home(),
        Path("/data"),
        Path("/opt"),
        Path("/srv"),
    ]:
        if not broad_root.is_dir():
            continue

        try:
            children = list(
                broad_root.iterdir()
            )
        except PermissionError:
            continue

        for child in children:
            if not child.is_dir():
                continue

            lowered = (
                child.name.casefold()
            )

            if any(
                keyword in lowered
                for keyword in (
                    DATABASE_DIRECTORY_KEYWORDS
                )
            ):
                roots.add(
                    child.resolve()
                )

    return sorted(
        {
            path
            for path in roots
            if path.exists()
            and not path_is_pruned(path)
        },
        key=lambda path:
            path.as_posix(),
    )


def inventory_databases(
    environments: list[Path],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for variable in (
        DATABASE_ENVIRONMENT_VARIABLES
    ):
        value = os.environ.get(
            variable,
            "",
        ).strip()

        records.append(
            {
                "record_type":
                    "environment_variable",
                "resource_name":
                    variable,
                "path":
                    value,
                "exists":
                    bool(
                        value
                        and Path(
                            value
                        ).expanduser().exists()
                    ),
                "size_bytes":
                    "",
                "notes":
                    "set"
                    if value
                    else "not_set",
            }
        )

    roots = candidate_database_roots(
        environments
    )

    seen_paths: set[str] = set()

    for root in roots:
        maximum_depth = 5

        for current_text, directories, files in os.walk(
            root,
            topdown=True,
        ):
            current = Path(
                current_text
            )

            try:
                relative_depth = len(
                    current.relative_to(
                        root
                    ).parts
                )
            except ValueError:
                continue

            filtered_directories = []

            for directory_name in directories:
                candidate = (
                    current
                    / directory_name
                )

                if (
                    directory_name
                    in PRUNE_DIRECTORY_NAMES
                    or path_is_pruned(
                        candidate
                    )
                ):
                    continue

                filtered_directories.append(
                    directory_name
                )

            directories[:] = (
                filtered_directories
            )

            if relative_depth >= maximum_depth:
                directories[:] = []

            current_lower = (
                current.name.casefold()
            )

            if any(
                keyword in current_lower
                for keyword in (
                    DATABASE_DIRECTORY_KEYWORDS
                )
            ):
                path_text = (
                    current.resolve().as_posix()
                )

                if path_text not in seen_paths:
                    seen_paths.add(
                        path_text
                    )

                    records.append(
                        {
                            "record_type":
                                "candidate_directory",
                            "resource_name":
                                current.name,
                            "path":
                                path_text,
                            "exists":
                                True,
                            "size_bytes":
                                "",
                            "notes":
                                f"root={root}",
                        }
                    )

            for filename in files:
                lowered = filename.casefold()

                is_marker = (
                    lowered
                    in DATABASE_FILE_MARKERS
                )

                is_suffix = any(
                    lowered.endswith(
                        suffix
                    )
                    for suffix in (
                        DATABASE_FILE_SUFFIXES
                    )
                )

                is_gtdb_metadata = (
                    lowered.startswith(
                        "bac120_metadata"
                    )
                    or lowered.startswith(
                        "ar53_metadata"
                    )
                )

                if not (
                    is_marker
                    or is_suffix
                    or is_gtdb_metadata
                ):
                    continue

                path = (
                    current / filename
                )

                path_text = (
                    path.resolve().as_posix()
                )

                if path_text in seen_paths:
                    continue

                seen_paths.add(
                    path_text
                )

                try:
                    size_bytes: int | str = (
                        path.stat().st_size
                    )
                except OSError:
                    size_bytes = ""

                records.append(
                    {
                        "record_type":
                            "candidate_file",
                        "resource_name":
                            filename,
                        "path":
                            path_text,
                        "exists":
                            True,
                        "size_bytes":
                            size_bytes,
                        "notes":
                            f"root={root}",
                    }
                )

    columns = [
        "record_type",
        "resource_name",
        "path",
        "exists",
        "size_bytes",
        "notes",
    ]

    return (
        pd.DataFrame(
            records,
            columns=columns,
        )
        .drop_duplicates()
        .sort_values(
            [
                "record_type",
                "resource_name",
                "path",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    print(
        "===== SCRIPT 49 SEQUENCE-TAXONOMY "
        "RESOURCE INVENTORY ====="
    )

    if not SCRIPT48_CORE.is_file():
        raise FileNotFoundError(
            f"Missing frozen Script 48 core: "
            f"{SCRIPT48_CORE}"
        )

    fasta_files = list(
        FASTA_ROOT.glob("*.fna")
    )

    partial_files = list(
        FASTA_ROOT.glob("*.part")
    )

    if len(fasta_files) != EXPECTED_FASTAS:
        raise RuntimeError(
            "Expected 23,632 frozen production "
            f"FASTAs; found {len(fasta_files):,}."
        )

    if partial_files:
        raise RuntimeError(
            "Partial production FASTA files remain."
        )

    environments = (
        discover_conda_environments()
    )

    tools = inventory_tools(
        environments
    )

    packages = inventory_conda_packages(
        environments
    )

    databases = inventory_databases(
        environments
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tools_path = (
        OUTPUT_ROOT
        / "sequence_taxonomy_tool_inventory.tsv"
    )

    packages_path = (
        OUTPUT_ROOT
        / "sequence_taxonomy_conda_package_inventory.tsv"
    )

    databases_path = (
        OUTPUT_ROOT
        / "sequence_taxonomy_database_inventory.tsv"
    )

    environments_path = (
        OUTPUT_ROOT
        / "sequence_taxonomy_conda_environment_inventory.tsv"
    )

    summary_path = (
        TABLE_ROOT
        / "sequence_taxonomy_resource_summary.tsv"
    )

    environment_table = pd.DataFrame(
        {
            "environment": [
                environment_name(path)
                for path in environments
            ],
            "environment_path": [
                path.as_posix()
                for path in environments
            ],
            "has_bin_directory": [
                (path / "bin").is_dir()
                for path in environments
            ],
            "has_conda_metadata": [
                (path / "conda-meta").is_dir()
                for path in environments
            ],
        }
    )

    environment_table.to_csv(
        environments_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    tools.to_csv(
        tools_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    packages.to_csv(
        packages_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    databases.to_csv(
        databases_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    database_candidates = databases[
        databases[
            "record_type"
        ].isin(
            [
                "candidate_directory",
                "candidate_file",
            ]
        )
    ]

    set_database_variables = databases[
        databases[
            "record_type"
        ].eq(
            "environment_variable"
        )
        & databases[
            "notes"
        ].eq("set")
    ]

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "production_fastas",
                "value":
                    len(fasta_files),
            },
            {
                "metric":
                    "production_partial_fastas",
                "value":
                    len(partial_files),
            },
            {
                "metric":
                    "conda_environments",
                "value":
                    len(environments),
            },
            {
                "metric":
                    "installed_tool_records",
                "value":
                    len(tools),
            },
            {
                "metric":
                    "unique_installed_tools",
                "value":
                    tools[
                        "tool"
                    ].nunique(),
            },
            {
                "metric":
                    "relevant_conda_packages",
                "value":
                    len(packages),
            },
            {
                "metric":
                    "candidate_database_entries",
                "value":
                    len(database_candidates),
            },
            {
                "metric":
                    "set_database_environment_variables",
                "value":
                    len(set_database_variables),
            },
        ]
    )

    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    output_paths = [
        environments_path,
        tools_path,
        packages_path,
        databases_path,
        summary_path,
    ]

    checksum_path = (
        OUTPUT_ROOT
        / "script49_inventory_outputs_sha256.txt"
    )

    with checksum_path.open(
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
                f"{path.as_posix()}\n"
            )

    print(
        "Production FASTA files:",
        f"{len(fasta_files):,}",
    )

    print(
        "Partial production FASTA files:",
        len(partial_files),
    )

    print(
        "Conda environments discovered:",
        len(environments),
    )

    print(
        "Installed tool records:",
        len(tools),
    )

    print(
        "Unique installed taxonomy-related tools:",
        tools[
            "tool"
        ].nunique(),
    )

    print(
        "Relevant conda package records:",
        len(packages),
    )

    print(
        "Candidate database/reference entries:",
        len(database_candidates),
    )

    print(
        "Set database environment variables:",
        len(set_database_variables),
    )

    print()
    print(
        "===== INSTALLED TOOL INVENTORY ====="
    )

    if tools.empty:
        print(
            "<NO CANDIDATE TOOLS DISCOVERED>"
        )
    else:
        print(
            tools[
                [
                    "environment",
                    "tool",
                    "executable_path",
                    "version_return_code",
                    "version_output",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "===== RELEVANT CONDA PACKAGES ====="
    )

    if packages.empty:
        print(
            "<NO RELEVANT CONDA PACKAGES DISCOVERED>"
        )
    else:
        print(
            packages[
                [
                    "environment",
                    "package_name",
                    "version",
                    "build",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "===== DATABASE ENVIRONMENT VARIABLES ====="
    )

    print(
        databases.loc[
            databases[
                "record_type"
            ].eq(
                "environment_variable"
            ),
            [
                "resource_name",
                "path",
                "exists",
                "notes",
            ],
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Frozen Script 48 core modified:",
        "NO",
    )

    print(
        "Production FASTA files modified:",
        "NO",
    )

    print(
        "Packages installed or upgraded:",
        "NO",
    )

    print(
        "Reference databases downloaded:",
        "NO",
    )

    print(
        "Sequence-taxonomy decisions made:",
        "NO",
    )

    print()
    print(
        "STATUS: SCRIPT 49 SEQUENCE-TAXONOMY "
        "RESOURCE INVENTORY COMPLETE"
    )


if __name__ == "__main__":
    main()
