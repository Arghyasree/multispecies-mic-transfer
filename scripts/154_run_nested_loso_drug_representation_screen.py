#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

BASE_RUNNER_PATH = (
    PROJECT
    / "scripts/"
      "143_run_nested_loso_full_kmer_grid.py"
)

EXPECTED_BASE_RUNNER_SHA256 = (
    "d82435bc05f13fcc330632e6e8b27460"
    "139d24ab812ef4c46a5741ddebb18b80"
)

SCRIPT143_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script143_validated_full_grid_runner_sha256.txt"
)

SCRIPT153_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script153_successful_genome_selection_core_sha256.txt"
)

RUN_PLAN_PATH_NEW = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "drug_representation_screen_v1/"
      "nested_loso_drug_representation_screen_run_plan_v1.tsv"
)

SELECTED_GENOME_REGISTRY = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "drug_representation_screen_v1/"
      "nested_loso_selected_genome_representation_registry_v1.tsv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


for required_path in [
    BASE_RUNNER_PATH,
    SCRIPT143_FREEZE,
    SCRIPT153_FREEZE,
    RUN_PLAN_PATH_NEW,
    SELECTED_GENOME_REGISTRY,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

observed_base_sha = sha256_file(BASE_RUNNER_PATH)

if observed_base_sha != EXPECTED_BASE_RUNNER_SHA256:
    raise RuntimeError(
        "Validated Script 143 SHA mismatch: "
        f"{observed_base_sha}"
    )

registry = pd.read_csv(
    SELECTED_GENOME_REGISTRY,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

required_registry_columns = {
    "outer_target_code",
    "matrix_path",
}

missing_columns = sorted(
    required_registry_columns.difference(
        registry.columns
    )
)

if missing_columns:
    raise RuntimeError(
        "Missing selected-genome registry columns: "
        + "|".join(missing_columns)
    )

matrix_paths = {
    str(row.outer_target_code):
        PROJECT / str(row.matrix_path)
    for row in registry.itertuples(index=False)
}

if set(matrix_paths) != {"ec", "kp", "se"}:
    raise RuntimeError(
        "Selected-genome registry does not cover all outer targets."
    )

for path in matrix_paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)

source = BASE_RUNNER_PATH.read_text(
    encoding="utf-8"
)

source = source.replace(
    "SCRIPT 143",
    "SCRIPT 154",
)

source = source.replace(
    "FULL K-MER GRID",
    "DRUG REPRESENTATION SCREEN",
)

input_marker = (
    "        Path(__file__).resolve(),\n"
    "        OBSERVATION_INDEX_PATH,\n"
)

input_replacement = (
    "        Path(__file__).resolve(),\n"
    "        BASE_RUNNER_PATH,\n"
    "        SCRIPT153_FREEZE,\n"
    "        SELECTED_GENOME_REGISTRY,\n"
    "        OBSERVATION_INDEX_PATH,\n"
)

if input_marker not in source:
    raise RuntimeError(
        "Could not extend the per-run input manifest."
    )

source = source.replace(
    input_marker,
    input_replacement,
    1,
)

namespace = {
    "__name__": "script154_base",
    "__file__": str(Path(__file__).resolve()),
    "BASE_RUNNER_PATH": BASE_RUNNER_PATH,
    "SCRIPT153_FREEZE": SCRIPT153_FREEZE,
    "SELECTED_GENOME_REGISTRY": SELECTED_GENOME_REGISTRY,
}

exec(
    compile(
        source,
        str(Path(__file__).resolve()),
        "exec",
    ),
    namespace,
)

namespace.update(
    {
        "RUN_PLAN_PATH": RUN_PLAN_PATH_NEW,
        "SCRIPT142_OUTPUTS_PATH": SCRIPT153_FREEZE,
        "SCRIPT142_FROZEN_PATH": SCRIPT153_FREEZE,
        "KMER_PATHS": {
            "selected_genome_representation": matrix_paths["ec"],
        },
        "RESULT_ROOT": (
            PROJECT
            / "results/tables/config_selection/nested_loso_v1/"
              "drug_representation_screen_runs_v1"
        ),
        "METADATA_ROOT": (
            PROJECT
            / "metadata/config_selection/nested_loso_v1/"
              "drug_representation_screen_runs_v1"
        ),
        "AGGREGATE_ROOT": (
            PROJECT
            / "results/tables/config_selection/nested_loso_v1/"
              "drug_representation_screen_aggregate_v1"
        ),
        "EXPECTED_RUNS": 108,
    }
)

verify_sha_manifest = namespace[
    "verify_sha_manifest"
]

verify_sha_manifest(
    SCRIPT143_FREEZE
)

verify_sha_manifest(
    SCRIPT153_FREEZE
)

base_execute_run = namespace[
    "execute_run"
]


def execute_drug_screen_run(
    run_row,
    observations,
    device,
):
    outer = str(
        run_row["outer_target_code"]
    )

    if outer not in matrix_paths:
        raise RuntimeError(
            f"Unexpected outer target: {outer}"
        )

    namespace[
        "KMER_PATHS"
    ][
        "selected_genome_representation"
    ] = matrix_paths[outer]

    return base_execute_run(
        run_row,
        observations,
        device,
    )


namespace[
    "execute_run"
] = execute_drug_screen_run

namespace["main"]()
