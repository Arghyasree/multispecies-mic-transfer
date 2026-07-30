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

SCRIPT151_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script151_successful_selected_kmer_and_matrix_core_sha256.txt"
)

RUN_PLAN_PATH_NEW = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_genome_representation_screen_run_plan_v1.tsv"
)

MATRIX_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "genome_representation_screen_v1/"
      "nested_loso_selected_kmer_plus_common_amr_matrix_registry_v1.tsv"
)

AMR_ROOT = (
    PROJECT
    / "features/genome_representation/nested_loso_v1/"
      "common_cross_species_amr"
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
    SCRIPT151_FREEZE,
    RUN_PLAN_PATH_NEW,
    MATRIX_REGISTRY_PATH,
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
    MATRIX_REGISTRY_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

required_registry_columns = {
    "outer_target_code",
    "fused_matrix_path",
}

missing = sorted(
    required_registry_columns.difference(
        registry.columns
    )
)

if missing:
    raise RuntimeError(
        "Missing matrix-registry columns: "
        + "|".join(missing)
    )

fused_paths = {
    str(row.outer_target_code):
        PROJECT / str(row.fused_matrix_path)
    for row in registry.itertuples(index=False)
}

amr_paths = {
    outer: (
        AMR_ROOT
        / (
            f"outer_{outer}_common_cross_species_"
            "amr_binary_v1.npy"
        )
    )
    for outer in ["ec", "kp", "se"]
}

for path in [*fused_paths.values(), *amr_paths.values()]:
    if not path.is_file():
        raise FileNotFoundError(path)

source = BASE_RUNNER_PATH.read_text(encoding="utf-8")

source = source.replace(
    "SCRIPT 143",
    "SCRIPT 152",
)

source = source.replace(
    "FULL K-MER GRID",
    "GENOME REPRESENTATION SCREEN",
)

input_marker = '''        Path(__file__).resolve(),
        OBSERVATION_INDEX_PATH,
'''

input_replacement = '''        Path(__file__).resolve(),
        BASE_RUNNER_PATH,
        SCRIPT151_FREEZE,
        MATRIX_REGISTRY_PATH,
        OBSERVATION_INDEX_PATH,
'''

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
    "__name__": "script152_base",
    "__file__": str(Path(__file__).resolve()),
    "BASE_RUNNER_PATH": BASE_RUNNER_PATH,
    "SCRIPT151_FREEZE": SCRIPT151_FREEZE,
    "MATRIX_REGISTRY_PATH": MATRIX_REGISTRY_PATH,
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
        "SCRIPT142_OUTPUTS_PATH": SCRIPT151_FREEZE,
        "SCRIPT142_FROZEN_PATH": SCRIPT151_FREEZE,
        "KMER_PATHS": {
            "common_cross_species_AMR": amr_paths["ec"],
            "selected_kmer_plus_common_AMR": fused_paths["ec"],
        },
        "RESULT_ROOT": (
            PROJECT
            / "results/tables/config_selection/nested_loso_v1/"
              "genome_representation_screen_runs_v1"
        ),
        "METADATA_ROOT": (
            PROJECT
            / "metadata/config_selection/nested_loso_v1/"
              "genome_representation_screen_runs_v1"
        ),
        "AGGREGATE_ROOT": (
            PROJECT
            / "results/tables/config_selection/nested_loso_v1/"
              "genome_representation_screen_aggregate_v1"
        ),
        "EXPECTED_RUNS": 36,
    }
)

verify_sha_manifest = namespace["verify_sha_manifest"]

verify_sha_manifest(SCRIPT143_FREEZE)
verify_sha_manifest(SCRIPT151_FREEZE)

base_execute_run = namespace["execute_run"]


def execute_representation_run(
    run_row,
    observations,
    device,
):
    outer = str(
        run_row["outer_target_code"]
    )

    representation = str(
        run_row["genome_representation"]
    )

    if representation == "common_cross_species_AMR":
        matrix_path = amr_paths[outer]

    elif representation == "selected_kmer_plus_common_AMR":
        matrix_path = fused_paths[outer]

    else:
        raise RuntimeError(
            "Unexpected genome representation: "
            f"{representation}"
        )

    namespace[
        "KMER_PATHS"
    ][
        representation
    ] = matrix_path

    return base_execute_run(
        run_row,
        observations,
        device,
    )


namespace["execute_run"] = execute_representation_run
namespace["main"]()
