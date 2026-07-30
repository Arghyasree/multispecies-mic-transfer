#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT = Path(
    __file__
).resolve().parents[1]

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

SCREEN_PROTOCOL_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "kmer_length_screen_v1/"
      "nested_loso_kmer_length_screen_protocol_v1.tsv"
)


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


for required_path in [
    BASE_RUNNER_PATH,
    SCRIPT143_FREEZE,
    SCREEN_PROTOCOL_PATH,
]:
    if not required_path.is_file():
        raise FileNotFoundError(
            required_path
        )

observed_base_sha = sha256_file(
    BASE_RUNNER_PATH
)

if observed_base_sha != (
    EXPECTED_BASE_RUNNER_SHA256
):
    raise RuntimeError(
        "Validated Script 143 SHA mismatch: "
        f"{observed_base_sha}"
    )

source = BASE_RUNNER_PATH.read_text(
    encoding="utf-8"
)

source = source.replace(
    "SCRIPT 143",
    "SCRIPT 150",
)

source = source.replace(
    "FULL K-MER GRID",
    "K-MER LENGTH SCREEN",
)

input_marker = '''        Path(__file__).resolve(),
        OBSERVATION_INDEX_PATH,
'''

input_replacement = '''        Path(__file__).resolve(),
        BASE_RUNNER_PATH,
        SCREEN_PROTOCOL_PATH,
        OBSERVATION_INDEX_PATH,
'''

if input_marker not in source:
    raise RuntimeError(
        "Could not extend the screen "
        "input manifest."
    )

source = source.replace(
    input_marker,
    input_replacement,
    1,
)

namespace = {
    "__name__": "script150_base",
    "__file__": str(
        Path(__file__).resolve()
    ),
    "BASE_RUNNER_PATH":
        BASE_RUNNER_PATH,
    "SCREEN_PROTOCOL_PATH":
        SCREEN_PROTOCOL_PATH,
}

exec(
    compile(
        source,
        str(
            Path(__file__).resolve()
        ),
        "exec",
    ),
    namespace,
)

namespace.update(
    {
        "RESULT_ROOT": (
            PROJECT
            / "results/tables/"
              "config_selection/"
              "nested_loso_v1/"
              "kmer_length_screen_runs_v1"
        ),
        "METADATA_ROOT": (
            PROJECT
            / "metadata/config_selection/"
              "nested_loso_v1/"
              "kmer_length_screen_runs_v1"
        ),
        "AGGREGATE_ROOT": (
            PROJECT
            / "results/tables/"
              "config_selection/"
              "nested_loso_v1/"
              "kmer_length_screen_aggregate_v1"
        ),
    }
)

verify_sha_manifest = namespace[
    "verify_sha_manifest"
]

verify_sha_manifest(
    SCRIPT143_FREEZE
)

base_parse_arguments = namespace[
    "parse_arguments"
]


def parse_screen_arguments():
    arguments = base_parse_arguments()

    # Fixed a priori so that the screen isolates
    # only the effect of k-mer length.
    arguments.drug_representation = "Morgan"
    arguments.architecture = (
        "projected_concatenation_MLP"
    )

    return arguments


namespace[
    "parse_arguments"
] = parse_screen_arguments

namespace["main"]()
