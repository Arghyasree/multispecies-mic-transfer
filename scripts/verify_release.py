#!/usr/bin/env python3
"""Validate the released benchmark, matrices, split definitions, and code.

The default verification is lightweight and requires only NumPy and pandas.
Use ``--full`` after installing all dependencies to import the frozen model
implementation and execute one CPU forward pass for each outer-target model.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_OBSERVATIONS = {
    "ec": 68_881,
    "kp": 50_299,
    "se": 49_183,
}
EXPECTED_GENOMES = {
    "ec": 6_673,
    "kp": 5_602,
    "se": 9_119,
}
EXPECTED_ANTIBIOTICS = {
    "ec": 19,
    "kp": 17,
    "se": 8,
}
EXPECTED_GENOME_MATRICES = {
    "ec": (
        "features/genome_representation/nested_loso_v1/"
        "selected_kmer_plus_common_amr/"
        "outer_ec_selected_4mer_plus_common_amr_float32_v1.npy",
        (21_394, 228),
    ),
    "kp": (
        "features/genome_representation/nested_loso_v1/"
        "common_cross_species_amr/"
        "outer_kp_common_cross_species_amr_binary_v1.npy",
        (21_394, 91),
    ),
    "se": (
        "features/genome_representation/nested_loso_v1/"
        "common_cross_species_amr/"
        "outer_se_common_cross_species_amr_binary_v1.npy",
        (21_394, 108),
    ),
}
EXPECTED_DRUG_MATRICES = {
    "morgan": ("features/drug/morgan_radius2_2048_chiral.npy", (19, 2048)),
    "rdkit": ("features/drug/rdkit_descriptors.npy", (19, 27)),
    "chemberta_mean": ("features/drug/chemberta_mean.npy", (19, 384)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Import the frozen models and run one CPU forward pass per target.",
    )
    return parser.parse_args()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def compile_python(root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        py_compile.compile(str(path), doraise=True)
        count += 1
    return count


def verify_documentation(root: Path) -> None:
    readme = require_file(root, "README.md").read_text(encoding="utf-8")
    reproducibility = require_file(
        root, "docs/reproducibility.md"
    ).read_text(encoding="utf-8")
    execution_map = require_file(
        root, "docs/execution_map.md"
    ).read_text(encoding="utf-8")

    forbidden = [
        "Verify the released files before running experiments:",
        "SHA256SUMS.txt",
        "RELEASE_VALIDATION.txt",
    ]
    for term in forbidden:
        if term in readme or term in reproducibility:
            raise RuntimeError(f"Stale documentation reference: {term}")

    if re.search(r"```bash\s*```", readme):
        raise RuntimeError("README contains an empty bash code block.")

    for required in [
        "## Quick verification",
        "## Reproducibility scope",
        "docs/execution_map.md",
    ]:
        if required not in readme:
            raise RuntimeError(f"README is missing: {required}")

    required_execution_map_fragments = [
        "Exact released script numbers are shown",
        "Taxonomy-verified modelling precursor",
        "Nested-LOSO configuration split and feature-index freeze",
        "Final protocol, three-species paper benchmark, and split freeze",
        "common-six fields from an earlier preregistration",
        "`01`, `02`, `03`",
        "`06`, `07`, `08`",
        "`13`, `14`, `15`, `16`, `20`, `22`, `25`, `26`",
        "`150`–`167`, `170`–`173`",
    ]
    for fragment in required_execution_map_fragments:
        if fragment not in execution_map:
            raise RuntimeError(
                f"Execution map is missing required mapping text: {fragment}"
            )

    forbidden_execution_map_fragments = [
        "`01`–`08`",
        "`13`–`28`",
        "`45`–`52`",
        "final three-species modelling cohort",
    ]
    for fragment in forbidden_execution_map_fragments:
        if fragment in execution_map:
            raise RuntimeError(
                f"Execution map contains misleading mapping text: {fragment}"
            )

    private_path = re.compile(r"/home/[^/\s]+|Path\.home\(\).*arghyasree")
    for path in sorted((root / "scripts").glob("*.py")):
        # Do not scan this verifier itself: it necessarily contains the
        # private-path detection expression used for checking other scripts.
        if path.resolve() == Path(__file__).resolve():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if private_path.search(text):
            raise RuntimeError(
                f"Private machine path remains in {path.relative_to(root)}"
            )


def verify_matrices(root: Path) -> tuple[int, int]:
    genome_rows = -1
    for target, (relative, expected_shape) in EXPECTED_GENOME_MATRICES.items():
        path = require_file(root, relative)
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(matrix.shape) != expected_shape:
            raise RuntimeError(
                f"{target} genome matrix shape {matrix.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise RuntimeError(f"Non-numeric genome matrix: {relative}")
        genome_rows = matrix.shape[0]

    drug_rows = -1
    for name, (relative, expected_shape) in EXPECTED_DRUG_MATRICES.items():
        path = require_file(root, relative)
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(matrix.shape) != expected_shape:
            raise RuntimeError(
                f"{name} drug matrix shape {matrix.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise RuntimeError(f"Non-numeric drug matrix: {relative}")
        drug_rows = matrix.shape[0]

    return genome_rows, drug_rows


def verify_observations(root: Path, genome_rows: int, drug_rows: int) -> pd.DataFrame:
    path = require_file(
        root,
        "metadata/final_transfer/nested_loso_v1/splits_v1/"
        "final_transfer_observation_feature_index_v1.tsv.gz",
    )
    frame = read_tsv(path)
    require_columns(
        frame,
        [
            "observation_id",
            "species_code",
            "genome_id",
            "normalized_antibiotic",
            "mic_target_log2_mg_per_l",
            "genome_feature_row",
            "drug_feature_row",
        ],
        "observation index",
    )

    if len(frame) != sum(EXPECTED_OBSERVATIONS.values()):
        raise RuntimeError(
            f"Observation rows={len(frame)}; expected {sum(EXPECTED_OBSERVATIONS.values())}"
        )
    if frame["observation_id"].duplicated().any():
        raise RuntimeError("Observation IDs are not unique.")
    if frame.duplicated(
        ["species_code", "genome_id", "normalized_antibiotic"]
    ).any():
        raise RuntimeError("More than one final observation exists per genome–antibiotic pair.")

    for code in ("ec", "kp", "se"):
        subset = frame.loc[frame["species_code"].eq(code)]
        observed = (
            len(subset),
            subset["genome_id"].nunique(),
            subset["normalized_antibiotic"].nunique(),
        )
        expected = (
            EXPECTED_OBSERVATIONS[code],
            EXPECTED_GENOMES[code],
            EXPECTED_ANTIBIOTICS[code],
        )
        if observed != expected:
            raise RuntimeError(
                f"{code} benchmark counts={observed}; expected={expected}"
            )

    targets = pd.to_numeric(frame["mic_target_log2_mg_per_l"], errors="raise")
    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite MIC targets detected.")

    genome_index = pd.to_numeric(frame["genome_feature_row"], errors="raise").astype(int)
    drug_index = pd.to_numeric(frame["drug_feature_row"], errors="raise").astype(int)
    if genome_index.min() < 0 or genome_index.max() >= genome_rows:
        raise RuntimeError("Genome feature-row index is out of bounds.")
    if drug_index.min() < 0 or drug_index.max() >= drug_rows:
        raise RuntimeError("Drug feature-row index is out of bounds.")

    return frame


def verify_splits(root: Path, observations: pd.DataFrame) -> None:
    split_root = root / "metadata/final_transfer/nested_loso_v1/splits_v1"
    query = read_tsv(require_file(
        root,
        str((split_root / "target_query_membership_v1.tsv.gz").relative_to(root)),
    ))
    support = read_tsv(require_file(
        root,
        str((split_root / "target_nested_support_membership_v1.tsv.gz").relative_to(root)),
    ))

    require_columns(
        query,
        ["target_species_code", "query_id", "observation_id"],
        "query membership",
    )
    require_columns(
        support,
        [
            "target_species_code",
            "query_id",
            "support_budget_percent",
            "observation_id",
        ],
        "support membership",
    )

    valid_ids = set(observations["observation_id"].astype(str))
    unknown_query = set(query["observation_id"].astype(str)) - valid_ids
    unknown_support = set(support["observation_id"].astype(str)) - valid_ids
    if unknown_query:
        raise RuntimeError(f"Unknown query observation IDs: {list(unknown_query)[:3]}")
    if unknown_support:
        raise RuntimeError(f"Unknown support observation IDs: {list(unknown_support)[:3]}")

    query_sets = {
        (str(target), str(query_id)): set(group["observation_id"].astype(str))
        for (target, query_id), group in query.groupby(
            ["target_species_code", "query_id"], sort=False
        )
    }

    for (target, query_id, budget), group in support.groupby(
        ["target_species_code", "query_id", "support_budget_percent"],
        sort=False,
    ):
        key = (str(target), str(query_id))
        if key not in query_sets:
            raise RuntimeError(f"Support group has no matching query: {key}")
        overlap = query_sets[key].intersection(group["observation_id"].astype(str))
        if overlap:
            raise RuntimeError(
                f"Query/support leakage for {key}, budget={budget}: {list(overlap)[:3]}"
            )

    for key, group in support.groupby(
        ["target_species_code", "query_id"], sort=False
    ):
        budget_sets = {
            int(float(budget)): set(rows["observation_id"].astype(str))
            for budget, rows in group.groupby("support_budget_percent", sort=False)
        }
        if {1, 5, 10}.issubset(budget_sets):
            if not budget_sets[1].issubset(budget_sets[5]):
                raise RuntimeError(f"1% support is not nested in 5% for {key}")
            if not budget_sets[5].issubset(budget_sets[10]):
                raise RuntimeError(f"5% support is not nested in 10% for {key}")

    for name in [
        "target_random_pair_fold_registry_v1.tsv.gz",
        "target_genome_disjoint_fold_registry_v1.tsv.gz",
        "target_drug_holdout_registry_v1.tsv.gz",
    ]:
        require_file(root, str((split_root / name).relative_to(root)))


def verify_configurations(root: Path) -> None:
    final = read_tsv(require_file(root, "config/final/outer_target_configurations.tsv"))
    shared = read_tsv(require_file(root, "config/final/shared_hyperparameters.tsv"))
    if set(final["outer_target_code"]) != {"ec", "kp", "se"}:
        raise RuntimeError("Final configuration registry does not contain exactly EC/KP/SE.")
    if final["outer_target_code"].duplicated().any():
        raise RuntimeError("Duplicate final outer-target configuration.")
    if set(shared["outer_target_code"]) != {"ec", "kp", "se"}:
        raise RuntimeError("Shared-hyperparameter registry does not contain EC/KP/SE.")

    for directory in ["results/evaluation", "results/model_selection"]:
        files = list((root / directory).rglob("*.tsv"))
        if not files:
            raise RuntimeError(f"No aggregate TSV files found under {directory}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_model_forward_passes(root: Path) -> None:
    os.environ["MIC_TRANSFER_PROJECT"] = str(root)
    import torch

    arch = load_module(
        root / "src/mic_transfer/final_architectures.py",
        "release_final_architectures",
    )
    selected = read_tsv(
        require_file(
            root,
            "results/tables/config_selection/nested_loso_v1/"
            "corrective_architecture_screen_aggregate_v2/"
            "selected_corrective_architecture_registry.tsv",
        )
    )

    integer_fields = {
        "low_rank_interaction_rank",
        "drug_view_low_rank",
        "latent_width",
        "genome_hidden_multiplier",
        "drug_hidden_multiplier",
        "fusion_hidden_multiplier",
        "batch_size",
        "maximum_epochs",
        "early_stopping_patience",
    }
    float_fields = {
        "dropout",
        "learning_rate",
        "weight_decay",
        "minimum_rmse_improvement",
        "gradient_clip_norm",
    }

    for _, row in selected.iterrows():
        representation = str(row["genome_representation"])
        spec = dict(arch.final165.CONFIG_BY_REPRESENTATION[representation])
        spec.update(row.to_dict())
        for field in integer_fields:
            if field in spec and str(spec[field]).strip():
                spec[field] = int(float(spec[field]))
        for field in float_fields:
            if field in spec and str(spec[field]).strip():
                spec[field] = float(spec[field])

        arch.final165.CURRENT_SPEC = spec
        arch.final165.set_current_hyperparameters(spec)

        genome_path = Path(spec["genome_matrix_path"])
        if not genome_path.is_absolute():
            genome_path = root / genome_path
        genome_matrix = np.load(genome_path, mmap_mode="r", allow_pickle=False)

        representation_id = str(row["drug_representation"])
        views = arch.backend.DRUG_REPRESENTATION_VIEWS[representation_id]
        drug_matrices = {
            view: np.load(
                arch.backend.DRUG_VIEW_PATHS[view],
                mmap_mode="r",
                allow_pickle=False,
            )
            for view in views
        }
        architecture_id = arch.backend.ARCHITECTURE_NAME_TO_ID[
            str(row["cross_modal_architecture"])
        ]
        model = arch.CorrectiveArchitectureNetwork(
            genome_dimension=genome_matrix.shape[1],
            drug_matrices=drug_matrices,
            architecture_id=architecture_id,
            spec=spec,
        ).cpu().eval()

        genome = torch.from_numpy(
            np.array(genome_matrix[:2], dtype=np.float32, copy=True)
        )
        drug_inputs = {
            view: torch.from_numpy(np.array(matrix[:2], dtype=np.float32, copy=True))
            for view, matrix in drug_matrices.items()
        }
        with torch.no_grad():
            output = model(genome, drug_inputs)
        if tuple(output.shape) not in {(2,), (2, 1)}:
            raise RuntimeError(
                f"Unexpected forward-pass shape for {row['outer_target_code']}: {output.shape}"
            )



def verify_dependency_manifest(root: Path) -> None:
    obsolete_manifest = root / ("requirements" + "-frozen.txt")
    if obsolete_manifest.exists():
        raise RuntimeError(
            "Obsolete duplicate dependency manifest remains: "
            f"{obsolete_manifest.name}"
        )

    manifests = sorted(path.name for path in root.glob("requirements*.txt"))
    if manifests != ["requirements.txt"]:
        raise RuntimeError(
            f"Expected only requirements.txt; found dependency manifests: {manifests}"
        )

    pyproject = require_file(root, "pyproject.toml").read_text(encoding="utf-8")
    required_fragments = [
        'requires-python = ">=3.11"',
        'dynamic = ["dependencies"]',
        'dependencies = {file = ["requirements.txt"]}',
    ]
    for fragment in required_fragments:
        if fragment not in pyproject:
            raise RuntimeError(
                f"pyproject.toml is missing canonical dependency metadata: {fragment}"
            )

    public_paths = [
        root / "README.md",
        root / "data/README.md",
        root / "docs/computational_environment.md",
        root / "docs/public_pipeline.md",
        root / "docs/reproducibility.md",
    ]
    obsolete_name = "requirements" + "-frozen.txt"
    for document in public_paths:
        content = require_file(root, str(document.relative_to(root))).read_text(
            encoding="utf-8"
        )
        if obsolete_name in content:
            raise RuntimeError(
                f"Obsolete dependency-manifest reference remains in "
                f"{document.relative_to(root)}"
            )
        if "conceptual workflow above" in content:
            raise RuntimeError(
                f"Stale workflow wording remains in {document.relative_to(root)}"
            )

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / ".git").is_dir():
        raise RuntimeError(f"Not a Git repository: {root}")

    verify_dependency_manifest(root)

    required_root = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        "docs/execution_map.md",
        "docs/public_pipeline.md",
        "docs/reproducibility.md",
    ]
    for relative in required_root:
        require_file(root, relative)

    compiled = compile_python(root)
    verify_documentation(root)
    genome_rows, drug_rows = verify_matrices(root)
    observations = verify_observations(root, genome_rows, drug_rows)
    verify_splits(root, observations)
    verify_configurations(root)

    if args.full:
        verify_model_forward_passes(root)

    print("===== RELEASE VERIFICATION =====")
    print(f"Python files compiled: {compiled}")
    print(f"Benchmark observations: {len(observations):,}")
    print(f"Genome feature rows: {genome_rows:,}")
    print(f"Antibiotic feature rows: {drug_rows:,}")
    print("Documentation checks: PASS")
    print("Matrix and row-index checks: PASS")
    print("Split leakage and nested-support checks: PASS")
    print("Configuration and aggregate-result checks: PASS")
    if args.full:
        print("Frozen-model CPU forward passes: PASS")
    print("STATUS: RELEASE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
