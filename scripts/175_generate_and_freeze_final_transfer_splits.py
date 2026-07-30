#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import math
import os
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()

SCRIPT174_FREEZE = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1/"
      "script174_successful_final_transfer_preregistration_core_sha256.txt"
)

PROTOCOL_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1/"
      "final_transfer_protocol_v1.tsv"
)

SOURCE_REGIME_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1/"
      "final_transfer_source_regime_registry_v1.tsv"
)

TARGET_PANEL_PATH = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/preregistration_v1/"
      "final_transfer_target_panel_registry_v1.tsv"
)

COHORT_PATH = (
    PROJECT
    / "data/processed/modelling/"
      "multispecies_taxonomy_verified_finalized_panel_mic_cohort.tsv"
)

KMER_ROWS_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/genome_features/"
      "canonical_kmer_v1/nested_loso_all_species_kmer_feature_rows_v1.tsv"
)

DRUG_ROWS_PATH = (
    PROJECT
    / "metadata/drug_representation/drug_feature_rows_v1.tsv"
)

DUPLICATE_GROUP_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/genome_features/"
      "canonical_kmer_v1/nested_loso_all_species_duplicate_8mer_profile_groups_v1.tsv"
)

FINAL_CONFIGURATION_PATH = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_architecture_screen_aggregate_v2/"
      "selected_corrective_architecture_registry.tsv"
)

OUTPUT_ROOT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/splits_v1"
)

TABLE_ROOT = (
    PROJECT
    / "results/tables/final_transfer/nested_loso_v1/splits_v1"
)

OBSERVATION_OUTPUT = (
    OUTPUT_ROOT / "final_transfer_observation_feature_index_v1.tsv"
)
RANDOM_FOLD_OUTPUT = (
    OUTPUT_ROOT / "target_random_pair_fold_registry_v1.tsv"
)
GENOME_FOLD_OUTPUT = (
    OUTPUT_ROOT / "target_genome_disjoint_fold_registry_v1.tsv"
)
DRUG_HOLDOUT_OUTPUT = (
    OUTPUT_ROOT / "target_drug_holdout_registry_v1.tsv"
)
QUERY_OUTPUT = (
    OUTPUT_ROOT / "target_query_membership_v1.tsv"
)
SUPPORT_OUTPUT = (
    OUTPUT_ROOT / "target_nested_support_membership_v1.tsv"
)
DRUG_FAMILIARITY_OUTPUT = (
    OUTPUT_ROOT / "source_target_drug_familiarity_v1.tsv"
)
SPLIT_PROTOCOL_OUTPUT = (
    OUTPUT_ROOT / "final_transfer_split_generation_protocol_v1.tsv"
)
INPUT_MANIFEST_OUTPUT = (
    OUTPUT_ROOT / "script175_input_manifest.tsv"
)
OUTPUT_MANIFEST = (
    OUTPUT_ROOT / "script175_outputs_sha256.txt"
)
FREEZE_OUTPUT = (
    PROJECT
    / "metadata/final_transfer/nested_loso_v1/"
      "script175_successful_final_transfer_splits_core_sha256.txt"
)

TARGET_SUMMARY_OUTPUT = (
    TABLE_ROOT / "target_full_panel_summary_v1.tsv"
)
RANDOM_BALANCE_OUTPUT = (
    TABLE_ROOT / "target_random_pair_fold_balance_v1.tsv"
)
GENOME_BALANCE_OUTPUT = (
    TABLE_ROOT / "target_genome_disjoint_fold_balance_v1.tsv"
)
SUPPORT_SUMMARY_OUTPUT = (
    TABLE_ROOT / "target_nested_support_summary_v1.tsv"
)
LEAKAGE_AUDIT_OUTPUT = (
    TABLE_ROOT / "target_split_leakage_audit_v1.tsv"
)
DRUG_FAMILIARITY_SUMMARY_OUTPUT = (
    TABLE_ROOT / "source_target_drug_familiarity_summary_v1.tsv"
)

N_FOLDS = 5
SPLIT_SEED = 20260814
SUPPORT_BUDGETS = (1, 5, 10)

SPECIES_NAME_TO_CODE = {
    "Klebsiella pneumoniae": "kp",
    "Escherichia coli": "ec",
    "Salmonella enterica": "se",
}
SPECIES_CODE_TO_NAME = {
    value: key for key, value in SPECIES_NAME_TO_CODE.items()
}

KP_PANEL = (
    "amikacin",
    "aztreonam",
    "cefepime",
    "cefmetazole",
    "cefotaxime",
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "cefuroxime",
    "ciprofloxacin",
    "imipenem",
    "levofloxacin",
    "meropenem",
    "minocycline",
    "tetracycline",
    "tigecycline",
    "tobramycin",
)
EC_PANEL = tuple(sorted(set(KP_PANEL).union({"ampicillin", "chloramphenicol"})))
SE_PANEL = (
    "ampicillin",
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "chloramphenicol",
    "ciprofloxacin",
    "meropenem",
    "tetracycline",
)

EXPECTED_PANELS = {
    "kp": KP_PANEL,
    "ec": EC_PANEL,
    "se": SE_PANEL,
}

COMMON_SIX = {
    "cefoxitin",
    "ceftazidime",
    "ceftriaxone",
    "ciprofloxacin",
    "meropenem",
    "tetracycline",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(value.strip())
    return path if path.is_absolute() else PROJECT / path


def verify_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)

    verified: list[Path] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed SHA line {line_number}: {path}")
        expected, value = parts
        candidate = project_path(value)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != expected:
            raise RuntimeError(f"SHA mismatch: {candidate}")
        verified.append(candidate)

    if not verified:
        raise RuntimeError(f"Empty SHA manifest: {path}")
    return verified


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def parse_binary_indicator(
    series: pd.Series,
    label: str,
) -> pd.Series:
    normalised = (
        series.astype(str)
        .str.strip()
        .str.casefold()
    )

    mapping = {
        "1": 1,
        "true": 1,
        "t": 1,
        "yes": 1,
        "y": 1,
        "0": 0,
        "false": 0,
        "f": 0,
        "no": 0,
        "n": 0,
    }

    unknown = sorted(
        set(normalised) - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"{label} contains unsupported binary values: "
            f"{unknown[:10]}"
        )

    return (
        normalised.map(mapping)
        .astype(np.int8)
    )


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def write_manifest(paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for candidate in sorted(
            {path.resolve() for path in paths},
            key=lambda value: value.as_posix(),
        ):
            try:
                display = candidate.relative_to(PROJECT)
            except ValueError:
                display = candidate
            handle.write(f"{sha256_file(candidate)}  {display}\n")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def normalise_genome_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def fold_name(index: int) -> str:
    return f"fold_{index + 1:02d}"


def assign_random_pair_folds(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for species_code, species_frame in observations.groupby(
        "species_code",
        sort=True,
    ):
        for antibiotic, drug_frame in species_frame.groupby(
            "normalized_antibiotic",
            sort=True,
        ):
            ordered = drug_frame[["observation_id"]].copy()
            ordered["stable_key"] = [
                stable_int(
                    f"random_pair|{SPLIT_SEED}|{species_code}|"
                    f"{antibiotic}|{observation_id}"
                )
                for observation_id in ordered["observation_id"].astype(str)
            ]
            ordered = ordered.sort_values(
                ["stable_key", "observation_id"],
                kind="stable",
            ).reset_index(drop=True)

            for position, record in enumerate(
                ordered.to_dict(orient="records")
            ):
                rows.append(
                    {
                        "target_species_code": species_code,
                        "observation_id": record["observation_id"],
                        "normalized_antibiotic": antibiotic,
                        "random_pair_fold": fold_name(position % N_FOLDS),
                        "assignment_seed": SPLIT_SEED,
                        "assignment_method": (
                            "per-antibiotic deterministic hash order followed "
                            "by round-robin fivefold assignment"
                        ),
                    }
                )

    output = pd.DataFrame(rows)
    if output["observation_id"].duplicated().any():
        raise RuntimeError("Observation assigned to multiple random-pair folds.")
    if len(output) != len(observations):
        raise RuntimeError("Random-pair fold row-count mismatch.")
    return output


def assign_genome_group_folds(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for species_code, species_frame in observations.groupby(
        "species_code",
        sort=True,
    ):
        drugs = sorted(
            species_frame["normalized_antibiotic"].unique().tolist()
        )
        group_drug = (
            species_frame.groupby(
                ["genome_group_id", "normalized_antibiotic"]
            )
            .size()
            .unstack(fill_value=0)
            .reindex(columns=drugs, fill_value=0)
            .astype(np.int64)
        )
        group_genomes = (
            species_frame.groupby("genome_group_id")["genome_id"]
            .nunique()
            .reindex(group_drug.index)
            .astype(np.int64)
        )
        group_total = group_drug.sum(axis=1).astype(np.int64)

        order = pd.DataFrame(
            {
                "genome_group_id": group_drug.index.astype(str),
                "group_observations": group_total.to_numpy(),
                "group_unique_genomes": group_genomes.to_numpy(),
                "group_max_drug": group_drug.max(axis=1).to_numpy(),
            }
        )
        order["tie"] = [
            stable_int(
                f"genome_disjoint|{SPLIT_SEED}|{species_code}|{group_id}"
            )
            for group_id in order["genome_group_id"]
        ]
        order = order.sort_values(
            [
                "group_observations",
                "group_max_drug",
                "group_unique_genomes",
                "tie",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        ).reset_index(drop=True)

        target_drug = group_drug.sum(axis=0).to_numpy(dtype=float) / N_FOLDS
        target_total = float(group_total.sum()) / N_FOLDS
        target_genomes = float(group_genomes.sum()) / N_FOLDS

        fold_drug = np.zeros((N_FOLDS, len(drugs)), dtype=float)
        fold_total = np.zeros(N_FOLDS, dtype=float)
        fold_genomes = np.zeros(N_FOLDS, dtype=float)
        fold_groups = np.zeros(N_FOLDS, dtype=float)

        if len(order) < N_FOLDS:
            raise RuntimeError(
                f"Species {species_code} has only {len(order)} "
                f"genome groups for {N_FOLDS} folds."
            )

        for assignment_index, record in enumerate(
            order.itertuples(index=False)
        ):
            group_id = str(record.genome_group_id)
            vector = group_drug.loc[
                group_id
            ].to_numpy(dtype=float)

            group_observations = float(
                record.group_observations
            )
            group_unique_genomes = float(
                record.group_unique_genomes
            )

            # Deterministically seed the five largest genome groups
            # into different folds. This guarantees that every fold
            # is represented before the greedy balancing begins.
            if assignment_index < N_FOLDS:
                chosen = assignment_index
            else:
                scores: list[
                    tuple[float, float, float, int]
                ] = []

                for fold_index in range(N_FOLDS):
                    current_drug = fold_drug[
                        fold_index
                    ]
                    proposed_drug = (
                        current_drug + vector
                    )

                    drug_delta = float(
                        np.sum(
                            (
                                (
                                    proposed_drug
                                    - target_drug
                                )
                                ** 2
                                -
                                (
                                    current_drug
                                    - target_drug
                                )
                                ** 2
                            )
                            / (target_drug + 1.0)
                        )
                    )

                    current_total = fold_total[
                        fold_index
                    ]
                    proposed_total = (
                        current_total
                        + group_observations
                    )

                    total_delta = float(
                        (
                            (
                                proposed_total
                                - target_total
                            )
                            ** 2
                            -
                            (
                                current_total
                                - target_total
                            )
                            ** 2
                        )
                        / (target_total + 1.0)
                    )

                    current_genomes = fold_genomes[
                        fold_index
                    ]
                    proposed_genomes = (
                        current_genomes
                        + group_unique_genomes
                    )

                    genome_delta = float(
                        (
                            (
                                proposed_genomes
                                - target_genomes
                            )
                            ** 2
                            -
                            (
                                current_genomes
                                - target_genomes
                            )
                            ** 2
                        )
                        / (target_genomes + 1.0)
                    )

                    incremental_global_error = (
                        drug_delta
                        + 0.25 * total_delta
                        + 0.10 * genome_delta
                    )

                    scores.append(
                        (
                            incremental_global_error,
                            fold_groups[fold_index],
                            fold_total[fold_index],
                            fold_index,
                        )
                    )

                _, _, _, chosen = min(scores)

            fold_drug[chosen] += vector
            fold_total[chosen] += (
                group_observations
            )
            fold_genomes[chosen] += (
                group_unique_genomes
            )
            fold_groups[chosen] += 1

            rows.append(
                {
                    "target_species_code": species_code,
                    "genome_group_id": group_id,
                    "genome_disjoint_fold": fold_name(
                        chosen
                    ),
                    "assignment_seed": SPLIT_SEED,
                    "assignment_method": (
                        "deterministic fivefold seeding followed "
                        "by incremental global-error minimisation "
                        "over per-antibiotic counts, total "
                        "observations and genome counts"
                    ),
                }
            )

    output = pd.DataFrame(rows)
    if output.duplicated(
        ["target_species_code", "genome_group_id"]
    ).any():
        raise RuntimeError("Genome group assigned to multiple folds.")
    for species_code, group in output.groupby("target_species_code"):
        if group["genome_disjoint_fold"].nunique() != N_FOLDS:
            raise RuntimeError(
                f"Species {species_code} does not use all genome folds."
            )
    return output


def balanced_support_order(
    support_pool: pd.DataFrame,
    context: str,
) -> list[str]:
    queues: dict[str, deque[str]] = {}

    for antibiotic, group in support_pool.groupby(
        "normalized_antibiotic",
        sort=True,
    ):
        ordered = group[["observation_id"]].copy()
        ordered["stable_key"] = [
            stable_int(
                f"support|{SPLIT_SEED}|{context}|{antibiotic}|"
                f"{observation_id}"
            )
            for observation_id in ordered["observation_id"].astype(str)
        ]
        ordered = ordered.sort_values(
            ["stable_key", "observation_id"],
            kind="stable",
        )
        queues[str(antibiotic)] = deque(
            ordered["observation_id"].astype(str).tolist()
        )

    drug_order = sorted(
        queues,
        key=lambda antibiotic: (
            stable_int(f"support_drug|{SPLIT_SEED}|{context}|{antibiotic}"),
            antibiotic,
        ),
    )

    output: list[str] = []
    while True:
        added = False
        for antibiotic in drug_order:
            if queues[antibiotic]:
                output.append(queues[antibiotic].popleft())
                added = True
        if not added:
            break

    if len(output) != len(support_pool):
        raise RuntimeError("Support-order row-count mismatch.")
    if len(output) != len(set(output)):
        raise RuntimeError("Duplicate observation in support order.")
    return output


def support_target_size(pool_size: int, drug_count: int, budget: int) -> int:
    nominal = int(math.floor(pool_size * budget / 100.0 + 0.5))
    minimum = drug_count if pool_size >= drug_count else pool_size
    return min(pool_size, max(1, nominal, minimum))


def main() -> None:
    required = [
        SCRIPT174_FREEZE,
        PROTOCOL_PATH,
        SOURCE_REGIME_PATH,
        TARGET_PANEL_PATH,
        COHORT_PATH,
        KMER_ROWS_PATH,
        DRUG_ROWS_PATH,
        DUPLICATE_GROUP_PATH,
        FINAL_CONFIGURATION_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified_preregistration = verify_manifest(SCRIPT174_FREEZE)

    protocol = read_tsv(PROTOCOL_PATH)
    source_regimes = read_tsv(SOURCE_REGIME_PATH)
    target_panels = read_tsv(TARGET_PANEL_PATH)
    final_configurations = read_tsv(FINAL_CONFIGURATION_PATH)

    cohort = read_tsv(COHORT_PATH)
    require_columns(
        cohort,
        [
            "provisional_species",
            "genome_id",
            "normalized_antibiotic",
            "observation_id",
            "mic_target_log2_mg_per_l",
            "is_exact_observation",
            "is_censored_observation",
        ],
        "full multispecies cohort",
    )

    cohort = cohort.loc[
        cohort["provisional_species"].isin(SPECIES_NAME_TO_CODE)
    ].copy()
    cohort["species_code"] = cohort["provisional_species"].map(
        SPECIES_NAME_TO_CODE
    )
    cohort["genome_id"] = cohort["genome_id"].map(normalise_genome_id)
    cohort["normalized_antibiotic"] = (
        cohort["normalized_antibiotic"].astype(str).str.strip().str.casefold()
    )
    cohort["observation_id"] = cohort["observation_id"].astype(str).str.strip()
    cohort["mic_target_log2_mg_per_l"] = pd.to_numeric(
        cohort["mic_target_log2_mg_per_l"],
        errors="raise",
    )
    cohort["is_exact_observation"] = parse_binary_indicator(
        cohort["is_exact_observation"],
        "is_exact_observation",
    )
    cohort["is_censored_observation"] = parse_binary_indicator(
        cohort["is_censored_observation"],
        "is_censored_observation",
    )

    if not np.isfinite(
        cohort["mic_target_log2_mg_per_l"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite MIC targets in final cohort.")

    cohort = cohort.loc[
        [
            antibiotic in set(EXPECTED_PANELS[species_code])
            for species_code, antibiotic in zip(
                cohort["species_code"],
                cohort["normalized_antibiotic"],
            )
        ]
    ].copy()

    if cohort["observation_id"].duplicated().any():
        raise RuntimeError("Observation IDs are not globally unique.")
    if cohort.duplicated(
        ["species_code", "genome_id", "normalized_antibiotic"]
    ).any():
        raise RuntimeError(
            "Final cohort contains duplicate species/genome/drug pairs."
        )

    observed_panels = {
        species_code: tuple(
            sorted(group["normalized_antibiotic"].unique().tolist())
        )
        for species_code, group in cohort.groupby("species_code")
    }
    for species_code, expected_panel in EXPECTED_PANELS.items():
        if observed_panels.get(species_code) != tuple(sorted(expected_panel)):
            raise RuntimeError(
                f"Full target panel mismatch for {species_code}: "
                f"{observed_panels.get(species_code)}"
            )

    kmer = read_tsv(KMER_ROWS_PATH)
    require_columns(
        kmer,
        ["feature_row", "species_code", "species", "genome_id"],
        "k-mer feature-row registry",
    )
    kmer["genome_id"] = kmer["genome_id"].map(normalise_genome_id)
    if kmer["genome_id"].duplicated().any():
        raise RuntimeError("Duplicate genome in k-mer feature registry.")
    kmer_map = kmer[
        ["feature_row", "species_code", "species", "genome_id"]
    ].rename(
        columns={
            "feature_row": "genome_feature_row",
            "species_code": "feature_species_code",
            "species": "feature_species",
        }
    )

    drug = read_tsv(DRUG_ROWS_PATH)
    require_columns(
        drug,
        [
            "feature_row",
            "antibiotic",
            "identity_feature_row",
            "morgan_feature_row",
            "rdkit_feature_row",
            "chemberta_mean_feature_row",
            "chemberta_first_feature_row",
        ],
        "drug feature-row registry",
    )
    drug_map = drug[
        [
            "feature_row",
            "antibiotic",
            "identity_feature_row",
            "morgan_feature_row",
            "rdkit_feature_row",
            "chemberta_mean_feature_row",
            "chemberta_first_feature_row",
        ]
    ].rename(
        columns={
            "feature_row": "drug_feature_row",
            "antibiotic": "normalized_antibiotic",
        }
    )
    drug_map["normalized_antibiotic"] = (
        drug_map["normalized_antibiotic"].astype(str).str.strip().str.casefold()
    )
    if drug_map["normalized_antibiotic"].duplicated().any():
        raise RuntimeError("Duplicate drug in feature-row registry.")

    observations = cohort.merge(
        kmer_map,
        on="genome_id",
        how="left",
        validate="many_to_one",
    )
    observations = observations.merge(
        drug_map,
        on="normalized_antibiotic",
        how="left",
        validate="many_to_one",
    )

    if not observations["species_code"].eq(
        observations["feature_species_code"]
    ).all():
        raise RuntimeError("Species mismatch between cohort and genome features.")

    mapped_columns = [
        "genome_feature_row",
        "drug_feature_row",
        "identity_feature_row",
        "morgan_feature_row",
        "rdkit_feature_row",
        "chemberta_mean_feature_row",
        "chemberta_first_feature_row",
    ]
    for column in mapped_columns:
        if observations[column].isna().any() or observations[column].eq("").any():
            raise RuntimeError(f"Missing feature mapping: {column}")
        observations[column] = pd.to_numeric(
            observations[column],
            errors="raise",
        ).astype(np.int64)

    duplicate_groups = read_tsv(DUPLICATE_GROUP_PATH)
    require_columns(
        duplicate_groups,
        [
            "duplicate_profile_group_id",
            "species_code",
            "genome_id",
            "group_size",
        ],
        "duplicate-profile registry",
    )
    duplicate_groups["genome_id"] = duplicate_groups["genome_id"].map(
        normalise_genome_id
    )
    if duplicate_groups["genome_id"].duplicated().any():
        raise RuntimeError("Genome occurs in multiple duplicate-profile groups.")

    duplicate_map = duplicate_groups[
        [
            "genome_id",
            "species_code",
            "duplicate_profile_group_id",
            "group_size",
        ]
    ].rename(columns={"species_code": "duplicate_species_code"})

    observations = observations.merge(
        duplicate_map,
        on="genome_id",
        how="left",
        validate="many_to_one",
    )
    has_duplicate = (
        observations["duplicate_profile_group_id"].notna()
        & observations["duplicate_profile_group_id"].ne("")
    )
    if not observations.loc[has_duplicate, "species_code"].eq(
        observations.loc[has_duplicate, "duplicate_species_code"]
    ).all():
        raise RuntimeError("Duplicate-profile group crosses species.")

    observations["genome_group_id"] = np.where(
        has_duplicate,
        observations["duplicate_profile_group_id"],
        "singleton::"
        + observations["species_code"].astype(str)
        + "::"
        + observations["genome_id"].astype(str),
    )
    observations["duplicate_profile_group_id"] = (
        observations["duplicate_profile_group_id"]
        .fillna("")
        .replace("", "not_duplicate")
    )
    observations["duplicate_profile_group_size"] = pd.to_numeric(
        observations["group_size"],
        errors="coerce",
    ).fillna(1).astype(np.int64)

    observations = observations[
        [
            "species_code",
            "provisional_species",
            "observation_id",
            "genome_id",
            "normalized_antibiotic",
            "mic_target_log2_mg_per_l",
            "is_exact_observation",
            "is_censored_observation",
            "genome_feature_row",
            "drug_feature_row",
            "identity_feature_row",
            "morgan_feature_row",
            "rdkit_feature_row",
            "chemberta_mean_feature_row",
            "chemberta_first_feature_row",
            "genome_group_id",
            "duplicate_profile_group_id",
            "duplicate_profile_group_size",
        ]
    ].sort_values(
        ["species_code", "genome_id", "normalized_antibiotic", "observation_id"],
        kind="stable",
    ).reset_index(drop=True)
    observations.insert(
        0,
        "final_transfer_observation_row",
        np.arange(len(observations), dtype=np.int64),
    )

    random_folds = assign_random_pair_folds(observations)
    genome_group_folds = assign_genome_group_folds(observations)

    observations = observations.merge(
        random_folds[["observation_id", "random_pair_fold"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    observations = observations.merge(
        genome_group_folds[
            [
                "target_species_code",
                "genome_group_id",
                "genome_disjoint_fold",
            ]
        ],
        left_on=["species_code", "genome_group_id"],
        right_on=["target_species_code", "genome_group_id"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["target_species_code"])

    if observations[
        ["random_pair_fold", "genome_disjoint_fold"]
    ].isna().any().any():
        raise RuntimeError("Missing target fold assignment.")

    genome_fold_registry = (
        observations[
            [
                "species_code",
                "provisional_species",
                "genome_id",
                "genome_feature_row",
                "genome_group_id",
                "duplicate_profile_group_id",
                "duplicate_profile_group_size",
                "genome_disjoint_fold",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["species_code", "genome_disjoint_fold", "genome_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if genome_fold_registry.duplicated(
        ["species_code", "genome_id"]
    ).any():
        raise RuntimeError("Genome assigned to more than one genome fold.")

    query_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []
    support_summaries: list[dict[str, object]] = []
    leakage_records: list[dict[str, object]] = []
    drug_holdout_records: list[dict[str, object]] = []

    for species_code, target in observations.groupby(
        "species_code",
        sort=True,
    ):
        target = target.copy().reset_index(drop=True)

        protocol_definitions: list[tuple[str, str, pd.Series]] = []
        for current_fold in [fold_name(index) for index in range(N_FOLDS)]:
            protocol_definitions.append(
                (
                    "random_pair",
                    f"random_pair__{current_fold}",
                    target["random_pair_fold"].eq(current_fold),
                )
            )
            protocol_definitions.append(
                (
                    "genome_disjoint",
                    f"genome_disjoint__{current_fold}",
                    target["genome_disjoint_fold"].eq(current_fold),
                )
            )

        for held_out_drug in sorted(
            target["normalized_antibiotic"].unique().tolist()
        ):
            protocol_definitions.append(
                (
                    "drug_held_out",
                    f"drug_held_out__{held_out_drug}",
                    target["normalized_antibiotic"].eq(held_out_drug),
                )
            )

        for protocol_id, query_id, query_mask in protocol_definitions:
            query = target.loc[query_mask].copy()
            support_pool = target.loc[~query_mask].copy()

            if query.empty or support_pool.empty:
                raise RuntimeError(
                    f"Empty query/support for {species_code}/{query_id}"
                )

            query_frame = query[
                [
                    "final_transfer_observation_row",
                    "observation_id",
                    "genome_id",
                    "normalized_antibiotic",
                ]
            ].copy()
            query_frame.insert(0, "query_id", query_id)
            query_frame.insert(0, "target_protocol", protocol_id)
            query_frame.insert(0, "target_species_code", species_code)
            query_frames.append(query_frame)

            order = balanced_support_order(
                support_pool,
                f"{species_code}|{protocol_id}|{query_id}",
            )
            support_lookup = support_pool.set_index("observation_id")
            support_drug_count = support_pool["normalized_antibiotic"].nunique()

            previous_support: set[str] = set()
            for budget in SUPPORT_BUDGETS:
                target_size = support_target_size(
                    pool_size=len(support_pool),
                    drug_count=support_drug_count,
                    budget=budget,
                )
                selected_ids = order[:target_size]
                selected_set = set(selected_ids)

                if not previous_support.issubset(selected_set):
                    raise RuntimeError(
                        f"Non-nested support sets for {species_code}/{query_id}"
                    )
                previous_support = selected_set

                selected = support_lookup.loc[selected_ids].reset_index()
                selected_frame = selected[
                    [
                        "final_transfer_observation_row",
                        "observation_id",
                        "genome_id",
                        "normalized_antibiotic",
                    ]
                ].copy()
                selected_frame.insert(0, "support_budget_percent", budget)
                selected_frame.insert(0, "query_id", query_id)
                selected_frame.insert(0, "target_protocol", protocol_id)
                selected_frame.insert(0, "target_species_code", species_code)
                selected_frame["support_rank"] = np.arange(
                    1,
                    len(selected_frame) + 1,
                    dtype=np.int64,
                )
                support_frames.append(selected_frame)

                support_summaries.append(
                    {
                        "target_species_code": species_code,
                        "target_protocol": protocol_id,
                        "query_id": query_id,
                        "support_budget_percent": budget,
                        "support_pool_observations": len(support_pool),
                        "support_pool_unique_genomes": support_pool[
                            "genome_id"
                        ].nunique(),
                        "support_pool_unique_antibiotics": support_drug_count,
                        "selected_support_observations": len(selected_frame),
                        "selected_support_unique_genomes": selected_frame[
                            "genome_id"
                        ].nunique(),
                        "selected_support_unique_antibiotics": selected_frame[
                            "normalized_antibiotic"
                        ].nunique(),
                        "actual_support_percent_of_pool": (
                            100.0 * len(selected_frame) / len(support_pool)
                        ),
                        "nested_support_policy": "prefix_of_one_balanced_order",
                    }
                )

                overlap = set(query["observation_id"]).intersection(selected_set)
                genome_overlap = set(query["genome_id"]).intersection(
                    set(selected_frame["genome_id"])
                )
                held_out_drug_overlap = ""
                if protocol_id == "drug_held_out":
                    held_out_drug = query_id.split("__", 1)[1]
                    held_out_drug_overlap = str(
                        held_out_drug
                        in set(selected_frame["normalized_antibiotic"])
                    )

                if overlap:
                    raise RuntimeError(
                        f"Query/support observation leakage: {species_code}/{query_id}"
                    )
                if protocol_id == "genome_disjoint" and genome_overlap:
                    raise RuntimeError(
                        f"Genome leakage: {species_code}/{query_id}"
                    )
                if protocol_id == "drug_held_out" and held_out_drug_overlap == "True":
                    raise RuntimeError(
                        f"Held-out drug leakage: {species_code}/{query_id}"
                    )

                leakage_records.append(
                    {
                        "target_species_code": species_code,
                        "target_protocol": protocol_id,
                        "query_id": query_id,
                        "support_budget_percent": budget,
                        "query_observations": len(query),
                        "selected_support_observations": len(selected_frame),
                        "observation_overlap_count": len(overlap),
                        "genome_overlap_count": len(genome_overlap),
                        "held_out_drug_present_in_support": held_out_drug_overlap,
                        "leakage_status": "PASS",
                    }
                )

            if protocol_id == "drug_held_out":
                held_out_drug = query_id.split("__", 1)[1]
                drug_holdout_records.append(
                    {
                        "target_species_code": species_code,
                        "target_species": SPECIES_CODE_TO_NAME[species_code],
                        "held_out_drug": held_out_drug,
                        "query_id": query_id,
                        "query_observations": len(query),
                        "query_unique_genomes": query["genome_id"].nunique(),
                        "all_other_drugs_support_pool_observations": len(
                            support_pool
                        ),
                        "all_other_drugs_support_pool_unique_genomes": (
                            support_pool["genome_id"].nunique()
                        ),
                        "all_other_drugs_count": support_drug_count,
                    }
                )

    query_membership = pd.concat(query_frames, ignore_index=True).sort_values(
        [
            "target_species_code",
            "target_protocol",
            "query_id",
            "observation_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    support_membership = pd.concat(
        support_frames,
        ignore_index=True,
    ).sort_values(
        [
            "target_species_code",
            "target_protocol",
            "query_id",
            "support_budget_percent",
            "support_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)

    support_summary = pd.DataFrame(support_summaries).sort_values(
        [
            "target_species_code",
            "target_protocol",
            "query_id",
            "support_budget_percent",
        ],
        kind="stable",
    ).reset_index(drop=True)

    leakage_audit = pd.DataFrame(leakage_records).sort_values(
        [
            "target_species_code",
            "target_protocol",
            "query_id",
            "support_budget_percent",
        ],
        kind="stable",
    ).reset_index(drop=True)

    drug_holdout_registry = pd.DataFrame(drug_holdout_records).sort_values(
        ["target_species_code", "held_out_drug"],
        kind="stable",
    ).reset_index(drop=True)

    familiarity_records: list[dict[str, object]] = []
    source_panels = {
        species_code: set(panel)
        for species_code, panel in EXPECTED_PANELS.items()
    }

    for regime in source_regimes.to_dict(orient="records"):
        target_species_code = str(regime["outer_target_code"])
        source_species_codes = [
            value
            for value in str(regime["source_species_codes"]).split("|")
            if value
        ]
        source_union = set().union(
            *(source_panels[source] for source in source_species_codes)
        )
        target_panel = set(EXPECTED_PANELS[target_species_code])

        for antibiotic in sorted(target_panel):
            familiarity_records.append(
                {
                    "outer_target_code": target_species_code,
                    "source_regime_id": regime["source_regime_id"],
                    "source_species_codes": regime["source_species_codes"],
                    "target_antibiotic": antibiotic,
                    "source_mic_supervision_status": (
                        "source_seen" if antibiotic in source_union else "source_unseen"
                    ),
                    "in_common_six_panel": (
                        "YES" if antibiotic in COMMON_SIX else "NO"
                    ),
                    "in_full_target_panel": "YES",
                }
            )

    drug_familiarity = pd.DataFrame(familiarity_records).sort_values(
        ["outer_target_code", "source_regime_id", "target_antibiotic"],
        kind="stable",
    ).reset_index(drop=True)

    target_summary = (
        observations.groupby(
            ["species_code", "provisional_species"],
            as_index=False,
        )
        .agg(
            observations=("observation_id", "size"),
            unique_genomes=("genome_id", "nunique"),
            unique_genome_groups=("genome_group_id", "nunique"),
            unique_antibiotics=("normalized_antibiotic", "nunique"),
            exact_observations=("is_exact_observation", "sum"),
            censored_observations=("is_censored_observation", "sum"),
        )
        .sort_values("species_code")
        .reset_index(drop=True)
    )
    target_summary["eligible_antibiotics"] = [
        "|".join(sorted(EXPECTED_PANELS[species_code]))
        for species_code in target_summary["species_code"]
    ]

    random_balance = (
        observations.groupby(
            ["species_code", "random_pair_fold", "normalized_antibiotic"],
            as_index=False,
        )
        .agg(
            observations=("observation_id", "size"),
            unique_genomes=("genome_id", "nunique"),
        )
        .sort_values(
            ["species_code", "random_pair_fold", "normalized_antibiotic"]
        )
        .reset_index(drop=True)
    )

    genome_balance = (
        observations.groupby(
            [
                "species_code",
                "genome_disjoint_fold",
                "normalized_antibiotic",
            ],
            as_index=False,
        )
        .agg(
            observations=("observation_id", "size"),
            unique_genomes=("genome_id", "nunique"),
            unique_genome_groups=("genome_group_id", "nunique"),
        )
        .sort_values(
            [
                "species_code",
                "genome_disjoint_fold",
                "normalized_antibiotic",
            ]
        )
        .reset_index(drop=True)
    )

    familiarity_summary = (
        drug_familiarity.groupby(
            [
                "outer_target_code",
                "source_regime_id",
                "source_mic_supervision_status",
            ],
            as_index=False,
        )
        .agg(drug_count=("target_antibiotic", "nunique"))
        .sort_values(
            [
                "outer_target_code",
                "source_regime_id",
                "source_mic_supervision_status",
            ]
        )
        .reset_index(drop=True)
    )

    split_protocol = pd.DataFrame(
        [
            {
                "item": "analysis_stage",
                "value": "final target split and nested support generation",
            },
            {
                "item": "target_species",
                "value": "kp|ec|se",
            },
            {
                "item": "target_protocols",
                "value": "random_pair|genome_disjoint|drug_held_out",
            },
            {
                "item": "random_pair_assignment",
                "value": (
                    "per-antibiotic deterministic hash order and round-robin "
                    "fivefold assignment; same genome may occur in support and query"
                ),
            },
            {
                "item": "genome_disjoint_assignment",
                "value": (
                    "complete genome groups assigned to five folds; duplicate "
                    "8-mer profiles remain together"
                ),
            },
            {
                "item": "drug_held_out_assignment",
                "value": (
                    "leave one target antibiotic out; no target label for that "
                    "antibiotic enters support"
                ),
            },
            {
                "item": "support_budgets_percent",
                "value": "1|5|10",
            },
            {
                "item": "support_nesting",
                "value": "S_1_subset_S_5_subset_S_10",
            },
            {
                "item": "support_sampling",
                "value": (
                    "one deterministic per-query drug-balanced order; budget "
                    "sets are prefixes reused across all source regimes and model seeds"
                ),
            },
            {
                "item": "outcome_use_in_assignment",
                "value": "none; MIC values are retained but never used for split assignment",
            },
            {
                "item": "zero_shot_query_policy",
                "value": "same frozen query cohorts used for zero-shot and few-shot",
            },
            {
                "item": "unseen_drug_terminology",
                "value": "antibiotic unseen in source MIC training",
            },
            {
                "item": "models_trained_by_script175",
                "value": "NO",
            },
            {
                "item": "split_seed",
                "value": str(SPLIT_SEED),
            },
        ]
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    write_tsv(observations, OBSERVATION_OUTPUT)
    write_tsv(random_folds, RANDOM_FOLD_OUTPUT)
    write_tsv(genome_fold_registry, GENOME_FOLD_OUTPUT)
    write_tsv(drug_holdout_registry, DRUG_HOLDOUT_OUTPUT)
    write_tsv(query_membership, QUERY_OUTPUT)
    write_tsv(support_membership, SUPPORT_OUTPUT)
    write_tsv(drug_familiarity, DRUG_FAMILIARITY_OUTPUT)
    write_tsv(split_protocol, SPLIT_PROTOCOL_OUTPUT)
    write_tsv(target_summary, TARGET_SUMMARY_OUTPUT)
    write_tsv(random_balance, RANDOM_BALANCE_OUTPUT)
    write_tsv(genome_balance, GENOME_BALANCE_OUTPUT)
    write_tsv(support_summary, SUPPORT_SUMMARY_OUTPUT)
    write_tsv(leakage_audit, LEAKAGE_AUDIT_OUTPUT)
    write_tsv(familiarity_summary, DRUG_FAMILIARITY_SUMMARY_OUTPUT)

    input_paths = [
        Path(__file__).resolve(),
        *required,
    ]
    input_manifest = pd.DataFrame(
        [
            {
                "file_path": str(path.relative_to(PROJECT)),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(
                {candidate.resolve() for candidate in input_paths},
                key=lambda value: value.as_posix(),
            )
        ]
    )
    write_tsv(input_manifest, INPUT_MANIFEST_OUTPUT)

    output_paths = [
        OBSERVATION_OUTPUT,
        RANDOM_FOLD_OUTPUT,
        GENOME_FOLD_OUTPUT,
        DRUG_HOLDOUT_OUTPUT,
        QUERY_OUTPUT,
        SUPPORT_OUTPUT,
        DRUG_FAMILIARITY_OUTPUT,
        SPLIT_PROTOCOL_OUTPUT,
        INPUT_MANIFEST_OUTPUT,
        TARGET_SUMMARY_OUTPUT,
        RANDOM_BALANCE_OUTPUT,
        GENOME_BALANCE_OUTPUT,
        SUPPORT_SUMMARY_OUTPUT,
        LEAKAGE_AUDIT_OUTPUT,
        DRUG_FAMILIARITY_SUMMARY_OUTPUT,
    ]
    write_manifest(output_paths, OUTPUT_MANIFEST)
    verify_manifest(OUTPUT_MANIFEST)

    freeze_paths = [
        Path(__file__).resolve(),
        OUTPUT_MANIFEST,
        *output_paths,
        SCRIPT174_FREEZE,
        *verified_preregistration,
    ]
    write_manifest(freeze_paths, FREEZE_OUTPUT)
    verify_manifest(FREEZE_OUTPUT)

    print("===== SCRIPT 175 FINAL TRANSFER SPLITS =====")
    print(target_summary.to_string(index=False))
    print()
    print("Source regimes:", len(source_regimes))
    print("Target protocols: random_pair|genome_disjoint|drug_held_out")
    print("Random-pair query folds:", 3 * N_FOLDS)
    print("Genome-disjoint query folds:", 3 * N_FOLDS)
    print("Drug-held-out query folds:", len(drug_holdout_registry))
    print("Nested support budgets: 1|5|10")
    print("Query-membership rows:", len(query_membership))
    print("Support-membership rows:", len(support_membership))
    print("Leakage audit rows:", len(leakage_audit))
    print("Leakage failures:", int(leakage_audit["leakage_status"].ne("PASS").sum()))
    print("Models trained: NO")
    print()
    print("STATUS: SCRIPT 175 FINAL TRANSFER SPLITS GENERATED AND FROZEN")


if __name__ == "__main__":
    main()
