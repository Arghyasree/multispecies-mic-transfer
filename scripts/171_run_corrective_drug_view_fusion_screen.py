#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(
    os.environ.get(
        "MIC_TRANSFER_PROJECT",
        Path.home()
        / "arghyasree/ISI_Research/"
          "multispecies_mic_transfer",
    )
).expanduser().resolve()

SCRIPT167_PATH = (
    PROJECT
    / "scripts/"
      "167_run_corrective_drug_representation_screen.py"
)

EXPECTED_SCRIPT167_SHA256 = (
    "89de8c4fd4fcd58f42c5e57b51481fac"
    "165193db6fff0b245a388e4526088c9f"
)

SCRIPT170_FREEZE = (
    PROJECT
    / "metadata/config_selection/"
      "script170_successful_corrective_drug_view_fusion_preregistration_core_sha256.txt"
)

RUN_PLAN_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_v1/"
      "corrective_drug_view_fusion_run_plan_v1.tsv"
)

CONFIGURATION_REGISTRY_PATH = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_v1/"
      "corrective_drug_view_fusion_configuration_registry_v1.tsv"
)

RESULT_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_runs_v1"
)

METADATA_ROOT = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_runs_v1"
)

AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_view_fusion_screen_aggregate_v1"
)

SCRIPT167_AGGREGATE_ROOT = (
    PROJECT
    / "results/tables/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_aggregate_v2"
)

SCRIPT167_AGGREGATE_MANIFEST = (
    PROJECT
    / "metadata/config_selection/nested_loso_v1/"
      "corrective_drug_representation_screen_runs_v2/"
      "aggregate_outputs_sha256.txt"
)

EXPECTED_RUNS = 72

CURRENT_SPEC: dict[str, Any] | None = None
CURRENT_FUSION_METHOD: str | None = None
CURRENT_DRUG_VIEW_LOW_RANK: int = 0
CURRENT_PARAMETER_COUNT: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    path = Path(str(value).strip())
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
            raise RuntimeError(
                f"Malformed SHA line {line_number}: {path}"
            )

        expected, value = parts
        candidate = project_path(value)

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        if sha256_file(candidate) != expected:
            raise RuntimeError(f"SHA mismatch: {candidate}")

        verified.append(candidate)

    if not verified:
        raise RuntimeError(f"Empty SHA manifest: {path}")

    return verified


def load_script167():
    if not SCRIPT167_PATH.is_file():
        raise FileNotFoundError(SCRIPT167_PATH)

    observed = sha256_file(SCRIPT167_PATH)
    if observed != EXPECTED_SCRIPT167_SHA256:
        raise RuntimeError(
            "Script 167 SHA mismatch: "
            f"{observed}"
        )

    specification = importlib.util.spec_from_file_location(
        "corrective_drug_screen_167",
        SCRIPT167_PATH,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "Could not load corrected Script 167."
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


screen167 = load_script167()
final165 = screen167.final165
backend = screen167.backend


class SeparateDrugViewFusionNetwork(
    final165.FinalGenomeConfirmationNetwork
):
    def __init__(
        self,
        genome_dimension: int,
        drug_matrices: dict[str, np.ndarray],
        spec: dict[str, Any],
        fusion_method: str,
        low_rank: int,
    ) -> None:
        if len(drug_matrices) < 2:
            raise RuntimeError(
                "Corrective drug-view fusion requires at least two views."
            )

        if fusion_method not in {
            "separate_encoder_projected",
            "separate_encoder_low_rank",
        }:
            raise RuntimeError(
                f"Unsupported drug-view fusion method: {fusion_method}"
            )

        self.drug_view_order = tuple(
            drug_matrices.keys()
        )
        self.drug_view_fusion_method = fusion_method
        self.drug_view_low_rank = int(low_rank)

        total_dimension = int(
            sum(
                matrix.shape[1]
                for matrix in drug_matrices.values()
            )
        )

        super().__init__(
            genome_dimension=genome_dimension,
            drug_dimension=total_dimension,
            spec=spec,
        )

        latent = int(spec["latent_width"])
        dropout = float(spec["dropout"])
        drug_multiplier = int(
            spec["drug_hidden_multiplier"]
        )
        fusion_multiplier = int(
            spec["fusion_hidden_multiplier"]
        )

        self.drug_view_encoders = nn.ModuleDict(
            {
                view_id: final165.FreshViewEncoder(
                    input_dimension=int(
                        drug_matrices[view_id].shape[1]
                    ),
                    latent_width=latent,
                    hidden_multiplier=drug_multiplier,
                    dropout=dropout,
                )
                for view_id in self.drug_view_order
            }
        )

        fusion_hidden = max(
            latent,
            fusion_multiplier * latent,
        )

        self.drug_projected_base = nn.Sequential(
            nn.Linear(
                len(self.drug_view_order) * latent,
                fusion_hidden,
            ),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_hidden,
                latent,
            ),
            nn.LayerNorm(latent),
            nn.GELU(),
        )

        if (
            self.drug_view_fusion_method
            == "separate_encoder_low_rank"
        ):
            if self.drug_view_low_rank <= 0:
                raise RuntimeError(
                    "Low-rank drug fusion requires a positive rank."
                )

            self.drug_low_rank = nn.ModuleDict(
                {
                    view_id: nn.Linear(
                        latent,
                        self.drug_view_low_rank,
                        bias=False,
                    )
                    for view_id in self.drug_view_order
                }
            )

            self.drug_view_pairs = tuple(
                itertools.combinations(
                    self.drug_view_order,
                    2,
                )
            )

            if not self.drug_view_pairs:
                raise RuntimeError(
                    "No pairwise drug-view interactions were formed."
                )

            self.drug_pairwise_to_latent = nn.Linear(
                len(self.drug_view_pairs)
                * self.drug_view_low_rank,
                latent,
                bias=False,
            )

            self.drug_residual_norm = nn.LayerNorm(
                latent
            )

        # The single raw-concatenation encoder made by the parent is
        # intentionally not used in this corrective stage.
        del self.drug_encoder

    def encode_drug(
        self,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if set(drug_views) != set(
            self.drug_view_order
        ):
            raise RuntimeError(
                "Drug-view mismatch. Expected "
                f"{self.drug_view_order}; observed "
                f"{tuple(drug_views)}."
            )

        latents = {
            view_id: self.drug_view_encoders[
                view_id
            ](
                drug_views[view_id]
            )
            for view_id in self.drug_view_order
        }

        base = self.drug_projected_base(
            torch.cat(
                [
                    latents[view_id]
                    for view_id
                    in self.drug_view_order
                ],
                dim=1,
            )
        )

        if (
            self.drug_view_fusion_method
            == "separate_encoder_projected"
        ):
            return base

        projected = {
            view_id: self.drug_low_rank[
                view_id
            ](
                latents[view_id]
            )
            for view_id in self.drug_view_order
        }

        pairwise = torch.cat(
            [
                projected[left]
                * projected[right]
                for left, right
                in self.drug_view_pairs
            ],
            dim=1,
        )

        residual = self.drug_pairwise_to_latent(
            pairwise
        )

        return torch.nn.functional.gelu(
            self.drug_residual_norm(
                base + residual
            )
        )

    def forward(
        self,
        genome: torch.Tensor,
        drug_views: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        genome_latent = self.encode_genome(
            genome
        )
        drug_latent = self.encode_drug(
            drug_views
        )

        additive = (
            self.intercept
            + self.genome_head(
                genome_latent
            ).squeeze(1)
            + self.drug_head(
                drug_latent
            ).squeeze(1)
        )

        interaction = self.interaction_head(
            torch.cat(
                [
                    genome_latent,
                    drug_latent,
                ],
                dim=1,
            )
        ).squeeze(1)

        return additive + interaction


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


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        lineterminator="\n",
    )


def write_sha_manifest(
    paths: list[Path],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique = sorted(
        {path.resolve() for path in paths},
        key=lambda value: value.as_posix(),
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for path in unique:
            try:
                display = path.relative_to(PROJECT)
            except ValueError:
                display = path
            handle.write(
                f"{sha256_file(path)}  {display}\n"
            )


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()
    return (
        float(numeric.mean())
        if len(numeric)
        else float("nan")
    )


def sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()
    return (
        float(numeric.std(ddof=1))
        if len(numeric) >= 2
        else float("nan")
    )


def build_model(
    genome_matrix: np.ndarray,
    drug_matrices: dict[str, np.ndarray],
    architecture_id: str,
    target_mean: float,
    device: torch.device,
) -> nn.Module:
    global CURRENT_PARAMETER_COUNT

    if CURRENT_SPEC is None:
        raise RuntimeError(
            "Current drug-view-fusion specification is unset."
        )

    if CURRENT_FUSION_METHOD is None:
        raise RuntimeError(
            "Current drug-view-fusion method is unset."
        )

    if architecture_id not in {
        "cross_modal_projected_concat",
        "projected_concatenation_MLP",
    }:
        raise RuntimeError(
            "Drug-view fusion screen fixes the cross-modal "
            "architecture to projected concatenation."
        )

    model = SeparateDrugViewFusionNetwork(
        genome_dimension=genome_matrix.shape[1],
        drug_matrices=drug_matrices,
        spec=CURRENT_SPEC,
        fusion_method=CURRENT_FUSION_METHOD,
        low_rank=CURRENT_DRUG_VIEW_LOW_RANK,
    ).to(device)

    CURRENT_PARAMETER_COUNT = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )
    final165.CURRENT_PARAMETER_COUNT = (
        CURRENT_PARAMETER_COUNT
    )

    with torch.no_grad():
        model.intercept.fill_(
            float(target_mean)
        )

    return model


def patch_run_outputs(
    run_id: str,
    run_row: pd.Series,
) -> None:
    if CURRENT_PARAMETER_COUNT is None:
        raise RuntimeError(
            "Model parameter count is unavailable."
        )

    result_directory = RESULT_ROOT / run_id
    metadata_directory = METADATA_ROOT / run_id

    summary_path = (
        result_directory
        / "run_summary.tsv"
    )
    configuration_path = (
        metadata_directory
        / "configuration.tsv"
    )
    manifest_path = (
        metadata_directory
        / "outputs_sha256.txt"
    )

    additions = {
        "drug_view_fusion_method": run_row[
            "drug_view_fusion_method"
        ],
        "drug_view_low_rank": run_row[
            "drug_view_low_rank"
        ],
        "model_parameter_count": (
            CURRENT_PARAMETER_COUNT
        ),
        "corrective_analysis_stage": (
            "corrective_drug_view_fusion_screen_v1"
        ),
    }

    summary = backend.read_tsv(
        summary_path
    )

    for key, value in additions.items():
        summary[key] = value

    backend.write_tsv(
        summary,
        summary_path,
    )

    configuration = backend.read_tsv(
        configuration_path
    )

    addition_frame = pd.DataFrame(
        [
            {
                "item": key,
                "value": value,
            }
            for key, value in additions.items()
        ]
    )

    configuration = configuration.loc[
        ~configuration["item"].isin(
            addition_frame["item"]
        )
    ]

    configuration = pd.concat(
        [
            configuration,
            addition_frame,
        ],
        ignore_index=True,
    )

    backend.write_tsv(
        configuration,
        configuration_path,
    )

    output_paths: list[Path] = []
    for line in manifest_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        _, value = line.split(maxsplit=1)
        output_paths.append(
            project_path(value)
        )

    backend.write_sha_manifest(
        output_paths,
        manifest_path,
    )
    backend.verify_sha_manifest(
        manifest_path
    )


def execute_run(
    run_row: pd.Series,
    observations: pd.DataFrame,
    device: torch.device,
) -> dict[str, object]:
    global CURRENT_SPEC
    global CURRENT_FUSION_METHOD
    global CURRENT_DRUG_VIEW_LOW_RANK

    representation = str(
        run_row["genome_representation"]
    )

    if representation not in (
        final165.CONFIG_BY_REPRESENTATION
    ):
        raise RuntimeError(
            "Unregistered final genome representation: "
            f"{representation}"
        )

    CURRENT_SPEC = (
        final165.CONFIG_BY_REPRESENTATION[
            representation
        ]
    )
    final165.CURRENT_SPEC = CURRENT_SPEC

    CURRENT_FUSION_METHOD = str(
        run_row[
            "drug_view_fusion_method"
        ]
    )
    CURRENT_DRUG_VIEW_LOW_RANK = int(
        float(
            run_row[
                "drug_view_low_rank"
            ]
        )
    )

    final165.set_current_hyperparameters(
        CURRENT_SPEC
    )

    summary = final165.ORIGINAL_EXECUTE_RUN(
        run_row,
        observations,
        device,
    )

    patch_run_outputs(
        str(run_row["run_id"]),
        run_row,
    )

    return summary


def add_worst_direction_and_metadata() -> pd.DataFrame:
    all_runs_path = (
        AGGREGATE_ROOT
        / "all_direction_seed_metrics.tsv"
    )
    configuration_path = (
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )

    plan = read_tsv(RUN_PLAN_PATH)
    registry = read_tsv(
        CONFIGURATION_REGISTRY_PATH
    )
    all_runs = read_tsv(
        all_runs_path
    )

    all_runs["macro_rmse"] = pd.to_numeric(
        all_runs["macro_rmse"],
        errors="raise",
    )

    worst_seed = (
        all_runs.groupby(
            [
                "outer_target_code",
                "configuration_id",
                "genome_representation",
                "drug_representation",
                "cross_modal_architecture",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            direction_count=(
                "source_species_code",
                "size",
            ),
            seedwise_worst_direction_macro_rmse=(
                "macro_rmse",
                "max",
            ),
        )
        .reset_index()
    )

    if not worst_seed[
        "direction_count"
    ].eq(2).all():
        raise RuntimeError(
            "Every seed must contain exactly two directions."
        )

    worst_seed_path = (
        AGGREGATE_ROOT
        / "seedwise_worst_direction_macro_rmse.tsv"
    )
    write_tsv(worst_seed, worst_seed_path)

    worst_records: list[dict[str, object]] = []
    group_columns = [
        "outer_target_code",
        "configuration_id",
        "genome_representation",
        "drug_representation",
        "cross_modal_architecture",
    ]

    for keys, group in worst_seed.groupby(
        group_columns,
        dropna=False,
    ):
        record = dict(zip(group_columns, keys))
        values = group[
            "seedwise_worst_direction_macro_rmse"
        ]
        record["seed_count"] = group[
            "seed"
        ].nunique()
        record[
            "worst_direction_macro_rmse_mean"
        ] = safe_mean(values)
        record[
            "worst_direction_macro_rmse_sd"
        ] = sample_sd(values)
        worst_records.append(record)

    worst_summary = pd.DataFrame(
        worst_records
    )
    worst_summary_path = (
        AGGREGATE_ROOT
        / "configuration_seedwise_worst_direction_mean_sd.tsv"
    )
    write_tsv(worst_summary, worst_summary_path)

    configuration = read_tsv(
        configuration_path
    )

    configuration = configuration.merge(
        worst_summary[
            [
                "outer_target_code",
                "configuration_id",
                "worst_direction_macro_rmse_mean",
                "worst_direction_macro_rmse_sd",
            ]
        ],
        on=[
            "outer_target_code",
            "configuration_id",
        ],
        how="left",
        validate="one_to_one",
    )

    metadata_columns = [
        column
        for column in [
            "configuration_id",
            "corrective_genome_variant",
            "shared_hp_id",
            "low_rank_interaction_rank",
            "drug_view_fusion_method",
            "drug_view_low_rank",
            "selection_eligible",
        ]
        if column in registry.columns
    ]

    configuration = configuration.merge(
        registry[
            metadata_columns
        ].drop_duplicates(
            "configuration_id"
        ),
        on="configuration_id",
        how="left",
        validate="one_to_one",
    )

    parameter_records = []
    first_runs = (
        plan.sort_values(
            [
                "configuration_id",
                "run_id",
            ]
        )
        .drop_duplicates(
            "configuration_id"
        )
    )

    for row in first_runs.to_dict(
        orient="records"
    ):
        summary_path = (
            RESULT_ROOT
            / str(row["run_id"])
            / "run_summary.tsv"
        )

        run_summary = backend.read_tsv(
            summary_path
        )

        parameter_records.append(
            {
                "configuration_id": row[
                    "configuration_id"
                ],
                "model_parameter_count": int(
                    float(
                        run_summary[
                            "model_parameter_count"
                        ].iloc[0]
                    )
                ),
            }
        )

    configuration = configuration.merge(
        pd.DataFrame(parameter_records),
        on="configuration_id",
        how="left",
        validate="one_to_one",
    )

    write_tsv(
        configuration,
        configuration_path,
    )

    return configuration


def combine_with_script167(
    new_configuration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_path = (
        SCRIPT167_AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
    )

    old = read_tsv(old_path)

    combined_representations = {
        "ChemBERTa_mean_plus_Morgan",
        "ChemBERTa_mean_plus_Morgan_plus_RDKit",
    }

    old[
        "drug_view_fusion_method"
    ] = np.where(
        old[
            "drug_representation"
        ].isin(combined_representations),
        "raw_single_encoder_concatenation",
        "single_view",
    )

    old["drug_view_low_rank"] = 0
    old[
        "corrective_analysis_stage"
    ] = (
        "corrective_drug_representation_screen_v2"
    )

    if "selection_eligible" not in old.columns:
        old["selection_eligible"] = np.where(
            old[
                "drug_representation"
            ].eq(
                "identity_seen_drug_control"
            ),
            "NO",
            "YES",
        )

    all_columns = sorted(
        set(old.columns).union(
            new_configuration.columns
        )
    )

    for column in all_columns:
        if column not in old.columns:
            old[column] = pd.NA
        if column not in new_configuration.columns:
            new_configuration[column] = pd.NA

    combined = pd.concat(
        [
            old[all_columns],
            new_configuration[all_columns],
        ],
        ignore_index=True,
    )

    combined[
        "candidate_id"
    ] = (
        combined[
            "drug_representation"
        ].astype(str)
        + "__"
        + combined[
            "drug_view_fusion_method"
        ].astype(str)
    )

    if combined.duplicated(
        [
            "outer_target_code",
            "candidate_id",
        ]
    ).any():
        duplicates = combined.loc[
            combined.duplicated(
                [
                    "outer_target_code",
                    "candidate_id",
                ],
                keep=False,
            ),
            [
                "outer_target_code",
                "candidate_id",
            ],
        ]
        raise RuntimeError(
            "Duplicate combined drug candidates:\n"
            + duplicates.to_string(index=False)
        )

    for column in [
        "bidirectional_macro_rmse_mean",
        "worst_direction_macro_rmse_mean",
        "model_parameter_count",
    ]:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="raise",
        )

    combined["selection_rank"] = pd.NA

    eligible_mask = combined[
        "selection_eligible"
    ].astype(str).str.upper().eq("YES")

    eligible = combined.loc[
        eligible_mask
    ].sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "worst_direction_macro_rmse_mean",
            "model_parameter_count",
            "candidate_id",
        ],
        kind="stable",
    )

    ranks = (
        eligible.groupby(
            "outer_target_code"
        )
        .cumcount()
        .add(1)
    )

    combined.loc[
        eligible.index,
        "selection_rank",
    ] = ranks.astype(int)

    combined = combined.sort_values(
        [
            "outer_target_code",
            "selection_eligible",
            "selection_rank",
            "bidirectional_macro_rmse_mean",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    selected = (
        combined.loc[
            combined[
                "selection_rank"
            ].astype(str).eq("1")
        ][
            [
                column
                for column in [
                    "outer_target_code",
                    "genome_representation",
                    "corrective_genome_variant",
                    "shared_hp_id",
                    "low_rank_interaction_rank",
                    "drug_representation",
                    "drug_view_fusion_method",
                    "drug_view_low_rank",
                    "bidirectional_macro_rmse_mean",
                    "bidirectional_macro_rmse_sd",
                    "worst_direction_macro_rmse_mean",
                    "worst_direction_macro_rmse_sd",
                    "model_parameter_count",
                ]
                if column in combined.columns
            ]
        ]
        .sort_values(
            "outer_target_code"
        )
        .reset_index(drop=True)
    )

    if len(selected) != 3:
        raise RuntimeError(
            "Expected one final drug candidate per outer target."
        )

    if selected[
        "drug_representation"
    ].eq(
        "identity_seen_drug_control"
    ).any():
        raise RuntimeError(
            "Identity control was incorrectly selected."
        )

    return combined, selected


def aggregate_completed_runs() -> None:
    final165.ORIGINAL_AGGREGATE()

    new_configuration = (
        add_worst_direction_and_metadata()
    )

    combined, selected = combine_with_script167(
        new_configuration.copy()
    )

    combined_path = (
        AGGREGATE_ROOT
        / "complete_corrective_drug_representation_and_fusion_ranking_v1.tsv"
    )

    selected_path = (
        AGGREGATE_ROOT
        / "selected_final_corrective_drug_representation_registry_v1.tsv"
    )

    write_tsv(combined, combined_path)
    write_tsv(selected, selected_path)

    protocol_path = (
        METADATA_ROOT
        / "aggregate_protocol.tsv"
    )

    protocol = pd.DataFrame(
        [
            {
                "item": "primary_metric",
                "value": (
                    "bidirectional-average per-antibiotic macro RMSE"
                ),
            },
            {
                "item": "secondary_metric",
                "value": (
                    "seedwise worst-direction macro RMSE, mean and sample SD"
                ),
            },
            {
                "item": "complete_candidate_pool",
                "value": (
                    "single views; raw single-encoder concatenation; "
                    "separate-encoder projected drug fusion; "
                    "separate-encoder low-rank drug fusion"
                ),
            },
            {
                "item": "selection_rule",
                "value": (
                    "minimum primary mean among eligible candidates per "
                    "outer target; exact ties resolved by lower worst-"
                    "direction mean, then lower parameter count"
                ),
            },
            {
                "item": "identity_control_policy",
                "value": (
                    "reported but not eligible for selection"
                ),
            },
            {
                "item": "sample_sd",
                "value": "ddof=1",
            },
            {
                "item": "outer_target_label_policy",
                "value": (
                    "held-out outer-target MIC labels not used"
                ),
            },
        ]
    )

    write_tsv(protocol, protocol_path)

    aggregate_paths = [
        AGGREGATE_ROOT
        / "all_direction_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "direction_three_seed_mean_sd.tsv",
        AGGREGATE_ROOT
        / "bidirectional_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv",
        AGGREGATE_ROOT
        / "all_per_antibiotic_seed_metrics.tsv",
        AGGREGATE_ROOT
        / "per_antibiotic_three_seed_mean_sd.tsv",
        AGGREGATE_ROOT
        / "seedwise_worst_direction_macro_rmse.tsv",
        AGGREGATE_ROOT
        / "configuration_seedwise_worst_direction_mean_sd.tsv",
        combined_path,
        selected_path,
        protocol_path,
    ]

    for path in aggregate_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_path = (
        METADATA_ROOT
        / "aggregate_outputs_sha256.txt"
    )

    write_sha_manifest(
        aggregate_paths,
        manifest_path,
    )
    verify_manifest(manifest_path)

    print(
        "===== FINAL CORRECTIVE DRUG REPRESENTATIONS AFTER "
        "WITHIN-DRUG FUSION ====="
    )
    print(
        selected.to_string(index=False)
    )


def configure_backend() -> None:
    final165.RUN_PLAN_PATH = RUN_PLAN_PATH
    final165.RESULT_ROOT = RESULT_ROOT
    final165.METADATA_ROOT = METADATA_ROOT
    final165.AGGREGATE_ROOT = AGGREGATE_ROOT
    final165.EXPECTED_RUNS = EXPECTED_RUNS

    backend.__file__ = str(
        Path(__file__).resolve()
    )
    backend.RUN_PLAN_PATH = RUN_PLAN_PATH
    backend.SCRIPT142_OUTPUTS_PATH = (
        SCRIPT170_FREEZE
    )
    backend.SCRIPT142_FROZEN_PATH = (
        SCRIPT170_FREEZE
    )
    backend.IMPLEMENTATION_BUNDLE_PATH = (
        final165.IMPLEMENTATION_SPEC_PATH
    )
    backend.EXPECTED_IMPLEMENTATION_BUNDLE_SHA256 = (
        sha256_file(
            final165.IMPLEMENTATION_SPEC_PATH
        )
    )
    backend.RESULT_ROOT = RESULT_ROOT
    backend.METADATA_ROOT = METADATA_ROOT
    backend.AGGREGATE_ROOT = AGGREGATE_ROOT
    backend.EXPECTED_RUNS = EXPECTED_RUNS
    backend.KMER_PATHS = {
        representation: Path(
            spec[
                "genome_matrix_path"
            ]
        )
        for representation, spec
        in final165.CONFIG_BY_REPRESENTATION.items()
    }
    backend.build_model = build_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outer-target",
        choices=[
            "all",
            "ec",
            "kp",
            "se",
        ],
        default="all",
    )
    parser.add_argument(
        "--drug-representation",
        default="all",
    )
    parser.add_argument(
        "--fusion-method",
        choices=[
            "all",
            "separate_encoder_projected",
            "separate_encoder_low_rank",
        ],
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            20260811,
            20260812,
            20260813,
        ],
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )
    parser.add_argument(
        "--aggregate-every",
        type=int,
        default=6,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    required = [
        SCRIPT167_PATH,
        SCRIPT170_FREEZE,
        RUN_PLAN_PATH,
        CONFIGURATION_REGISTRY_PATH,
        SCRIPT167_AGGREGATE_MANIFEST,
        final165.IMPLEMENTATION_SPEC_PATH,
        backend.OBSERVATION_INDEX_PATH,
        backend.SCRIPT140_FROZEN_PATH,
        *backend.DRUG_VIEW_PATHS.values(),
        *[
            Path(
                spec[
                    "genome_matrix_path"
                ]
            )
            for spec in (
                final165.CONFIG_BY_REPRESENTATION.values()
            )
        ],
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verify_manifest(SCRIPT170_FREEZE)
    verify_manifest(SCRIPT167_AGGREGATE_MANIFEST)

    configure_backend()

    observations, plan = backend.load_inputs()

    if arguments.outer_target != "all":
        plan = plan.loc[
            plan[
                "outer_target_code"
            ].eq(
                arguments.outer_target
            )
        ]

    if arguments.drug_representation != "all":
        plan = plan.loc[
            plan[
                "drug_representation"
            ].eq(
                arguments.drug_representation
            )
        ]

    if arguments.fusion_method != "all":
        plan = plan.loc[
            plan[
                "drug_view_fusion_method"
            ].eq(
                arguments.fusion_method
            )
        ]

    plan = plan.loc[
        pd.to_numeric(
            plan["seed"],
            errors="raise",
        ).isin(arguments.seeds)
    ].copy()

    plan = plan.sort_values(
        [
            "outer_target_code",
            "drug_representation",
            "drug_view_fusion_method",
            "source_species_code",
            "seed",
        ]
    ).reset_index(drop=True)

    plan["already_complete"] = [
        backend.run_complete(
            str(run_id)
        )
        for run_id in plan["run_id"]
    ]

    print(
        "===== SCRIPT 171 RUN PLAN ====="
    )
    print(
        plan.groupby(
            [
                "outer_target_code",
                "drug_representation",
                "drug_view_fusion_method",
            ]
        )
        .agg(
            planned_runs=(
                "run_id",
                "size",
            ),
            completed_runs=(
                "already_complete",
                "sum",
            ),
        )
        .reset_index()
        .to_string(index=False)
    )
    print()
    print("Selected planned runs:", len(plan))
    print(
        "Already complete:",
        int(plan["already_complete"].sum()),
    )
    print(
        "New runs remaining:",
        int((~plan["already_complete"]).sum()),
    )

    if arguments.aggregate_only:
        aggregate_completed_runs()
        print(
            "STATUS: SCRIPT 171 AGGREGATE-ONLY COMPLETE"
        )
        return

    if arguments.dry_run:
        print(
            "STATUS: SCRIPT 171 DRY RUN COMPLETE"
        )
        return

    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)

    new_runs = 0

    for row in plan.to_dict(orient="records"):
        run_id = str(row["run_id"])

        if backend.run_complete(run_id):
            print(
                f"SKIP VERIFIED: {run_id}",
                flush=True,
            )
            continue

        execute_run(
            pd.Series(row),
            observations,
            device,
        )

        new_runs += 1

        if (
            not arguments.worker_only
            and arguments.aggregate_every > 0
            and new_runs
            % arguments.aggregate_every
            == 0
        ):
            aggregate_completed_runs()

        if (
            arguments.max_new_runs > 0
            and new_runs
            >= arguments.max_new_runs
        ):
            if not arguments.worker_only:
                aggregate_completed_runs()

            print(
                "STATUS: SCRIPT 171 PARTIAL RUN COMPLETE"
            )
            return

    if arguments.worker_only:
        print(
            "STATUS: SCRIPT 171 WORKER PARTITION COMPLETE"
        )
        return

    aggregate_completed_runs()

    print(
        "STATUS: SCRIPT 171 CORRECTIVE DRUG-VIEW "
        "FUSION SCREEN COMPLETE"
    )


if __name__ == "__main__":
    main()
