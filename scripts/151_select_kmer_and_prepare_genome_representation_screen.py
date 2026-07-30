#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get(
    "MIC_TRANSFER_PROJECT",
    Path(__file__).resolve().parents[1],
)).expanduser().resolve()

SCRIPT150 = PROJECT / "scripts/150_run_nested_loso_kmer_length_screen.py"
SCREEN_PROTOCOL = PROJECT / (
    "metadata/config_selection/nested_loso_v1/"
    "kmer_length_screen_v1/"
    "nested_loso_kmer_length_screen_protocol_v1.tsv"
)
SCREEN_METADATA_ROOT = PROJECT / (
    "metadata/config_selection/nested_loso_v1/"
    "kmer_length_screen_runs_v1"
)
SCREEN_AGGREGATE_ROOT = PROJECT / (
    "results/tables/config_selection/nested_loso_v1/"
    "kmer_length_screen_aggregate_v1"
)
SCREEN_AGGREGATE_MANIFEST = (
    SCREEN_METADATA_ROOT / "aggregate_outputs_sha256.txt"
)
SCREEN_RANKING = SCREEN_AGGREGATE_ROOT / (
    "configuration_bidirectional_three_seed_mean_sd_and_ranking.tsv"
)
FULL_KMER_RUN_PLAN = PROJECT / (
    "metadata/config_selection/nested_loso_v1/"
    "full_kmer_grid_v1/"
    "nested_loso_full_kmer_run_plan_v1.tsv"
)
KMER_ROOT = PROJECT / (
    "features/genome_representation/nested_loso_v1/canonical_kmer"
)
AMR_ROOT = PROJECT / (
    "features/genome_representation/nested_loso_v1/"
    "common_cross_species_amr"
)
FUSED_ROOT = PROJECT / (
    "features/genome_representation/nested_loso_v1/"
    "selected_kmer_plus_common_amr"
)
OUTPUT_ROOT = PROJECT / (
    "metadata/config_selection/nested_loso_v1/"
    "genome_representation_screen_v1"
)
TABLE_ROOT = PROJECT / (
    "results/tables/config_selection/nested_loso_v1/"
    "genome_representation_screen_v1"
)
OUTPUT_MANIFEST = OUTPUT_ROOT / "script151_outputs_sha256.txt"
FREEZE_MANIFEST = PROJECT / (
    "metadata/config_selection/"
    "script151_successful_selected_kmer_and_matrix_core_sha256.txt"
)

EXPECTED_RUNS = 90
EXPECTED_GENOMES = 21394
EXPECTED_OUTERS = {"ec", "kp", "se"}
EXPECTED_KMERS = {
    "canonical_4mer", "canonical_5mer", "canonical_6mer",
    "canonical_7mer", "canonical_8mer",
}
KMER_DIMENSIONS = {
    "canonical_4mer": 136,
    "canonical_5mer": 512,
    "canonical_6mer": 2080,
    "canonical_7mer": 8192,
    "canonical_8mer": 32896,
}
AMR_DIMENSIONS = {"ec": 92, "kp": 91, "se": 108}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project_path(text: str) -> Path:
    path = Path(text.strip())
    return path if path.is_absolute() else PROJECT / path


def verify_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checked = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed manifest line {i}: {path}")
        expected, value = parts
        candidate = project_path(value)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        observed = sha256_file(candidate)
        if observed != expected:
            raise RuntimeError(f"SHA mismatch: {candidate}")
        checked.append(candidate)
    if not checked:
        raise RuntimeError(f"Empty manifest: {path}")
    return checked


def write_manifest(paths: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candidate in sorted(set(paths), key=lambda x: x.as_posix()):
            f.write(
                f"{sha256_file(candidate)}  "
                f"{candidate.relative_to(PROJECT)}\n"
            )


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path, sep="\t", dtype=str,
        keep_default_na=False, low_memory=False,
    )


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path, sep="\t", index=False, lineterminator="\n"
    )


def kmer_number(rep: str) -> int:
    return int(rep.removeprefix("canonical_").removesuffix("mer"))


def kmer_path(rep: str) -> Path:
    k = kmer_number(rep)
    return KMER_ROOT / (
        f"nested_loso_all_species_canonical_{k}mer_"
        "relative_frequency_v1.npy"
    )


def amr_path(outer: str) -> Path:
    return AMR_ROOT / (
        f"outer_{outer}_common_cross_species_amr_binary_v1.npy"
    )


def fused_path(outer: str, rep: str) -> Path:
    k = kmer_number(rep)
    return FUSED_ROOT / (
        f"outer_{outer}_selected_{k}mer_plus_common_amr_float32_v1.npy"
    )


def verify_screen_runs() -> list[Path]:
    flags = sorted(SCREEN_METADATA_ROOT.glob("*/RUN_COMPLETE"))
    if len(flags) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} completed runs; observed {len(flags)}."
        )
    verified = []
    for flag in flags:
        if flag.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"Nonzero RUN_COMPLETE: {flag}")
        manifest = flag.parent / "outputs_sha256.txt"
        verified.extend(verify_manifest(manifest))
        verified.extend([flag, manifest])
    return verified


def select_kmers() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = read_tsv(SCREEN_RANKING)
    required = {
        "outer_target_code", "configuration_id",
        "genome_representation", "drug_representation",
        "cross_modal_architecture", "seed_count",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    }
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise RuntimeError("Missing ranking columns: " + "|".join(missing))
    ranking["bidirectional_macro_rmse_mean"] = pd.to_numeric(
        ranking["bidirectional_macro_rmse_mean"], errors="raise"
    )
    ranking["bidirectional_macro_rmse_sd"] = pd.to_numeric(
        ranking["bidirectional_macro_rmse_sd"], errors="coerce"
    )
    ranking["seed_count"] = pd.to_numeric(
        ranking["seed_count"], errors="raise"
    ).astype(int)
    if len(ranking) != 15:
        raise RuntimeError(f"Expected 15 rows; observed {len(ranking)}.")
    if set(ranking["outer_target_code"]) != EXPECTED_OUTERS:
        raise RuntimeError("Unexpected outer targets.")
    if set(ranking["genome_representation"]) != EXPECTED_KMERS:
        raise RuntimeError("Unexpected k-mer candidates.")
    if set(ranking["drug_representation"]) != {"Morgan"}:
        raise RuntimeError("Morgan was not fixed.")
    if set(ranking["cross_modal_architecture"]) != {
        "projected_concatenation_MLP"
    }:
        raise RuntimeError("Projected-concatenation MLP was not fixed.")
    if not ranking["seed_count"].eq(3).all():
        raise RuntimeError("Not all configurations have three seeds.")
    ranking["kmer_length"] = ranking[
        "genome_representation"
    ].map(kmer_number)
    ranking = ranking.sort_values(
        [
            "outer_target_code",
            "bidirectional_macro_rmse_mean",
            "kmer_length",
        ]
    ).reset_index(drop=True)
    ranking["selection_rank"] = (
        ranking.groupby("outer_target_code").cumcount() + 1
    )
    selected = ranking.loc[
        ranking["selection_rank"].eq(1)
    ].copy().reset_index(drop=True)
    if len(selected) != 3:
        raise RuntimeError("Expected one selected k-mer per outer target.")
    selected["selected_kmer_dimension"] = selected[
        "genome_representation"
    ].map(KMER_DIMENSIONS)
    selected["selection_metric"] = (
        "three_seed_mean_bidirectional_macro_rmse"
    )
    selected["exact_tie_rule"] = "smaller_kmer_length"
    selected["outer_target_labels_used"] = "NO"
    return ranking, selected


def build_fused(selected: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    FUSED_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    outputs = []
    for row in selected.itertuples(index=False):
        outer = str(row.outer_target_code)
        rep = str(row.genome_representation)
        kp, ap, fp = kmer_path(rep), amr_path(outer), fused_path(outer, rep)
        for path in [kp, ap]:
            if not path.is_file():
                raise FileNotFoundError(path)
        kmer = np.load(kp, mmap_mode="r", allow_pickle=False)
        amr = np.load(ap, mmap_mode="r", allow_pickle=False)
        kd, ad = KMER_DIMENSIONS[rep], AMR_DIMENSIONS[outer]
        if kmer.shape != (EXPECTED_GENOMES, kd):
            raise RuntimeError(f"K-mer shape mismatch: {kp} {kmer.shape}")
        if amr.shape != (EXPECTED_GENOMES, ad):
            raise RuntimeError(f"AMR shape mismatch: {ap} {amr.shape}")
        shape = (EXPECTED_GENOMES, kd + ad)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fused = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=shape
        )
        for start in range(0, EXPECTED_GENOMES, 256):
            stop = min(start + 256, EXPECTED_GENOMES)
            fused[start:stop, :kd] = np.asarray(
                kmer[start:stop], dtype=np.float32
            )
            fused[start:stop, kd:] = np.asarray(
                amr[start:stop], dtype=np.float32
            )
        fused.flush()
        del fused
        tmp.replace(fp)
        check = np.load(fp, mmap_mode="r", allow_pickle=False)
        if check.shape != shape or check.dtype != np.float32:
            raise RuntimeError(f"Fused validation failed: {fp}")
        records.append({
            "outer_target_code": outer,
            "selected_kmer_representation": rep,
            "selected_kmer_dimension": kd,
            "common_amr_dimension": ad,
            "fused_dimension": kd + ad,
            "rows": EXPECTED_GENOMES,
            "dtype": "float32",
            "kmer_matrix_path": str(kp.relative_to(PROJECT)),
            "common_amr_matrix_path": str(ap.relative_to(PROJECT)),
            "fused_matrix_path": str(fp.relative_to(PROJECT)),
            "fused_matrix_sha256": sha256_file(fp),
        })
        outputs.append(fp)
    return pd.DataFrame(records), outputs


def next_plan(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = read_tsv(FULL_KMER_RUN_PLAN)
    template = full.loc[
        full["genome_representation"].eq("canonical_4mer")
        & full["drug_representation"].eq("Morgan")
        & full["cross_modal_architecture"].eq(
            "projected_concatenation_MLP"
        )
    ].copy()
    if len(template) != 18:
        raise RuntimeError(f"Expected 18 template rows; observed {len(template)}.")
    lookup = dict(zip(
        selected["outer_target_code"],
        selected["genome_representation"],
    ))
    frames = []
    for rep in [
        "common_cross_species_AMR",
        "selected_kmer_plus_common_AMR",
    ]:
        frame = template.copy()
        frame["genome_representation"] = rep
        frame["selected_kmer_representation"] = frame[
            "outer_target_code"
        ].map(lookup)
        frame["configuration_id"] = (
            "outer_" + frame["outer_target_code"].astype(str)
            + "__" + rep
            + "__Morgan__projected_concatenation_MLP"
        )
        frame["run_id"] = (
            frame["configuration_id"].astype(str)
            + "__" + frame["source_species_code"].astype(str)
            + "_to_" + frame["evaluation_species_code"].astype(str)
            + "__seed_" + frame["seed"].astype(str)
        )
        frames.append(frame)
    run_plan = pd.concat(frames, ignore_index=True)
    if len(run_plan) != 36 or run_plan["run_id"].duplicated().any():
        raise RuntimeError("Invalid 36-run genome screen plan.")
    run_plan = run_plan.sort_values(
        [
            "outer_target_code", "genome_representation",
            "source_species_code", "seed",
        ]
    ).reset_index(drop=True)
    config = run_plan[[
        "configuration_id", "outer_target_code",
        "genome_representation", "selected_kmer_representation",
        "drug_representation", "cross_modal_architecture",
    ]].drop_duplicates().reset_index(drop=True)
    if len(config) != 6:
        raise RuntimeError("Expected six next-stage configurations.")
    return run_plan, config


def main() -> None:
    for path in [
        SCRIPT150, SCREEN_PROTOCOL, SCREEN_AGGREGATE_MANIFEST,
        SCREEN_RANKING, FULL_KMER_RUN_PLAN,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    verified_runs = verify_screen_runs()
    verified_aggregate = verify_manifest(SCREEN_AGGREGATE_MANIFEST)
    ranking, selected = select_kmers()
    fused_registry, fused_outputs = build_fused(selected)
    run_plan, config = next_plan(selected)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    ranking_path = TABLE_ROOT / "nested_loso_kmer_length_complete_ranking_v1.tsv"
    selected_path = OUTPUT_ROOT / "nested_loso_selected_kmer_registry_v1.tsv"
    fused_registry_path = OUTPUT_ROOT / (
        "nested_loso_selected_kmer_plus_common_amr_matrix_registry_v1.tsv"
    )
    run_plan_path = OUTPUT_ROOT / (
        "nested_loso_genome_representation_screen_run_plan_v1.tsv"
    )
    config_path = OUTPUT_ROOT / (
        "nested_loso_genome_representation_screen_configuration_registry_v1.tsv"
    )
    protocol_path = OUTPUT_ROOT / (
        "nested_loso_genome_representation_screen_protocol_v1.tsv"
    )
    protocol = pd.DataFrame([
        {"item": "stage_objective", "value": (
            "compare selected canonical k-mer, common cross-species AMR, "
            "and selected-kmer-plus-common-AMR"
        )},
        {"item": "fixed_drug_representation", "value": "Morgan"},
        {"item": "fixed_cross_modal_architecture",
         "value": "projected_concatenation_MLP"},
        {"item": "new_training_fits", "value": 36},
        {"item": "selected_kmer_metrics_reused", "value": "YES"},
        {"item": "primary_metric",
         "value": "bidirectional per-antibiotic macro RMSE"},
        {"item": "selection_scope",
         "value": "separate per outer target"},
        {"item": "outer_target_labels_used", "value": "NO"},
        {"item": "full_cartesian_product", "value": "NO"},
        {"item": "models_trained_by_script151", "value": "NO"},
    ])
    write_tsv(ranking, ranking_path)
    write_tsv(selected, selected_path)
    write_tsv(fused_registry, fused_registry_path)
    write_tsv(run_plan, run_plan_path)
    write_tsv(config, config_path)
    write_tsv(protocol, protocol_path)
    input_manifest_path = OUTPUT_ROOT / "script151_input_manifest.tsv"
    inputs = [
        Path(__file__).resolve(), SCRIPT150, SCREEN_PROTOCOL,
        SCREEN_AGGREGATE_MANIFEST, SCREEN_RANKING, FULL_KMER_RUN_PLAN,
    ]
    write_tsv(pd.DataFrame([
        {
            "file_path": str(path.relative_to(PROJECT)),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(inputs, key=lambda x: x.as_posix())
    ]), input_manifest_path)
    outputs = [
        ranking_path, selected_path, fused_registry_path,
        run_plan_path, config_path, protocol_path,
        input_manifest_path, *fused_outputs,
    ]
    write_manifest(outputs, OUTPUT_MANIFEST)
    verify_manifest(OUTPUT_MANIFEST)
    freeze = [
        Path(__file__).resolve(), OUTPUT_MANIFEST, *outputs,
        SCREEN_AGGREGATE_MANIFEST, *verified_aggregate,
    ]
    write_manifest(freeze, FREEZE_MANIFEST)
    verify_manifest(FREEZE_MANIFEST)
    print("===== SCRIPT 151 K-MER SELECTION =====")
    print(selected[[
        "outer_target_code", "genome_representation",
        "selected_kmer_dimension",
        "bidirectional_macro_rmse_mean",
        "bidirectional_macro_rmse_sd",
    ]].to_string(index=False))
    print()
    print("===== FUSED MATRIX REGISTRY =====")
    print(fused_registry.to_string(index=False))
    print()
    print("Verified completed screen runs:", EXPECTED_RUNS)
    print("Verified screen run files:", len(verified_runs))
    print("Next-stage configurations:", len(config))
    print("Next-stage new training fits:", len(run_plan))
    print("Models trained: NO")
    print()
    print(
        "STATUS: SCRIPT 151 K-MERS SELECTED AND "
        "GENOME-REPRESENTATION SCREEN PREPARED"
    )


if __name__ == "__main__":
    main()
