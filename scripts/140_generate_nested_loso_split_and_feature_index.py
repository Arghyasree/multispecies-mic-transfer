#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PAIRWISE = Path('metadata/config_selection/nested_loso_v1/feature_asset_audit_v1/pairwise_development_observation_manifest.tsv')
OUTER = Path('metadata/config_selection/nested_loso_v1/nested_loso_outer_target_registry_v1.tsv')
KMER = Path('metadata/config_selection/nested_loso_v1/genome_features/canonical_kmer_v1/nested_loso_all_species_kmer_feature_rows_v1.tsv')
DRUG = Path('metadata/drug_representation/drug_feature_rows_v1.tsv')
DUP = Path('results/tables/config_selection/nested_loso_v1/genome_features/canonical_kmer_v1/nested_loso_all_species_duplicate_8mer_profile_groups_v1.tsv')
FROZEN136 = Path('metadata/config_selection/script136_successful_run_core_sha256.txt')

OUTDIR = Path('metadata/config_selection/nested_loso_v1/configuration_splits_v1')
TABDIR = Path('results/tables/config_selection/nested_loso_v1/configuration_splits_v1')
OBS_OUT = OUTDIR / 'nested_loso_observation_feature_index_v1.tsv'
GENOME_OUT = OUTDIR / 'nested_loso_genome_fold_registry_v1.tsv'
TASK_OUT = OUTDIR / 'nested_loso_configuration_task_registry_v1.tsv'
PROTOCOL_OUT = OUTDIR / 'nested_loso_configuration_split_protocol_v1.tsv'
INPUT_OUT = OUTDIR / 'script140_input_manifest.tsv'
SHA_OUT = OUTDIR / 'script140_outputs_sha256.txt'
BALANCE_OUT = TABDIR / 'nested_loso_within_species_fold_balance_v1.tsv'
SUMMARY_OUT = TABDIR / 'nested_loso_within_species_fold_summary_v1.tsv'
LEAK_OUT = TABDIR / 'nested_loso_duplicate_profile_fold_leakage_audit_v1.tsv'

N_FOLDS = 5
SEED = 20260727
EXPECTED_ROWS = 247_719
EXPECTED_GENOMES = 21_394
EXPECTED_DRUGS = 19
EXPECTED_DUP_ROWS = 74
EXPECTED_DUP_GROUPS = 37
EXPECTED_PANEL_COUNTS = {'ec': 6, 'se': 17, 'kp': 8}
NAME_TO_CODE = {
    'Klebsiella pneumoniae': 'kp',
    'Escherichia coli': 'ec',
    'Salmonella enterica': 'se',
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sha_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding='utf-8').splitlines():
        expected, file_text = line.split(maxsplit=1)
        file_path = Path(file_text.strip())
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if sha256_file(file_path) != expected:
            raise RuntimeError(f'SHA mismatch: {file_path}')


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f'{label} missing columns: {sorted(missing)}')


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, lineterminator='\n')


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)


def pipe_values(value: str) -> list[str]:
    return [x.strip().casefold() for x in str(value).split('|') if x.strip()]


def assign_group_folds(obs: pd.DataFrame, outer: str, species: str, drugs: list[str]) -> pd.DataFrame:
    sub = obs.loc[
        obs['outer_target_code'].eq(outer)
        & obs['development_species_code'].eq(species)
    ].copy()
    if sub.empty:
        raise RuntimeError(f'No rows for {outer}/{species}')
    observed_drugs = sorted(sub['normalized_antibiotic'].unique().tolist())
    if observed_drugs != sorted(drugs):
        raise RuntimeError(f'Drug mismatch for {outer}/{species}')

    gd = (
        sub.groupby(['genome_group_id', 'normalized_antibiotic'])
        .size().unstack(fill_value=0).reindex(columns=drugs, fill_value=0).astype(np.int64)
    )
    gg = sub.groupby('genome_group_id')['genome_id'].nunique().reindex(gd.index).astype(np.int64)
    gt = gd.sum(axis=1).astype(np.int64)

    order = pd.DataFrame({
        'genome_group_id': gd.index.astype(str),
        'group_observations': gt.to_numpy(),
        'group_unique_genomes': gg.to_numpy(),
        'group_max_drug': gd.max(axis=1).to_numpy(),
    })
    order['tie'] = [stable_int(f'{SEED}|{outer}|{species}|{g}') for g in order['genome_group_id']]
    order = order.sort_values(
        ['group_observations', 'group_max_drug', 'group_unique_genomes', 'tie'],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    target_drug = gd.sum(axis=0).to_numpy(dtype=float) / N_FOLDS
    target_total = float(gt.sum()) / N_FOLDS
    target_genomes = float(gg.sum()) / N_FOLDS
    fold_drug = np.zeros((N_FOLDS, len(drugs)), dtype=float)
    fold_total = np.zeros(N_FOLDS, dtype=float)
    fold_genomes = np.zeros(N_FOLDS, dtype=float)
    fold_groups = np.zeros(N_FOLDS, dtype=float)
    rows = []

    for r in order.itertuples(index=False):
        gid = str(r.genome_group_id)
        vec = gd.loc[gid].to_numpy(dtype=float)
        scores = []
        for f in range(N_FOLDS):
            pdrug = fold_drug[f] + vec
            ptotal = fold_total[f] + float(r.group_observations)
            pgen = fold_genomes[f] + float(r.group_unique_genomes)
            drug_load = float(np.sum((pdrug / (target_drug + 1.0)) ** 2))
            total_load = (ptotal / (target_total + 1.0)) ** 2
            genome_load = (pgen / (target_genomes + 1.0)) ** 2
            score = drug_load + 0.25 * total_load + 0.10 * genome_load
            scores.append((score, fold_groups[f], fold_total[f], f))
        _, _, _, chosen = min(scores)
        fold_drug[chosen] += vec
        fold_total[chosen] += float(r.group_observations)
        fold_genomes[chosen] += float(r.group_unique_genomes)
        fold_groups[chosen] += 1
        rows.append({
            'outer_target_code': outer,
            'development_species_code': species,
            'genome_group_id': gid,
            'within_species_fold': f'fold_{chosen + 1:02d}',
            'assignment_seed': SEED,
            'assignment_method': 'deterministic_greedy_drug_count_balance',
        })

    out = pd.DataFrame(rows)
    if out['genome_group_id'].duplicated().any() or out['within_species_fold'].nunique() != N_FOLDS:
        raise RuntimeError(f'Invalid fold assignment for {outer}/{species}')
    return out


def main() -> None:
    for path in [PAIRWISE, OUTER, KMER, DRUG, DUP, FROZEN136]:
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_sha_manifest(FROZEN136)

    pairwise = read_tsv(PAIRWISE)
    outer = read_tsv(OUTER)
    kmer = read_tsv(KMER)
    drug = read_tsv(DRUG)
    dup = read_tsv(DUP)

    require_columns(pairwise, [
        'outer_target_code', 'development_species_code', 'development_species',
        'observation_id', 'genome_id', 'normalized_antibiotic',
        'mic_target_log2_mg_per_l'
    ], 'pairwise')
    require_columns(outer, [
        'outer_target_code', 'development_species_a', 'development_species_b',
        'development_drug_count', 'development_antibiotics', 'selected_configuration_id'
    ], 'outer registry')
    require_columns(kmer, ['feature_row', 'species_code', 'species', 'genome_id'], 'kmer')
    require_columns(drug, [
        'feature_row', 'antibiotic', 'identity_feature_row', 'morgan_feature_row',
        'rdkit_feature_row', 'chemberta_mean_feature_row', 'chemberta_first_feature_row'
    ], 'drug')
    require_columns(dup, [
        'duplicate_profile_group_id', 'species_code', 'genome_id', 'group_size'
    ], 'duplicate groups')

    if len(pairwise) != EXPECTED_ROWS or len(kmer) != EXPECTED_GENOMES or len(drug) != EXPECTED_DRUGS:
        raise RuntimeError('Primary input row-count mismatch')
    if len(dup) != EXPECTED_DUP_ROWS or dup['duplicate_profile_group_id'].nunique() != EXPECTED_DUP_GROUPS:
        raise RuntimeError('Duplicate-profile count mismatch')

    for c in ['outer_target_code', 'development_species_code', 'development_species', 'observation_id', 'genome_id', 'normalized_antibiotic']:
        pairwise[c] = pairwise[c].astype(str).str.strip()
    pairwise['outer_target_code'] = pairwise['outer_target_code'].str.casefold()
    pairwise['development_species_code'] = pairwise['development_species_code'].str.casefold()
    pairwise['normalized_antibiotic'] = pairwise['normalized_antibiotic'].str.casefold()
    pairwise['mic_target_log2_mg_per_l'] = pd.to_numeric(pairwise['mic_target_log2_mg_per_l'], errors='raise')
    if not np.isfinite(pairwise['mic_target_log2_mg_per_l'].to_numpy(float)).all():
        raise RuntimeError('Non-finite MIC target')
    if pairwise.duplicated().any() or pairwise.duplicated(['outer_target_code', 'observation_id']).any():
        raise RuntimeError('Duplicate observation row or outer-target observation ID')

    kmap = kmer[['feature_row', 'species_code', 'species', 'genome_id']].rename(columns={
        'feature_row': 'genome_feature_row', 'species_code': 'kmer_species_code', 'species': 'kmer_species'
    })
    if kmap['genome_id'].duplicated().any():
        raise RuntimeError('Duplicate k-mer genome ID')

    dmap = drug[[
        'feature_row', 'antibiotic', 'identity_feature_row', 'morgan_feature_row',
        'rdkit_feature_row', 'chemberta_mean_feature_row', 'chemberta_first_feature_row'
    ]].rename(columns={'feature_row': 'drug_feature_row', 'antibiotic': 'normalized_antibiotic'})
    dmap['normalized_antibiotic'] = dmap['normalized_antibiotic'].astype(str).str.strip().str.casefold()
    if dmap['normalized_antibiotic'].duplicated().any():
        raise RuntimeError('Duplicate drug row')

    indexed = pairwise.merge(kmap, on='genome_id', how='left', validate='many_to_one')
    indexed = indexed.merge(dmap, on='normalized_antibiotic', how='left', validate='many_to_one')
    mapped_cols = [
        'genome_feature_row', 'drug_feature_row', 'identity_feature_row', 'morgan_feature_row',
        'rdkit_feature_row', 'chemberta_mean_feature_row', 'chemberta_first_feature_row'
    ]
    for c in mapped_cols:
        if indexed[c].isna().any() or indexed[c].eq('').any():
            raise RuntimeError(f'Missing feature mapping: {c}')
        indexed[c] = pd.to_numeric(indexed[c], errors='raise').astype(np.int64)
    if not indexed['development_species_code'].eq(indexed['kmer_species_code']).all():
        raise RuntimeError('Species mismatch between observation and k-mer rows')

    dup_map = dup[['genome_id', 'species_code', 'duplicate_profile_group_id', 'group_size']].rename(columns={'species_code': 'dup_species_code'})
    if dup_map['genome_id'].duplicated().any():
        raise RuntimeError('Genome in multiple duplicate groups')
    if not dup_map.groupby('duplicate_profile_group_id')['dup_species_code'].nunique().eq(1).all():
        raise RuntimeError('Duplicate 8-mer group spans species')
    indexed = indexed.merge(dup_map, on='genome_id', how='left', validate='many_to_one')
    has_dup = indexed['duplicate_profile_group_id'].notna() & indexed['duplicate_profile_group_id'].ne('')
    if not indexed.loc[has_dup, 'development_species_code'].eq(indexed.loc[has_dup, 'dup_species_code']).all():
        raise RuntimeError('Duplicate-group species mismatch')
    indexed['genome_group_id'] = np.where(
        has_dup,
        indexed['duplicate_profile_group_id'],
        'singleton::' + indexed['development_species_code'] + '::' + indexed['genome_id'],
    )
    indexed['duplicate_profile_group_id'] = indexed['duplicate_profile_group_id'].fillna('').replace('', 'not_duplicate')
    indexed['duplicate_profile_group_size'] = pd.to_numeric(indexed['group_size'], errors='coerce').fillna(1).astype(np.int64)

    panel_by_outer: dict[str, list[str]] = {}
    fold_frames = []
    task_rows = []
    for r in outer.itertuples(index=False):
        ot = str(r.outer_target_code).strip().casefold()
        drugs = pipe_values(r.development_antibiotics)
        if ot not in EXPECTED_PANEL_COUNTS or len(drugs) != EXPECTED_PANEL_COUNTS[ot] or int(r.development_drug_count) != EXPECTED_PANEL_COUNTS[ot]:
            raise RuntimeError(f'Outer panel mismatch: {ot}')
        panel_by_outer[ot] = drugs
        a = NAME_TO_CODE[str(r.development_species_a)]
        b = NAME_TO_CODE[str(r.development_species_b)]
        observed_species = sorted(indexed.loc[indexed['outer_target_code'].eq(ot), 'development_species_code'].unique().tolist())
        if observed_species != sorted([a, b]):
            raise RuntimeError(f'Development species mismatch: {ot}')
        for s in [a, b]:
            fold_frames.append(assign_group_folds(indexed, ot, s, drugs))
            subset = indexed.loc[indexed['outer_target_code'].eq(ot) & indexed['development_species_code'].eq(s)]
            task_rows.append({
                'outer_target_code': ot,
                'selected_configuration_id': str(r.selected_configuration_id),
                'task_id': f'{ot}__within_{s}',
                'task_kind': 'within_species_fivefold_guardrail',
                'source_species_code': s,
                'evaluation_species_code': s,
                'source_observations': len(subset),
                'evaluation_observations': len(subset),
                'development_drug_count': len(drugs),
                'development_antibiotics': '|'.join(drugs),
                'fold_policy': 'fivefold genome-group disjoint CV',
                'primary_selection_role': 'guardrail_and_tiebreaker',
            })
        for source, target in [(a, b), (b, a)]:
            source_n = len(indexed.loc[indexed['outer_target_code'].eq(ot) & indexed['development_species_code'].eq(source)])
            target_n = len(indexed.loc[indexed['outer_target_code'].eq(ot) & indexed['development_species_code'].eq(target)])
            task_rows.append({
                'outer_target_code': ot,
                'selected_configuration_id': str(r.selected_configuration_id),
                'task_id': f'{ot}__{source}_to_{target}',
                'task_kind': 'cross_species_configuration_validation',
                'source_species_code': source,
                'evaluation_species_code': target,
                'source_observations': source_n,
                'evaluation_observations': target_n,
                'development_drug_count': len(drugs),
                'development_antibiotics': '|'.join(drugs),
                'fold_policy': 'all source observations train; all opposite-development observations evaluate',
                'primary_selection_role': 'primary_bidirectional_rmse',
            })

    group_fold = pd.concat(fold_frames, ignore_index=True)
    indexed = indexed.merge(
        group_fold[['outer_target_code', 'development_species_code', 'genome_group_id', 'within_species_fold']],
        on=['outer_target_code', 'development_species_code', 'genome_group_id'],
        how='left', validate='many_to_one'
    )
    if indexed['within_species_fold'].isna().any() or indexed['within_species_fold'].eq('').any():
        raise RuntimeError('Missing fold assignment')
    if not indexed.groupby(['outer_target_code', 'development_species_code', 'genome_group_id'])['within_species_fold'].nunique().eq(1).all():
        raise RuntimeError('Genome group split across folds')

    genome_fold = indexed[[
        'outer_target_code', 'development_species_code', 'development_species', 'genome_id',
        'genome_feature_row', 'genome_group_id', 'duplicate_profile_group_id',
        'duplicate_profile_group_size', 'within_species_fold'
    ]].drop_duplicates().sort_values([
        'outer_target_code', 'development_species_code', 'within_species_fold', 'genome_id'
    ]).reset_index(drop=True)
    if genome_fold.duplicated(['outer_target_code', 'development_species_code', 'genome_id']).any():
        raise RuntimeError('Genome assigned to multiple folds')

    leak = (
        genome_fold.loc[genome_fold['duplicate_profile_group_id'].ne('not_duplicate')]
        .groupby(['outer_target_code', 'development_species_code', 'duplicate_profile_group_id'])
        .agg(
            group_genomes=('genome_id', 'nunique'),
            assigned_folds=('within_species_fold', 'nunique'),
            fold_values=('within_species_fold', lambda x: '|'.join(sorted(set(x)))),
        ).reset_index()
    )
    if not leak['assigned_folds'].eq(1).all():
        raise RuntimeError('Duplicate-profile leakage across folds')

    balance = (
        indexed.groupby([
            'outer_target_code', 'development_species_code', 'development_species',
            'within_species_fold', 'normalized_antibiotic'
        ]).agg(
            observations=('observation_id', 'size'),
            unique_genomes=('genome_id', 'nunique'),
            unique_genome_groups=('genome_group_id', 'nunique'),
            mean_mic_log2=('mic_target_log2_mg_per_l', 'mean'),
            sd_mic_log2=('mic_target_log2_mg_per_l', 'std'),
        ).reset_index()
    )
    summary = (
        indexed.groupby([
            'outer_target_code', 'development_species_code', 'development_species', 'within_species_fold'
        ]).agg(
            observations=('observation_id', 'size'),
            unique_observations=('observation_id', 'nunique'),
            unique_genomes=('genome_id', 'nunique'),
            unique_genome_groups=('genome_group_id', 'nunique'),
            unique_antibiotics=('normalized_antibiotic', 'nunique'),
        ).reset_index()
    )

    task_registry = pd.DataFrame(task_rows).sort_values(['outer_target_code', 'task_kind', 'task_id']).reset_index(drop=True)
    indexed = indexed[[
        'outer_target_code', 'development_species_code', 'development_species', 'observation_id',
        'genome_id', 'normalized_antibiotic', 'mic_target_log2_mg_per_l', 'genome_feature_row',
        'drug_feature_row', 'identity_feature_row', 'morgan_feature_row', 'rdkit_feature_row',
        'chemberta_mean_feature_row', 'chemberta_first_feature_row', 'genome_group_id',
        'duplicate_profile_group_id', 'duplicate_profile_group_size', 'within_species_fold'
    ]].sort_values([
        'outer_target_code', 'development_species_code', 'genome_id',
        'normalized_antibiotic', 'observation_id'
    ]).reset_index(drop=True)
    indexed.insert(0, 'configuration_observation_row', np.arange(len(indexed), dtype=np.int64))

    protocol = pd.DataFrame([
        {'item': 'split_protocol_id', 'value': 'nested_loso_configuration_splits_v1'},
        {'item': 'outer_target_policy', 'value': 'outer-target MIC labels are absent from these manifests'},
        {'item': 'within_species_folds', 'value': N_FOLDS},
        {'item': 'split_seed', 'value': SEED},
        {'item': 'assignment_unit', 'value': 'genome group; duplicate 8-mer profiles remain together'},
        {'item': 'assignment_objective', 'value': 'deterministic greedy balance of per-antibiotic observation counts, totals and genome counts'},
        {'item': 'outcome_use_in_assignment', 'value': 'none; MIC values are not used to assign folds'},
        {'item': 'cross_species_task_policy', 'value': 'all observations from one development species train; all matched-drug observations from the other evaluate'},
        {'item': 'feature_alignment', 'value': 'frozen all-species k-mer registry plus frozen 19-drug registry'},
        {'item': 'models_trained', 'value': 'none'},
    ])

    input_paths = [Path(__file__).resolve(), PAIRWISE, OUTER, KMER, DRUG, DUP, FROZEN136]
    input_manifest = pd.DataFrame([
        {'file_path': str(p), 'file_size_bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        for p in sorted(input_paths, key=lambda x: x.as_posix())
    ])

    write_tsv(indexed, OBS_OUT)
    write_tsv(genome_fold, GENOME_OUT)
    write_tsv(task_registry, TASK_OUT)
    write_tsv(protocol, PROTOCOL_OUT)
    write_tsv(input_manifest, INPUT_OUT)
    write_tsv(balance, BALANCE_OUT)
    write_tsv(summary, SUMMARY_OUT)
    write_tsv(leak, LEAK_OUT)

    outputs = [OBS_OUT, GENOME_OUT, TASK_OUT, PROTOCOL_OUT, INPUT_OUT, BALANCE_OUT, SUMMARY_OUT, LEAK_OUT]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with SHA_OUT.open('w', encoding='utf-8') as f:
        for p in sorted(outputs, key=lambda x: x.as_posix()):
            f.write(f'{sha256_file(p)}  {p}\n')
    verify_sha_manifest(SHA_OUT)

    print('===== SCRIPT 140 SPLIT SUMMARY =====')
    print(summary.to_string(index=False))
    print('\n===== CONFIGURATION TASKS =====')
    print(task_registry[[
        'outer_target_code', 'task_id', 'task_kind', 'source_species_code',
        'evaluation_species_code', 'source_observations', 'evaluation_observations',
        'development_drug_count', 'primary_selection_role'
    ]].to_string(index=False))
    print('\n===== DUPLICATE-PROFILE LEAKAGE AUDIT =====')
    print('Audited duplicate-profile outer/species groups:', len(leak))
    print('Groups spanning multiple folds:', int(leak['assigned_folds'].gt(1).sum()))
    print('\nObservation-index rows:', len(indexed))
    print('Genome-fold rows:', len(genome_fold))
    print('Models trained: NO')
    print('\nSTATUS: SCRIPT 140 NESTED-LOSO SPLITS AND FEATURE INDEX COMPLETE')


if __name__ == '__main__':
    main()
