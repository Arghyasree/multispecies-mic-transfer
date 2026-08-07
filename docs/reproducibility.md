# Reproducibility

This document explains what can be reproduced directly from the released repository and what requires external raw data or software.

The repository supports three levels of reproducibility:

1. checking the released study artifacts;
2. rerunning the final cross-species transfer evaluation;
3. reconstructing the study from the original public data resources.

## 1. Released Study Artifacts

The repository includes the main artifacts used for the reported experiments:

- the final quantitative MIC benchmark with **168,363 unique genome–antibiotic observations**;
- **21,394 quality-controlled and sequence-verified genomes** represented in the benchmark;
- the selected genome feature matrices;
- the antibiotic feature matrices;
- target-excluded model configurations and numerical hyperparameters;
- random-pair, genome-disjoint, and leave-one-antibiotic-out (LOAO) query definitions;
- nested 1%, 5%, and 10% target-support memberships;
- single-source and balanced multisource run plans;
- training, adaptation, aggregation, and metric code;
- aggregate model-selection and final-evaluation results.

The benchmark is stored at:

`data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz`

The main evaluation metadata is stored under:

`metadata/final_transfer/nested_loso_v1/`

The selected configurations are stored under:

`config/final/`

The aggregate results are stored under:

- `results/model_selection/`
- `results/evaluation/`

## Checking the Released Artifacts

After installing the Python dependencies, the repository can be checked with:

```bash
python scripts/verify_release.py
```

A more complete check, including model imports and CPU forward passes for the selected target-specific configurations, can be run with:

```bash
python scripts/verify_release.py --full
```

These commands check the released benchmark, feature-row mappings, feature matrices, evaluation splits, nested target-support sets, model configurations, aggregate results, documentation, and Python code.

## 2. Rerunning the Final Evaluation

The released feature matrices, model configurations, run plans, split memberships, and evaluation code support rerunning the final cross-species transfer experiments.

The evaluation covers:

- zero-target-label transfer;
- single-source training;
- balanced multisource training;
- random-pair limited-label adaptation;
- genome-disjoint limited-label adaptation;
- leave-one-antibiotic-out (LOAO) limited-label adaptation;
- target-only scratch baselines;
- the target-only full-support reference used for LOAO;
- source-shared and source-unseen antibiotic analysis.

### Step 1: Set the Project Root

From the repository root, run:

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"
```

### Step 2: Train the Source Models

Run:

```bash
python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py
```

`train_zero_target.py` trains the source models used for zero-target-label transfer.

For each target species, it uses:

- two single-source regimes;
- one balanced multisource regime;
- three evaluation seeds.

The resulting source checkpoints are required for the limited-label adaptation stages.

### Step 3: Random-Pair Adaptation

Run:

```bash
python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py
```

Random-pair evaluation uses one of five observation folds as the query set. The remaining non-query observations form the target-support pool.

The same genome may occur in both query and support data, but the same observation cannot occur in both.

### Step 4: Genome-Disjoint Adaptation

Run:

```bash
python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py
```

Genome-disjoint evaluation keeps all observations from the same genome in one fold.

Genomes with identical canonical 8-mer profiles are also kept together.

Query genomes are therefore absent from the target-support set.

### Step 5: Leave-One-Antibiotic-Out Adaptation

Run:

```bash
python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

In leave-one-antibiotic-out (LOAO) evaluation, all observations for one target antibiotic form the query set.

All remaining target antibiotics form the support pool.

The query antibiotic therefore has no target-species MIC labels in the support data.

### Target-Support Budgets

Limited-label transfer uses nested:

- 1% target support;
- 5% target support;
- 10% target support.

The support sets are constructed from non-query target observations.

Target-query labels are used only for final evaluation and are not used for adaptation-epoch selection.

### Target-Only Comparisons

Each adapted source-trained model is compared with a **target-only scratch baseline** using the same:

- architecture;
- target support;
- query set;
- seed;
- inner validation split.

The target-only scratch baseline starts from random initialization and uses support-fitted scalers.

LOAO evaluation also includes a **target-only full-support reference** trained on all non-query target observations.

## 3. Reconstruction from Raw Public Resources

Reconstructing the complete study from the original public resources is not self-contained.

The complete raw BV-BRC snapshot and genome-assembly collection are not stored in this repository.

Raw reconstruction requires:

- BV-BRC quantitative antimicrobial susceptibility records;
- BV-BRC genome metadata;
- genome assemblies;
- Kleborate;
- AMRFinderPlus;
- external molecular resources;
- regeneration of historical intermediate assets that are not distributed.

The BV-BRC snapshot used for the released benchmark was retrieved on **22 July 2026**.

Because BV-BRC is a live public resource, records downloaded at a later date may differ from the data available on that date.

For this reason, the released benchmark, feature matrices, split definitions, model configurations, and aggregate results should be used when reproducing the reported final evaluation.

## Public Audit Trail

The numbered scripts preserve the main public audit trail for:

- BV-BRC record acquisition;
- quantitative MIC filtering and harmonization;
- genome quality control;
- repeated-record reconciliation;
- antibiotic eligibility;
- genome acquisition;
- sequence-based species verification;
- AMRFinderPlus annotation;
- feature preparation;
- target-excluded nested leave-one-species-out (LOSO) model selection;
- final benchmark construction;
- evaluation split construction.

However, these numbered scripts are not one uninterrupted raw-data-to-results command chain.

Some historical scripts depend on intermediate manifests or candidate feature assets whose generating steps are not included in the public release.

For the script-to-stage mapping, see:

`docs/execution_map.md`

## Molecular Feature Provenance

The final antibiotic feature matrices are stored under:

`features/drug/`

Additional metadata records how these molecular representations were produced.

The relevant files are:

- `metadata/drug_representation/chemberta_checkpoint_spec.tsv`
- `metadata/drug_representation/rdkit_descriptor_columns_v1.tsv`
- `metadata/drug_representation/drug_feature_generation_protocol_v1.tsv`

These files record:

- the ChemBERTa model and revision used in the study;
- the ordered 27-descriptor RDKit schema;
- the molecular-feature generation protocol.

## External Software

Raw-data reconstruction requires separately installed tools.

### Kleborate

Kleborate is used for sequence-based species verification.

Project documentation:

https://github.com/klebgenomics/Kleborate

### AMRFinderPlus

AMRFinderPlus is used for antimicrobial resistance determinant annotation.

Project documentation:

https://github.com/ncbi/amr

Executable paths can be supplied with:

```bash
export KLEBORATE_EXECUTABLE=/path/to/kleborate
export AMRFINDER_EXECUTABLE=/path/to/amrfinder
```

If these variables are not set, the relevant scripts look for `kleborate` and `amrfinder` on `PATH`.

## Path Portability

The final evaluation scripts use repository-relative paths.

If the project root needs to be supplied explicitly, set:

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"
```

Some historical model-selection scripts retain older machine-specific fallback paths.

When running those historical scripts, setting `MIC_TRANSFER_PROJECT` avoids relying on those fallback paths.

These historical paths are retained so that the released model-selection scripts remain consistent with the study audit trail.

Gzip-compressed TSV files are read directly by pandas and do not need to be extracted first.

## Computational Environment

The software and hardware environment used for the final experiments is recorded in:

`docs/computational_environment.md`

The released code supports Python 3.11 or newer.

The portable dependency file uses:

`torch==2.10.0`

The final experiments used:

`torch==2.10.0+cu130`

Users on other systems should install a PyTorch build that is compatible with their operating system, hardware, and CUDA installation.

## Related Documentation

For additional details, see:

- `README.md`
- `data/README.md`
- `docs/benchmark_schema.md`
- `docs/model_selection.md`
- `docs/public_pipeline.md`
- `docs/execution_map.md`
- `docs/computational_environment.md`
