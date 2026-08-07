# Public Reproducibility Pipeline

This document summarizes the public workflow of the study and points to the main released scripts and resources.

For the detailed script-to-stage mapping, see:

`docs/execution_map.md`

## Study Workflow

The public study workflow follows these main stages:

1. quantitative MIC data curation and benchmark construction;
2. construction of the three-species quantitative MIC benchmark;
3. target-excluded nested leave-one-species-out (LOSO) model selection;
4. construction of random-pair, genome-disjoint, and leave-one-antibiotic-out (LOAO) evaluation sets;
5. zero-target-label transfer using single-source and balanced multisource training;
6. limited-label transfer using nested 1%, 5%, and 10% target-support sets;
7. comparison with target-only scratch baselines;
8. target-only full-support reference evaluation for LOAO;
9. source-shared and source-unseen antibiotic analysis;
10. aggregation of the paper-facing results.

Target-query labels are used only for final evaluation.

## Benchmark Construction

The quantitative MIC benchmark was constructed from public BV-BRC antimicrobial susceptibility records and associated genome assemblies.

The final benchmark contains:

- **168,363** unique genome–antibiotic observations;
- **21,394** quality-controlled and sequence-verified genomes;
- *Escherichia coli* (Ec);
- *Klebsiella pneumoniae* (Kp);
- *Salmonella enterica* (Se);
- **19** antibiotics across the complete cross-species antibiotic vocabulary.

The released benchmark is:

`data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz`

Column definitions are provided in:

`docs/benchmark_schema.md`

## Target-Excluded Nested LOSO Model Selection

Model selection is performed separately for each held-out target species.

MIC outcome labels from the target species are not used to select its model configuration.

For each outer target, the other two species are used as development species:

| Held-out target | Development transfer | Shared-antibiotic cohort |
| --- | --- | ---: |
| Ec | Kp ↔ Se | 6 |
| Kp | Ec ↔ Se | 8 |
| Se | Ec ↔ Kp | 17 |

Candidate genome representations, antibiotic representations, architectures, and hyperparameters are compared through bidirectional transfer between the two development species.

The main model-selection metric is per-antibiotic macro-RMSE.

More details are provided in:

`docs/model_selection.md`

## Cross-Species Transfer

After model selection, one target-specific configuration is used for each target species.

For each target, we train:

- two single-source models;
- one balanced multisource model.

The final source regimes are:

| Target | Source regimes |
| --- | --- |
| Ec | Kp→Ec, Se→Ec, Kp+Se→Ec |
| Kp | Ec→Kp, Se→Kp, Ec+Se→Kp |
| Se | Kp→Se, Ec→Se, Kp+Ec→Se |

### Zero-Target-Label Transfer

In zero-target-label transfer, the model is trained using source-species observations only.

No MIC labels from the target species are used for training.

This corresponds to the zero-shot transfer setting used in the manuscript.

### Limited-Label Transfer

Limited-label transfer starts from the source-trained model and adapts it using nested:

- 1% target support;
- 5% target support;
- 10% target support.

The target-support sets contain only non-query observations.

Target-query labels are not used for adaptation-epoch selection.

## Evaluation Protocols

Three target evaluation protocols are used.

### Random-Pair Evaluation

Random-pair evaluation divides observations into five folds within each antibiotic.

A genome may occur in both query and support sets, but the same observation cannot occur in both.

### Genome-Disjoint Evaluation

Genome-disjoint evaluation keeps all observations from the same genome in one fold.

Genomes with identical canonical 8-mer profiles are also kept together.

Query genomes are absent from target support.

### Leave-One-Antibiotic-Out (LOAO) Evaluation

LOAO evaluation uses all observations for one target antibiotic as the query set.

The remaining target antibiotics form the support pool.

The query antibiotic therefore has no target-species MIC labels in the support data.

## Target-Only Comparisons

Limited-label transfer is compared with a **target-only scratch baseline**.

The target-only scratch baseline uses the same architecture, target support, query set, seed, and inner validation split as the source-pretrained model, but starts from random initialization.

LOAO evaluation also includes a **target-only full-support reference** trained on all non-query target observations.

## Source-Shared and Source-Unseen Antibiotics

Target antibiotics are also grouped according to their source MIC supervision.

- **Source-shared:** MIC supervision for the target antibiotic is available in at least one source species.
- **Source-unseen:** the target antibiotic is absent from source MIC supervision.

The six antibiotics shared across Ec, Kp, and Se form an additional matched comparison panel.

## External Data and Software

The complete raw BV-BRC snapshot and genome-assembly collection are not redistributed.

Raw-data reconstruction requires access to:

- BV-BRC antimicrobial susceptibility records;
- BV-BRC genome metadata;
- genome assemblies;
- Kleborate;
- AMRFinderPlus;
- external molecular resources.

BV-BRC resources are available from:

- https://www.bv-brc.org/
- https://www.bv-brc.org/docs/

The Python packages required by the released scripts are listed in:

`requirements.txt`

## Kleborate and AMRFinderPlus

Kleborate is used for sequence-based species verification.

Project documentation:

https://github.com/klebgenomics/Kleborate

AMRFinderPlus is used to annotate antimicrobial resistance determinants.

Project documentation:

https://github.com/ncbi/amr

These tools must be installed separately when reconstructing the corresponding stages from raw data.

Their executable paths can be supplied with:

```bash
export KLEBORATE_EXECUTABLE=/path/to/kleborate
export AMRFINDER_EXECUTABLE=/path/to/amrfinder
```

If these variables are not set, the relevant scripts look for `kleborate` and `amrfinder` on `PATH`.

## Molecular Feature Resources

The released antibiotic representations are stored under:

`features/drug/`

The repository also records information needed to identify how the molecular representations were generated.

The relevant metadata files are:

- `metadata/drug_representation/chemberta_checkpoint_spec.tsv`
- `metadata/drug_representation/rdkit_descriptor_columns_v1.tsv`
- `metadata/drug_representation/drug_feature_generation_protocol_v1.tsv`

These files record:

- the ChemBERTa model and revision used in the study;
- the ordered 27-descriptor RDKit schema;
- the molecular-feature generation protocol.

## Final Evaluation Entry Points

### Zero-Target-Label Transfer

```bash
python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py
```

### Random-Pair Adaptation

```bash
python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py
```

### Genome-Disjoint Adaptation

```bash
python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py
```

### Leave-One-Antibiotic-Out Adaptation

```bash
python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

The adaptation stages require the source checkpoints produced by `train_zero_target.py`.

For a small execution test, the supported scripts can be run with:

```bash
--max-new-runs 1
```

Run:

```bash
python <script> --help
```

to see the exact options supported by each script.

## Release Verification

Before running expensive experiments, the released benchmark, feature mappings, split definitions, configurations, results, and code can be checked with:

```bash
python scripts/verify_release.py
```

A more complete check is available with:

```bash
python scripts/verify_release.py --full
```

## Reproducibility Boundary

The released feature matrices, configurations, run plans, split definitions, and evaluation code support rerunning the final evaluation.

Reconstruction from the original raw public resources is not fully self-contained because some raw and historical intermediate assets are not stored in this repository.

The numbered scripts preserve the study's public audit trail, but they should not be interpreted as one uninterrupted raw-data-to-results command chain.

For more detail, see:

- `README.md`
- `docs/execution_map.md`
- `docs/model_selection.md`
- `docs/reproducibility.md`
- `docs/benchmark_schema.md`
