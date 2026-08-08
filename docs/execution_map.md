# Execution Map

This document shows how the main stages of the study correspond to the scripts and files in this repository.

The numbered scripts preserve the study's development record. The script numbers are not continuous because some historical or internal steps are not included in the public repository.

The named training and adaptation scripts are the main entry points for rerunning the final evaluation.

Leave-one-species-out (LOSO) refers to the target-excluded configuration-selection design used in the study.

## Study Stages and Scripts

| Stage | Released scripts | Main inputs | Main outputs |
| --- | --- | --- | --- |
| BV-BRC laboratory-record acquisition and profiling | `01`, `02`, `03` | BV-BRC `genome_amr` records | laboratory-method records, quantitative MIC profiles, and provisional species–antibiotic coverage summaries |
| Genome metadata and assembly quality control | `06`, `07`, `08` | MIC-linked candidate genomes and BV-BRC genome metadata | taxonomy checks, assembly-quality summaries, and retained-genome manifests |
| Quantitative MIC, source, mapping, and identity curation | `13`, `14`, `15`, `16`, `20`, `22`, `25`, `26` | quality-controlled laboratory antimicrobial resistance (AMR) records | scalar single-antibiotic MIC records, repeated-record checks, and unique quantitative MIC observations after resolving repeated measurements |
| Coverage-based antibiotic eligibility | `28` | unique species–genome–antibiotic observations after resolving repeated measurements | species–antibiotic coverage definitions |
| Molecular eligibility and genome acquisition | `45`, `46` | coverage-eligible MIC observations and antibiotic structure registry | modelling cohort and genome-acquisition manifest |
| Sequence-based taxonomy preparation and verification | `49`, `50`, `51`, `52` | candidate genome assemblies and reference resources | sequence-based species assignments and retained-genome registries |
| Taxonomy-verified modelling precursor | `63` | molecularly eligible observations and retained-genome registries | taxonomy-filtered modelling cohorts used to construct the final benchmark |
| AMRFinderPlus annotation | `137` | retained genome assemblies | AMR determinant annotations for each genome |
| Nested LOSO configuration splits and feature indices | `140` | pairwise development-species observations, genome groups, k-mer rows, and antibiotic rows | development folds, feature indices, task definitions, balance summaries, and leakage checks |
| Common-AMR vocabulary and matrix construction | `144`, `145`, `146`, `147` | AMRFinderPlus annotations and development-species definitions | target-excluded common-AMR vocabularies and binary feature matrices |
| Target-excluded nested LOSO model selection | `150`–`167`, `170`–`173` | development-species observations and candidate representations | ranked representation and architecture candidates and one selected configuration for each held-out target |
| Final three-species benchmark and evaluation splits | `174`, `175` | taxonomy-verified modelling precursor, selected configurations, and feature-row registries | final Ec/Kp/Se observation index, query sets, nested target-support sets, and leakage checks |
| Zero-target-label source training and evaluation | `train_zero_target.py`, `aggregate_zero_target.py` | final feature matrices, selected configurations, and source run plan | source checkpoints, target predictions, and zero-target-label aggregate results |
| Random-pair adaptation | `adapt_random_pair.py`, `aggregate_random_pair.py` | source checkpoints and random-pair support/query sets | adapted-model and target-only scratch results |
| Genome-disjoint adaptation | `adapt_genome_disjoint.py`, `aggregate_genome_disjoint.py` | source checkpoints and genome-disjoint support/query sets | unseen-genome adapted-model and target-only scratch results |
| Leave-one-antibiotic-out (LOAO) adaptation | `adapt_antibiotic_held_out.py`, `aggregate_antibiotic_held_out.py` | source checkpoints and LOAO support/query sets | per-antibiotic results, source-shared/source-unseen results, target-only scratch results, and target-only full-support reference results |

## How the Study Fits Together

The public workflow can be read in four broad parts.

### 1. Benchmark Construction

Scripts `01` through `63` cover the main steps used to move from public BV-BRC antimicrobial susceptibility records to the taxonomy-verified modelling cohort.

These stages include:

- quantitative MIC filtering and harmonization;
- genome assembly quality control;
- resolution of repeated measurements;
- species–antibiotic coverage requirements;
- antibiotic molecular-structure eligibility;
- genome acquisition;
- sequence-based species verification.

The final released benchmark contains **168,363 unique genome–antibiotic observations** from **21,394 quality-controlled and sequence-verified genomes** across Ec, Kp, and Se.

### 2. Feature Construction and Target-Excluded Model Selection

Scripts `137` through `173` cover AMR annotation, feature preparation, and **target-excluded nested leave-one-species-out (LOSO) model selection**.

For each held-out target species, MIC labels from that target are not used to choose the model configuration.

Instead, representation, architecture, and hyperparameter choices are evaluated by bidirectional transfer between the other two development species.

This produces one selected configuration for each held-out target.

### 3. Final Benchmark and Evaluation Splits

Scripts `174` and `175` define the final three-species benchmark used by the paper and construct the evaluation memberships.

These include:

- random-pair query folds;
- genome-disjoint query folds;
- leave-one-antibiotic-out (LOAO) query sets;
- nested 1%, 5%, and 10% target-support sets.

Target-query labels are used only for final evaluation.

### 4. Cross-Species Transfer and Adaptation

The named scripts run the paper-facing cross-species transfer experiments.

For each target species, the selected configuration is trained under:

- two single-source regimes;
- one balanced multisource regime.

The evaluation then covers:

- zero-target-label transfer;
- random-pair limited-label adaptation;
- genome-disjoint limited-label adaptation;
- LOAO limited-label adaptation;
- target-only scratch baselines;
- the target-only full-support reference used in LOAO evaluation;
- source-shared and source-unseen antibiotic analysis.

## Reproducibility Scope

The repository supports three levels of reproducibility.

### 1. Released-Artifact Verification

The released benchmark index, feature matrices, split memberships, model configurations, and aggregate results can be checked with:

```bash
python scripts/verify_release.py
```

A more complete check, including model imports and CPU forward passes, can be run with:

```bash
python scripts/verify_release.py --full
```

### 2. Final-Evaluation Training

The final evaluation can be rerun using the released feature matrices, model configurations, run plans, and split definitions.

`train_zero_target.py` creates the source checkpoints needed by the adaptation stages.

### 3. Reconstruction from Raw Public Resources

Reconstruction from the original public resources is documented but is not self-contained.

It requires resources that are not stored in the repository, including:

- current BV-BRC records and genome assemblies;
- Kleborate;
- AMRFinderPlus;
- external molecular resources;
- historical intermediate files that are not distributed.

The numbered scripts preserve the released workflow record, but they should not be interpreted as one uninterrupted raw-data-to-results command chain.

Some historical scripts use intermediate manifests or candidate feature files whose generating steps are not part of the public release.

## Historical Metadata Fields

Scripts `174` and `175` retain common-six fields from an earlier preregistration stage.

These fields are retained as historical metadata. They are not part of the final evaluation panel.

The paper-facing evaluation reports:

- the complete target antibiotic panel;
- source-shared antibiotics;
- source-unseen antibiotics.

The six antibiotics shared across all three species are used as an additional matched comparison panel.

## Recommended Order for the Final Evaluation

Set the repository root:

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"
```

Run the release check:

```bash
python scripts/verify_release.py --full
```

Run zero-target-label source training:

```bash
python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py
```

Run random-pair adaptation:

```bash
python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py
```

Run genome-disjoint adaptation:

```bash
python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py
```

Run leave-one-antibiotic-out adaptation:

```bash
python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

The three adaptation stages require the source checkpoints produced by `train_zero_target.py`.

`train_zero_target.py` performs aggregation automatically after all selected source runs are complete. Running `aggregate_zero_target.py` afterwards provides a separate completeness check of the result files.

The execution scripts also provide filters for items such as target species, source regime, seed, target-support budget, query, and `--max-new-runs`, where applicable.

Run:

```bash
python <script> --help
```

to see the exact options for a particular script.

## Related Documentation

For more detail, see:

- `README.md`
- `docs/model_selection.md`
- `docs/public_pipeline.md`
- `docs/reproducibility.md`
- `docs/benchmark_schema.md`
