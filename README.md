# Cross-Species Transferability of Multi-View Genome–Drug Models for Quantitative MIC Prediction

This repository contains the benchmark, frozen configurations, evaluation splits, code, and aggregate results for cross-species quantitative minimum inhibitory concentration (MIC) prediction in *Escherichia coli* (Ec), *Klebsiella pneumoniae* (Kp), and *Salmonella enterica* (Se).

The study evaluates whether genome–drug models trained on one or two source species transfer to a held-out target species with zero labelled target MIC observations, and how performance changes after adaptation with 1%, 5%, or 10% labelled target data.

K-mer and AMR features are genome views; RDKit, Morgan, and ChemBERTa features are drug views. All drugs considered in this study are antibiotics. Genome and drug constitute the two model modalities.

## Benchmark

The benchmark was curated from the public
[BV-BRC](https://www.bv-brc.org/) `genome_amr` collection and associated
genome assemblies retrieved on 22 July 2026.

| Species | Genomes | MIC observations | Exact | Censored | Antibiotics |
| --- | ---: | ---: | ---: | ---: | ---: |
| *E. coli* | 6,673 | 68,881 | 25,742 | 43,139 | 19 |
| *K. pneumoniae* | 5,602 | 50,299 | 13,582 | 36,717 | 17 |
| *S. enterica* | 9,119 | 49,183 | 20,644 | 28,539 | 8 |

The complete derived benchmark contains 168,363 observations from 21,394
genomes. It is released as a
[gzip-compressed TSV](data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz),
with definitions of all 62 fields provided in the
[benchmark schema](docs/benchmark_schema.md).

Main curation criteria:

* Laboratory-reported, scalar, single-antibiotic MIC measurements
* BV-BRC genome quality `Good`
* CheckM completeness ≥95% and contamination ≤5%
* ≤500 contigs and N50 ≥20 kb
* ≥500 genomes and ≥200 exact MIC observations per eligible species–antibiotic cell
* One reproducible, connected molecular structure per antibiotic
* Sequence-based species verification using [Kleborate](https://github.com/klebgenomics/Kleborate)

MIC values were harmonized to log₂ mg/L while retaining exact and censored
measurement metadata. Compatible repeated genome–antibiotic records were
reconciled, while conflicting records were excluded.

The complete raw BV-BRC snapshot and genome assemblies are not redistributed.
Detailed curation and reconstruction information is available in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Target-Excluded Model Selection

One species was held out as the outer target. Representations, architecture, and hyperparameters were selected using only bidirectional transfer between the other two development species.

| Held-out target | Development transfer | Shared-drug cohort used for selection | Final source regimes   |
| --------------- | -------------------- | ------------------------------------: | ---------------------- |
| Ec              | Kp ↔ Se              |                                     6 | Kp→Ec, Se→Ec, Kp+Se→Ec |
| Kp              | Ec ↔ Se              |                                     8 | Ec→Kp, Se→Kp, Ec+Se→Kp |
| Se              | Kp ↔ Ec              |                                    17 | Kp→Se, Ec→Se, Kp+Ec→Se |

The shared-drug cohorts were used only for configuration selection. Final evaluation used the complete eligible antibiotic panel of each target species.

### Candidates Compared

| Component                      | Candidate families                                                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Genome representation          | canonical 4–8-mer composition; common AMR determinants; input-level k-mer+AMR concatenation; projected and low-rank separate-encoder fusion |
| Antibiotic representation      | RDKit descriptors; Morgan fingerprints; ChemBERTa embeddings; input-level and separate-encoder multi-view combinations                      |
| Genome–antibiotic architecture | additive effects; projection–concatenation MLP; dual-tower interaction; GMU; low-rank bilinear interaction; drug-to-genome FiLM             |

### Frozen Configurations

| Held-out target | Genome representation                                                | Antibiotic representation                                      | Architecture                   |
| --------------- | -------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------ |
| Ec              | 4-mer + common AMR with separate encoders and rank-8 low-rank fusion | RDKit descriptors                                              | Drug-to-genome FiLM            |
| Kp              | Common AMR                                                           | ChemBERTa mean + Morgan + RDKit with input-level concatenation | Drug-to-genome FiLM            |
| Se              | Common AMR                                                           | ChemBERTa mean                                                 | Dual-tower interaction network |

Exact settings are provided in:

* [`config/final/outer_target_configurations.tsv`](config/final/outer_target_configurations.tsv)
* [`config/final/shared_hyperparameters.tsv`](config/final/shared_hyperparameters.tsv)
* [`results/model_selection/`](results/model_selection/)

## Final Evaluation

Each target species was evaluated using two single-source models and one multi-source model.

| Target | Source regimes         |
| ------ | ---------------------- |
| Ec     | Kp→Ec, Se→Ec, Kp+Se→Ec |
| Kp     | Ec→Kp, Se→Kp, Ec+Se→Kp |
| Se     | Kp→Se, Ec→Se, Kp+Ec→Se |

The evaluation includes:

* **Zero-target-label transfer:** source-only training without labelled target-species MIC observations
* **Limited-label target adaptation:** adaptation using nested 1%, 5%, and 10% target-support sets
* **Pair-level random evaluation:** five folds of genome–antibiotic observations
* **Genome-disjoint evaluation:** five folds in which query genomes are absent from target support
* **Leave-one-antibiotic-out evaluation:** the query antibiotic is absent from target support
* **Same-support target-only baseline:** the same architecture trained from random initialization using identical target support
* **Target-only all-other-antibiotics reference:** trained on all non-query target antibiotics for leave-one-antibiotic-out evaluation
* **Source-seen/source-unseen analysis:** based on whether an antibiotic appeared in source-species MIC supervision

“Zero-target-label” refers specifically to the absence of target-species MIC supervision. Antibiotic molecular structures remain available, and ChemBERTa uses external molecular pretraining.

## Metrics

The primary metric is per-antibiotic macro-RMSE on the log₂ MIC scale. Additional metrics include:

* Macro-MAE
* R²
* Pearson correlation
* Spearman correlation
* Within-one-dilution accuracy, defined as absolute error ≤1 log₂ dilution

| Evaluation                                   | Uncertainty reporting                                                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Zero-target-label full target panel          | Mean ± sample SD across three model seeds                                                       |
| Pair-level random and genome-disjoint        | Seeds averaged within each fold, followed by mean ± sample SD across five folds                 |
| Leave-one-antibiotic-out                     | Seeds averaged within each antibiotic, followed by mean ± sample SD across held-out antibiotics |

Paper-facing results are available in [`results/evaluation/`](results/evaluation/).

## Repository Layout

```text
config/final/             Frozen configurations and hyperparameters
features/drug/            Molecular features and antibiotic registry
features/genome_representation/
                          Selected genome features
metadata/                 Benchmark index, run plans, and split registries
results/model_selection/  Target-excluded selection rankings
results/evaluation/       Paper-facing aggregate results
scripts/                  Curation, training, adaptation, and aggregation
src/mic_transfer/         Models, preprocessing, and metrics
docs/                     Execution and reproducibility documentation
```

The final benchmark index is available at:

[`metadata/final_transfer/nested_loso_v1/splits_v1/final_transfer_observation_feature_index_v1.tsv.gz`](metadata/final_transfer/nested_loso_v1/splits_v1/final_transfer_observation_feature_index_v1.tsv.gz)

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/Arghyasree/multispecies-mic-transfer.git
cd multispecies-mic-transfer

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Verification

```bash
python scripts/verify_release.py
python scripts/verify_release.py --full
```

The portable dependency file pins `torch==2.10.0`; the final experiments used `torch==2.10.0+cu130`. See [`docs/computational_environment.md`](docs/computational_environment.md) for the recorded environment.

## Reproducing the Frozen Evaluation

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"

python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py

python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py

python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py

python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

The adaptation stages require the source checkpoints produced by `train_zero_target.py`. Use `--device cpu` if CUDA is unavailable and `--max-new-runs 1` for a small execution test. See [`docs/execution_map.md`](docs/execution_map.md) for stage-specific inputs and outputs.

## Reproducibility Scope

The released benchmark index, feature matrices, splits, configurations, code, and aggregate tables support verification and rerunning of the frozen final evaluation.

Full raw-data reconstruction additionally requires BV-BRC records and assemblies, Kleborate, AMRFinderPlus, and external molecular resources.

## License and Citation

Repository code and documentation are released under the [MIT License](LICENSE). Third-party data, software, models, and databases retain their original licenses.

A `CITATION.cff` file will be added after the manuscript metadata and persistent identifier are finalized. Until then, cite the corresponding manuscript and this repository.
