# Cross-Species Transferability of Multi-View Genome–Antibiotic Models for Quantitative MIC Prediction

This repository contains the frozen benchmark definitions, model configurations,
source/target run plans, evaluation code, molecular features, and aggregate results
for a three-species study of quantitative minimum inhibitory concentration (MIC)
prediction involving *Escherichia coli*, *Klebsiella pneumoniae*, and
*Salmonella enterica*.

The study asks whether a genome–antibiotic model trained on one or two bacterial
species can transfer to a held-out species with no labelled target-species MIC
observations, and how performance changes after limited-label target adaptation.

## Terminology

In this repository, a **view** is an alternative representation of the same
entity, whereas a **modality** identifies the two different model inputs:

- **Genome views:** canonical k-mer composition and common AMR determinants.
- **Antibiotic views:** RDKit descriptors, Morgan fingerprints, and frozen
  ChemBERTa embeddings.
- **Modalities:** the bacterial genome and the tested antibiotic.

## Data source

The benchmark was derived from the public [BV-BRC](https://www.bv-brc.org/) `genome_amr` collection and
associated BV-BRC genome records and assemblies. The frozen source snapshot was
retrieved on **22 July 2026** through the BV-BRC REST API using
`evidence = Laboratory Method`. Only laboratory-reported antimicrobial
susceptibility measurements linked to the selected bacterial genomes were
considered.

The original BV-BRC downloads and genome assemblies are not redistributed in
this repository. The released observation index contains the final benchmark
identifiers, harmonized quantitative targets, feature-row mappings, and frozen
split assignments.

## External resources and software

| Resource | Role in this study |
|---|---|
| [BV-BRC](https://www.bv-brc.org/) | Source of quantitative MIC records, genome metadata, and genome assemblies |
| [BV-BRC documentation](https://www.bv-brc.org/docs/) | Documentation of BV-BRC data types, fields, and provenance |
| [BV-BRC Data API documentation](https://www.bv-brc.org/docs/system_documentation/system_architecture.html#data-api) | Programmatic REST access used for data retrieval |
| [Kleborate](https://github.com/klebgenomics/Kleborate) and its [documentation](https://kleborate.readthedocs.io/) | Sequence-based species verification |
| [AMRFinderPlus](https://github.com/ncbi/amr) and its [documentation](https://github.com/ncbi/amr/wiki) | AMR determinant annotation |
| [PyTorch installation guide](https://pytorch.org/get-started/locally/) | Platform-specific CPU or CUDA installation |
| [RDKit documentation](https://www.rdkit.org/docs/) | Molecular descriptors and Morgan fingerprints |
| [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/) | Loading the frozen ChemBERTa molecular encoder |

Python package versions are pinned in
[`requirements.txt`](requirements.txt). The final compute environment is
summarized in
[`docs/computational_environment.md`](docs/computational_environment.md);
external executable, model-checkpoint, and database versions should also be
reported in the manuscript and release metadata.

## Benchmark construction

The benchmark was constructed using the following frozen curation procedure.

1. **BV-BRC acquisition and genome quality control.** Records were downloaded
   from the BV-BRC `genome_amr` collection on 22 July 2026 using
   `evidence = Laboratory Method`; computationally predicted phenotypes were
   therefore excluded. Genome metadata and assemblies were obtained for
   MIC-linked candidate genomes. Genomes were required to have BV-BRC genome
   quality `Good`, CheckM completeness ≥95%, CheckM contamination ≤5%,
   ≤500 contigs, and contig N50 ≥20,000 bp. Taxonomic metadata, lineage
   information, genome-name consistency, and genome mappings were also audited.

2. **Quantitative MIC and source filtering.** MIC values were read from the
   structured `measurement_value` field and censoring signs from
   `measurement_sign`. Records were retained only when they contained a
   positive scalar value, a unit equivalent to mg/L or μg/mL, and one of the
   supported signs: blank/`=`, `<`, `<=`, `>`, or `>=`. Slash-separated paired
   concentrations were excluded. Records identified through source and
   method audits as inhibition-zone diameters rather than MICs, invalid
   sentinel values, and unresolved source-specific anomalies were also
   excluded.

3. **Antibiotic normalization and monotherapy filtering.** Raw antibiotic
   labels and spelling variants were mapped to normalized identities.
   Thirteen combination-drug identities, including slash-separated
   combinations and two known space-separated combinations, were excluded
   because the model requires one defined antibiotic structure per
   observation. Nine records carrying an invalid non-drug label were also
   removed. This stage retained 83 normalized monotherapy identities before
   coverage and molecular filtering.

4. **Source, mapping, and repeated-record reconciliation.** Frozen source-policy
   and genome-mapping audits were applied before repeated observations were
   reconciled. Repeated records for the same
   species–genome–antibiotic combination were represented as MIC constraints.
   Compatible constraints were reduced through interval intersection, whereas
   combinations with an empty intersection were excluded as conflicts.
   From 310,048 filtered source records, this produced 285,797 unique
   genome–antibiotic observations; 21,348 compatible repeated combinations
   were collapsed and 817 conflicting combinations were excluded.

5. **Coverage-based antibiotic selection.** Coverage was calculated after
   repeated-record reconciliation. A species–antibiotic cell was eligible only
   when it contained at least 500 unique genomes and at least 200 exact (`=`)
   MIC observations. Antibiotics considered for the cross-species benchmark
   were additionally required to have an eligible cell in at least two
   species. This produced 21 coverage-eligible cross-species antibiotic
   candidates.

6. **Molecular eligibility and final antibiotic panels.** Coverage-eligible
   antibiotics were subjected to chemical-identity and structure
   adjudication. Each retained identity required one reproducible, connected
   single-compound structure with a resolved parent compound, SMILES, InChI,
   and InChIKey. Gentamicin and colistin were excluded because their database
   labels could not be represented by one defensible single-compound
   structure. The remaining 19 antibiotics formed the global benchmark
   vocabulary. Within each species, only antibiotics whose
   species–antibiotic cell satisfied the coverage rule were retained, yielding
   19 antibiotics for *E. coli*, 17 for *K. pneumoniae*, and 8 for
   *S. enterica*.

7. **Point-target construction with censoring retained.** Original MIC signs
   and thresholds were preserved in the released observation index. Exact and
   inclusive measurements (`=`, `<=`, and `>=`) were represented at the
   reported threshold. Strict `<` measurements were represented one twofold
   dilution below the threshold, and strict `>` measurements one twofold
   dilution above it. The resulting positive MIC value in mg/L was transformed
   to log₂ MIC.

8. **Sequence-based species verification.** Candidate *E. coli*,
   *K. pneumoniae*, and *S. enterica* assemblies were evaluated using the
   Kleborate `enterobacterales__species` module. Only strong sequence-based
   calls concordant with the intended species were retained. This excluded
   14 provisional *E. coli* genomes and 70 provisional
   *K. pneumoniae* genomes; no *S. enterica* genome was excluded.

## Final benchmark

| Species | Genomes | MIC observations | Exact | Censored | Antibiotics |
|---|---:|---:|---:|---:|---:|
| *E. coli* | 6,673 | 68,881 | 25,742 | 43,139 | 19 |
| *K. pneumoniae* | 5,602 | 50,299 | 13,582 | 36,717 | 17 |
| *S. enterica* | 9,119 | 49,183 | 20,644 | 28,539 | 8 |
| **Total** | **21,394** | **168,363** | **59,968** | **108,395** | — |

## Representations compared

Model selection was staged rather than a simultaneous full-factorial search.
For every outer target, candidate choices were compared without using MIC labels
from that target species.

### Genome representations

The genome screen compared:

1. **Canonical k-mer composition only.** Canonical reverse-complement-collapsed
   relative-frequency vectors were screened for k = 4, 5, 6, 7, and 8. The
   target-excluded screens selected 4-mer, 5-mer, and 6-mer representations for
   the outer *E. coli*, *K. pneumoniae*, and *S. enterica* loops,
   respectively.
2. **Common AMR view only.** Binary genome-level presence of transferable AMR
   determinants derived separately for each outer loop using only its two
   development species. A determinant had to occur in both development species,
   in at least five genomes per species, and have pooled prevalence no greater
   than 0.99. Copy multiplicity was ignored.
3. **Input-level k-mer + AMR concatenation.** Raw k-mer and AMR features were
   concatenated before one genome encoder.
4. **Projected latent concatenation.** K-mer and AMR views used separate
   encoders, followed by concatenation of their projected latent vectors.
5. **Factorized low-rank bilinear genome-view fusion.** Separate k-mer and AMR
   encoders were coupled by a low-rank bilinear interaction in addition to the
   projected latent base.

### Antibiotic representations

The antibiotic screen compared:

1. **RDKit descriptors:** a frozen 27-descriptor physicochemical vector.
2. **Morgan fingerprint:** radius 2, 2,048 bits, with chirality.
3. **ChemBERTa mean embedding:** a 384-dimensional frozen embedding obtained by
   mean-pooling attention-valid non-special tokens.
4. **ChemBERTa first-token embedding:** a 384-dimensional pooling ablation.
5. **Two-view input-level concatenation:** ChemBERTa mean + Morgan.
6. **Three-view input-level concatenation:** ChemBERTa mean + Morgan + RDKit.
7. **Projected latent multi-view fusion:** separate molecular-view encoders
   followed by latent concatenation for the two-view and three-view bundles.
8. **Factorized low-rank bilinear molecular-view fusion:** separate
   molecular-view encoders with a rank-16 bilinear interaction for the two-view
   and three-view bundles.

Continuous molecular views were scaled using training-partition statistics;
the Morgan view remained binary. The ChemBERTa encoder was frozen.

### Genome–antibiotic architectures

With the selected genome and antibiotic representations fixed, the cross-modal
screen compared:

- a separable additive genome–antibiotic effects baseline;
- a projection–concatenation multilayer perceptron;
- a dual-tower interaction network;
- a cross-modal gated multimodal unit (GMU);
- a factorized low-rank bilinear interaction model; and
- drug-to-genome feature-wise linear modulation (FiLM).

## Target-excluded nested leave-one-species-out selection

For each outer target species, all of its MIC outcome labels were excluded from
representation, architecture, and hyperparameter selection. Candidates were
ranked by bidirectional transfer between the other two development species using
all eligible observations from their pairwise shared-antibiotic cohort. The
selected configuration was then frozen, reinitialized, and trained under each
prescribed source regime before evaluation on the held-out target.

| Held-out outer target | Development transfer used for selection | Shared antibiotics used only for inner selection | Final transfer regimes |
|---|---|---:|---|
| *E. coli* | *K. pneumoniae* ↔ *S. enterica* | 6 | KP→EC, SE→EC, KP+SE→EC |
| *K. pneumoniae* | *E. coli* ↔ *S. enterica* | 8 | EC→KP, SE→KP, EC+SE→KP |
| *S. enterica* | *K. pneumoniae* ↔ *E. coli* | 17 | KP→SE, EC→SE, KP+EC→SE |

The pairwise shared cohorts above were used only for target-excluded model
selection. Final held-out evaluation used each target species' complete eligible
antibiotic panel.

## Frozen final configurations

| Held-out target | Genome representation | Antibiotic representation | Cross-modal architecture |
|---|---|---|---|
| *E. coli* | Separate selected 4-mer and common-AMR encoders with rank-8 factorized low-rank bilinear genome-view fusion | RDKit descriptors | Drug-to-genome FiLM |
| *K. pneumoniae* | Common-AMR binary view | Input-level concatenation of ChemBERTa mean, Morgan, and RDKit views | Drug-to-genome FiLM |
| *S. enterica* | Common-AMR binary view | ChemBERTa mean embedding | Dual-tower interaction network |

The exact target-excluded numerical settings are stored in
`config/final/shared_hyperparameters.tsv`; the complete frozen configurations are
stored in `config/final/outer_target_configurations.tsv`.

## Final evaluation

The final study includes:

- **zero-target-label transfer:** source-only training with no labelled MIC
  observation from the held-out target species;
- **single-source and multi-source limited-label adaptation:** full-model
  adaptation with nested 1%, 5%, and 10% target-support sets;
- **same-support target-only from-scratch baseline:** the same frozen
  outer-target architecture trained from random initialization using exactly the
  target labels available to the corresponding adapted model;
- **pair-level random split:** target observations are partitioned into five
  folds; support and query may contain different antibiotic records from the
  same genome;
- **genome-disjoint evaluation:** every observation from a genome group is kept
  in one of five folds, so query genomes do not occur in target support;
- **leave-one-antibiotic-out evaluation:** every target observation for one
  antibiotic is held out, while target support contains only the other
  antibiotics;
- **target-only all-other-antibiotics reference:** a target-only model trained on
  the complete non-query support pool for leave-one-antibiotic-out evaluation;
- **source MIC-supervision familiarity analysis:** held-out antibiotics are
  labelled source-seen or source-unseen according to whether they appeared in
  the source-species MIC training data.

“Zero-target-label” refers specifically to the absence of labelled target-species
MIC observations during source training. It does not imply that an antibiotic or
related chemical structure was absent from external molecular pretraining.

## Metrics and uncertainty

The primary metric is per-antibiotic macro-RMSE on the log₂ MIC scale. The
repository also reports macro-MAE, R², Pearson correlation, Spearman correlation,
and within-one-dilution accuracy (1-tier accuracy), defined as the proportion of
predictions within ±1 log₂ dilution of the harmonized evaluation target.

Uncertainty was summarized as follows:

- **Zero-target-label full-panel results:** mean ± sample standard deviation
  across three model seeds.
- **Pair-level random and genome-disjoint results:** the three seeds were first
  averaged within each target fold, followed by mean ± sample standard deviation
  across the five fold-level values.
- **Leave-one-antibiotic-out results:** the three seeds were first averaged
  within each held-out antibiotic, followed by mean ± sample standard deviation
  across held-out antibiotics.

## Repository contents

```text
config/                  Frozen target-excluded configurations and hyperparameters
data/                    Data-access and reconstruction notes
features/drug/           Released molecular feature matrices and row registry
features/genome_representation/
                         Three final selected genome feature matrices
metadata/                Final benchmark index, run plans, query/support memberships, and split registries
results/model_selection/ Aggregate target-excluded model-selection evidence
results/evaluation/      Aggregate held-out target results
scripts/                 Final training, adaptation, and aggregation entry points
src/mic_transfer/        Reusable model architectures, preprocessing, and metric code
```

The observation-level benchmark is released as:

```text
metadata/final_transfer/nested_loso_v1/splits_v1/
final_transfer_observation_feature_index_v1.tsv.gz
```

It contains the species, BV-BRC genome identifier, normalized antibiotic,
harmonized log₂ MIC target, exact/censored indicator, feature-row mappings,
duplicate-profile group, and frozen random-pair and genome-disjoint fold
assignments for all 168,363 final observations.

The nested support and query memberships used in the paper are released in the
same directory. The final molecular matrices and their row registry are included
under `features/drug/`.

## Installation

Use Python **3.11 or newer**. Clone the repository, create an isolated
environment, and install the single canonical pinned dependency manifest:

```bash
git clone https://github.com/Arghyasree/multispecies-mic-transfer.git
cd multispecies-mic-transfer
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) is the only Python dependency manifest
used by this release. It pins the portable `torch==2.10.0` package version.
The final experiments used the CUDA-specific build `torch==2.10.0+cu130`,
which is recorded in
[`docs/computational_environment.md`](docs/computational_environment.md).
For another CPU or CUDA platform, follow the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/) while retaining the
remaining pinned package versions.

## Quick verification

Before running training:

```bash
python scripts/verify_release.py
```

After installing the complete environment, also run the model smoke test:

```bash
python scripts/verify_release.py --full
```

Both commands are read-only with respect to the released benchmark and result
tables.

## Reproducibility scope

The included benchmark index, selected genome and antibiotic matrices, split
memberships, configurations, run plans, code, and aggregate tables are
sufficient to validate and rerun the **frozen final evaluation**. Raw BV-BRC
downloads, assemblies, discarded candidate matrices, checkpoints, and per-run
training histories are not distributed. Reconstructing the full raw-data and
model-selection pipeline requires the external resources described in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Reproducing the final evaluation

Set the project root and verify the release:

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"
python scripts/verify_release.py --full
```

Then run the final stages in order:

```bash
python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py
python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py
python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py
python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

The adaptation stages require the source checkpoints produced by
`train_zero_target.py`. On a machine without CUDA, use `--device cpu`; full runs
will be substantially slower. Use `--max-new-runs 1` and the target/seed filters
shown by `python <script> --help` for a small execution test.

The three selected genome matrices are included under
`features/genome_representation/`; final molecular matrices are included under
`features/drug/`. See [`docs/execution_map.md`](docs/execution_map.md) for the
stage-by-stage inputs and outputs.

## Results

Paper-facing aggregate results are available under `results/evaluation/`.
Detailed frozen aggregate tables are retained under
`results/tables/final_transfer/nested_loso_v1/`. Model-selection rankings are
available under `results/model_selection/`.

Figures are intentionally omitted until the manuscript figure set and captions
are finalized.

## License

The original software and documentation in this repository are released under
the MIT License; see `LICENSE`.

The MIT License does not relicense BV-BRC records or genome assemblies, the
ChemBERTa checkpoint, RDKit, Kleborate, AMRFinderPlus, or any other third-party
resource. Those materials remain subject to their respective licenses and terms
of use.

## Citation

A `CITATION.cff` file will be added when the manuscript title, author list,
venue, and persistent identifier are finalized. Until then, please cite the
corresponding manuscript and this repository URL.

## Public reproducibility pipeline

The study stages are mapped to the existing repository layout in
[`docs/execution_map.md`](docs/execution_map.md). The current physical folder
structure is retained because the frozen scripts and registries use those
repository-relative paths.
