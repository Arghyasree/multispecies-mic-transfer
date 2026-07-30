# Cross-Species Transferability of Multi-View Genome–Antibiotic Models for Quantitative MIC Prediction

This repository contains the code, frozen configurations, split definitions,
and aggregate results for a three-species quantitative minimum inhibitory
concentration (MIC) transfer study involving *Escherichia coli*, *Klebsiella
pneumoniae*, and *Salmonella enterica*.

## Study scope

Each observation pairs a bacterial genome with an antibiotic and a harmonized
log2 MIC target. In this repository, **view** denotes an alternative
representation of the same entity (for example, k-mer and curated AMR genome
views), whereas **modality** denotes the genome and antibiotic inputs.

The final study uses nested leave-one-species-out model selection. For each
outer target species, its MIC outcome labels were excluded from representation,
architecture, and hyperparameter selection. Candidate configurations were
compared by bidirectional transfer between the two remaining development
species on their pairwise shared-antibiotic cohort. The selected configuration
was then frozen and trained under each prescribed single-source or multi-source
regime before held-out target evaluation.

## Benchmark

| Species | Genomes | MIC observations | Antibiotics |
|---|---:|---:|---:|
| *E. coli* | 6,673 | 68,881 | 19 |
| *K. pneumoniae* | 5,602 | 50,299 | 17 |
| *S. enterica* | 9,119 | 49,183 | 8 |

## Final outer-target configurations

| Held-out target | Genome representation | Antibiotic representation | Cross-modal architecture |
|---|---|---|---|
| *E. coli* | Separate encoders for selected 4-mer and common AMR views with low-rank bilinear fusion (rank 8) | RDKit descriptors | Drug-conditioned feature-wise linear modulation (FiLM) |
| *K. pneumoniae* | Single-view common AMR representation | Input-level concatenation of ChemBERTa mean embedding, Morgan fingerprint, and RDKit descriptors | Drug-conditioned FiLM |
| *S. enterica* | Single-view common AMR representation | ChemBERTa mean embedding | Dual-encoder interaction network |

Numerical settings are read from the frozen shared-hyperparameter registry in
`config/final/shared_hyperparameters.tsv`; no pilot numerical defaults are used.

## Evaluation

The final held-out target experiments comprise:

- zero-target-label cross-species transfer from each single source and from the
  two-source joint regime;
- limited-label target adaptation using nested 1%, 5%, and 10% target-support
  sets for both single-source and multi-source pretrained models;
- target-only from-scratch baselines using the same target-support observations
  and query sets;
- random genome–antibiotic-pair, genome-disjoint, and
  leave-one-antibiotic-out evaluation;
- a target-only all-other-antibiotics reference for leave-one-antibiotic-out
  evaluation;
- stratification by whether an antibiotic was seen or unseen in source MIC
  supervision.

Here, zero-target-label means that no labelled target-species MIC observations
were used for training or adaptation. It does not imply that an antibiotic
structure was unavailable to the molecular representation model.

## Primary metrics

The primary metric is per-antibiotic macro-RMSE on the log2 MIC scale. The
repository also reports macro-MAE, R2, Pearson correlation, Spearman
correlation, and within-one-dilution accuracy (1-tier accuracy), defined as the
proportion of predictions within ±1 log2 dilution of the harmonized evaluation
target.

## Repository layout

```text
config/                  Frozen model-selection and final configurations
data/                    Data-access and reconstruction notes
features/drug/           Small final molecular feature matrices
metadata/                Frozen run plans and release metadata
results/model_selection/ Aggregate development-selection evidence
results/evaluation/      Aggregate held-out target results
scripts/                 Final evaluation and aggregation entry points
src/mic_transfer/        Reusable model, scaling, and metric code
```

Large genome assemblies, genome feature matrices, source checkpoints, and
per-run training outputs are not stored in Git. See `data/README.md` and
`docs/reproducibility.md` for the expected asset layout and reconstruction
procedure.

## Figures

Figures are intentionally omitted from this release. Final publication figures will be added after manuscript figure selection and caption finalization.

## Reproducing the final evaluation

1. Create the environment described in `requirements.txt`.
2. Place the large genome feature matrices at the paths recorded in
   `config/final/outer_target_configurations.tsv`.
3. Use the compressed frozen split definitions already stored under
   `metadata/final_transfer/nested_loso_v1/splits_v1/`; pandas reads the
   `.tsv.gz` files directly.
4. Train zero-target-label source checkpoints with
   `scripts/train_zero_target.py`.
5. Run the three limited-label evaluation scripts and their aggregation scripts.
6. Use the aggregate TSV files in `results/model_selection/` and
   `results/evaluation/` when preparing manuscript tables and figures.

Figures are intentionally not included in this release. They will be added only
after the manuscript figure set and captions are finalized.

The internal filenames retained inside run plans are implementation identifiers;
the publication terminology used above is authoritative.
