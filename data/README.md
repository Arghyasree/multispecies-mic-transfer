# Data Access

The quantitative MIC records and genome metadata used in this study were obtained from [BV-BRC](https://www.bv-brc.org/). Genome assemblies were also downloaded for MIC-linked candidate genomes.

The complete raw BV-BRC snapshot and the full genome-assembly collection are not redistributed in this repository.

## Released Benchmark

The complete derived benchmark is available as:

[`benchmark/final_quantitative_mic_benchmark_v1.tsv.gz`](benchmark/final_quantitative_mic_benchmark_v1.tsv.gz)

The benchmark contains:

- **168,363** unique genome–antibiotic observations;
- **21,394** quality-controlled and sequence-verified genomes;
- three bacterial species: *Escherichia coli* (Ec), *Klebsiella pneumoniae* (Kp), and *Salmonella enterica* (Se);
- **19** antibiotics across the complete cross-species antibiotic vocabulary.

Each row represents one unique species–genome–antibiotic observation.

The table includes:

- the quantitative MIC regression target;
- the original exact or censored MIC information;
- the final MIC interval after resolving repeated measurements;
- provenance for the contributing BV-BRC source records;
- genome and antibiotic feature-row indices;
- random-pair fold assignments;
- genome-disjoint fold assignments.

Definitions of all 62 columns are provided in:

[`../docs/benchmark_schema.md`](../docs/benchmark_schema.md)


## Evaluation Splits and Target-Support Sets

The released metadata contains the observation index and the split definitions used in the final evaluation.

These files identify:

- target species;
- genomes;
- antibiotics;
- query-set membership;
- random-pair folds;
- genome-disjoint folds;
- leave-one-antibiotic-out (LOAO) queries;
- nested 1%, 5%, and 10% target-support sets.

The 1%, 5%, and 10% target-support sets are nested. Target-query labels are not used during adaptation-epoch selection.

The final observation and feature index is located at:

[`../metadata/final_transfer/nested_loso_v1/splits_v1/final_transfer_observation_feature_index_v1.tsv.gz`](../metadata/final_transfer/nested_loso_v1/splits_v1/final_transfer_observation_feature_index_v1.tsv.gz)

## Genome Representations

The genome feature matrices used by the selected target-specific configurations are included under:

[`../features/genome_representation/`](../features/genome_representation/)

The selected genome representations use canonical k-mers, common antimicrobial resistance (AMR) determinants, or their selected combination, depending on the held-out target species.

These representations were selected through target-excluded nested leave-one-species-out (LOSO) model selection.

## Antibiotic Representations

The molecular antibiotic representations used in the selected models are included under:

[`../features/drug/`](../features/drug/)

The released files include the representations used in the study, together with the antibiotic row registry and antibiotic-structure registry.

The candidate antibiotic representations considered during model selection include:

- RDKit descriptors;
- Morgan fingerprints;
- ChemBERTa embeddings;
- selected multi-view combinations.

## Raw-Data Reconstruction

Reconstructing the benchmark from the original public resources is not self-contained because the raw BV-BRC snapshot and genome assemblies are not stored in this repository.

Raw reconstruction requires additional resources and software, including:

- BV-BRC records and genome assemblies;
- Kleborate for sequence-based species verification;
- AMRFinderPlus for AMR determinant annotation;
- the benchmark-curation procedures described in the manuscript;
- the feature-generation procedures described in the repository documentation.

See:

- [`../docs/reproducibility.md`](../docs/reproducibility.md)
- [`../docs/public_pipeline.md`](../docs/public_pipeline.md)

The released benchmark, feature matrices, split definitions, configurations, and aggregate results are the data artifacts used for the reported experiments.
