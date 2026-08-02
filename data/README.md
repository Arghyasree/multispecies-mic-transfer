# Data Access

Quantitative MIC records and genome metadata were obtained from
[BV-BRC](https://www.bv-brc.org/), and the corresponding genome assemblies
were downloaded for MIC-linked candidate genomes. The repository does not
redistribute the complete raw BV-BRC snapshot or genome-assembly collection.

## Released Benchmark

The complete derived benchmark is available as a gzip-compressed TSV:

[`benchmark/final_quantitative_mic_benchmark_v1.tsv.gz`](benchmark/final_quantitative_mic_benchmark_v1.tsv.gz)

It contains 168,363 unique species–genome–antibiotic observations from 21,394
quality-controlled and sequence-verified genomes across 19 antibiotics. The
table includes the regression target, original censoring information,
reconciled MIC interval, source-record provenance, feature-row indices, and
random-pair and genome-disjoint fold assignments.

Definitions of all 62 columns are provided in
[`docs/benchmark_schema.md`](../docs/benchmark_schema.md). The accompanying
per-file checksum is:

[`benchmark/final_quantitative_mic_benchmark_v1.tsv.gz.sha256`](benchmark/final_quantitative_mic_benchmark_v1.tsv.gz.sha256)

## Supporting Release Data

The released observation index and frozen split definitions identify the
benchmark observations, species, genomes, antibiotics, query memberships, and
nested support memberships. The three genome feature matrices used by the
frozen final configurations are included under
[`features/genome_representation/`](../features/genome_representation/).

Reconstruction from raw BV-BRC records and genome assemblies is not
self-contained. It requires the external data, software, benchmark-curation
steps, and genome-feature generation procedures described in the manuscript,
[`docs/reproducibility.md`](../docs/reproducibility.md), and
[`docs/public_pipeline.md`](../docs/public_pipeline.md).

The final molecular feature matrices used by the selected models are included
under [`features/drug/`](../features/drug/), together with their row registry
and the frozen antibiotic-structure registry used in this study.