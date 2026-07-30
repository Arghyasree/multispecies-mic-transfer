# Data access

Quantitative MIC records and genome metadata were obtained from
[BV-BRC](https://www.bv-brc.org/), and the corresponding genome assemblies were downloaded for
MIC-linked candidate genomes. The repository does not redistribute these large
raw source files.

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
and the frozen antibiotic structure registry used in this study.
