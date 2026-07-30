# Data access

Quantitative MIC records and genome metadata were obtained from BV-BRC, and the
corresponding genome assemblies were downloaded for the selected records. The
repository does not redistribute those large source files.

The frozen public split definitions identify observations, target species,
genomes, antibiotics, query memberships, and nested support memberships. The
three genome feature matrices used by the final frozen configurations are
included under `features/genome_representation/`. Reconstruction from raw
assemblies still requires the benchmark-curation and genome-feature generation
procedure described in the manuscript and public pipeline.

The small final molecular feature matrices used by the selected models are
included under `features/drug/` together with their row registry and the
authoritative antibiotic structure registry.
