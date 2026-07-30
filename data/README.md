# Data access

Quantitative MIC records and genome metadata were obtained from BV-BRC, and the
corresponding genome assemblies were downloaded for the selected records. The
repository does not redistribute those large source files.

The frozen public split definitions identify observations, target species,
genomes, antibiotics, query memberships, and nested support memberships. To
rerun training, reconstruct the curated observation table and genome feature
matrices following the manuscript methods, then preserve the row mappings
recorded by the released observation index and feature registries.

The small final molecular feature matrices used by the selected models are
included under `features/drug/` together with their row registry and the
authoritative antibiotic structure registry.
