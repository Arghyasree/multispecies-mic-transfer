# Reproducibility notes

The public repository preserves the frozen final configurations, run plans,
support/query memberships, aggregate tables, and reusable model code. It does
not include raw BV-BRC downloads, genome assemblies, large genome feature
matrices, source checkpoints, or per-run training histories.

All paths are repository-relative and may be relocated by setting
`MIC_TRANSFER_PROJECT` to the repository root. The split files are distributed
as gzip-compressed TSV files; pandas reads them directly when the `.tsv.gz`
paths in the public scripts are retained.

The public release SHA manifest is written to `SHA256SUMS.txt`. This manifest
covers the release candidate itself and is independent of laboratory-internal
freeze manifests whose source paths are not distributed.
