# Reproducibility scope

The release supports two distinct levels of reproducibility.

## 1. Frozen final evaluation — self-contained

The repository includes:

- the 168,363-observation final benchmark index;
- frozen query and nested support memberships;
- the three selected genome feature matrices;
- all final molecular feature matrices;
- target-excluded configurations and numerical hyperparameters;
- single-source and multi-source run plans;
- final training, adaptation, aggregation, and metric code;
- aggregate model-selection and held-out evaluation tables.

After installing the dependencies, run:

```bash
python scripts/verify_release.py
python scripts/verify_release.py --full
```

The first command validates the benchmark, matrices, row mappings, split
leakage, nested support sets, configurations, results, and Python syntax. The
second additionally imports the frozen model stack and performs one CPU forward
pass for each outer-target configuration.

## 2. Reconstruction from raw public resources — documented but not bundled

Raw BV-BRC downloads, genome assemblies, discarded candidate feature matrices,
source checkpoints, per-run predictions, and training histories are not stored
in Git. The numbered scripts document the benchmark-curation, species-checking,
AMR-feature, model-selection, and split-freezing stages. Reconstructing those
assets requires fresh BV-BRC downloads plus separately installed Kleborate and
AMRFinderPlus.

Because BV-BRC is a live resource, a fresh download may not be byte-identical to
the 22 July 2026 snapshot. The released observation index, feature matrices,
splits, configurations, and aggregate tables are therefore the authoritative
paper artifact.

All public paths are repository-relative. Set `MIC_TRANSFER_PROJECT` to relocate
the project root. Gzip-compressed TSV files are read directly by pandas.
