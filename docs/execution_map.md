# Execution map

The repository separates the **conceptual study workflow** from the physical
file layout. Numbered scripts preserve the chronological research pipeline;
named scripts provide the final paper-facing execution entry points.

| Stage | Scripts | Main inputs | Main outputs |
|---|---|---|---|
| BV-BRC acquisition and profiling | `01`–`08` | BV-BRC `genome_amr`, genome metadata | laboratory-method records, QC manifests |
| Quantitative MIC curation | `13`–`28` | laboratory MIC records | reconciled scalar monotherapy MIC cohort |
| Molecular eligibility and genome acquisition | `45`–`52` | curated MIC cohort, structure registry | eligible antibiotic panel, genome/assembly manifests |
| Final taxonomy-verified benchmark | `63` | curated MIC records, species verification | final three-species modelling cohort |
| AMR feature construction | `137`, `140`, `144`–`147` | assemblies, AMRFinderPlus | target-excluded common-AMR vocabularies and matrices |
| Target-blind model selection | `150`–`173` | development-species observations and features | ranked candidates and one frozen configuration per target |
| Final protocol and split freeze | `174`–`175` | final cohort and frozen configurations | query/support memberships and leakage audits |
| Zero-target-label transfer | `train_zero_target.py`, `aggregate_zero_target.py` | final matrices, source run plan | source checkpoints, target predictions, aggregate tables |
| Pair-level random adaptation | `adapt_random_pair.py`, `aggregate_random_pair.py` | source checkpoints, frozen support/query sets | adapted and scratch-baseline results |
| Genome-disjoint adaptation | `adapt_genome_disjoint.py`, `aggregate_genome_disjoint.py` | source checkpoints, genome-disjoint sets | adapted and scratch-baseline results |
| Leave-one-antibiotic-out adaptation | `adapt_antibiotic_held_out.py`, `aggregate_antibiotic_held_out.py` | source checkpoints, antibiotic-held-out sets | per-antibiotic and aggregate results |

## Reproducibility boundary

The repository supports three distinct levels of reproduction:

1. **Released-artifact verification is self-contained.** The benchmark index,
   frozen split memberships, selected genome and antibiotic matrices, frozen
   configurations, and aggregate tables can be checked with
   `python scripts/verify_release.py`.

2. **Frozen final-evaluation training is reproducible from the released
   matrices.** `train_zero_target.py` generates the required source
   checkpoints, after which the three adaptation pipelines can be run.

3. **Raw-data reconstruction is not self-contained.** Rebuilding the benchmark
   from the beginning requires fresh BV-BRC records and assemblies, Kleborate,
   AMRFinderPlus, and the external molecular resources documented elsewhere.
   Raw downloads and assemblies are not redistributed.

The released model-selection scripts and aggregate rankings preserve the
target-excluded selection audit trail. However, discarded candidate k-mer
matrices and the historical candidate k-mer generator are not distributed.
Consequently, reproducing every model-selection candidate from raw assemblies
requires regenerating those candidate matrices according to the documented
representation specification.

## Recommended order for the frozen final evaluation

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"
python scripts/verify_release.py
python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py
python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py
python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py
python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

The adaptation stages depend on source checkpoints produced by
`train_zero_target.py`. Each execution script supports filters such as target,
source regime, seed, budget, and `--max-new-runs`; run `python <script> --help`
for the exact options.
