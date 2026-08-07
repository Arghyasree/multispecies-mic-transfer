# Execution map

This document maps the released study stages to the physical repository layout.
Exact released script numbers are shown; gaps correspond to historical or
internal steps that are not included in the public repository. Named scripts
provide the final paper-facing execution entry points.

| Stage | Released scripts | Main inputs | Main outputs |
|---|---|---|---|
| BV-BRC laboratory-record acquisition and profiling | `01`, `02`, `03` | BV-BRC `genome_amr` records | laboratory-method download, quantitative profile, and provisional species–antibiotic support audit |
| Genome metadata and assembly quality control | `06`, `07`, `08` | MIC-linked candidate genomes and BV-BRC genome metadata | taxonomy and assembly-QC audits and retained-genome manifest |
| Quantitative MIC, source, mapping, and identity curation | `13`, `14`, `15`, `16`, `20`, `22`, `25`, `26` | quality-controlled laboratory AMR records | source-clean scalar monotherapy records, repeated-record audits, and reconciled quantitative MIC cohort |
| Coverage-based antibiotic eligibility | `28` | reconciled species–genome–antibiotic observations | species–antibiotic coverage manifest |
| Molecular eligibility and genome acquisition | `45`, `46` | coverage-eligible MIC cohort and structure registry | single-structure modelling cohort and genome-acquisition manifest |
| Sequence-based taxonomy preparation and verification | `49`, `50`, `51`, `52` | candidate assemblies and reference resources | sequence-species calls and adjudicated retained-genome registries |
| Taxonomy-verified modelling precursor | `63` | molecularly eligible observations and retained-genome registries | taxonomy-filtered modelling cohorts used by the later benchmark-freezing stage |
| AMRFinderPlus annotation | `137` | retained genome assemblies | per-genome AMR determinant annotations |
| Nested-LOSO configuration split and feature-index freeze | `140` | pairwise development observations, genome groups, k-mer rows, and antibiotic rows | development-species folds, feature index, task registry, balance tables, and leakage audit |
| Common-AMR vocabulary and matrix construction | `144`, `145`, `146`, `147` | AMRFinderPlus outputs and frozen development-species definitions | target-excluded common-AMR vocabularies and binary matrices |
| Target-excluded nested-LOSO model selection | `150`–`167`, `170`–`173` | development-species observations and candidate features | ranked representation and architecture candidates and one frozen configuration per target |
| Final protocol, three-species paper benchmark, and split freeze | `174`, `175` | taxonomy-verified precursor, frozen configurations, and feature-row registries | final *E. coli*/*K. pneumoniae*/*S. enterica* observation index, query sets, nested supports, and leakage audits |
| Zero-target-label source training and evaluation | `train_zero_target.py`, `aggregate_zero_target.py` | final matrices, frozen configurations, and source run plan | source checkpoints, target predictions, and zero-target-label aggregate tables |
| Random-pair adaptation | `adapt_random_pair.py`, `aggregate_random_pair.py` | source checkpoints and frozen random-pair support/query sets | adapted-model and target-only scratch results |
| Genome-disjoint adaptation | `adapt_genome_disjoint.py`, `aggregate_genome_disjoint.py` | source checkpoints and genome-disjoint support/query sets | unseen-genome adaptation and target-only results |
| Leave-one-antibiotic-out (LOAO) adaptation | `adapt_antibiotic_held_out.py`, `aggregate_antibiotic_held_out.py` | source checkpoints and antibiotic-held-out support/query sets | per-antibiotic, source-shared/source-unseen, adaptation, target-only scratch, and target-only full-support reference results |

## Reproducibility boundary

The repository supports three distinct levels of reproduction.

1. **Released-artifact verification is self-contained.** The benchmark index,
   frozen split memberships, selected genome and antibiotic matrices, frozen
   configurations, and aggregate tables can be checked with:

   ```bash
   python scripts/verify_release.py
   ```

2. **Frozen final-evaluation training is reproducible from the released
   matrices.** `train_zero_target.py` generates the source checkpoints required
   by the three adaptation pipelines.

3. **Raw-data reconstruction is documented but not self-contained.** It
   requires fresh BV-BRC records and assemblies, Kleborate, AMRFinderPlus,
   external molecular resources, and regeneration of historical intermediate
   assets that are not distributed.

The numbered scripts preserve the released audit trail, but they are not one
uninterrupted raw-to-results command chain. Some scripts consume frozen
historical manifests or candidate feature assets whose generating steps are
not part of the public release.

The released model-selection scripts and aggregate rankings preserve the
target-excluded selection evidence. Discarded candidate k-mer matrices and the
historical candidate k-mer generator are not distributed, so reproducing every
selection candidate from raw assemblies requires regenerating those matrices
according to the documented representation specification.

## Historical audit fields

Scripts `174` and `175` retain common-six fields from an earlier preregistration
stage. These fields are retained only as historical audit metadata. The final
paper-facing evaluation runners report the complete target panel,
source-shared antibiotics, and antibiotics unseen in source MIC supervision;
the common-six subset is not a final evaluation panel.

## Recommended order for the frozen final evaluation

```bash
export MIC_TRANSFER_PROJECT="$(pwd)"

python scripts/verify_release.py --full

python scripts/train_zero_target.py --device cuda
python scripts/aggregate_zero_target.py

python scripts/adapt_random_pair.py --device cuda
python scripts/aggregate_random_pair.py

python scripts/adapt_genome_disjoint.py --device cuda
python scripts/aggregate_genome_disjoint.py

python scripts/adapt_antibiotic_held_out.py --device cuda
python scripts/aggregate_antibiotic_held_out.py
```

`train_zero_target.py` aggregates automatically after all selected source runs
are complete. Running `aggregate_zero_target.py` afterwards provides an
independent manifest and completeness check.

The adaptation stages require source checkpoints produced by
`train_zero_target.py`. Each execution script supports target, source-regime,
seed, budget, query, and `--max-new-runs` filters where applicable. Run
`python <script> --help` for the exact options.
