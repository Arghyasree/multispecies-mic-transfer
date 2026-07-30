# Public reproducibility pipeline

This repository contains the publication-relevant stages of the final workflow.

## Benchmark construction

Numbered scripts 01–63 cover BV-BRC laboratory-method MIC retrieval and
profiling, quantitative MIC parsing, scalar monotherapy filtering, repeated
genome–antibiotic reconciliation, genome metadata and assembly-quality
screening, molecular-identity eligibility, sequence-based species verification,
and final taxonomy-verified cohort construction.

## Genome representations

Scripts 137 and 140–147 cover AMRFinderPlus annotation, target-excluded AMR
vocabulary construction, common-AMR matrices, and frozen feature-row alignment.

The three selected final genome matrices are distributed directly under
`features/genome_representation/`. The historical k-mer generator is not
included because it also contained compatibility-only checks unrelated to the
final paper.

## Target-blind model selection

Scripts 150–173 preserve the staged selection of k-mer length, genome views,
antibiotic views, shared numerical hyperparameters, within-entity fusion, and
cross-modal genome–antibiotic architecture. Outer-target MIC labels were
excluded from these selections.

## Final transfer definitions

Scripts 174–175 preserve final experiment preregistration and the frozen
pair-level random, genome-disjoint, and leave-one-antibiotic-out query/support
definitions.

The cleaned final execution entry points are `train_zero_target.py`,
`adapt_random_pair.py`, `adapt_genome_disjoint.py`, and
`adapt_antibiotic_held_out.py`, together with their aggregation scripts.

## External software

Kleborate and AMRFinderPlus must be installed separately. Supply their
executables through:

```bash
export KLEBORATE_EXECUTABLE=/path/to/kleborate
export AMRFINDER_EXECUTABLE=/path/to/amrfinder
```

When these variables are unset, the scripts search for `kleborate` and
`amrfinder` on `PATH`.
