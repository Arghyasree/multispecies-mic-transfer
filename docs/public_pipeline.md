# Public reproducibility pipeline

The public artifact follows this sequence:

1. dataset curation and benchmark construction;
2. three-species quantitative MIC benchmark;
3. target-excluded nested leave-one-species-out model selection;
4. frozen pair-level random, genome-disjoint, and leave-one-antibiotic-out cohorts;
5. zero-target-label single-source and multi-source transfer;
6. 1%, 5%, and 10% limited-label target adaptation;
7. same-support target-only from-scratch controls;
8. source-MIC-seen and source-MIC-unseen antibiotic analysis;
9. manuscript-ready aggregate tables.

See [`execution_map.md`](execution_map.md) for the exact script-to-stage mapping.

## External resources

Raw BV-BRC records and assemblies are not redistributed. Kleborate and
AMRFinderPlus must be installed separately for raw-data reconstruction. Their
executables can be supplied as:

```bash
export KLEBORATE_EXECUTABLE=/path/to/kleborate
export AMRFINDER_EXECUTABLE=/path/to/amrfinder
```

When unset, the relevant scripts search for `kleborate` and `amrfinder` on
`PATH`.

## Final evaluation entry points

The frozen final evaluation uses:

- `train_zero_target.py` and `aggregate_zero_target.py`;
- `adapt_random_pair.py` and `aggregate_random_pair.py`;
- `adapt_genome_disjoint.py` and `aggregate_genome_disjoint.py`;
- `adapt_antibiotic_held_out.py` and `aggregate_antibiotic_held_out.py`.

Run `python scripts/verify_release.py` before launching expensive experiments.
