# Public Quantitative MIC Benchmark

This document explains the released quantitative MIC benchmark and every column in the benchmark table.

## Benchmark File

The benchmark is stored at:

`data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz`

The compressed TSV contains one row for each unique species–genome–antibiotic observation.

| Species | Genomes | MIC observations | Antibiotics | Exact | Censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Escherichia coli* (Ec) | 6,673 | 68,881 | 19 | 25,742 | 43,139 |
| *Klebsiella pneumoniae* (Kp) | 5,602 | 50,299 | 17 | 13,582 | 36,717 |
| *Salmonella enterica* (Se) | 9,119 | 49,183 | 8 | 20,644 | 28,539 |
| **Total** | **21,394** | **168,363** | **19 unique** | **59,968** | **108,395** |

The benchmark was constructed from **179,385 source records** in the BV-BRC snapshot retrieved on **22 July 2026**.

Compatible repeated records for the same genome–antibiotic pair were resolved into one observation. The released benchmark contains **11,005 observations** built from more than one source record. At most three source records contribute to a single released observation.

## Reading the Benchmark

Use a TSV-aware reader. For example:

```python
import pandas as pd

benchmark = pd.read_csv(
    "data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz",
    sep="\t",
    dtype=str,
    na_filter=False,
    compression="gzip",
)
```

The machine-readable species codes are:

- `ec` for *Escherichia coli*;
- `kp` for *Klebsiella pneumoniae*;
- `se` for *Salmonella enterica*.

These lowercase codes are used throughout the released feature matrices, split definitions, and model configurations.

## Observation and Target Fields

These columns identify each benchmark observation and its quantitative MIC target.

| Column | Type | Description |
| --- | --- | --- |
| `final_transfer_observation_row` | integer | Zero-based row index in the final transfer benchmark. |
| `species_code` | string | Machine-readable species code: `ec`, `kp`, or `se`. |
| `provisional_species` | string | Harmonized species assignment used during benchmark construction. |
| `observation_id` | string | Stable identifier for the final quantitative MIC observation. |
| `genome_id` | string | BV-BRC genome identifier. |
| `normalized_antibiotic` | string | Harmonized antibiotic name used throughout the study. |
| `mic_target_log2_mg_per_l` | float | Regression target: base-2 logarithm of the point MIC in mg/L. |
| `is_exact_observation` | Boolean | `True` when the final MIC observation is exact. |
| `is_censored_observation` | Boolean | `True` when the observation is left- or right-censored. |

Every observation is either exact or censored, but never both.

Each species–genome–antibiotic combination appears only once in the released benchmark.

## MIC Constraint Fields

These columns describe how the original MIC measurement, or compatible repeated measurements, were converted into the final quantitative observation.

| Column | Type | Description |
| --- | --- | --- |
| `reconciliation_status` | string | Result of handling a single record or resolving repeated records. |
| `constraint_origin` | string | Source category of the final MIC constraint. |
| `duplicate_class` | string | Class describing the source-record multiplicity or duplication pattern. |
| `reduced_constraint_type` | string | Final MIC constraint type after resolving repeated measurements. |
| `reduced_sign` | string | Final relation sign: `=`, `<`, `<=`, `>`, or `>=`. |
| `reduced_mic_value` | float | MIC threshold associated with the final relation sign. |
| `intersection_lower` | optional float | Lower endpoint of the final MIC interval. Blank when there is no lower bound. |
| `intersection_lower_closed` | optional Boolean | Indicates whether the lower interval endpoint is inclusive. |
| `intersection_upper` | optional float | Upper endpoint of the final MIC interval. Blank when there is no upper bound. |
| `intersection_upper_closed` | optional Boolean | Indicates whether the upper interval endpoint is inclusive. |
| `intersection_notation` | string | Human-readable form of the final MIC interval. |
| `mic_target_point_mg_per_l` | float | Positive point MIC used as the regression target, in mg/L. |
| `mic_target_substitution_rule` | string | Rule used to convert the final constraint into a point MIC target. |
| `censoring_direction` | string | Censoring direction: none, left, or right. |
| `censoring_strictness` | string | Indicates whether the censoring relation is strict or inclusive. |
| `point_target_version` | string | Version identifier for the point-target construction rule. |
| `normalized_unit` | string | Common MIC unit. All released MIC values use `mg/L`. |

### Point Targets for Censored MIC Values

Exact measurements and inclusive bounds keep the reported threshold.

For strict censored measurements:

- `<c` is represented by `c/2`;
- `>c` is represented by `2c`.

The original relation sign and final interval remain in the benchmark, so the censoring information is retained.

## Source-Record Information Fields

These columns preserve information about the original BV-BRC records that contributed to each released observation.

| Column | Type | Description |
| --- | --- | --- |
| `source_record_count` | integer | Number of original BV-BRC records contributing to the released observation. |
| `source_record_ids` | string | Pipe-delimited BV-BRC record identifiers in source-record order. |
| `source_genome_names` | string | Genome names reported by the contributing source records. |
| `source_taxon_ids` | string | Taxon identifiers reported by the contributing source records. |
| `source_antibiotic_labels` | string | Original antibiotic labels before name harmonization. |
| `source_measurements` | string | Original MIC measurement strings. |
| `source_measurement_signs` | string | Original MIC relation signs. |
| `source_measurement_values` | string | Original numeric MIC values. |
| `source_normalized_signs` | string | Relation signs after quantitative parsing and normalization. |
| `source_mic_values` | string | Parsed MIC values on the common quantitative scale. |
| `source_methods` | string | Laboratory methods reported by the contributing records. |
| `source_method_versions` | string | Laboratory-method versions reported by the contributing records. |
| `source_platforms` | string | Laboratory platforms reported by the contributing records. |
| `source_vendors` | string | Testing vendors reported by the contributing records. |
| `source_testing_standards` | string | Susceptibility-testing standards reported by the contributing records. |
| `source_testing_standard_years` | string | Years associated with the reported testing standards. |
| `source_pmids` | string | Publication identifiers reported by BV-BRC. |
| `source_insertion_dates` | string | BV-BRC insertion dates for the contributing records. |
| `source_context_count` | integer | Number of retained source contexts for the released observation. |
| `source_contexts` | string | Serialized source-context information retained while resolving repeated records. |
| `source_measurement_units` | string | Original MIC units, positionally aligned with `source_record_ids`. |
| `source_resistant_phenotypes` | string | Resistant-phenotype values, positionally aligned with `source_record_ids`. |
| `source_modification_dates` | string | BV-BRC modification dates, positionally aligned with `source_record_ids`. |
| `source_evidence` | string | BV-BRC evidence values, positionally aligned with `source_record_ids`. |

For the positionally aligned fields, item `j` corresponds to item `j` in `source_record_ids`.

Empty pipe-delimited entries are kept when an individual source record does not contain a value.

Every released source record has `Laboratory Method` evidence.

The complete raw BV-BRC snapshot and genome assemblies are not copied into this repository. The released benchmark is a derived table that retains the source-record information needed to identify its contributing BV-BRC records.

## Feature-Alignment Fields

These columns connect benchmark observations to the genome and antibiotic representations used by the models.

| Column | Type | Description |
| --- | --- | --- |
| `genome_feature_row` | integer | Row index linking the observation to the genome registry and genome feature matrices. |
| `drug_feature_row` | integer | Row index linking the observation to the antibiotic registry. |
| `identity_feature_row` | integer | Row index for the antibiotic-identity control representation. |
| `morgan_feature_row` | integer | Row index in the Morgan fingerprint matrix. |
| `rdkit_feature_row` | integer | Row index in the RDKit descriptor matrix. |
| `chemberta_mean_feature_row` | integer | Row index in the mean-pooled ChemBERTa representation. |
| `chemberta_first_feature_row` | integer | Row index in the ChemBERTa first-token ablation representation. |

All feature-row indices are zero-based.

Observations for the same genome point to the same genome feature row. Observations for the same antibiotic point to the same corresponding antibiotic feature row.

## Group and Fold Fields

These columns support the random-pair and genome-disjoint evaluation protocols.

| Column | Type | Description |
| --- | --- | --- |
| `genome_group_id` | string | Genome-group identifier used during split construction. |
| `duplicate_profile_group_id` | string | Group identifier for genomes with identical canonical 8-mer profiles. |
| `duplicate_profile_group_size` | integer | Number of genomes in the identical-profile group. |
| `random_pair_fold` | integer | Deterministic random-pair query-fold assignment. |
| `genome_disjoint_fold` | integer | Deterministic genome-group-disjoint query-fold assignment. |

Complete query memberships and nested 1%, 5%, and 10% target-support memberships are stored under:

`metadata/final_transfer/nested_loso_v1/splits_v1/`

For **leave-one-antibiotic-out (LOAO) evaluation**, query and support sets are defined by separate membership files rather than by a single column in the benchmark table.

## How the Benchmark Connects to the Evaluation

The benchmark table provides the common observation index used by the released evaluation protocols.

- **Random-pair evaluation** uses `random_pair_fold`.
- **Genome-disjoint evaluation** uses `genome_disjoint_fold` together with the genome-group definitions.
- **Leave-one-antibiotic-out (LOAO) evaluation** uses the released LOAO query and support membership files.
- **Limited-label transfer** uses the released nested 1%, 5%, and 10% target-support memberships.

Target-query labels are used only for final evaluation.

For a broader description of the evaluation workflow, see:

- `docs/public_pipeline.md`
- `docs/reproducibility.md`
- `README.md`
