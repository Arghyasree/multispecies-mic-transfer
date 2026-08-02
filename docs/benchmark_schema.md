# Public Quantitative MIC Benchmark

## Artifact

The released benchmark table is:

```text
data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz
```

Its checksum is:

```text
data/benchmark/final_quantitative_mic_benchmark_v1.tsv.gz.sha256
```

The gzip-compressed TSV contains one row per unique
species–genome–antibiotic observation.

| Species | Genomes | MIC observations | Antibiotics | Exact | Censored |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Escherichia coli* (Ec) | 6,673 | 68,881 | 19 | 25,742 | 43,139 |
| *Klebsiella pneumoniae* (Kp) | 5,602 | 50,299 | 17 | 13,582 | 36,717 |
| *Salmonella enterica* (Se) | 9,119 | 49,183 | 8 | 20,644 | 28,539 |
| **Total** | **21,394** | **168,363** | **19 unique** | **59,968** | **108,395** |

The benchmark represents 179,385 source records from the BV-BRC snapshot
retrieved on 22 July 2026. Compatible repeated records were reconciled into
one observation. The released table contains 11,005 observations derived
from more than one source record, with at most three source records
contributing to one observation.

## Reading the Table

The file follows standard quoted TSV conventions and should be read with a
TSV-aware parser:

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

The machine-readable species codes are `ec`, `kp`, and `se`. These lowercase
codes are retained for compatibility with the released matrices, splits, and
configurations.

## Observation and Target Fields

| Column | Type | Description |
| --- | --- | --- |
| `final_transfer_observation_row` | integer | Zero-based row index in the frozen final-transfer table. |
| `species_code` | string | Machine-readable species code: `ec`, `kp`, or `se`. |
| `provisional_species` | string | Harmonized species assignment used during benchmark construction. |
| `observation_id` | string | Stable identifier for the reconciled quantitative MIC observation. |
| `genome_id` | string | BV-BRC genome identifier. |
| `normalized_antibiotic` | string | Harmonized antibiotic identity used throughout the study. |
| `mic_target_log2_mg_per_l` | float | Regression target: the base-2 logarithm of the point MIC in mg/L. |
| `is_exact_observation` | Boolean | Indicates that the reconciled MIC observation is exact. |
| `is_censored_observation` | Boolean | Indicates that the observation is left- or right-censored. |

Every row is exactly one of exact or censored. No
species–genome–antibiotic key occurs more than once.

## Reconciled MIC-Constraint Fields

| Column | Type | Description |
| --- | --- | --- |
| `reconciliation_status` | string | Outcome of single-record handling or repeated-record reconciliation. |
| `constraint_origin` | string | Provenance category of the final MIC constraint. |
| `duplicate_class` | string | Classification of the source-record multiplicity or duplication pattern. |
| `reduced_constraint_type` | string | Final constraint type after reconciliation. |
| `reduced_sign` | string | Final relation sign: `=`, `<`, `<=`, `>`, or `>=`. |
| `reduced_mic_value` | float | Reported threshold associated with the final relation sign. |
| `intersection_lower` | optional float | Lower endpoint of the reconciled interval; blank when unbounded. |
| `intersection_lower_closed` | optional Boolean | Indicates whether the lower endpoint is inclusive. |
| `intersection_upper` | optional float | Upper endpoint of the reconciled interval; blank when unbounded. |
| `intersection_upper_closed` | optional Boolean | Indicates whether the upper endpoint is inclusive. |
| `intersection_notation` | string | Human-readable representation of the reconciled interval. |
| `mic_target_point_mg_per_l` | float | Positive point MIC used for regression, expressed in mg/L. |
| `mic_target_substitution_rule` | string | Rule used to derive the point target from the reconciled constraint. |
| `censoring_direction` | string | Censoring direction: none, left, or right. |
| `censoring_strictness` | string | Whether the censoring relation is strict or inclusive. |
| `point_target_version` | string | Version identifier for the point-target construction policy. |
| `normalized_unit` | string | Common MIC unit; all released values use `mg/L`. |

Exact measurements and inclusive bounds retain the reported threshold. A
strict left-censored value `<c` is represented by `c/2`, and a strict
right-censored value `>c` is represented by `2c`. The original sign and
interval remain available for censor-aware analyses.

## Source-Record Provenance Fields

| Column | Type | Description |
| --- | --- | --- |
| `source_record_count` | integer | Number of original BV-BRC records contributing to the observation. |
| `source_record_ids` | string | Pipe-delimited BV-BRC record identifiers in source-record order. |
| `source_genome_names` | string | Genome-name provenance retained from the contributing records. |
| `source_taxon_ids` | string | Taxon-identifier provenance retained from the contributing records. |
| `source_antibiotic_labels` | string | Original antibiotic labels before name harmonization. |
| `source_measurements` | string | Original MIC measurement strings. |
| `source_measurement_signs` | string | Original relation-sign provenance. |
| `source_measurement_values` | string | Original numeric measurement-value provenance. |
| `source_normalized_signs` | string | Relation signs after quantitative parsing and normalization. |
| `source_mic_values` | string | Parsed MIC-value provenance on the common quantitative scale. |
| `source_methods` | string | Laboratory-method provenance. |
| `source_method_versions` | string | Laboratory-method version provenance. |
| `source_platforms` | string | Laboratory-platform provenance. |
| `source_vendors` | string | Testing-vendor provenance. |
| `source_testing_standards` | string | Susceptibility-testing-standard provenance. |
| `source_testing_standard_years` | string | Testing-standard year provenance. |
| `source_pmids` | string | Publication identifiers reported by BV-BRC. |
| `source_insertion_dates` | string | BV-BRC insertion-date provenance. |
| `source_context_count` | integer | Number of retained source contexts for the observation. |
| `source_contexts` | string | Serialized source-context provenance retained during reconciliation. |
| `source_measurement_units` | string | Original measurement units, positionally aligned with `source_record_ids`. |
| `source_resistant_phenotypes` | string | Resistant-phenotype values, positionally aligned with `source_record_ids`. |
| `source_modification_dates` | string | BV-BRC modification dates, positionally aligned with `source_record_ids`. |
| `source_evidence` | string | BV-BRC evidence values, positionally aligned with `source_record_ids`. |

For the four positionally aligned fields, pipe-delimited element `j`
corresponds to source-record identifier `j` in `source_record_ids`. Empty
tokens are retained when an individual source record lacks a value. Every
released source record has `Laboratory Method` evidence.

The complete raw BV-BRC snapshot and genome assemblies are not copied into
this repository. The benchmark is a derived, reconciled table containing the
provenance required to identify its contributing source records.

## Feature-Alignment Fields

| Column | Type | Description |
| --- | --- | --- |
| `genome_feature_row` | integer | Row index linking the observation to the genome registry and released genome matrices. |
| `drug_feature_row` | integer | Row index linking the observation to the frozen antibiotic registry. |
| `identity_feature_row` | integer | Row index for the antibiotic-identity control representation. |
| `morgan_feature_row` | integer | Row index in the Morgan fingerprint matrix. |
| `rdkit_feature_row` | integer | Row index in the RDKit descriptor matrix. |
| `chemberta_mean_feature_row` | integer | Row index in the mean-pooled ChemBERTa matrix. |
| `chemberta_first_feature_row` | integer | Row index in the ChemBERTa first-token ablation matrix. |

The feature-row indices are zero-based. Repeated observations for the same
genome or antibiotic point to the same corresponding feature row.

## Group and Fold Fields

| Column | Type | Description |
| --- | --- | --- |
| `genome_group_id` | string | Genome-group identifier used during split construction. |
| `duplicate_profile_group_id` | string | Group identifier for genomes with identical canonical 8-mer profiles. |
| `duplicate_profile_group_size` | integer | Number of genomes in the identical-profile group. |
| `random_pair_fold` | integer | Deterministic random-pair query-fold assignment. |
| `genome_disjoint_fold` | integer | Deterministic genome-group-disjoint query-fold assignment. |

Complete query and nested-support memberships are released under:

```text
metadata/final_transfer/nested_loso_v1/splits_v1/
```

The drug-held-out protocol is defined through its leave-one-antibiotic-out
query and support membership files rather than a single column in this table.