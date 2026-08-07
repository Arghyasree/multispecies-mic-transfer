# Target-Excluded Nested LOSO Model Selection

This document explains how the model configuration was selected for each held-out target species.

The main rule is that **MIC outcome labels from the target species are not used during model selection**.

## Outer Target Loops

Model selection is performed separately for each target species.

| Held-out target | Development species | Shared-antibiotic cohort |
| --- | --- | ---: |
| *Escherichia coli* (Ec) | Kp and Se | 6 |
| *Klebsiella pneumoniae* (Kp) | Ec and Se | 8 |
| *Salmonella enterica* (Se) | Ec and Kp | 17 |

For each outer target, the other two species are used as the development species.

Candidate configurations are evaluated by **bidirectional transfer** between those two development species. Only their shared-antibiotic cohort is used for this model-selection stage.

For example, when Ec is the held-out target, model selection uses Kp→Se and Se→Kp transfer. Ec MIC labels are not used to choose the configuration.

This target-excluded design prevents target-species MIC labels from influencing representation, architecture, or hyperparameter selection.

## Model-Selection Stages

The model-development procedure is staged rather than one large simultaneous search.

The stages are:

1. canonical k-mer length screening;
2. genome-representation screening;
3. numerical hyperparameter and low-rank genome-fusion screening;
4. confirmation of the leading genome representations;
5. antibiotic-representation screening;
6. cross-modal genome–antibiotic architecture screening;
7. final target-excluded configuration confirmation.

Each stage uses the selected result from the earlier stage before moving to the next comparison.

This keeps the search manageable while preserving the target-excluded nested LOSO design.

## What Is Selected

The procedure compares choices for three main parts of the model.

### Genome Representations

Candidate genome representations include:

- canonical k-mer composition;
- common antimicrobial resistance (AMR) determinants;
- combined k-mer and common-AMR representations;
- separate-encoder genome-view fusion where evaluated.

### Antibiotic Representations

Candidate antibiotic representations include:

- RDKit descriptors;
- Morgan fingerprints;
- ChemBERTa embeddings;
- multi-view combinations of these representations.

The antibiotic-identity control is also evaluated during model development, but it is not used for final selection because it cannot represent source-unseen antibiotics.

### Genome–Antibiotic Architectures

The cross-modal architecture comparison includes the candidate genome–antibiotic interaction models used in the study, including:

- additive genome–antibiotic effects;
- projection–concatenation MLP;
- dual-tower interaction;
- gated multimodal unit (GMU);
- low-rank bilinear interaction;
- drug-to-genome FiLM.

## How Candidates Are Compared

For each outer target loop, candidate configurations are evaluated in both transfer directions between the two development species.

The main model-selection metric is **per-antibiotic macro-RMSE**.

The bidirectional development results are used to rank candidate representations, fusion choices, architectures, and hyperparameters.

These are **model-selection results**. They are not held-out outer-target evaluation results.

The held-out target species is evaluated only after the target-specific configuration has been selected.

## Selected Target-Specific Configurations

The model-selection procedure produces one selected configuration for each held-out target species.

| Held-out target | Genome representation | Antibiotic representation | Architecture |
| --- | --- | --- | --- |
| Ec | 4-mer + common AMR with separate encoders and rank-8 low-rank fusion | RDKit descriptors | Drug-to-genome FiLM |
| Kp | Common AMR | ChemBERTa mean + Morgan + RDKit with input-level concatenation | Drug-to-genome FiLM |
| Se | Common AMR | ChemBERTa mean | Dual-tower interaction network |

The exact machine-readable settings are stored in:

- `config/final/outer_target_configurations.tsv`
- `config/final/genome_representations.tsv`
- `config/final/antibiotic_representations.tsv`
- `config/final/cross_modal_architectures.tsv`
- `config/final/shared_hyperparameters.tsv`

The aggregate model-selection results are stored under:

`results/model_selection/`

## Fusion Terminology

The repository uses the following terms for representation and architecture comparisons.

### Input-Level Feature Concatenation

The input views are concatenated first and then passed through a common encoder.

### Projected Latent Concatenation

Each view is encoded separately. The resulting latent representations are then concatenated.

### Low-Rank Bilinear Fusion

The views are encoded separately and combined through factorized bilinear interactions.

This allows multiplicative interactions between the views without using a full bilinear parameter matrix.

### Additive Genome–Antibiotic Effects Baseline

This is a separable predictor of the form:

`b + f_genome(G) + f_drug(D)`

The genome and antibiotic effects are added, with no later joint nonlinear genome–antibiotic layer.

### Projection–Concatenation MLP

The genome and antibiotic representations are first projected into latent vectors.

The projected vectors are then concatenated and processed jointly by a nonlinear multilayer perceptron.

## Relation to Final Evaluation

After a target-specific configuration is selected, it is used for:

- two single-source training regimes;
- one balanced multisource training regime;
- zero-target-label transfer;
- random-pair limited-label adaptation;
- genome-disjoint limited-label adaptation;
- leave-one-antibiotic-out (LOAO) limited-label adaptation.

The model-selection stage and the final target evaluation are therefore separate.

Target-species MIC labels are excluded from model selection and are used only according to the support and query definitions of the final evaluation protocols.

For the full execution workflow, see:

- `README.md`
- `docs/execution_map.md`
- `docs/public_pipeline.md`
- `docs/reproducibility.md`
