# Model-selection protocol

The model-development procedure was staged rather than a simultaneous full
factorial search:

1. canonical k-mer length screening;
2. genome-representation screening;
3. antibiotic-representation screening;
4. cross-modal genome–antibiotic architecture screening;
5. targeted refinement of within-genome and within-antibiotic view fusion,
   together with numerical hyperparameter confirmation;
6. final outer-target-label-excluded configuration confirmation and freeze;
7. held-out outer-species transfer evaluation.

For an outer target species T, all MIC outcome labels from T were excluded from
model selection. Candidate configurations were evaluated in both transfer
directions between the two development species using their pairwise
shared-antibiotic cohort. This produced one frozen configuration for each outer
target loop. Inner bidirectional development results are model-selection
results, not held-out outer-target test results.

## Fusion terminology

- **Input-level feature concatenation:** raw views are concatenated before a
  common encoder.
- **Projected latent concatenation:** views are encoded separately and their
  latent representations are concatenated.
- **Low-rank bilinear fusion:** separate view encoders are coupled by factorized
  bilinear interactions.
- **Additive genome–antibiotic effects baseline:** a separable predictor of the
  form `b + f_genome(G) + f_drug(D)` with no subsequent joint nonlinear layer.
- **Projection–concatenation MLP:** projected genome and antibiotic vectors are
  concatenated and processed jointly by a nonlinear MLP.
