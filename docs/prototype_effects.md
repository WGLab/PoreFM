# Context-conditioned effect model

Prototype models learn modification prototypes from small labelled datasets
and use footprinting to account for neighboring effects when calling
modifications. This page describes the fitted model and its scoring objective.
To apply a released bundle, start with the
[prototype guide](guides/residual-prototypes.md).

| Task | Reference |
| --- | --- |
| Fit an artifact from matched modified and canonical samples | [Fit a custom effect model](reference/prototype-fitting.md) |
| Score custom artifacts or tune memory and devices | [Scoring and resources](reference/prototype-runtime.md) |
| Understand stored fields or write MM/ML BAM tags | [Output files and BAM tags](reference/prototype-outputs.md) |
| Preview, filter, and summarize calls | [Output API](outputs.md) |

## Model and score

In the default recipe, for canonical base family b, fit the expected embedding μ_b(c) from canonical
context means. A ridge regression on noncentral one-hot bases supplies an unseen
context fallback. A supported exact context mean receives weight n/(n+s), where
s is `canonical_strength`; the remainder is the regression prediction. The
canonical context can be 9 bases even when effect training uses 5 bases.

In the default effect recipe, fit Δ_m(c) from **matched modified–control means of the
same k-mer**, using equal-context ridge regression. `effect_shrinkage` blends
this context-dependent prediction toward the global modification effect. There
is no modified-target-context lookup in regression mode. Empirical and mixed
modes instead retain and use the matched effect table. The center prototype is
μ_b(query context) + Δ_m(query 5-mer). Different modifications retain different
effect norms; weaker m6A separation is not normalized to match stronger effects.

Default canonical noise is pooled **within context**, centering each canonical reservoir
group on its own sample mean. Its covariance is shrunk toward trace(Σ)/D times
the identity, with an absolute diagonal floor. Row whitening is W=L^(-T), where
LLᵀ=Σ. Both observations and candidate effects use the **receiving base's** W,
or its receiving **5-mer's** W in hierarchical mode.
Modified reads never supply noise observations. This is vector whitening, not
reuse of a scalar null z-score or an angular residual covariance in raw space.

Off-center templates F_m(offset, source flanks, receiving base) are fitted from
paired modified/control reference-position means. All BED events in an oligo
enter the additive design simultaneously. The separately fitted center effect
is fixed and subtracted during this fit. A receiving-base-specific background
absorbs sample-wide shifts and is **not** used as a biological footprint at
inference. Footprint coefficients are shrunk toward an offset/receiving-base
model without source-flank dependence. Offset zero is always reserved for Δ.

For a read configuration z (canonical or a compatible modification at each
base), the relative score is:

```
Q(z) = -0.5 Σ_j ||(h_j - μ_j - Σ_i F_{z_i}(j-i)) W_j||²
       - penalty × number_of_modified_bases
       + 0.5 Σ_j ||(h_j - μ_j) W_j||²
```

F includes Δ at offset zero and is zero for canonical states. Pairwise overlap
terms ensure each observation is used once in this objective. Finite-range beam
search followed by alternating coordinate ascent approximates the best
configuration. Multiple nearby modifications, including different types, are
allowed; there is no one-modification-per-window constraint and no oracle event
positions at inference. A reported class score holds the other inferred states
fixed and compares the candidate center state against canonical.

This is a first-order additive model with independent receiving-position noise,
not a learned model of modification–modification interactions or temporal noise
covariance. Single-type oligos do not establish accuracy on mixed-modification
reads. The beam search is approximate even when coordinate ascent converges.

## Evaluation considerations

Before biological use, choose hyperparameters using held-out whole constructs
and separate reads/samples, then freeze them. Compare center-only and neighbor
models on identical callable positions; measure false positives by offset and
base family, sensitivity separately for each modification, and performance on
held-out IVT. Evaluate inferred-neighbor correction without using target BEDs
as inference input. Mixed-modification nonadditivity, new 9-mer contexts, noisy
basecalls, and sample-wide shifts need explicit validation. Synthetic checks
alone do not establish held-out biological accuracy.
