# Fit a custom effect model

Released prototype bundles are ready for inference. Use this reference when
fitting effects from your own matched modified and canonical data. Read the
[scoring model](../prototype_effects.md#model-and-score) before selecting a recipe.

## Fit inputs

Create a JSON manifest using the fields below. Relative paths are resolved
against the manifest directory. The neighborhood files must be extracted first.
Paths in the command examples are placeholders for your own data, checkpoints,
and manifests.

Each `canonical` family has:

- `label`: output class name; U is encoded as T internally.
- `prototypes`: nonempty list of canonical context tables, all with the same
  odd k. Repeated contexts are pooled by observation count.
- `noise_prototypes`: optional list of canonical reservoirs, defaulting to
  `prototypes`. Noise contexts may use a different k from the baseline.

All four canonical families are required, including families without a modeled
modification, because they receive other bases' footprints. Each `modifications`
entry has `label`, `canonical`, `prototype`, `control_prototype`, and
`footprint_pairs`. Each pair supplies modified and control neighborhood files
plus a single-base **BED6**, with 0-based half-open coordinates and explicit
strand. BED name/score fields are not interpreted as class labels. The caller
must supply every known event of that type, not just evaluation target sites.
Control samples must actually be unmodified; filenames cannot prove this.

To use IVT 9-mer data, put canonical-only 9-mer reservoirs from
`prototype extract-kmer-prototypes --kmer_size 9 --reservoir_size ...` in each family's
`prototypes` and `noise_prototypes`. A shared all-base reservoir may be listed
once per family; the fitter selects that family's center base. Keep each
modification's matched **5-mer oligo control** in `control_prototype`. Do not
subtract unrelated IVT means to estimate the modification effect. This
implements the 9-mer canonical baseline plus 5-mer transferable-effect model;
it does not infer modification-specific outer-flank interactions absent from
the labeled data.

Relevant manifest defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `kmer_size` | 5 | Effect context length |
| `footprint_radius` | 4 | Read-oriented footprint support; 0 fits center-only |
| `canonical_mean_mode` | `shrunken` | `empirical`, `regression`, or count-shrunken blend |
| `canonical_unseen` | `regression` | Fallback for unseen empirical/shrunken contexts; alternatively `no-call` |
| `canonical_ridge` | 10 | Canonical sequence-regression ridge |
| `effect_mode` | `regression` | `empirical`, `regression`, or `mixed` |
| `effect_ridge` | 10 | Center-effect regression ridge |
| `effect_shrinkage` | 0.5 | Global-center shrinkage α, as defined below; 0 disables it |
| `effect_mix_weight` | 0.5 | Empirical weight u in mixed mode; not estimated from read counts |
| `effect_unseen` | `no-call` | Empirical-only missing-context policy; alternatively `global` |
| `footprint_ridge` | 10 | Independent off-center footprint ridge |
| `footprint_shrinkage` | 0.5 | Independent footprint shrinkage toward offset/receiving-base effects |
| `canonical_strength` | 32 | Pseudocount favoring the canonical sequence regression |
| `noise_mode` | `base` | `base`, `hierarchical-5mer`, or `none` (identity metric) |
| `noise_shrinkage` | 0.1 | Isotropic covariance shrinkage |
| `noise_floor` | 1e-6 | Absolute covariance diagonal floor |
| `noise_local_strength` | 512 | Pseudodegrees of freedom for local-to-base covariance pooling |
| `noise_local_shrinkage` | null | Optional fixed local-to-base weight in [0,1], replacing the count heuristic |
| `noise_samples_per_context` | 256 | Deterministic uniform subsample cap per source/context |
| `min_count` | 2 | Minimum canonical/paired mean support |
| `modification_penalty` | 4 | Per-event cost in whitened Gaussian score units |
| `seed` | 0 | Noise subsampling seed |

These defaults are starting values, not empirically calibrated thresholds.
Reservoir contribution is observation-weighted up to the cap, not uniformly
weighted over all contexts or samples. Avoid duplicated input reads/tables.

## Choose fitting settings

These are **fit-time** controls, available as manifest keys and identically
named `prototype fit-effect-prototypes` CLI flags. CLI values override manifest globals.
Each modification entry may additionally override `effect_mode`,
`effect_shrinkage`, `effect_mix_weight`, `effect_unseen`, and `effect_ridge`;
these more-specific settings take precedence over globals. All resolved values
are stored in the artifact and reported during fitting.

### Canonical mean

- `--canonical_mean_mode empirical`: observed means without shrinkage. On
  observed contexts this equals setting `canonical_strength=0` in the default
  recipe. By default an unseen context still uses regression; add
  `--canonical_unseen no-call` to disable that fallback and skip canonical
  regression training entirely.
- `--canonical_mean_mode regression`: use the regression for every context.
- `--canonical_mean_mode shrunken`: a count-weighted blend of empirical and
  regression estimates.

### Center modification effect

Let **d** be the matched empirical modified−control difference for one 5-mer,
**dhat** the unshrunk ridge prediction for that 5-mer, and **dbar** the global
average of empirical differences across paired 5-mers. **α** is
`effect_shrinkage`; **u** is `effect_mix_weight`. They are scalar weights, not
learned predictors. The three recipes are:

| `effect_mode` | Effect used on an observed paired 5-mer |
| --- | --- |
| `empirical` | `(1−α) d + α dbar` |
| `regression` | `(1−α) dhat + α dbar` |
| `mixed` | `u d + (1−u) [(1−α) dhat + α dbar]` |

Thus `empirical` does not force a choice between raw and shrunken effects:
α=0 gives raw **d**, and α>0 shrinks **d** toward **dbar**. Mixed mode does
**not** shrink its empirical branch a second time. Its weights on d, dhat, and
dbar are u, (1−u)(1−α), and (1−u)α, respectively.

On unseen 5-mers, regression and mixed use the regression/global estimate
(mixed sets u=0). Empirical mode defaults to a no-call, or uses dbar when
`--effect_unseen global` is specified. If any compatible class lacks a required
empirical effect, the position is a no-call, not a forced canonical call. A
missing empirical hypothesis also masks observations in its possible footprint,
so overlapping candidate footprints are conservatively unscorable (up to 2R
away). This prevents an unsupported neighbor from being silently treated as
canonical; compare recipes on shared callable positions. A
footprint training event with an unsupported center effect raises an error;
provide paired support or explicitly select a fallback.

For initial comparisons, use **regression**, **raw empirical**,
and optionally **empirical with α=0.5**. Add mixed mode if neither suffices;
start with fixed u=0.5. Large read counts from one construct do not establish
that its effects transfer reliably to other sequence contexts.

Center-effect changes also change what is subtracted during footprint fitting,
so **refit the artifact when changing these modes**. The independent
`footprint_ridge` and `footprint_shrinkage` controls prevent a center-effect
ablation from silently changing footprint regularization. Empirical center
mode disables center regression, not the separate off-center footprint fit.
Likewise `canonical_ridge` is separate from `effect_ridge`.

### Canonical noise

`--noise_mode none` uses identity whitening and does not require saved noise
reservoirs. `base` uses one metric per canonical base. `hierarchical-5mer`
**requires kmer_size=9 canonical noise reservoirs**; 5-mer input is rejected.

For each source/9-mer group, subtract its own sampled mean first. Pool the
resulting scatter and degrees of freedom by central 5-mer. Let **S_c** denote
that 5-mer's empirical covariance, **C_b** the already regularized base-level
covariance, **ε** the configured noise floor, and **I** the identity matrix.
The local covariance is:

```
C_c = (1−η_c) (S_c + ε I) + η_c C_b
```

By default `η_c = noise_local_strength / (noise_local_strength + df_c)`;
`df_c` is the sum of `(sampled_count−1)` over contributing source/9-mer groups.
Alternatively, `--noise_local_shrinkage VALUE` fixes η for all supported 5-mers.
η=0 gives local covariance plus floor, and η=1 exactly reproduces the pooled
base metric. This is a configurable shrinkage heuristic, not an estimated
Ledoit-Wolf optimum. Isotropic `noise_shrinkage` applies to the base target.

An unseen complete receiving 5-mer falls back to C_b; an incomplete 5-mer is
unavailable. The same receiving-position matrix whitens observations, center
effects, and all incoming footprints. No covariance is estimated from queries.
Prefer a 9-mer canonical baseline alongside this noise model, so unmodeled
outer-flank mean shifts are not mistaken for modifications. The covariance
could have good support even if individual 9-mers are sparse, because many
9-mers share the same central 5-mer.

All 1,024 float32 512×512 local whitening matrices occupy about **1 GiB**.
Scoring uses shared matrix views, not an expanded read-length×512×512 array.
Fitting also needs temporary scatter storage; only one base family is fitted
at a time. Inspect the saved local counts/degrees/pooling weights when judging
support. Centering before pooling prevents 9-mer mean differences from being
counted as canonical noise.

### Example fit overrides

Keep all source paths in a manifest, then change only these flags between fits:

```bash
# No canonical shrinkage; raw empirical modification effects.
porefm prototype fit-effect-prototypes \
  --manifest /path/to/effect-manifest.json \
  --canonical_mean_mode empirical --effect_mode empirical --effect_shrinkage 0 \
  --output outputs/prototype_effects/empirical.pt

# Requires a manifest whose noise_prototypes point to genuine IVT 9-mer files.
porefm prototype fit-effect-prototypes \
  --manifest /path/to/ivt9-manifest.json \
  --effect_mode mixed --effect_mix_weight 0.5 --effect_shrinkage 0.5 \
  --noise_mode hierarchical-5mer --noise_local_strength 512 \
  --output outputs/prototype_effects/mixed_hierarchical.pt
```

The fitted artifact records the resolved settings, empirical support, and
local noise tables.

## Extract neighborhoods and fit

Fitting and scoring saved embeddings use CPU. Fresh backbone extraction supports
CPU but a GPU is preferable for volume.
The following extracts the four neighborhood files used in the example. It
can be limited with `--max_batches` for a small trial run.
Full references are restricted to 10 Mb and statistics to `--max_sites`.

```bash
data=/path/to/oligo-data
backbone=/path/to/backbone-checkpoint

for sample in control_A m6A control_C m5C; do
  case "$sample" in control_A|m6A) base=A ;; control_C|m5C) base=C ;; esac
  porefm prototype extract-effect-neighborhoods \
    --bam_path "$data/signal_bams/$sample.signal.bam" \
    --ref "$data/references/${base}_reference.fasta" \
    --model_path "$backbone" \
    --size 2048 --step 1800 --num_trim 8 \
    --min_qscore 10 --min_base_quality 10 --kmer_size 5 --radius 4 \
    --embedding_key contextual_base_seq --device cuda --disable_compile \
    --output "outputs/prototype_effects/neighborhoods/$sample.pt"
done

porefm prototype fit-effect-prototypes \
  --manifest /path/to/effect-manifest.json \
  --output outputs/prototype_effects/model.pt --threads 4
```

Neighborhood extraction retains only exact contiguous k-mers whose bases pass
the quality threshold, with one reference-position observation per read. Both
strands use genomic coordinates while sequences and effect offsets follow read
orientation. Matched statistics must record identical QC settings. It uses the
production signal normalization/chunking path, not cached basecalls alone.
