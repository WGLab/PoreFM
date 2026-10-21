# Other prototype scorers

For the released joint-footprinting model, use the
[prototype guide](guides/residual-prototypes.md). This page covers alternative
per-k-mer scorers and packaging custom artifacts.

For the per-k-mer scorers, `porefm prototype classify` fits one conditional
classifier for every eligible k-mer. Classes are first partitioned by the
canonical center base of their prototype k-mers, so only modifications
applicable to the same center base are compared.

The command group also provides `extract-kmer-prototypes`,
`extract-effect-neighborhoods`, and `fit-effect-prototypes`. Run
`porefm prototype --help` to list them.

## Single-file model bundles

`--model-path` and `--model_path` accept either the usual pretrained checkpoint
or a complete PoreFM inference bundle. A bundle selects its scorer and overrides
separate `--effect_model`, `--prototype_embeddings`,
and `--class_labels` arguments. The residual-effects bundle also ignores
class-prior, metric, and classifier-output options belonging to other scorers.
Runtime and input/output options remain available.

For a residual-effect model:

```bash
porefm utils bundle-model --kind prototype \
  --model_path pretraining/model.100 --effect_model neighbors.pt \
  --output RNA004_prototype.pt

porefm prototype classify --model-path RNA004_prototype.pt \
  --effect_penalty 50 \
  --bam_path reads.signal.bam --read_based --output_dir predictions
```

Effect penalty, neighbor selection, and decoder settings remain runtime options;
the bundle does not save overrides for them. Omitting `--effect_penalty` retains
the value saved in the fitted artifact.
Other scorers can be packaged with `--scorer`, `--prototype_embeddings`
and matching `--class_labels` instead of `--effect_model`. Their fitting and
distance options remain inference CLI settings.

The destination must not exist. Bundles contain the inference backbone and
fitted artifacts, so source files are not needed on the inference machine.
Load a bundle with the inference CLI rather than `torch.load`. Inference
verifies component checksums and extracts temporary copies under `TMPDIR`
(or the system temporary directory). Allow space for the complete component
set; the temporary files are cleaned up when inference exits.

## Per-k-mer scorers

Select a scorer with `--scorer`:

- `nearest-prototype`: smallest raw cosine distance (default) or Euclidean
  distance selected with `--metric l2`. It uses saved exact means, with no
  temperature, covariance weighting, or prior adjustment. Ties select the
  first applicable class in `--class_labels` order.
- `centroid-l2`: negative Euclidean distance divided by
  `--score_temperature`.
- `centroid-cosine`: cosine similarity divided by `--score_temperature`.
- `centroid-rbf`: negative squared Euclidean distance divided by twice
  `--rbf_bandwidth` squared.
- `diagonal-gaussian`: class-specific feature variances estimated from each
  reservoir and shrunk toward one family-pooled feature variance using
  `--diagonal_shrinkage`.
- `per-kmer-lda`: the full locally shrunk, shared-covariance LDA scorer.

The centroid, diagonal-Gaussian, and LDA scorers add the applicable class
log-prior. Uniform priors are used unless `--class_priors` is supplied.
Their Parquet outputs retain scores;
they do not generate probabilities.

Nearest-prototype accepts omitted or uniform priors only; nonuniform priors
are rejected. `--metric` is only valid with this scorer.

Nearest-prototype and centroid scorers consume the exact `mean_embeddings`
and `counts` stored in a prototype file, so they also accept files extracted
with `--reservoir_size 0`.
Diagonal Gaussian and LDA require aligned reservoir embeddings, offsets, and
counts. Supply these files with `--prototype_embeddings`.

Each prototype file must contain eligible k-mers with one canonical center
base, and each family must have at least two classes. Within a family, only
k-mers shared by all classes and meeting `--min_prototype_count` are scored.
These requirements also apply to nearest-prototype; single-prototype and
mixed-center inputs are not accepted.

## Read-based motifs and optional genome coordinates

Add `--read_based` (also spelled `--read-based`) to select targets from the
called read sequence. Both `--ref` and `--chrom` are optional and can be used
independently. Without `--chrom`, this mode streams mapped and unmapped primary
records from unsorted/unindexed signal BAMs, with the usual read-quality,
length and flag filters. With `--chrom`, only reads mapped to the requested
contigs are fetched, requiring sorted, indexed signal BAMs. Direct BAM + POD5
uses a read-ID index instead. Raw signal is required through `SG` tags or
`--pod5`; move tags are preferred, with inferred chunking available for
supported chemistries. FASTQ or a basecall-only BAM without POD5 is insufficient.

By default it selects all canonical read bases (`--motif 'N 0'`). An explicit
motif, such as `--motif 'CG 0'`, is matched in the read sequence. Reference
position files and reference-guided polishing cannot be combined with this
mode. K-mer lookup and model inputs continue to use the called read sequence.
The option supports all scorers, including `residual-effects`; fitted effect
models can be reused without refitting.

`read_pos` is the zero-based index in the model's oriented read sequence.
Supply `--ref` to retain zero-based genomic `position` coordinates on both
strands. Motif matching still uses the read, even where it differs from the
reference. For example, add these options to a classification command:

```bash
--read_based --motif 'A 0;C 0' --ref reference.fa --chrom chr1 chr2
```

Negative `position` values indicate absent reference coordinates: -1 for
insertions/unmapped positions and -2 for soft clips. Unmapped records also
have `chrom='*'` and `strand=-1`. Without `--ref`, mapped records retain their
BAM chromosome/strand but their `position` is -1, including when `--chrom`
filters the input.
Use `(readname, read_pos)` to identify calls lacking genomic coordinates.
Aligned mode requires both `--ref` and `--chrom`.

## Excluding genomic regions

Add `--exclude-bed excluded.bed` (or `--exclude_bed excluded.bed`) to
`porefm prototype classify`. This uses
the same whole-read exclusion as pairwise comparison, including all classifier
scorers and residual-effects Parquet/BAM output. Exclusion defaults to disabled.

The file accepts `chrom start end` or `chrom start end strand`, with zero-based,
half-open coordinates. Three-column rows exclude both strands; optional column
4 must be `+` or `-` and matches the BAM alignment strand, including for RNA:

```text
chr9 3000000 3100000
chr9 5000000 5100000 -
```

Any matching overlap removes the entire query read before signal decoding,
including calls outside the interval. The alignment span includes introns and
deletions; touching endpoints do not overlap. Contig names must match the BAM.
Blank lines and `#` comments are allowed. This is BED3 with an optional strand
column, not standard BED4 names or BED6. See the
[pairwise exclusion documentation](embedding-comparisons.md#direct-bam-input)
for the input format.

With `--read_based`, mapped reads can be filtered without `--ref` because the
overlap uses BAM alignment coordinates; unmapped reads remain eligible. Saved
prototype/effect artifacts are unaffected: this option filters the query BAMs.
The BED path and SHA-256 are recorded in the run manifest.

## Output contract

The fitted artifact records `format_version=4`, the selected `scorer_type`,
embedding method and dimension, k-mer size, global class order, center-base
families, class priors, and scorer-specific fitted parameters.

For centroid, diagonal-Gaussian, and LDA scorers, each Parquet row contains:

- `score` for the predicted class;
- `predicted_class_index` and `predicted_class`;
- `score_<class>` for every global class;
- read, reference, strand, position, and k-mer identity columns.

Nearest-prototype instead writes `distance`, `closest_prototype_index`,
`closest_prototype`, and `distance_<class>`, alongside the same identity and
coordinate columns. Smaller distances are closer; no score or probability
columns are emitted. Its fitted artifact also records the metric and means.

Only columns belonging to the query k-mer's center-base family are finite.
Other class columns are `NaN` because those classes were not compared.
Parquet output contains scores or distances, not probabilities.

For `--scorer residual-effects`, `--output_format bam` or `both` additionally
supports RNA MM/ML output using conditional family softmax probabilities.
`parquet` remains the default; other scorers support only Parquet. See
[effect BAM output](reference/prototype-outputs.md#optional-mmml-bam-output) for the SAM
code table, temperature option, and probability semantics.

Run `porefm prototype classify --help` for the complete fitting and BAM
query options.
