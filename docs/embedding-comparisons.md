# Pairwise sample comparison

Start with the [test-versus-control recipe](guides/pairwise.md) for a complete
command and output summary. This page describes the available options.

`porefm pairwise-comparison` compares control and test samples directly from
BAM inputs.

The workflow compiles the model with `torch.compile(dynamic=True)` by default.
`--disable-compile` selects
eager execution for diagnostics.

Unless `--step` is supplied explicitly, BAM collection chooses a downstream
chunk step from `--size`: DNA reserves 128 time steps of overlap and RNA
reserves 256. Windows too small for the normal overlap use a step equal to
three quarters of their size. The resolved choice is printed before data
collection; the small augmentation stride recorded by a pretraining checkpoint
is intentionally ignored.

## Analysis modes

| Mode | Control retained | Test retained | Parquet scores | BAM output |
|---|---|---|---|---|
| `--sitewise-embedding-shift` | Per-site count and sum | Per-site count and sum | Mean-embedding MSE and cosine distance per locus | No |
| `--sitewise-anomaly-scoring` | All canonical rows | All canonical rows | Per-test-read MSE, cosine distance, retained PC scores, squared Mahalanobis distance, and p-value; separate per-site PCA/null statistics | Anomaly Phred or control-mean cosine |
| `--readwise-anomaly-scoring` | Per-site count and sum | All canonical rows | Per-test-read MSE and cosine distance against the full control mean | Control-mean cosine |

`sitewise-anomaly-scoring` already includes each test read's direct comparison
with its sitewise control mean, so it cannot be selected together with
`readwise-anomaly-scoring`. `sitewise-embedding-shift` may be combined with
either read-scoring mode when both locus-level and read-level outputs are
needed. The minimum control and test coverage defaults are 15. Readwise
scoring requires the control threshold but intentionally permits one test read
at a site; the test threshold applies to the two sitewise modes.

All modes use the same ownership rule for overlapping model windows. Each
eligible read position is assigned to the chunk in which it lies furthest
from a real sequence edge; equal-margin ties use the earlier chunk. Thus a
read position contributes at most one embedding, regardless of mode.

For sitewise anomaly scoring, each test row stores exactly `pc_1` through
`pc_N`, where `N` is `--n-components`. These are the coordinates actually
used for its squared Mahalanobis distance: a separate control-standardized,
joint PCA space is fitted for every position and strand. Direct MSE and cosine
distance remain in the original selected embedding space.

## Direct BAM input

```bash
porefm pairwise-comparison \
  --model-path model.pt \
  --control-bam control.bam \
  --test-bam test.bam \
  --ref reference.fa \
  --motif-string "A 0" \
  --sitewise-anomaly-scoring \
  --sitewise-embedding-shift \
  --output comparison_run
```

Use sorted, indexed signal BAMs with `SG` tags. Sequence type, pore chemistry,
and expected k-mer currents are inferred from the model checkpoint.

When either population expands to more than one BAM, its layout must be
declared explicitly:

- `ordered-shards` naturally sorts numeric shard suffixes and reads the files
  as an ordered coordinate stream within each worker assignment.
- `independent-streams` performs a k-way merge of independently sorted BAM
  streams.

Pass `--exclude-bed excluded.bed` to skip reads overlapping unwanted regions
before signal decoding, chunking, and embedding accumulation. This applies to
both populations, including `--store-embeddings`, and defaults to disabled.
The file accepts three or four whitespace-separated columns:

```text
# chrom start end strand
chr9 3000000 3100000
chr9 5000000 5100000 -
```

Coordinates are zero-based and half-open. Three-column rows exclude both
strands. Optional column 4 must be `+` or `-` and is matched to the BAM alignment
strand, also for RNA. Rows with and without strand may be mixed.
Chromosome names must match the BAM exactly. A matching overlap discards
the entire read from collection, including its positions outside the region.
The reference span includes introns and deletions; touching endpoints do not
overlap. Blank lines and `#` comments are allowed. This is a custom four-column
strand extension to BED3, not standard BED4 with a name column or BED6 with
strand in column 6.

Use `--num-workers 4` for up to four loader processes **per population**,
or set `--control-num-workers 4 --test-num-workers 4` independently. The shared
default is 1 per population. An override of 0 reads that population in the main
process. Negative counts are rejected. Each positive request is capped at the
number of unique expanded BAM paths in that population; one BAM uses at most
one worker. Requested/effective counts are logged, and effective counts are
recorded in the run manifest.

`--prefetch-factor` defaults to 2 complete-read batches per worker and applies
only to subprocess loading. `--reads-per-chunk` controls reads per buffer;
`--forward-batch-size` separately caps model chunks per forward pass. Independent
streams also retain one batch per active BAM inside each worker, so their memory
use depends on the BAM count as well as the worker/prefetch settings. Worker
count changes can change batching and capped fit-sample selection; they preserve
complete-read ownership and observation counts, not bitwise score reproducibility.

Use `--store-embeddings` to export embeddings in genomic blocks while the
BAMs are being processed. A classifier checkpoint is also accepted; the selected
backbone embedding is still used for comparison.

## Classifier filtering

With a classifier checkpoint in `--model-path`, control and test can use
independent allowed class indices and probability thresholds. For example, add:

```bash
  --control-filter-class-index 1 2 \
  --control-filter-min-probability 0.95 0.96 \
  --test-filter-class-index 0 2 \
  --test-filter-min-probability 0.90 0.98
```

Indices are zero-based classifier output columns, using the label mapping of
that checkpoint. Each index is paired with the threshold in the same list
position. Both options must be supplied together for a population, with equal
nonzero list lengths, unique nonnegative indices, and finite thresholds in
`[0, 1]`.

A control position in this example passes when `P(class 1) >= 0.95` **or**
`P(class 2) >= 0.96`; test positions independently require `P(class 0) >= 0.90`
or `P(class 2) >= 0.98`. Probabilities use softmax over all classifier classes.
They are not summed or renormalized over allowed labels, and a passing class
does not have to be the highest-probability class. Passing multiple thresholds
still retains the position only once. Omit both options for a population to
leave it unfiltered.

Filtering selects individual read positions after canonical chunk ownership
is fixed. Rejected positions do not fall back to other overlapping chunks.
Coverage thresholds, means, anomaly fits, scores, and exported embeddings all
use the retained positions. Annotated BAMs still contain the original records;
only retained, eligible positions receive comparison scores. Both populations
use the same checkpoint.

Run manifests record both filters. Resuming a run requires the same labels
and thresholds for both populations.

## Performance and reproducibility

Reduce `--forward-batch-size` for GPU memory pressure. Loader workers are
assigned per population and capped by the number of BAM files; increasing
workers for a single BAM does not increase parallelism. Prefetching and
independent input streams also consume host memory.

Changing worker counts or processing order can change the capped fit-sample
selection and resulting scores. For repeatable comparisons, keep inputs,
settings, software, device, and processing order fixed. Background output
writes finish before successful completion or BAM annotation; write errors
fail the run.

## Outputs and BAM support

- `locus_metrics/*.parquet` stores sitewise embedding shifts.
- `read_scores/*_meta.parquet` stores `readname`, reference, strand, and the
  number of contiguous rows in the corresponding data table.
- `read_scores/*_data.parquet` stores flattened positions and scores in
  metadata-row order. This is the same per-read meta/data factoring used by
  classifier inference. Sitewise anomaly rows also contain `pc_1` through
  `pc_N`; their repeated site coverage is moved to `anomaly_site_metrics`
  instead of being duplicated here. Readwise direct scoring retains coverage
  in this table because it does not fit a PCA/null site table.
- `anomaly_site_metrics/*.parquet` stores one row per eligible position and
  strand for sitewise anomaly scoring. It records full control/test coverage,
  capped PCA-fit coverage, scored-test coverage, empirical-calibration
  coverage, and the control mean and regularized covariance in PCA space.
  `control_pca_covariance` is flattened row-major into a fixed-size list.
- `embeddings/*.pt` stores versioned canonical control and test rows when
  `--store-embeddings` is selected. Row embeddings are compact bfloat16, while
  float32 site means and counts are stored separately. Enabling export does
  not change the mean-based scores computed during BAM processing. These
  files are available for downstream analysis; the CLI does not rescore them.

For BAM annotation, pass `--annotated-bam-dir`, `--bam-base`,
`--bam-mod-code`, and one of these score selections:

Add `--drop-tags SG mv` to remove raw signal and move-table tags from every
annotated output record, including records without scores. Removal happens
after scoring and polishing; omitting this option retains those tags.

`--bam-base` is emitted as the MM fundamental base after uppercasing. RNA mode
does not rewrite `T` as `U`, so pass `--bam-base T` for consumers such as IGV
that require thymine in RNA BAM modification tags. Pass `U` explicitly when
uracil output is desired.

- `--bam-score anomaly-phred` converts each p-value to `Q = -10 log10(p)`,
  then writes `round(255 * clip(Q / --bam-phred-max, 0, 1))`. The default
  `--bam-phred-max 40` maps p-values 0.1, 0.01, 0.001, and 0.0001 to
  approximately 25%, 50%, 75%, and 100% intensity. It requires
  `--sitewise-anomaly-scoring`.
- `--bam-score control-mean-cosine` writes scaled cosine distance and works
  with either read-scoring mode.

The selected byte score is written to the standard `ML` array and duplicated
in `ES` by default. `CV` contains byte-clipped site coverage, and `MM`/`MN`
describe the encoded sites and sequence length. MSE remains Parquet-only.
Source test BAMs are re-read for annotation, so the mean-only
`sitewise-embedding-shift` mode cannot produce BAM output. When test sequence
polishing is enabled, the annotated BAM uses the same polished sequence and
coordinate maps used during embedding collection.

`--chrom-list` restricts both populations during scoring and
restricts annotated BAM records to mapped alignments on those references.
Indexed annotation inputs fetch only the selected references, in BAM header
order; unindexed inputs are filtered during a sequential scan. Selected
records without scores are still copied. Unmapped records are excluded when
a chromosome list is supplied. Without a chromosome restriction, annotation
copies all source records. Output headers retain the full reference dictionary.

## Annotating calibrated canonical-null Parquet

Canonical-null comparison writes flat per-read rows containing `readname`,
`chrom`, `position`, `strand`, and `pvalue`. Annotate these without rerunning
the model using:

```bash
python -m porefm.cli.pairwise_parquet_to_bam \
  --bam input.bam --parquet canonical_null_output/ \
  --output_bam annotated.bam --base N --mod_code N --seq_type dna \
  --canonical_null
```

This mode writes Phred-40-scaled `ML` bytes: `round(255 * clip(-10 log10(p) /
40, 0, 1))`. It writes no coverage tag and removes existing `CV` and `cv` tags
from every output record, even with `--preserve_existing_mod_tags`.
`calibration_count` is not read coverage and is never copied into a coverage
tag. Coverage-column overrides and `--require_coverage` are rejected in this
mode. Pairwise square-root encoding is the default when
`--canonical_null` is omitted. Use `--chrom_list chr2` to restrict output to
mapped records on selected references in indexed BAM inputs. Without this
option, the flat-Parquet annotator copies the entire input BAM. Add
`--drop_tags SG mv` to remove signal and move-table tags from output records.
The compact-score annotation Python API supports the same removal through
`drop_tags=("SG", "mv")`. Other input tags are retained apart from the score
and coverage tags replaced or removed by the selected annotation mode.
