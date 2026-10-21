# Zero-shot detection and output filtering

For a complete application recipe with matching null, whitening, scale, and
calibration artifacts, start with [zero-shot detection](guides/zero-shot.md).
This page covers additional modes and output options.

`porefm zero-shot detect` scores read positions using a unified canonical
reference model and empirical k-mer calibration. It supports cosine and
whitening scores, with the score family selected from calibration metadata
by default. Supply the matching whitening artifact for whitening detection.

## Single-file model bundles

Use `--model-path RNA004_zero-shot.pt` (or `--model_path`) to provide all
model artifacts in one file. This overrides `--canonical_null_model_path`,
`--calibration_path`, and `--whitening_artifact`, even if those paths are invalid.
Set input, output, filtering, and device options on the detection command.

Create a bundle once from matching artifacts:

```bash
porefm utils bundle-model --kind zero-shot \
  --canonical_null_model_path scale/model.best \
  --calibration_path calibration.pt \
  --whitening_artifact whitening.pt \
  --output RNA004_zero-shot.pt

porefm zero-shot detect --model-path RNA004_zero-shot.pt \
  --bam_path aligned.signal.bam --ref reference.fa \
  --motif 'N 0' --output_dir predictions/aligned
```

Whitening can be omitted when packaging cosine-only detection. The destination
must not exist. A bundle contains all required inference components; the source
files are not needed on the inference machine. Load it with the detection
command, rather than `torch.load`. Inference verifies component checksums and
extracts temporary files under `TMPDIR` (or the system temporary directory).
Allow enough disk space for the complete component set; these temporary files
are cleaned up when inference exits.

Detection always writes paired data/meta Parquets, factoring out
repeated read names, chromosomes, and strands. Read these files and
collapse them to reference loci with `porefm.outputs.open_output`; see the
[shared output API](outputs.md) for filters, DuckDB, and streamed site summaries.

## Optional deterministic execution

Add `--deterministic` to `porefm zero-shot detect` to request repeatable
numerical execution for identical inputs, model artifacts, batching, hardware,
and software versions. Compilation remains enabled unless `--disable_compile`
is also supplied; autocast and matmul precision follow the command settings.

This option enables PyTorch deterministic algorithms, fixes numerically
sensitive compiler kernel choices, disables cuDNN benchmarking, and defaults
`CUBLAS_WORKSPACE_CONFIG` to `:4096:8` when it is unset. PyTorch raises an error
for operations it identifies as lacking a deterministic implementation. The
requested option and effective numerical settings/software versions are
recorded in `run_manifest.json` under
`arguments.deterministic` and `arguments.numerical_policy`.

Deterministic execution can be slower and can produce different scores and
p-values from the default policy. It does not retrain the model, refit
whitening, or regenerate calibration. When using an existing calibration,
assess p-value behavior on canonical controls before relying on thresholds
under the selected policy. The option applies only to detection. Identical
results across devices, versions, or batch layouts are not guaranteed.

## Aligned, reference-based inference

Omit `--chrom` to process eligible aligned reads across all reference contigs.
Specify `--chrom chr1 chr2` to restrict the run. This mode still requires a
reference FASTA and aligned inputs. Signal-BAM input requires coordinate
sorting and indexing; direct BAM + POD5 uses its read-ID index instead (see
[input preparation](guides/inputs.md)). Motifs/position labels are
reference-based; `--motif "N 0"` selects all aligned bases without constructing
a genome-wide label dictionary.

```bash
porefm zero-shot detect \
  --bam_path aligned.signal.bam \
  --canonical_null_model_path scale/model.best \
  --calibration_path calibration.pt \
  --whitening_artifact whitening.pt \
  --ref reference.fa \
  --motif "N 0" \
  --output_dir predictions/aligned
```

## Unaligned or mixed aligned/unaligned samples

Use **`--read_based`** (also spelled `--read-based`). Without `--chrom`, this
streams BAM records in file order, including unmapped reads, without requiring
a BAM index or reference FASTA. It selects motifs in the called sequence, not
the reference. `--motif "N 0"` selects every eligible canonical read base; it is
expanded into A/C/G/T motifs per read, not reference-wide arrays.

```bash
porefm zero-shot detect \
  --bam_path unaligned.signal.bam \
  --canonical_null_model_path scale/model.best \
  --calibration_path calibration.pt \
  --whitening_artifact whitening.pt \
  --read_based --motif "N 0" \
  --output_dir predictions/read_based
```

Sequence type, pore chemistry, and k-mer currents come from checkpoint
metadata. Use matching model, whitening, and calibration artifacts. Raw signal
is required, either in `SG` tags or supplied with `--pod5`. Move tables are
preferred; supported chemistries can infer chunks when moves are absent.
Ordinary basecalls alone are not sufficient.

- `readname` and zero-based `read_pos` identify observations. Unmapped reads
  have `chrom="*"`, `position=-1`, and `strand=-1`. Do not group these rows by
  reference position as though they were genomic sites.
- Supplying `--ref` in read-based mode retains available reference coordinates
  for aligned reads. Without it, reference positions are unavailable even for
  mapped records; the BAM contig name/strand may still be recorded.
- An explicit `--chrom` restricts read-based mode to aligned records on those
  contigs and requires `--ref` (plus a coordinate index for signal-BAM input).
  Omit it to include unmapped
  reads. Supplying only `--ref` does not impose a chromosome filter.
- Read-based mode requires `--motif` and cannot use `--pos_file`. Reference-
  guided polishing requires `--ref` and suitable aligned reads; it is not an
  unaligned-read operation.
- Flag, read-length, quality, valid-kmer, trimming, and score filters
  apply. `--max_batches`, if supplied, limits the run. “All reads”
  removes chromosome restriction; it does not disable these filters.

Read-based mode is explicit rather than auto-enabled when `--ref` is absent,
so an accidentally omitted reference cannot silently change motif semantics.

## P-value filtering and peak output

`porefm zero-shot detect` supports an inclusive p-value
cutoff and optional peak finding. Add these options to an existing inference
command with its usual model, calibration, input, and output paths:

```bash
--motif "N 0" --max_pvalue 0.01 --peak_only --prune_radius 5
```

- `--max_pvalue P` writes only finite p-values **<= P**, with P in [0, 1].
  It replaces the raw-distance filter, so explicitly combining it with
  `--min_metric` is an error. This option works independently of peak finding
  and can be used with motif/position selections.
- `--peak_only` reports local p-value minima after overlapping chunks have
  been stitched into a read. It requires all-base selection (`--motif "N 0"`)
  and no `--pos_file`. It works in both reference-based and read-based modes.
- `--prune_radius W` optionally suppresses weaker qualifying peaks within
  **+/- W read bases, inclusive**. For W=5, retained peaks must be at least
  6 bases apart. The default, 0, disables pruning; a positive radius requires
  `--peak_only`. Distance uses `read_pos`, not genomic coordinates, including
  for reverse-strand, spliced, and unmapped reads.

Peak finding uses the original p-values, before applying the chosen output
threshold. Each peak or equal-valued plateau must have valid, strictly higher
p-values on both sides. Flat peaks use their midpoint rounded down: positions
100--104 select **102**, while 100--103 select **101**, not 100. Invalid k-mers,
unlabelled bases, nonfinite p-values, and missing positions break the profile;
their neighboring segment endpoints are not called as peaks. Gaps are never
compressed away. Chunk trimming determines available bases.

After peak detection, the selected p-value or raw-distance cutoff is applied.
If distance pruning is enabled, qualifying peaks are considered by increasing
p-value, with lower read position breaking ties. Each retained peak suppresses
its neighbors; suppressed peaks cannot suppress other peaks. Results are
written in read-position order with their original scores and p-values.
For example, peaks at 100, 104, and 108 with p-values 0.001, 0.002, and 0.003
and radius 5 retain 100 and 108. A smoothly declining shoulder is not a peak.

These options select output rows after scoring and calibration. Without either
threshold, the default `distance > 0.0` filter applies. Without `--peak_only`, all qualifying bases are emitted.
No smoothing or prominence filter is applied. Small secondary peaks can
survive, and distance pruning can merge distinct nearby modifications.

## Wider sequence context

Add `--output_context_radius 15` to `porefm zero-shot detect`
to save 31 bases around each retained position:

```bash
--motif "N 0" --max_pvalue 0.01 --peak_only --prune_radius 5 \
--output_context_radius 15
```

The default radius is 0 (disabled).
A positive radius R adds `sequence_context` and boolean `context_complete`.
Context works with either output threshold, with or without peak finding,
and with motif/position selections. It adds context to the selected rows
without rescoring them.

Context follows the **model-input sequence**, in read orientation, including
on reverse-strand reads. With `--use_polished_sequence`, this is the polished
sequence and its read coordinates, which can differ from original BAM query
offsets. The alphabet is A/C/G/T/N, including T for RNA, consistent with the
k-mer column. The reported base is at zero-based context index R.
Available neighboring bases are included even if they are unlabelled, trimmed
from scoring, or fail the output filter. Overlapping chunks use the same
midpoint stitching as the scores.

Unavailable read flanks or gaps between loaded chunks are padded with N, and
`context_complete` is false. An observed N counts as available and does not
by itself make context incomplete. No extra BAM/reference lookup is made to
fill gaps; context cannot recover sequence absent from the loaded chunks.

### Compact storage and decoding

The default `--output_context_format packed` stores lossless binary DNA:
two bits per A/C/G/T base plus an N mask only for contexts containing N.
A 31-base context takes **8 bytes**, or **12 bytes** with an N mask, before
Parquet overhead and compression. The output reader decodes packed contexts
automatically when the column is selected:

```python
from porefm.outputs import open_output

calls = open_output("outputs/zero-shot")
for table in calls.iter_tables(
    columns=["readname", "read_pos", "sequence_context", "context_complete"],
    reads_per_batch=32,
):
    print(table.to_pandas().head())
```

Use `--output_context_format text` to store directly readable DNA strings.
Keep the same format and radius across parts of one dataset. See the
[output API](outputs.md) for filtering and selecting columns without loading
the whole run into memory.
