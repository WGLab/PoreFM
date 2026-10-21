# Prototype scoring and resources

For a released bundle and a minimal command, use the
[prototype guide](../guides/residual-prototypes.md). This page covers custom
artifacts, read-based inputs, and resource tuning.

## Score a custom model

Use the artifact produced by [custom fitting](prototype-fitting.md), paired
with the exact backbone checkpoint used for that fit. Replace the query paths
and contig below with your sample:

```bash
backbone=/path/to/backbone-checkpoint

porefm prototype classify \
  --scorer residual-effects --effect_model outputs/prototype_effects/model.pt \
  --bam_path QUERY.signal.bam --ref QUERY.fasta --chrom CONTIG \
  --model_path "$backbone" \
  --size 2048 --step 1800 --num_trim 8 \
  --motif 'N 0' --device cuda --effect_device cuda --disable_compile \
  --effect_beam_width 32 --effect_max_sweeps 8 --effect_threads 4 \
  --output_dir outputs/prototype_effects/query_calls
```

For an unaligned signal BAM, add `--read_based` and omit `--ref` and `--chrom`:

```bash
porefm prototype classify \
  --scorer residual-effects --effect_model outputs/prototype_effects/model.pt \
  --bam_path QUERY.unaligned.signal.bam --read_based \
  --model_path "$backbone" \
  --size 2048 --step 1800 --num_trim 8 \
  --motif 'N 0' --device cuda --effect_device cuda --disable_compile \
  --output_dir outputs/prototype_effects/unaligned_calls
```

This streams unindexed BAM records, including unmapped reads. Raw signal is
required through `SG` tags or `--pod5`. Move tables are preferred; supported
chemistries can infer chunks when moves are absent. Sequence type, pore
chemistry, and k-mer currents come from the matched backbone's metadata.
All read chunks supply neighbor evidence; `--motif` only selects output
positions in the called sequence. Missing alignment does not invalidate a
read's embeddings. The usual context-support checks and edge trimming still
apply. Reference-based `--pos_file` and polishing options are incompatible.

Output uses zero-based oriented `read_pos` with `position=-1`; unmapped reads
also have `chrom='*'` and `strand=-1`. Mapped records streamed in this mode
without `--ref` retain chromosome/strand metadata but also use `position=-1`.
Identify calls lacking reference coordinates by `(readname, read_pos)`.

Read-based motifs also support optional `--ref` and `--chrom`, independently:

```bash
--read_based --motif 'A 0;C 0' --ref reference.fa --chrom chr1 chr2
```

This selects A/C positions in the called read sequence, fetches only reads
mapped to chr1/chr2 from sorted, indexed BAMs, and retains their zero-based
genomic coordinates on both strands. Insertions and soft clips remain read
targets with `position=-1` and `position=-2`, respectively. Without `--chrom`,
all records are streamed, including unmapped reads, even when `--ref` is supplied. Without `--ref`,
`--chrom` still filters input but genomic positions remain -1. All chunks of
each accepted read continue to supply neighbor evidence.

Do not pass `--prototype_embeddings`, `--class_labels`, `--class_priors`, or
`--classifier_output`: these come from the already fitted artifact. Use
`--effect_neighbors off` for a center-only ablation with the same fitted center
effects and noise metric. `--effect_penalty` explicitly overrides the event
cost. Fresh output paths are required; existing files are not overwritten.

## Devices and resource settings

Classification defaults to `--device cuda --effect_device cuda`. The first
option selects backbone execution; the second selects downstream effect scoring.
`--device cpu --effect_device cpu` requests CPU execution where supported by
model dependencies. `auto` opts into automatic device selection. See
[installation](../getting-started/installation.md) for the supported GPU setup.

| Option | Default | What it controls |
| --- | ---: | --- |
| `--batch_size` | See command help | Backbone chunks per forward pass; reduce for GPU memory pressure. |
| `--effect_batch_reads` | 16 | Maximum complete reads grouped for effect preparation. |
| `--effect_batch_bases` | 65536 | Base budget for a preparation group; a longer read is handled alone. |
| `--effect_block_size` | 4096 | Scoring tile size; smaller tiles reduce working memory. |
| `--effect_cache_mb` | 256 | MiB retained for reusable effect tensors; 0 disables the cache. |
| `--effect_queue_size` | 8 | Maximum outstanding reads sent to CPU decoding. |
| `--effect_queue_mb` | 256 | MiB of accounted queued scores and metadata, including the active read. |
| `--effect_decoder` | `auto` | Numba when available and enabled; `python` selects the reference decoder. |

The fitted model, canonical/noise tables, current tensors, Python overhead,
and output buffers use memory in addition to these budgets. A read exceeding
the queue byte limit fails with a request to raise it. Decoder traceback storage
scales with read length times effective beam width. Tiles do not split a read
into independent decoding problems or change the footprint radius.

Numba is installed with PoreFM. Its first use can take longer while kernels
compile. `--effect_decoder numba` requires it to be available with JIT enabled;
compilation/runtime errors are reported. The selected decoder is recorded in
`effect_run.json` as `decoder_backend`.

GPU effect scoring uses float32. CPU/GPU reduction order can produce small
rounding differences and change near-tied calls. For controlled comparisons,
keep the model, penalty, neighbors, search settings, batching, and software fixed.

Output-buffer controls are described in [prototype output files](prototype-outputs.md).
