# Prototype output files and BAM tags

For everyday analysis, start with [processing outputs](../guides/outputs.md).
This reference describes stored fields, completion checks, and optional BAM tags.

## Output and safeguards

All input chunks participate in inference, even if `--pos_file` or `--motif`
requests output for only one base family. Chunks are stitched by greatest edge
margin **before** classification; selection labels never identify the true
neighbors. In aligned mode, inserted and soft-clipped bases can supply observed
embeddings and read-derived contexts even though they have no reference
coordinate. Only reported rows require an aligned reference position; an
insertion alone does not invalidate neighboring calls. Neighborhood-statistics
extraction still applies its exact read/reference-match and base-quality filters.
`effect_run.json` records `observation_validity=observed_read_tokens_independent_of_reference_alignment`.

A candidate needs complete source/canonical sequence contexts and
valid observations throughout its footprint. Missing contexts give class index
`-1`, null scores, and `call_status=unavailable_context`, not a canonical call.
This includes missing empirical support when its configured policy is no-call.
Consequently the center-only ablation has more callable edge positions; compare
methods on their shared callable positions.

By default, inference writes paired `part_XXXXXX_data.parquet` and
`part_XXXXXX_meta.parquet` files using `effect_family_meta_data_v1`.
`effect_parts.json` records class labels, the local class order for each base
family, completed file pairs, and counts. Its `complete` flag is set only after
all writes succeed. Readers must use its part list, rather than glob all Parquet
files, to exclude stale files from explicitly overwritten runs.

The metadata file stores `readname`, `chrom`, `strand`, the full oriented read
`sequence`, `inference_converged`, and `length` once per read. `length` partitions
the flattened data rows in metadata order; it counts selected output positions,
not all bases in the sequence. Reads with no selected sites have no output row.
The data file stores `read_pos`, `position`, `family_index`,
`predicted_class_index`, `isolated_class_index`, `neighbor_count`,
`canonical_energy`, and float32 `scores` lists. Family indices
0–3 mean A/C/G/T, with -1 for an unrecognized base. For example, a C-centered
call stores only `[C, 5mC, 5hmC]` if those are its fitted classes; it stores no
A/G/T/6mA placeholders. An A-centered call may store `[A, 6mA]`. The actual
class order comes from `family_class_indices` in the manifest, not this example.
No-call rows have class indices -1, empty lists, and null canonical energy.

`--effect_write_rows 4000000` targets four million rows per part. Flushes occur
between complete reads. `--effect_write_mb 512` independently bounds the current
Arrow buffer and all queued/in-flight writer payloads; compression workspaces
and Python/allocator overhead are additional. A read larger than the byte limit
fails with a request to raise it. `--effect_writer_threads 2` allows concurrent
part writers; use 1 when a second writer does not help the storage system.
Write failures leave an incomplete manifest.
Files use lossless Zstandard level 1 compression, byte-stream splitting for
float payloads, and dictionary encoding for low-cardinality integer fields.
These storage encodings preserve the float32 values exactly.

## Read and summarize calls

Use `porefm.outputs.open_output` for previews, filters, DuckDB queries, and
site summaries. Start with the [prototype summary recipe](../guides/residual-prototypes.md#summarize-calls-within-each-base-family)
and see the [shared output API](../outputs.md) for complete options.

The reader reconstructs class names, k-mers, per-class scores, and call status
from the family-local layout. Off-family scores are null. Site summaries exclude
negative reference positions, missing chromosomes, and invalid strands. Keep
canonical calls in the denominator when calculating modification-call fractions.

For an active run, `open_output(path, allow_incomplete=True)` takes a snapshot of
finished file pairs. Reopen to see later parts. Completed runs use the manifest's
exact part list. Do not overwrite a directory while reading it.

The reconstructed `isolated_class`
uses that candidate's full footprint without any *other* events, not the
center-only ablation. `canonical_energy` is the center's uncorrected whitened
residual norm squared and is **not** a calibrated null p-value. Parquet output
contains scores rather than probabilities. Optional BAM output converts the
scores as described below. Scores
remain uncalibrated. A high canonical
energy with poor known-effect fit may indicate an unknown modification or
noise; the model does not assign a calibrated unknown class.

Artifacts record the manifest, input paths/metadata, BED hashes, and backbone
SHA256; inference also records the fitted artifact hash and run arguments.
Use the same backbone for prototype extraction and query inference; checkpoint
hash mismatches fail validation. Never load untrusted `.pt` pickle artifacts.

## Optional MM/ML BAM output

Add these options to an existing `prototype classify --scorer residual-effects`
command to write both the usual score Parquet files and `output.bam` under
`--output_dir`:

```bash
--output_format both --bam_threads 4 --effect_probability_temperature 1.0
```

Use `--output_format bam` for BAM alone; `parquet` remains the default.
The BAM output uses this RNA mapping from
[Dorado](https://software-docs.nanoporetech.com/dorado/latest/basecaller/mods/):

| Artifact modification label | SAM code |
| --- | --- |
| m5C | C+m |
| m6A | A+a |
| inosine | A+17596 |
| pseU | T+17802 |
| 2OmeC | C+19228 |
| 2OmeA | A+69426 |
| 2OmeG | G+19229 |
| 2OmeU | T+19227 |

Labels are case-insensitive (`Inosine` is accepted); `5mC` and `6mA` are also
accepted aliases. The fitted artifact supplies the class order and canonical
families. Unknown modification labels, mismatched families, and duplicate
codes fail before inference. Only modifications present in the artifact are
emitted. RNA uridine groups use T exactly as in the table.

For each callable selected position, probabilities are
`softmax(family_scores / effect_probability_temperature)`, including the
canonical score in the denominator. Temperature must be finite and positive;
the default is 1.
It affects only BAM probabilities, not decoding, predicted labels, or Parquet
scores. This normalization holds inferred neighbors fixed and is **not a
calibrated posterior or a marginal over beam paths**.

Every modeled modification gets a separate MM group, including low-probability
alternatives and sites predicted canonical. Canonical probability is implicit
as the remaining mass. Following the
[SAM tag specification](https://samtools.github.io/hts-specs/SAMtags.pdf), ML
stores `min(255, floor(256 * probability))` in group-major order. `MN` records
the full sequence length. Tags use as-sequenced coordinates and `+` even for
reverse alignments. The full source sequence supplies MM delta counts, including
clipped bases that were not scored.

The same motif/position selection used by Parquet determines BAM tag sites.
Unselected and unavailable-context sites are omitted with the MM `?` flag,
meaning unknown, not confidently canonical. Each processed read is retained,
including reads with no callable selected sites (which receive no MM/ML/MN).
Reads rejected by the input filters are not copied to the output.

Sequence, qualities, alignment, and unrelated tags are preserved; input
modification tags, raw signal SG, and move-table mv tags are removed,
as in supervised BAM output. With explicit sequence polishing, output uses
the polished sequence and alignment. Input BAM headers are
merged by reference name and RG/PG ID; conflicting definitions fail explicitly.
Output is **unsorted and unindexed** because workers may interleave reads; sort
and index it separately if needed for regional tools.

`effect_run.json` records SAM mappings, temperature, probability semantics,
read counts, and BAM completion. The writer publishes `output.bam` only after
successful completion; an interrupted attempt leaves `output.bam.partial`.
The usual output-overwrite safeguards apply. During forced reuse, an older
`output.bam` can remain until the new attempt succeeds, so check this run's
completion metadata. Stale indexes are not removed automatically.
