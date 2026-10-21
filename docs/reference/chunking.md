# Signal chunking

For ordinary inference, keep the default `auto` mode. This reference explains
when to select another method and how boundaries affect model inputs.

## Chunk-boundary selection

Data-consuming commands expose `--chunking_mode auto|mv|hierarchical|dtw|inferred`.
Omission defaults to `auto`: use move-based decoding when `mv`
exists, hierarchical anchors for DNA without moves, and mean6 DTW for RNA
without moves. Auto does not hide errors in malformed present move tags.

Explicit `mv` requires move tags.
Explicit `hierarchical` or `dtw` forces that method for either DNA or RNA,
even when moves exist.

Alternatively, pass `--chunking_config /path/to/chunking.json` with:

```json
{"chunking_mode": "dtw"}
```

This is a separate, run-wide data-processing JSON file, not a BAM manifest or
model configuration. An explicit `--chunking_mode` overrides the JSON value.
It applies to pretraining, supervised training, classifier inference,
canonical-null workflows, prototype extraction/scoring and `pairwise-comparison`.
Dataset/factory Python APIs accept the same modes, defaulting to
`chunking_mode="auto"`.

Both inferred methods support primary/unmapped DNA/RNA A/C/G/T reads with
the appropriate 9-mer current model. They use positive integer `ts` as the
signal start and `ns` as the exclusive end. A zero or absent `ts` triggers start estimation;
an absent `ns` triggers end estimation. If both are missing, both endpoints
are estimated. Character-valued `ts:A` alignment tags are ignored. Invalid
numeric bounds raise an error. Secondary/supplementary records are excluded.
Tags always describe acquisition-time coordinates; RNA signal is reversed
after trimming to follow original read sequence order. This does not depend
on the alignment strand. Other k-mer model sizes remain supported by mv only.

Both methods use six-sample non-overlapping means and quantile-based affine
calibration for matching. The defaults are:

- `hierarchical`: 192-base interior anchor queries, approximately 256-base
  anchor spacing, endpoint queries up to 384 bases, and piecewise interpolation.
  RNA search, duration and cost bounds scale with estimated signal dwell.
- `dtw`: whole-read one-to-many DTW with a sloped half-band of 1,000 averaged
  bins (6,000 raw samples), squared-current error, rolling cost rows and byte
  traceback pointers. Known ts/ns endpoints bypass anchor searches. Missing
  endpoints are estimated by the hierarchy before DTW. The traceback is capped
  at 256 MiB per read; oversized or unreachable alignments raise an error,
  without silently switching methods. Hierarchical is available for very long
  reads that exceed this allocation limit.

The model receives original signal samples at its configured row width.
Fewer than one row of terminal samples may be dropped to preserve that width.
Inferred boundaries do not supply move tables or measured per-base dwell times.
