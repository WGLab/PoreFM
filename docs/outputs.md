# Reading and summarizing PoreFM outputs

For a short introduction with previews and links to workflow-specific
summaries, start with [processing output files](guides/outputs.md).

`porefm.outputs` provides one entry point for read-level calls and reference-site
summaries. It runs without loading a model or allocating a GPU. Analysis
dependencies are included in the standard PoreFM installation.

```python
from porefm.outputs import open_output

calls = open_output(output_dir)  # format="auto" by default
print(calls.format, calls.schema)
```

| Format | Accepted layout | Basic site summary |
| --- | --- | --- |
| `canonical_null` | Paired `part_*_data.parquet` / `part_*_meta.parquet` | Valid p-value counts, anomaly fractions, Fisher combined p-value |
| `pairwise` | Comparison run with `read_scores/` | P-value counts and anomaly fractions |
| `prototypes` | Effect manifest plus family-local data/meta, or per-k-mer classifier parts | Hard-call frequencies and called/total counts |
| `supervised` | `checkpoint_*` or `rank_*_checkpoint_*` data/meta | Argmax frequencies and called/total counts |

Directories are inspected nonrecursively. Comparison roots automatically select
`read_scores/`; sibling site statistics are not treated as read observations.
Flat input can also be a file, glob, or explicit file list. Split file paths
resolve their companions. Mixed layouts and inconsistent schemas raise errors.
Use one inference/comparison run per dataset. Rows are not deduplicated.

Common logical column names are `readname`, `chrom`, `position`, `read_pos`,
`strand`, and `pvalue` where available. Stored `refname`, `refpos`,
`read_position`, and `p_value`/`p_val` are renamed to those common names. Other
fields retain their producer names and precision. Reference positions are
zero-based, strand is 0 forward/1 reverse, and read coordinates retain the
producer's orientation. No reference coordinates are inferred for unmapped
positions. Negative positions remain visible in read-level output.

For [SQL queries, specialized pairwise diagnostics, and label selection](reference/output-analysis.md),
see advanced output analysis.

## Filtered read-level access

For a single read, a set of read IDs, or a preview:

```python
filters = {
    "readname": ["read-id-1", "read-id-2"],
    "call_status": "called",
    "predicted_class": ["A", "6mA"],
}

for table in calls.iter_tables(
    columns=["readname", "chrom", "position", "read_pos", "predicted_class"],
    filters=filters,
    reads_per_batch=128,
):
    process(table)  # release each batch before advancing
```

Canonical-null, prototype and supervised readers default to **32 complete stored
reads** per batch. Set `reads_per_batch=N` to change this; rows and RAM usage vary
with read length. Metadata `length` values define split-file records.
Filtering keeps all matching stored rows of a read together; it does not restore
bases excluded during inference. Reads with no matches do not count toward the
batch size. Batches may span parts; only the final batch has fewer reads.
Repeated read names from duplicate inputs or separate alignments are not merged.

For strict row limits use `batch_size=N` instead (mutually exclusive with
`reads_per_batch`). Pairwise readers retain row batching, default 65,536 rows,
because genomic-block files can divide a read across parts; `reads_per_batch`
is not supported for pairwise output. DuckDB views do not impose read order.

Filters combine with AND; a list value means any of those values. An empty list
selects no rows. Filter fields need not appear in `columns`. Classification
filters apply only to formats that have class calls; use `pvalue`, `chrom`,
`readname`, etc. for null/pairwise results. Unknown column names raise. A class
label absent from the dataset matches no rows in the common API.

Comparison operators use a list of triples:

```python
filters = [
    ("chrom", "==", "chr1"),
    ("position", ">=", 1_000_000),
    ("position", "<", 1_100_000),
    ("pvalue", "<=", 0.05),
]
```

Supported operators are `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`.
Use `== None`/`!= None` for nulls. NaN is distinct from null. Both Arrow and
native SQL filters use Arrow's NaN comparison semantics: ordered comparisons
and equality to finite values are false; inequality is true.
Selecting only needed columns reduces decoding and memory use. Metadata for
one part is also held in memory. No-match results retain their schema.

For medium datasets, load Arrow once and convert to pandas once:

```python
import pandas as pd

table = calls.read_table(columns=columns, filters=filters)
df = table.to_pandas(types_mapper=pd.ArrowDtype)
```

Arrow concatenation shares batch buffers. All selected data still needs to fit
in RAM; do not accumulate batches if the dataset is too large. Packed null
`sequence_context` is decoded to text only when selected. Pass
`decode_context=False` to retain packed bytes and their encoding/length metadata.

Supervised calls derive `predicted_class`, `probability`, and `call_status` from
stored `prob_*` columns.
All stored logits/probabilities remain available. Ties pick the first stored
class; any missing, nonfinite or out-of-range probability makes the row
`unavailable_context`. Prototype score families are expanded with nulls for
inapplicable classes. Prototype output provides scores, not probabilities.

## Inspect an active job

```python
calls = open_output(output_dir, allow_incomplete=True)
```

This snapshots finished pairs. Reopen to include later parts. Prototype effect
manifests provide completion and stale-file checks. Other split layouts do not
have a completion manifest: strict mode requires both files for every discovered
stem; incomplete mode skips unpaired stems. A footer/schema error is surfaced,
not silently skipped. The supervised writer does not publish atomically, so
retry opening if its current file is still being written. Do not restart or
overwrite an input directory while a reader is using it.

## Collapse reference loci without loading all reads

```python
sites = calls.summarize_loci(
    filters={"call_status": "called"},  # classification formats
    classes=["A", "6mA", "C", "5mC", "5hmC"],
    memory_limit="8GB",
    threads=4,
)

# Or stream only the aggregated site batches to a separate Parquet file:
calls.summarize_loci(
    output_path=site_output_path,
    memory_limit="8GB",
    temp_directory=spill_directory,
    threads=4,
)
```

All summaries exclude `position < 0` (including -1 and -2), missing/`*`
chromosomes, and strands other than 0/1. Default grouping is
`chrom, position, strand`. An entirely reference-free run therefore returns an
empty site table. DuckDB aggregates across parts without retaining the full
read table in Python. Only site batches cross into Python. Memory limits govern
DuckDB operations, not every allocation or the size of a returned pandas table.
Keep site output outside the read-part directory so it cannot be mistaken for
another input part. Existing outputs require `overwrite=True`; input part files
cannot be overwritten by the shared summary function.

Summaries process read data in parts and may use temporary disk space under
`temp_directory` (or the system temporary directory). Temporary files are
cleaned up automatically. Millions of final loci can still form a large pandas table;
use `output_path` to write the final site batches directly to Parquet.

Classification summaries return `total_count`, `called_count`, and `freq_LABEL`.
The denominator is **all retained non-null class calls**, including classes not
listed in `classes`. That option chooses report columns, not a filter. Filtering
to modified classes alone changes the denominator and cannot estimate a
canonical-versus-modified fraction. Request `include_kmer=True` for a k-mer mode
with lexical tie breaking, or `group_by_kmer=True` to group by k-mer. Neither
reconstruction nor mode calculation runs by default. Request
`mean_probabilities=True` for supervised `mean_probability_LABEL` among called
rows, using the stored class probabilities.

For supervised output with exactly two classes, set a binary decision threshold
independently of row filters:

```python
calls = open_output(output_dir)
sites = calls.summarize_loci(
    binary_threshold=0.8,
    binary_positive_class="5mC",  # Exact stored class label; required.
    include_class_counts=True,
    threads=4,
)
```

Each valid call becomes 5mC when `P(5mC) >= 0.8`, otherwise the other stored
class (for example C). Both remain in `called_count`, so
`freq_5mC = count_5mC / called_count`. For probabilities `[0.2, 0.6, 0.8, 0.9]`
at one site, `count_5mC=2`, `called_count=4`, and `freq_5mC=0.5`.
Invalid probability rows stay no-calls and contribute only to `total_count`.
The threshold accepts finite numbers in `[0, 1]`, with an inclusive boundary.
It requires exactly two classes in the selected probability source, even if
`classes` requests only one report column. The positive class is explicit and
does not depend on column order. Omitting both options preserves argmax.

This decision applies only to summary class counts/frequencies; it does not
change read-level calls, stored probabilities, or probability means. The
`filters` run first with their original semantics, including argmax-derived
`predicted_class` and winning `probability`. Filtering on `prob_5mC >= 0.8`
would instead remove other reads from the denominator. The threshold supports
both per-part aggregation and direct Parquet summary output.

Prototype summaries can additionally report frequencies within each canonical
family and integer per-class counts:

```python
sites = calls.summarize_loci(
    filters={"chrom": "chr2"},
    relative_frequencies=True,
    include_class_counts=True,
)
```

For 10 U calls, 10 pseU calls, 20 C calls, and 10 no-calls at one site,
`total_count=50`, `called_count=40`, `count_pseU=10`, `freq_pseU=0.25`, and
`freq_pseU_rel=0.5`. The relative denominator includes **all called members of
the U family**, including other U modifications if present. It excludes other
families and no-calls; zero family coverage produces null/NaN. These are
prediction-family counts, not reference-base counts or deduplicated read IDs.
Selecting `classes=["pseU"]` restricts reported columns without restricting the
denominator. Read filters still change both denominators.

Both options default off. `include_class_counts=True` also works for supervised
outputs. Effect split outputs supply family assignments in their manifest.
Canonical T/U labels share a family. These options do not require reading
scores, reconstructing k-mers, or loading read-level data into pandas.

Use `mean_scores=True` to add prototype `mean_score_LABEL` columns:

```python
sites = calls.summarize_loci(
    classes=["pseU", "m6A"],
    include_class_counts=True,
    relative_frequencies=True,
    mean_scores=True,
)
```

`mean_score_pseU` averages the scores of calls **predicted as pseU** at that
locus. It does not average pseU candidate scores from U or other predictions.
Null, NaN and infinite scores are excluded from the mean without changing
`count_pseU`; no finite pseU scores gives null/NaN. Filters apply before the
mean, so a score threshold restricts the observations being averaged.
The option defaults off; enabling it reads score payloads and retains bounded
per-part aggregation, merging score sums and valid counts before division.
Effect summaries use the winning-class score. Scores retain their
original scorer semantics and are not converted to probabilities.

For canonical-null and pairwise summaries:

```python
null_calls = open_output(null_output_dir)
sites = null_calls.summarize_loci(
    modified_pvalue_cutoff=0.05,
    min_pvalue=1e-300,
    # Optional additions; omit for the lightweight defaults:
    pvalue_thresholds=(0.001, 0.01, 0.05),
    confidence=0.95,
    metric_columns=["distance"],
    metric_stats=("mean", "std"),
    threads=4,
)
```

`total_count` counts finite p-values in [0, 1]; `n_observations` includes invalid
p-values and `n_invalid_p` reports their count. `fraction_modified` uses the
**inclusive** cutoff (`p <= cutoff`). `combined_neglog10_pvalue` uses Fisher's statistic with the
given p-value floor/cap; zero valid observations produce null. Fisher combination
defaults on for canonical-null and off for pairwise; `combine_pvalues=True/False`
overrides this. Optional `pvalue_thresholds` add **strict** cutoff counts and
fractions. `confidence` enables Wilson bounds for primary and extra fractions.
Numeric summaries default off (`metric_stats=()`); request any of `count`, `mean`,
`std` (sample deviation), or exact `median`. `metric_columns` selects metrics,
otherwise available distances/PCs are discovered only when statistics are requested.
Exact medians can require substantial memory. Uncalibrated null output without
p-values returns observation counts plus requested numeric statistics. No conversion of anomaly fractions
or Fisher significance into calibrated modification proportions is implied.
