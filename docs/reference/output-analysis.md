# Advanced output analysis

Start with the [output guide](../guides/outputs.md) for a preview and a site
summary. This page covers SQL, specialized pairwise diagnostics, and selecting
labels for downstream analyses. All examples read completed output files.

## DuckDB

This example uses a completed prototype run. Adjust its path and class labels
to your model:

```python
from porefm.outputs import open_output
calls = open_output("outputs/prototype")
```

```python
import duckdb

with duckdb.connect() as con:
    con.execute("SET memory_limit='8GB'")
    calls.register(con, name="calls", filters={"call_status": "called"})
    df = con.sql("""
        SELECT chrom, position, predicted_class
        FROM calls
        WHERE readname = 'read-id-1'
    """).df()
```

These are reusable Parquet views. A split-file query
can do substantial join work even with `LIMIT`; use `iter_tables` for a quick
sequential preview. SQL leaves packed contexts as bytes. View rows have no
guaranteed order; use `ORDER BY` if needed.

## Pairwise diagnostics

Specialized pairwise diagnostics are available from `porefm.outputs`:

```python
from porefm.outputs import (
    load_pairwise_directory, summarize_pairwise_loci, add_reference_base,
    iter_pairwise_locus_summaries,
)

# Medium comparison runs use the refname/p_value column names.
pairwise_output_dir = "outputs/pairwise"
reads, site_metrics = load_pairwise_directory(pairwise_output_dir)
loci, annotated_reads = summarize_pairwise_loci(reads, site_metrics)

# Large runs: preserve exact quantiles, PCA fits and cross-fit diagnostics
# while holding only one complete partition at a time.
for stem, loci, _ in iter_pairwise_locus_summaries(
    pairwise_output_dir, return_annotated=False,
):
    consume(loci)

loci = add_reference_base(loci, reference_fasta)
```

These specialized functions use `refname` and `p_value` column names.
Grouped results exclude negative reference positions. Advanced
fits need the whole current partition in RAM; they cannot be combined from
arbitrary partial-locus batches. Loci repeated across partitions raise an error.
They preserve control PCA covariance checks and read/locus duplicate detection.
The simpler shared SQL summary does not compute exact quantiles or refit PCA.

Site tables written by the comparison can be read without scanning read rows:

```python
comparison = open_output(pairwise_output_dir)
site_metrics = comparison.read_site_table(kind="anomaly_site_metrics")
# comparison.iter_site_tables(...) provides bounded batches instead.
```

These select only companions of the snapshot's read parts. They contain original
producer coverage/covariance, not values recomputed after read filters.
`add_reference_base` accepts `chrom` or `refname`, complements bases
on reverse-strand sites, and retains zero-based genomic coordinates.

## Select read-position labels with bounded intermediates

`select_quantile_labels` and `select_threshold_labels` accept an existing
`open_output` snapshot and return sorted, deduplicated `readname`, `read_pos`,
`label` triples. Both reconstruct one source part at a time. They apply class,
status, and optional chromosome filters before selection, exclude nonfinite
scores, and use strict comparisons: positive scores above the cutoff and
negative scores below it. Class names, score columns, and output label IDs are
configurable; positive and negative class sets must be disjoint. The example
below assumes a DNA prototype run with C, 5mC, and 5hmC classes; adapt the labels
and score column for other models.

```python
from porefm.outputs import select_quantile_labels, select_threshold_labels

common = dict(
    score_column="score_5mC",
    positive_classes=["5mC", "5hmC"], negative_classes=["C"],
    chroms=None, positive_label=1, negative_label=0,
    threads=4, memory_limit="16GB",
    temp_directory="/path/to/node-local-scratch", progress=True,
)

final_df = select_quantile_labels(
    calls, **common, positive_quantile=.75, negative_quantile=.05,
    quantile_partitions=32,
)

# An output path avoids constructing the final pandas DataFrame as well.
label_path = select_threshold_labels(
    calls, **common, positive_threshold=300, negative_threshold=-300,
    negative_ratio=1.0, random_state=42,
    output_path="labels.parquet",
)
```

The quantile helper's first pass writes only group/cohort/score to temporary
Parquet, partitioned by a hash of `(group, cohort)`. Each partition pools all
source parts and computes exact interpolated quantiles. The second source pass
applies those thresholds and writes only selected labels. This preserves global
per-k-mer quantiles, including duplicate contributions before selection; it does
not average per-part quantiles or use approximate quantiles. `group_column`
defaults to `kmer`. More `quantile_partitions` reduce simultaneous aggregate
state at the cost of additional files; an exceptionally large individual group
still needs memory for its exact quantile state.

The fixed-threshold helper never projects k-mers or sequences. It writes eligible
label triples per part, then samples `floor(negative_ratio * n_positive)` negative
rows without replacement **before** deduplication. `negative_ratio=None` retains
all negatives; insufficient negatives raise. An integer seed makes the negative-row sampling reproducible for fixed inputs. The permutation requires 8 bytes of temporary disk per
eligible negative on a 64-bit system; mapped pages are managed by the OS.

Each invocation owns a unique work directory under `temp_directory`, including
its DuckDB spill directory, and cleans it on normal completion or exceptions.
When omitted, the parent is the system temporary directory rather than `.tmp`
under the working directory. Choose disk-backed node-local scratch with enough
space; NFS spill directories can fail with stale file handles. Abrupt process
termination can leave temporary files behind.

The final global sort/deduplication can spill to disk. Returning pandas still
requires RAM for the final result; `memory_limit` is not a hard process-RAM cap.
With `output_path`, sorted Parquet is published atomically and its absolute Path
is returned. The destination must not exist and must be outside input-part
directories. Input files must remain unchanged during a call, especially the
quantile helper's two passes; newly published parts are excluded from the fixed
snapshot.
