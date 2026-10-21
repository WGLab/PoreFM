# Process output files

All four guides use the same Python entry point, `porefm.outputs.open_output`.
It reads saved predictions without loading model weights or using a GPU.
Use a completed run directory and keep data/meta pairs and manifests together.

```text
completed run directory
        |
        v
   open_output
        |
        +--> iter_tables ------> preview/filter individual read observations
        |
        `--> summarize_loci ---> one row per chromosome, position, strand
                                  |
                                  `--> summary Parquet for plots/analysis
```

## Preview a few observations

```python
from porefm.outputs import open_output

calls = open_output("outputs/zero-shot")
print(calls.format)
print(calls.schema.names)
for table in calls.iter_tables(batch_size=5):
    print(table.to_pandas())
    break
```

Change the path to `outputs/supervised`, `outputs/pairwise`, or
`outputs/residual` to inspect another guide's result. Opening a pairwise run
selects `read_scores/`; its companion site tables are read separately.

## Choose the right site summary

| Workflow | What the summary measures | Example |
| --- | --- | --- |
| Supervised | Hard class counts/fractions, optionally mean class probabilities | [Supervised summary](supervised.md#read-the-results) |
| Zero-shot | Fraction of valid calls with a small canonical-null p-value | [Zero-shot summary](zero-shot.md#summarize-anomalies-at-reference-sites) |
| Pairwise | Fraction of valid test calls that deviate from the control | [Pairwise summary](pairwise.md#read-both-types-of-results) |
| Residual prototypes | Hard class counts, including within-family fractions | [Residual summary](residual-prototypes.md#summarize-calls-within-each-base-family) |

The snippets in those guides write directly to a summary Parquet instead of
loading all read observations into RAM. Use a new summary filename for each
analysis. The default memory budget is 8 GB; `memory_limit` and
`temp_directory` can be set for larger runs.

To preview a saved summary:

```python
import pyarrow.parquet as pq

summary = pq.ParquetFile("summaries/zero-shot.parquet")
for batch in summary.iter_batches(batch_size=5):
    print(batch.to_pandas())
    break
```

## Coordinates and denominators

Reference `position` is **zero-based**. `strand` is **0 for forward, 1 for
reverse**. `read_pos` follows the producer's oriented read sequence; it is not
a genomic coordinate. Site summaries exclude observations without valid
reference coordinates. They summarize retained predictions, so coverage can
be lower than BAM depth after input filtering, context checks, or edge trimming.

Filter by region when needed, but retain the calls that form the denominator:

```python
sites = calls.summarize_loci(
    filters=[("chrom", "==", "chr3"),
             ("position", ">=", 1_000_000),
             ("position", "<", 1_010_000)],
    modified_pvalue_cutoff=0.01,
)
print(sites.head())
```

This small-region example applies to the zero-shot run opened above. For
anomaly fractions, retain both significant and nonsignificant p-values. For
class fractions, retain canonical and modified calls. Filtering to positive
calls first makes the resulting fraction misleading.

The name `fraction_modified` in anomaly summaries means the fraction passing
the chosen p-value cutoff, not a calibrated biological modification fraction.
Likewise, effect scores and supervised probabilities have different meanings
and should not be compared as if they were the same quantity.

For selected columns, read filters, DuckDB queries, and detailed schemas, see
the [output API reference](../outputs.md).
