# Compare a test sample with a control

Use this recipe when you have a matched control population, such as an IVT
RNA control, and want to measure how a test sample differs at the same
reference positions. It produces both per-test-read anomaly scores and
differences between the mean test and control embeddings.

Follow [input and model preparation](inputs.md#prepare-signal-bams-for-pairwise-comparison).
You need sorted, indexed `signal/control.signal.bam` and
`signal/test.signal.bam`, and `reference.fa` with its `.fai`.
Both BAMs must contain raw signal (`SG` tags). The command downloads and caches
the backbone from `WGLab/PoreFM-RNA@v0.1.0`. See
[Models](../models.md#prefetching-and-offline-use) to prefetch it for offline use.

```text
control.signal.bam --+
                    +--> same backbone --> embeddings at each position/strand
test.signal.bam -----+                           |
                         +----------------------+---------------------+
                         |                                            |
                    mean embeddings                            joint local PCA
                         |                                     + control null
                  test/control shift                                  |
                         |                                  test-read p-values
                         v                                            v
                  locus_metrics/                                read_scores/
                                                     + anomaly_site_metrics/
```

## Basic usage

This example processes `chr3`. Set `--chrom-list` to the contigs of interest
in your BAM and reference.

```bash
porefm pairwise-comparison \
  --model WGLab/PoreFM-RNA@v0.1.0 \
  --control-bam signal/control.signal.bam \
  --test-bam signal/test.signal.bam \
  --ref reference.fa --chrom-list chr3 \
  --sitewise-anomaly-scoring --sitewise-embedding-shift \
  --output outputs/pairwise
```

`--sitewise-anomaly-scoring` scores individual test reads against the control.
`--sitewise-embedding-shift` also measures the difference between the two
populations' mean embeddings. Both operate at matching reference positions
and strands, without a motif restriction.
For a local checkpoint, replace `--model` with `--model-path models/backbone.pt`.

The basic command requires at least 15 retained observations per population
at a site. See the [pairwise reference](../embedding-comparisons.md) for
coverage settings and multiple BAM inputs.

## Read both types of results

```text
outputs/pairwise/
|-- read_scores/              # Paired data/meta Parquets for test reads
|-- locus_metrics/            # Mean test/control embedding differences
|-- anomaly_site_metrics/     # Coverage and fitted PCA/null statistics
`-- run_manifest.json
```

Summarize test-read anomalies after the comparison completes:

```python
from pathlib import Path
from porefm.outputs import open_output

Path("summaries").mkdir(exist_ok=True)
calls = open_output("outputs/pairwise")
calls.summarize_loci(
    modified_pvalue_cutoff=0.01,
    output_path="summaries/pairwise.parquet",
    threads=4,
)

# Preview the separately computed site-level embedding shifts.
for table in calls.iter_site_tables(kind="locus_metrics", batch_size=5):
    print(table.to_pandas())
    break
```

Here `fraction_modified` means the fraction of retained, valid **test-read**
p-values at each site that are <= 0.01. It measures disagreement with the
control, not the identity or stoichiometry of a modification. Experimental
differences between populations can also create a shift. The companion site
tables retain the original comparison statistics; later filtering of read
scores does not refit them.

See [processing outputs](outputs.md) for coordinates and read-level previews.

## Optional settings

??? example "Expanded command with window, filtering, and resource settings"

    This alternative requires 30 observations per population and fits on up to
    512 observations from each. It uses two principal components and a chi-square
    null. PCA is fitted separately at each position and strand, so PC coordinates
    cannot be compared across sites.

    The command also changes signal windows and read-quality filtering. Batch
    size, worker count, and block size control resource use; reduce
    `--forward-batch-size` if GPU memory is limiting.

    ```bash
    porefm pairwise-comparison \
      --model WGLab/PoreFM-RNA@v0.1.0 \
      --embedding-key contextual_base_seq \
      --control-bam signal/control.signal.bam \
      --test-bam signal/test.signal.bam \
      --ref reference.fa --chrom-list chr3 \
      --sitewise-anomaly-scoring --sitewise-embedding-shift \
      --min-control-coverage 30 --min-test-coverage 30 \
      --max-fit-coverage 512 --n-components 2 --pvalue-method chi2 \
      --size 4096 --step 3600 --num-trim 8 \
      --block-size 1000 --min-read-length 100 --min-qscore 0 \
      --forward-batch-size 16 --num-workers 1 --compile-model \
      --output outputs/pairwise
    ```
