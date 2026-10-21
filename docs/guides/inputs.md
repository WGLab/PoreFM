# Prepare inputs and models

The application guides use RNA004 examples and supplied, already trained
models. Start with an aligned basecall BAM, its original POD5 signal, and the
same reference FASTA used for alignment. Replace the example filenames with
your own files. Run commands from your analysis directory in Bash, with
PoreFM installed and your GPU environment active.

## Choose the input path

```text
Aligned basecalls.bam + original POD5 signal + reference.fa
    |
    +-- read directly --------------------> Supervised detection
    |                                  +--> Zero-shot detection
    |                                  +--> Residual prototype scoring
    |
    +-- attach signal --> sort --> index -> Pairwise comparison
                          signal.bam       (test and control)
```

An ordinary basecall BAM alone does not contain the raw signal PoreFM needs.
For the first three guides, PoreFM joins BAM records to POD5 reads by read ID.
`--pod5` accepts one file or a directory searched recursively. Supervised
detection also accepts these inputs through JSON. Use one BAM and one POD5 source per
run, without duplicated read IDs across POD5 files.

Direct BAM + POD5 input does not require a coordinate-sorted BAM or a BAI/CSI
index. PoreFM creates or reuses a read-ID index (`.pfmi`) automatically. Keep
the alignment in the BAM to obtain reference-site summaries.

Index the reference once:

```bash
samtools faidx reference.fa
```

The reference sequence names must match the BAM. The examples keep outputs
under `outputs/` and summaries under `summaries/`. Choose a fresh output
directory for each run; see [output safeguards](../run-output-safety.md).

## Move tables

Include `--emit-moves` when basecalling with Dorado and preserve the resulting
`mv` tags during alignment. These tags describe how basecalls map to signal
steps; see [Dorado's move-table documentation](https://software-docs.nanoporetech.com/dorado/latest/basecaller/move_table/).

PoreFM's default chunking uses move tables when present. Without them, direct
BAM + POD5 inference can infer chunks heuristically from the sequence and
signal. This fallback has given similar results in our analyses, but it does
not guarantee identical calls. These guides use RNA004; the RNA002 profile
requires move tables. See [preprocessing](../preprocessing.md) for details.

## Choose a model

The analysis guides use `--model OWNER/REPOSITORY@vVERSION`, for example
`--model WGLab/PoreFM-RNA_prototype@v0.1.0`. PoreFM downloads the matching
checkpoint once, verifies its SHA-256, and reuses its local cache. No Hugging
Face login or package is required. See [Models](../models.md#downloading-and-using-models)
for releases, checksums, cache configuration, and prefetching for offline jobs.

You can instead supply a local checkpoint with `--model-path` or `--model_path`.
For example, a manually downloaded RNA model directory might contain:

```text
models/
|-- supervised/model.pt      # Complete modification classifier
|-- RNA004_zero-shot.pt      # Backbone, null/scale models, whitening, calibration
|-- RNA004_prototype.pt      # Backbone and fitted residual effects
`-- backbone.pt              # Pretrained backbone for pairwise comparison
```

The released checkpoints supply sequence type, pore chemistry, and expected
k-mer currents through metadata. The classifier also supplies its canonical
base, class labels, and modification codes. The recipes therefore need no
manual chemistry, k-mer-table, or modification-code arguments.

Zero-shot and prototype bundles contain their matching model components.
Separate-file inputs remain supported for custom models: keep zero-shot
artifacts from the same release together, and pair fitted residual effects
with their exact backbone checkpoint. A model trained for a different
chemistry is not a drop-in replacement.

## Prepare signal BAMs for pairwise comparison

Skip this step for the three direct BAM + POD5 guides. For pairwise comparison,
prepare one signal BAM for each population. Here `control.bam` and `test.bam`
are aligned basecall BAMs, with their POD5 files in corresponding directories.

The current `pod5-to-bam` converter requires move tables and **skips reads
without `mv` tags**. Use basecalls produced with `--emit-moves` for this step.

```bash
mkdir -p signal
for sample in control test; do
  porefm utils pod5-to-bam \
    --bam "${sample}.bam" --input "pod5/${sample}" \
    --output signal --prefix "${sample}.unsorted" --threads 4
  samtools sort -@ 4 -o "signal/${sample}.signal.bam" \
    "signal/${sample}.unsorted.bam"
  samtools index "signal/${sample}.signal.bam"
done
```

Signal attachment writes an unsorted BAM containing `SG` signal tags. Sorting
and indexing make it ready for the coordinate-based pairwise workflow. Keep
both populations aligned to the same reference.

Continue with [supervised detection](supervised.md),
[zero-shot detection](zero-shot.md), [pairwise comparison](pairwise.md), or
[residual prototype scoring](residual-prototypes.md).
