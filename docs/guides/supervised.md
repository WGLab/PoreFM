# Supervised modification detection

Use a supplied fine-tuned classifier to call its trained modification classes
from contextual PoreFM representations. RNA classifiers are available for
m6A/inosine and pseudouridine; see [Models](../models.md) for supported classes.
The first recipe scores **A positions in the called read sequence** and writes
both read-level Parquet predictions and a BAM with modification tags.

First follow [input and model preparation](inputs.md). You need
`basecalls.bam`, `pod5/`, and `reference.fa` with its `.fai`. The command
downloads and caches `WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0`. See
[Models](../models.md#prefetching-and-offline-use) to prefetch it for offline use.

```text
basecalls.bam + POD5
        |
        v
  signal/sequence chunks
        |
        v
  fine-tuned classifier <--- model.pt (weights + class metadata)
        |
        +--> output.bam                 MM/ML modification tags
        `--> prediction Parquets -----> class counts/fractions at each site
                              reference.fa supplies aligned coordinates
```

## Basic usage

Provide the sample directly on the command line:

```bash
porefm detect-modification \
  --bam_path basecalls.bam --pod5 pod5/ \
  --read_based --ref reference.fa --motif 'A 0' \
  --model WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0 \
  --output_format both \
  --output outputs/supervised
```

`--read_based` selects motifs in basecalls, and `--ref` retains available
genomic coordinates for aligned reads. `A 0` selects the A itself. Use a
selector appropriate to the released model's target base. For an SG-tagged
signal BAM, omit `--pod5`.

The checkpoint supplies preprocessing settings, class labels, the canonical
base, and modification codes. `--output_format both` writes predictions and
a BAM with modification tags. For a local checkpoint, replace `--model` with
`--model-path models/supervised/model.pt`; `--model_load_path` remains supported.

Without `--read_based`, targets come from the reference: supply `--ref` and
either `--motif` or `--pos_file`. Use `--contigs chr1 chr2` (also `--chrom`) to
restrict contigs. Direct sample options accept hyphenated spellings too, such
as `--bam-path`, `--read-based`, and `--pos-file`.

Relative input paths are resolved from the directory where you run the command.
`run_manifest.json` records the direct options and their normalized defaults
in `arguments.resolved_input`, alongside the model release provenance.

## Pseudouridine detection

Use the pseU model to classify canonical U versus pseudouridine in RNA004 reads:

```bash
porefm detect-modification \
  --bam_path basecalls.bam --pod5 pod5/ \
  --read_based --ref reference.fa --motif 'T 0' \
  --model WGLab/PoreFM-RNA_pseU@v0.1.0 \
  --output_format both \
  --output outputs/supervised-pseu
```

RNA uridine is encoded as `T` in BAM sequences, so target `T 0` even though
the model's canonical class is named `U`. The saved class order is `U, pseU`;
the checkpoint supplies base `T` and modification code `17802` for BAM tags.
Keep each classifier's output in its own directory. The
[HeLa walkthrough](../examples.md#call-m6a-inosine-and-pseu-with-supervised-models)
shows both models and plots their calls together.

## JSON input alternative

JSON configs remain supported. Save the following as `sample.json` for the
same sample and A-base target selection as the first command:

```json
{
  "config_type": "supervised",
  "schema_version": 1,
  "labelled_samples": [
    {
      "bam_path": "basecalls.bam",
      "pod5": "pod5/",
      "read_based": true,
      "ref": "reference.fa",
      "motif": "A 0"
    }
  ]
}
```

`labelled_samples` selects prediction targets; known modification labels are
not required for inference.

```bash
porefm detect-modification \
  --bam_config sample.json \
  --model WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0 \
  --output_format both \
  --output outputs/supervised
```

Use either `--bam_config` or direct sample settings. They share validation and
defaults, and cannot be mixed. The global `--read_based` override for
JSON configs remains supported. JSON is useful for multiple samples with
different settings or tagged label sources. Multiple SG BAMs with shared
settings can also be passed as `--bam_path first.bam second.bam`; POD5 input
requires one BAM file. See the [input option mapping](../reference/inference-inputs.md#direct-supervised-inference-inputs)
for additional filters and polishing options.

## Read the results

```text
outputs/supervised/
|-- output.bam
|-- checkpoint_*_data.parquet  # Position-level logits and probabilities
|-- checkpoint_*_meta.parquet  # Read identities and alignment metadata
|-- run_manifest.json
`-- *.pfmi                    # Automatic BAM read-ID index, if needed
```

Keep each data/meta pair together. The BAM produced from POD5 input is
unsorted. For coordinate-based viewers or downstream tools, create a sorted,
indexed copy:

```bash
samtools sort -@ 4 -o outputs/supervised/calls.sorted.bam \
  outputs/supervised/output.bam
samtools index outputs/supervised/calls.sorted.bam
```

After inference finishes, summarize the Parquets without loading a model:

```python
from pathlib import Path
from porefm.outputs import open_output

Path("summaries").mkdir(exist_ok=True)
calls = open_output("outputs/supervised")
calls.summarize_loci(
    include_class_counts=True,
    mean_probabilities=True,
    output_path="summaries/supervised.parquet",
    threads=4,
)
```

Each reference site has `called_count`, `count_LABEL`, and `freq_LABEL`
columns using the checkpoint's actual labels. Frequencies count the most
probable class per call; mean probabilities are a separate statistic. Retain
canonical and modified calls when computing fractions. These are model call
fractions, not automatically calibrated modification stoichiometry.

See [processing outputs](outputs.md) for previews, coordinates, and filtering.

## Optional settings

??? example "Expanded command with window, filtering, and resource settings"

    Use this alternative when you want explicit signal windows and more loader
    workers. It also disables the default minimum read quality and length filters,
    so its retained calls can differ from the basic command.

    `--size`, `--step`, and `--num_trim` set signal windows and edge trimming.
    `--batch_size`, `--reads_per_chunk`, and loader settings control resource use;
    reduce the batch size if GPU memory is limiting. `--checkpoint_batch_step`
    controls prediction writes, not model checkpoints.

    ```bash
    porefm detect-modification \
      --bam_path basecalls.bam --pod5 pod5/ \
      --read_based --ref reference.fa --motif 'A 0' \
      --model WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0 \
      --size 4096 --step 3400 --num_trim 8 \
      --checkpoint_batch_step 500 --reads_per_chunk 10 \
      --dataloader_workers 8 --batch_size 256 --prefetch_factor 2 \
      --min_qscore 0 --min_read_length 0 \
      --output_format both \
      --output outputs/supervised
    ```
