# Models

Browse all releases in the [PoreFM collection on Hugging Face](https://huggingface.co/collections/WGLab/porefm).
Choose the model for your sequencing chemistry and analysis method.

## RNA

**Chemistry:** RNA004. **Backbone embedding dimension:** 512.

| Model on Hugging Face | Version | Checkpoint files | Description |
| --- | --- | --- | --- |
| [PoreFM-RNA](https://huggingface.co/WGLab/PoreFM-RNA) | 0.1.0 | `backbone.pt`, `pretraining.pt` | Pretrained RNA backbone for embeddings, pairwise comparison, and downstream training; also available with pretraining heads. |
| [PoreFM-RNA_zero-shot](https://huggingface.co/WGLab/PoreFM-RNA_zero-shot) | 0.1.0 | `RNA004_zero-shot.pt` | Canonical-null model with whitening and calibration for detecting deviations from canonical RNA signals. |
| [PoreFM-RNA_prototype](https://huggingface.co/WGLab/PoreFM-RNA_prototype) | 0.1.0 | `RNA004_prototype.pt` | Prototypes learned from small labelled datasets, with joint footprinting for m6A, inosine, m5C, pseudouridine, and 2′-O-methylated A/C/G/U. |
| [PoreFM-RNA_m6A-Inosine](https://huggingface.co/WGLab/PoreFM-RNA_m6A-Inosine) | 0.1.0 | `model.pt` | Supervised classifier distinguishing canonical A, m6A, and inosine. |
| [PoreFM-RNA_pseU](https://huggingface.co/WGLab/PoreFM-RNA_pseU) | 0.1.0 | `model.pt` | Supervised classifier distinguishing canonical U and pseudouridine (pseU). |

## DNA

**Chemistry:** R10.4.1 (E8.2, 400 bps). **Backbone embedding dimension:** 384.

| Model on Hugging Face | Version | Checkpoint files | Description |
| --- | --- | --- | --- |
| [PoreFM-DNA](https://huggingface.co/WGLab/PoreFM-DNA) | 0.1.0 | `backbone.pt`, `pretraining.pt` | Pretrained DNA backbone for embeddings, pairwise comparison, and downstream training; also available with pretraining heads. |
| [PoreFM-DNA_zero-shot](https://huggingface.co/WGLab/PoreFM-DNA_zero-shot) | 0.1.0 | `R10.4.1_zero-shot.pt` | Canonical-null model with whitening and calibration for detecting deviations from canonical DNA signals. |
| [PoreFM-DNA_prototype](https://huggingface.co/WGLab/PoreFM-DNA_prototype) | 0.1.0 | `R10.4.1_prototype.pt` | Prototypes learned from small labelled datasets, with joint footprinting for 5mC, 5hmC, and 6mA. |
| [PoreFM-DNA_BrdU-EdU](https://huggingface.co/WGLab/PoreFM-DNA_BrdU-EdU) | 0.1.0 | `model.pt` | Supervised classifier distinguishing canonical T, BrdU, and EdU. |

## Choosing a checkpoint

For modification identities, choose a supervised classifier or a prototype
model that includes your target classes. Supervised models use classifiers
trained on labelled data. Prototype models learn prototypes from small labelled
datasets and use footprinting to account for neighboring effects. Both use
contextual PoreFM representations. The released prototype models cover more
modification types, so a modification can have a prototype model even when
no supervised model is available.

- **`backbone.pt`** contains the pretrained backbone for embeddings, pairwise
  comparison, and downstream tasks.
- **`pretraining.pt`** contains the same backbone plus pretraining objective
  heads and the EMA target encoder. Optimizer and scheduler states are excluded.
- **Zero-shot and prototype bundles** contain their required backbone and fitted
  components; separate model-component paths are unnecessary.
- **Supervised `model.pt` files** contain the backbone, classifier, preprocessing,
  class labels, and base/modification-code defaults.

Derived models use the corresponding RNA or DNA pretrained backbone and include
the weights needed for inference. A separate base-model download is unnecessary
when using a derived model.

## Downloading and using models

Use `--model OWNER/REPOSITORY@vVERSION` to download a published release
automatically. The version is part of the same argument; there is no separate
version option. An explicit tag such as `@v0.1.0` is required.

```bash
porefm prototype classify \
  --model WGLab/PoreFM-RNA_prototype@v0.1.0 \
  --bam_path basecalls.bam --pod5 pod5/ --ref reference.fa \
  --chrom chr1 --motif 'N 0' \
  --effect_penalty 50 \
  --output_dir outputs/prototype
```

The four inference commands select the appropriate artifact from the release:

| Command | Example model argument | Artifact selected |
| --- | --- | --- |
| `detect-modification` | `--model WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0` | Complete supervised classifier. |
| `zero-shot detect` | `--model WGLab/PoreFM-RNA_zero-shot@v0.1.0` | Complete canonical-null bundle. |
| `prototype classify` | `--model WGLab/PoreFM-RNA_prototype@v0.1.0` | Complete prototype bundle. |
| `pairwise-comparison` | `--model WGLab/PoreFM-RNA@v0.1.0` | Pretrained backbone; supervised classifier releases can also supply embeddings and class filtering. |

For supervised pseudouridine detection, use
`--model WGLab/PoreFM-RNA_pseU@v0.1.0` with `--read_based --motif 'T 0'`.
Its class labels are `U` and `pseU`; RNA uridine is represented as `T` in BAM
sequences and modification tags. The checkpoint supplies SAM modification
code `17802`. The [HeLa example](examples.md#call-m6a-inosine-and-pseu-with-supervised-models)
runs both RNA supervised models.

Downloads use public HTTPS file URLs with Python's standard library. No
Hugging Face package, CLI, account, or login is required for these public
releases. PoreFM reads `manifest.json`, pins the checkpoint download to the
resolved repository commit, and verifies its size and SHA-256 before use.
It never silently substitutes the latest release for a missing version.

### Cache directory

The default cache is `~/.cache/porefm/models`, or
`$XDG_CACHE_HOME/porefm/models` when `XDG_CACHE_HOME` is set. Configure a shared
cache through the environment:

```bash
export POREFM_MODEL_CACHE=/path/to/shared/porefm-models
```

Alternatively, add `--model-cache-dir /path/to/cache` to an inference or
download command. This option takes precedence over the environment variable.
PoreFM verifies cached checkpoint checksums and reuses valid files without
contacting the server. Concurrent processes sharing a cache coordinate downloads.
The cache is independent of Hugging Face's SDK cache.

### Prefetching and offline use

Download on a machine with internet access before launching a GPU job:

```bash
porefm models download \
  --model WGLab/PoreFM-RNA_prototype@v0.1.0 \
  --model-cache-dir /path/to/shared/porefm-models
```

The command prints the verified local checkpoint path to stdout; download
progress goes to stderr. On the compute node, use the same `--model` and cache
directory, adding `--offline` to prohibit network access. A missing or corrupt
cached model then produces an error.

Base-model repositories default to `backbone.pt`. Request the pretraining
checkpoint explicitly when needed:

```bash
porefm models download --model WGLab/PoreFM-DNA@v0.1.0 --file pretraining.pt
```

### Local files and reproducibility

Load local checkpoints with `--model-path` or `--model_path`.
Use either a local path or `--model`, not both. Local paths never trigger a
download, and can point to files obtained manually or through another cache.
For training, pass the prefetched checkpoint path to the training command's
checkpoint-loading option.

For `--model` runs, `run_manifest.json` records the requested model reference,
version, resolved repository commit, selected filename, checkpoint SHA-256,
and release-manifest SHA-256 under `model_source`. The local checkpoint path
is also recorded. This provenance is retained when using cached models offline.

For setup and examples, see [inputs and models](guides/inputs.md),
[supervised detection](guides/supervised.md), [zero-shot detection](guides/zero-shot.md),
[pairwise comparison](guides/pairwise.md), and
[prototype-based joint footprinting](guides/residual-prototypes.md).

## Checkpoint checksums

These SHA-256 checksums identify the exact checkpoint files for version **0.1.0**.
Each Hugging Face repository also includes `SHA256SUMS` and `manifest.json`.
After downloading the checkpoints listed in that repository's `SHA256SUMS`, run
this command from their download directory:

```bash
sha256sum -c SHA256SUMS
```

### RNA checksums

??? info "PoreFM-RNA — 0.1.0"

    ```text
    0ed7a78abb862e9cf7fdbc4d167209014753c440226f5871bc06f8f3e72e6fae  backbone.pt
    895db92ecb38d4f505185e7e5c5f55f60caba8dd75ef6ec04b781a7ec3440795  pretraining.pt
    ```

??? info "PoreFM-RNA_zero-shot — 0.1.0"

    ```text
    5200a9333f507f3c26b13fdb32a7a08da416021b1646640406b6284fa44b372c  RNA004_zero-shot.pt
    ```

??? info "PoreFM-RNA_prototype — 0.1.0"

    ```text
    79bec4aa0d52bcadf433f97ece376936a70f0cde08420b522addfd698aca2043  RNA004_prototype.pt
    ```

??? info "PoreFM-RNA_m6A-Inosine — 0.1.0"

    ```text
    828123f81624cf257d0f700c5ad38fe1833c41ac96c2e3d03bb267e1ebb95902  model.pt
    ```

??? info "PoreFM-RNA_pseU — 0.1.0"

    ```text
    91e249d9d4fb11ae9c029bdd63ea5dd64e1e8cfde680bb017b32c7b68c4b0d03  model.pt
    ```

### DNA checksums

??? info "PoreFM-DNA — 0.1.0"

    ```text
    4367c9a7ef291108a4bfcf6ae8452723c06bb68aa89f715ad4b8115b560986ac  backbone.pt
    d120e0e9abba56fe8ec62cf8549e5fb969c84321982a0d55d4113fe670fdcf2f  pretraining.pt
    ```

??? info "PoreFM-DNA_zero-shot — 0.1.0"

    ```text
    e3e8198166464c3b84a3786abca00087b4b893a7a5e80298087906917b2c26a2  R10.4.1_zero-shot.pt
    ```

??? info "PoreFM-DNA_prototype — 0.1.0"

    ```text
    83b0fc7a56edd24280f768da367f5e267bc7435b674786184147f8da8087cd26  R10.4.1_prototype.pt
    ```

??? info "PoreFM-DNA_BrdU-EdU — 0.1.0"

    ```text
    0b5a351fc4fcfd03333ba3ddfc412af8e7d91dc53b51b0ead7b673cbf4ee54a3  model.pt
    ```
