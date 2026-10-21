# PoreFM

PoreFM is a foundation model for nanopore signal and sequence, designed to
detect DNA and RNA modifications and compare samples.

[![PoreFM applications: signal-contextualized base sequence embeddings, supervised modification calls, prototype modification calls, zero-shot anomalies, and test-versus-control comparison.](docs/assets/porefm-overview.svg)](docs/assets/porefm-overview.svg)

[Documentation](https://wglab.github.io/PoreFM/) ·
[Run the HeLa RNA example](docs/examples.md) ·
[Models](docs/models.md) ·
[Report an issue](https://github.com/WGLab/PoreFM/issues)

## Choose an analysis

| Your goal | Method | Control sample at inference | Result |
| --- | --- | --- | --- |
| Call modifications with a fine-tuned classifier | [Supervised detection](docs/guides/supervised.md) | No | Trained class calls and probabilities |
| Call a broader set of modifications using learned prototypes | [Prototype joint footprinting](docs/guides/residual-prototypes.md) | No | Supported modification calls and relative scores |
| Find departures from canonical signal expectations | [Zero-shot detection](docs/guides/zero-shot.md) | No | Anomaly scores and p-values |
| Compare a sample with a matched control | [Pairwise comparison](docs/guides/pairwise.md) | Yes | Embedding shifts and test-read anomaly scores |

Supervised models use a classifier trained on labelled data. Prototype models
learn modification prototypes from small labelled datasets and use footprinting
to account for neighboring effects. Both use contextual PoreFM representations.
The released prototype models cover more modification types, including some
without a supervised model; see the [model catalog](docs/models.md).

Choose a model matching your chemistry:
RNA004 or DNA R10.4.1 (E8.2, 400 bps). Anomaly scores do not identify a chemical
modification; site-level call fractions require validation before interpreting
them as modification stoichiometry.

## Getting started

1. [Install PoreFM](docs/getting-started/installation.md).
2. [Run the HeLa RNA example](docs/examples.md) to obtain calls and compare four methods.
3. [Prepare your own inputs](docs/guides/inputs.md), choose a guide above, and
   [summarize your results](docs/guides/outputs.md).

Inference uses Linux and a supported NVIDIA CUDA GPU. Inputs must include raw
signal: a basecall BAM alone is insufficient. The guides explain direct BAM +
POD5 input and signal-tagged BAMs. Saved outputs can be analyzed without a GPU.
For custom downstream analyses, see [exporting embeddings](docs/guides/embeddings.md).

## Citation and support

Use [CITATION.cff](CITATION.cff) to cite the software. Record the PoreFM version
or Git revision and the exact model release, such as `@v0.1.0`, used in your
analysis. [Support and reproducibility](docs/project.md) describes what to
include in an issue or methods section.

## License

PoreFM code is released under the [MIT license](LICENSE). Vendored k-mer
resources retain their [upstream license](src/porefm/resources/kmer_models/LICENSE.md)
and copyright notices. Model and example-data terms are provided by their
respective release repositories.
