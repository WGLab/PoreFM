# Support, citation, and license

## Get help

Search or open an issue in the [PoreFM issue tracker](https://github.com/WGLab/PoreFM/issues).
Include the command, error traceback, PoreFM version or Git revision, model
release, sequencing chemistry, GPU, and PyTorch/CUDA/FlashAttention versions.
For input problems, describe whether you used BAM + POD5 or an SG-tagged BAM,
and whether the reads were aligned and contained move tables.

A small reproducible example is helpful. Review logs and `run_manifest.json`
before sharing them: manifests can contain local paths, hostnames, and input
configuration snapshots.

## Cite PoreFM

Cite PoreFM as software using
[CITATION.cff](https://github.com/WGLab/PoreFM/blob/main/CITATION.cff).
In a methods section, report:

- The software version and Git revision.
- The model repository and release tag, and checkpoint checksum when available.
- Sequencing chemistry, target selection, read filters, and relevant scoring settings.
- How read-level results were summarized, including coverage and decision thresholds.

Keep `run_manifest.json` with your results. Model downloads record the resolved
release and checksum there. Cite the source datasets and reference assays
separately; the [HeLa example](examples.md#reference-annotations) includes its
GLORI and BACS references.

## License

PoreFM code is distributed under the
[MIT license](https://github.com/WGLab/PoreFM/blob/main/LICENSE).
The vendored Oxford Nanopore k-mer resources retain their
[upstream license](https://github.com/WGLab/PoreFM/blob/main/src/porefm/resources/kmer_models/LICENSE.md)
and copyright notices. Consult each model or dataset repository for the terms
that apply to those separately distributed files.
