# CLI and configuration reference

This guide explains how PoreFM command-line values are parsed and documents
the configuration-file formats consumed by maintained commands. Command help
is the source of truth for individual flags and current defaults:

```bash
porefm
porefm <command> --help
```

Running `porefm` or a leaf command with no additional arguments prints help and
exits successfully. Choices appear next to their option in braces, and defaults
appear in the option description when a meaningful default is set.

The maintained BAM and source configs described below use strict JSON
contracts. Validate a file without starting a model or opening BAMs with:

```bash
porefm utils validate-config validate /path/to/config.json
porefm utils validate-config validate /path/to/config.json --show-normalized
porefm utils validate-config validate /path/to/supervised.json --class-labels C 5mC 5hmC
porefm utils validate-config schema pretraining
```

The validator rejects duplicate or unknown fields, wrong types, unsupported
schema versions, and invalid relationships between fields. `--show-normalized`
displays the effective config after optional defaults are applied. Pass
`--class-labels` for supervised configs to validate class names and indices
against their CLI order. Supported schema names are `pretraining`,
`bam-sharding`, and `supervised`. Validation checks the config
contract, not whether paths exist or whether BAM/FASTA contents are compatible.
Structural validation uses the emitted Draft 2020-12 schemas; a small semantic
layer applies defaults and checks relationships involving external CLI values.

Paths in the examples are placeholders; replace them with your own inputs and
output destinations.

## Find the relevant options

| Task | Reference |
| --- | --- |
| Supply samples to inference; use POD5 or exclude reads | [Inference inputs](reference/inference-inputs.md) |
| Define supervised labels, class metadata, or supervision | [Supervised labels and training inputs](reference/training-inputs.md) |
| Change signal chunk boundaries | [Signal chunking](reference/chunking.md) |
| Configure pretraining modality dropout | [Signal dropout during training](reference/pretraining-policies.md) |

## Command-line value syntax

The shell splits arguments before PoreFM sees them. Follow these conventions:

- Multi-value options using `nargs` are **whitespace-separated**. A comma does
  not separate list items. For example:

  ```bash
  --bam_config control.json modified.json
  --class_labels canonical 5mC 5hmC
  --num_loss_trim 16 32
  ```

- A value that itself contains spaces must be quoted so it remains one
  argument. Motif sets are the common case:

  ```bash
  --motif_string 'CG 0;CH 0'
  --motif 'DRACH 2;CG 0'
  ```

- Do not write `--chrom chr1,chr2` for an option whose help says it accepts
  multiple values. Use `--chrom chr1 chr2`. Commas may still be part of one
  value's own syntax, as in `--mods C:m,h A:a`; explicitly documented
  single-string options may also accept comma-delimited content.
- Paths containing spaces must also be quoted.
- CLI booleans are flags such as `--disable_compile`; do not append `true` or
  `false` unless the help explicitly declares a value.

For distributed pretraining or fine-tuning, continue to use module mode because
TorchRun launches a Python module on every rank:

```bash
torchrun --module porefm.cli.finetune_modification [options]
```

## Configuration formats at a glance

| Format | Used by | Top-level structure |
|---|---|---|
| Pretraining JSON | `porefm pretrain`, `porefm zero-shot train`, `train-scale`, `fit-whitening`, `calibrate`, auxiliary supervised pretraining | Object containing `samples` |
| BAM-sharding JSON | `porefm utils shard-bam --bam_config` | Object containing `inputs` |
| Supervised/inference BAM JSON | `porefm finetune-modification`, `porefm detect-modification` | Object containing `labelled_samples` or `tagged_samples` |
| Canonical-null pipeline manifest and stage JSON | Written by `porefm zero-shot pipeline` | Output records; do not use as hand-written input configs |

Include `config_type` and `schema_version` in each input JSON, as shown in the
examples below. Use one object per sample or input in the appropriate array.

## Pretraining and canonical-null JSON

Use this format with `--bam_config` for pretraining and direct canonical-null
training/calibration. Supervised auxiliary pretraining accepts one or more of
these files through `--pretrain_bam_config`.

```json
{
  "config_type": "pretraining",
  "schema_version": 1,
  "samples": [
    {
      "bam_path": "/path/to/control.bam"
    },
    {
      "bam_path": "/path/to/modified.bam",
      "sampling_probability": 0.5
    }
  ]
}
```

Rules and strict assumptions:

- `samples` is required and must be a nonempty array.
- Each sample requires `bam_path`.
- `sampling_probability` is optional and defaults to `1.0`. It must be a
  finite number in `(0, 1]` and independently controls whether each eligible
  BAM record is retained. It is not a relative source or class weight.
- Unknown sample and top-level fields are rejected.
- Paths are used as written; environment variables and `~` are not expanded by
  the parser.
- For DDP, provide enough independent BAM files or shards for the configured
  rank/worker topology.
- Pretraining polishing remains run-wide: use `--use_polished_sequence`,
  `--polish_indel_limit`, and `--polish_vcf_path`. Per-sample polishing fields
  are deliberately not accepted because the current pretraining dataset uses
  one polishing policy for the entire run.

The canonical-null pipeline takes one config in this format, creates disjoint
sample-level pretraining JSON configs, and passes the generated configs to its
stages. Split provenance and fractions are recorded in `pipeline_manifest.json`.

## BAM-sharding JSON

Use this format with `porefm utils shard-bam --bam_config`:

```json
{
  "config_type": "bam-sharding",
  "schema_version": 1,
  "inputs": [
    {
      "bam_path": "/path/to/sample1.bam",
      "contigs": ["chr1", "chr2"]
    },
    {
      "bam_path": "/path/to/sample2.bam"
    }
  ]
}
```

Rules:

- `inputs` is required and must be a nonempty array.
- Each input requires `bam_path`.
- `contigs` is an optional array of unique nonempty strings and defaults to
  `[]`, which processes all records/contigs.
- Sampling probabilities are not accepted by this schema.
- Alternatively, use `--bam_path` for one BAM and omit `--bam_config`.
- Inputs selected by contig require coordinate-addressable aligned BAMs.

## Generated canonical-null JSON

`porefm zero-shot pipeline` writes:

- `pipeline_manifest.json`, recording inputs, deterministic split counts,
  artifacts, and stage states;
- `stage_args/<stage>.args.json`, recording the exact effective arguments for
  each stage.

These files are provenance outputs, not supported hand-written input schemas.
Do not edit them to resume or alter a run; use the pipeline's reuse/output
options and retain the generated manifest with the artifacts.

## Preflight checklist

Before a large analysis:

1. Validate JSON with `porefm utils validate-config validate CONFIG.json` and
   check the command's `--help` for available options.
2. Check input paths and the indexes needed by your input route:

   | Input route | Signal and indexing requirements |
   | --- | --- |
   | Direct BAM + POD5 inference | Matching read IDs; a `.pfmi` index is built automatically. A coordinate BAM index is unnecessary. |
   | Reference-based SG BAM, pairwise comparison, or explicit SG contig selection | Raw signal in `SG`; sorted, indexed alignments and a matching indexed FASTA. |
   | Unscoped read-based SG BAM | Raw signal in `SG`; coordinate sorting and a BAM index are unnecessary. |

3. Preserve sequence and quality information. Keep `mv` move tables and signal
   trimming tags when available; `pod5-to-bam` requires `mv`. Supported 9-mer
   chemistries can infer chunks without moves, while RNA002 requires moves.
   For move-based decoding, check the stride against the checkpoint's `ts_dim`.
4. When reference coordinates are needed, check that BAM and FASTA contig names
   and lengths agree. Reference-guided polishing additionally needs suitable
   alignments with CIGAR and MD information.
5. For training, verify class order across labels, position-file columns,
   modification codes, and model outputs.
6. Keep the command, input configs, model version, and run manifest with results.
