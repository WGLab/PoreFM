# Inference input options

Start with [input preparation](../guides/inputs.md) and the [supervised guide](../guides/supervised.md). This page covers direct options, JSON input, read exclusion, and POD5 indexing.

## Direct supervised inference inputs

`porefm detect-modification` accepts sample settings directly, as an alternative
to `--bam_config`. Both routes use the same supervised sample validation and
defaults. For example:

```bash
porefm detect-modification \
  --model WGLab/PoreFM-RNA_m6A-Inosine@v0.1.0 \
  --bam_path basecalls.bam --pod5 pod5/ \
  --read_based --ref reference.fa --motif 'A 0' \
  --output outputs/supervised
```

The direct options correspond to fields within a JSON `labelled_samples` entry:

| CLI option | JSON field | Meaning / default |
| --- | --- | --- |
| `--bam_path` | `bam_path` | BAM file or recursive SG BAM directory; multiple paths share the same settings. |
| `--pod5` | `pod5` | Raw signal file or recursive directory; otherwise use SG tags. Requires one BAM file. |
| `--bam_index` | `bam_index` | Optional saved read-ID index; requires POD5. |
| `--ref` | `ref` | Indexed reference FASTA. Required for reference-based targets. |
| `--contigs`, `--chrom` | `contigs` | Whitespace-separated contig names; default: all. |
| `--read_based` | `read_based` | Select motifs in basecalls; default: reference-based selection. |
| `--motif` | `motif` | Quoted motif/offset expression, e.g. `'A 0'` or `'DRACH 2'`. |
| `--pos_file` | `pos_file` | [Reference-position table](training-inputs.md#reference-position-files); mutually exclusive with `--motif` and incompatible with `--read_based`. |
| `--motif_label` | `motif_label` | Target class index or semicolon-delimited indices; default: `0`. Does not restrict predicted classes. |
| `--flag` | `flag` | Additional SAM flag bits to exclude; default: `0`. |
| `--sample_labels` | `sample_labels` | Per-chunk-occurrence target retention probability in `[0, 1]`; default: `1`. |
| `--polish`, `--use_polished_sequence` | `polish` | Enable reference-based sequence polishing; default: disabled. |
| `--polish_indel_limit` | `polish_indel_limit` | Maximum insertion/deletion length to polish; default: `5`. |
| `--polish_vcf_path` | `polish_vcf_path` | Variants to preserve during polishing; requires `--polish`. |

These sample options also accept hyphenated spellings, such as `--bam-path`
and `--read-based`. Paths are used relative to the working directory, just as
in JSON. Read-based selection requires `--motif`; reference-based selection
requires `--ref` and exactly one of `--motif` or `--pos_file`.

For read-based inference, a supplied `--ref` retains available reference
coordinates for aligned reads. Motifs still select positions in the called
read sequence. This differs from training label selection, where the unscoped
read-based motif loader does not use a reference to define labels.

Direct sample settings cannot be combined with `--bam_config` (also
`--bam-config`), except for its global `--read_based` override.
Use JSON for independently configured samples or tagged label sources.
Training-only label smoothing and supervision policies remain JSON/training
settings. Model, output, preprocessing, and resource options apply to either
input route. Direct runs record the normalized sample config and its defaults
under `arguments.resolved_input` in `run_manifest.json`.

## Excluding reads from detection

`porefm detect-modification` (alias `porefm infer`) accepts `--exclude-bed regions.bed`, also
spelled `--exclude_bed`. The same file applies to every configured sample,
including SG BAM and POD5 inputs, in reference-based or read-based mode.
Omitting the option disables region exclusion.

The file must contain three or four whitespace-separated columns:
`chrom start end [strand]`, with zero-based, half-open reference coordinates.
Three-column rows match both strands; column four, when present, must be `+`
or `-` and matches the BAM alignment strand, including for RNA. Blank lines
and `#` comments are allowed; named BED4 or BED6 files must first be converted
to this format.

Any overlap with a read's full reference span excludes the entire read before
signal decoding and model scoring, including overlaps inside deletions or
introns. Touching an interval endpoint is not an overlap. Unmapped reads are
not excluded by this option and remain subject to the other input filters.
The file is validated before checkpoint loading or output creation, and the
startup manifest records its path and SHA-256 digest. Region exclusion uses
`ncls`, which is included in the standard PoreFM installation.

## POD5 signal input for inference

`porefm detect-modification` accepts `--pod5` with direct `--bam_path` input,
or `pod5` on one JSON `labelled_samples` entry. A POD5 run must
contain exactly one sample and one BAM file: BAM directories, multiple BAMs
and mixed SG/POD5 samples are rejected. Supply one `.pod5` file or a directory;
directories are searched recursively for `*.pod5` files.

Build an optional reusable index before inference:

```bash
porefm utils index-bam --bam /path/to/basecalls.bam --threads 4
```

This creates `/path/to/basecalls.bam.pfmi` beside the BAM. The `.pfmi` suffix
is appended to the full filename. Use `--output /other/path/custom.pfmi` to
choose a different index destination.

`index-bam` includes **every BAM record**, including secondary, supplementary,
unmapped, low-quality and no-move-table records. It applies no inference filters.
It saves a SQLite lookup of parent/read IDs to BAM virtual offsets, without
signals or sequences. It needs no BAI and does not modify the BAM. The output
must be a new path; existing files are never overwritten. An output pointing
to the BAM itself, including via symbolic or hard links, is rejected. An
adjacent `<bam_path>.pfmi` is discovered automatically. To supply an index
explicitly, use `--bam_index` with direct inputs, or `bam_index` in the same
JSON sample entry:

```json
{
  "config_type": "supervised",
  "schema_version": 1,
  "labelled_samples": [
    {
      "bam_path": "/path/to/basecalls.bam",
      "pod5": "/path/to/pod5_directory",
      "bam_index": "/path/to/basecalls.bam.pfmi",
      "read_based": true,
      "motif": "A 0",
      "motif_label": 0
    }
  ]
}
```

The BAM supplies sequence, move tables and alignments; POD5 supplies raw int16
ADC signal. When `pod5` is present it takes precedence over any `SG` tag.
Omit `pod5` to read signal from `SG` tags. Reference-based motif/position samples
can also use `pod5` with their reference and target fields.

The same index can be reused with different quality, length, motif, region and
POD5-subset settings. Inference checks that the index still matches the BAM.
Rebuild after rewriting or recompressing the BAM, and do not modify it during
indexing or inference. `bam_index` requires `pod5`. Explicit indexes take
precedence over automatic discovery and must exist and validate.

Without `bam_index`, inference first checks for `<bam_path>.pfmi` alongside
the BAM. If absent, it reuses or builds an **unfiltered, reusable** index at
`OUTPUT_DIR/BAM_FILENAME.pfmi` before starting DataLoader workers. For example,
`/data/reads.bam` falls back to `OUTPUT_DIR/reads.bam.pfmi`. Automatic builds
write to the output directory. They leave the input BAM path and contents
unchanged. Distributed inference builds the index once and shares it across
workers. A valid automatically discovered index is reused; an invalid or stale
index raises an error.
Output-directory safety rules apply, including
`--force-overwrite` when intentionally reusing a run directory. The generated
index can also be supplied explicitly to a later run.
Unmatched POD5 reads are skipped without decoding signal. BAM sorting and a
BAI/CSI index are unnecessary for this path, including when `contigs` filters
the records. Secondary and supplementary alignments are excluded. Move tables
are preferred; in the default `auto` chunking mode, supported chemistries can
infer chunks for reads without moves. Explicit `mv` mode skips those reads.
Ordinary reads match by query name. Dorado child reads
match their `pi` parent ID, and their `sp` sample offsets are applied before
the usual `ts`/`mv` trimming and RNA orientation. Each parent's signal is
decoded once for all its eligible children. POD5 subsets emit only matching
BAM records. Files processed by different workers must contain disjoint read
IDs. Repeated IDs within one worker are suppressed; aliases of the same path
are removed during discovery.

POD5 inputs produce unordered results, and BAM output is marked `SO:unsorted`.
Training continues to require SG signal BAMs. The JSON option applies to
`porefm detect-modification` labelled samples, not tagged training sample definitions.

Zero-shot detection (`porefm zero-shot detect`)
and `porefm prototype classify` (including
`--scorer residual-effects`) accept the equivalent CLI options alongside
their scoring arguments:

```bash
--bam_path /path/to/basecalls.bam \
--pod5 /path/to/pod5_directory \
--bam_index /path/to/basecalls.bam.pfmi \
--output_dir /path/to/new_run
```

For these commands, omitting `--bam_index` uses the same adjacent-index
discovery and output-directory fallback before model loading and DataLoader
startup. `--bam_index` requires `--pod5`.
The single-BAM-file contract, child-read slicing, region/quality filters and
POD5-file worker assignment are shared with supervised inference. Zero-shot
detection and prototype scoring use one process/GPU with multiple DataLoader
workers. Prototype extraction and canonical-null training/calibration require
SG signal BAMs.

POD5 files are assigned disjointly to ranks and then DataLoader workers. The
requested `--dataloader_workers` (`--num_workers` in the scoring commands) is
capped to the number of files assigned to that rank: one POD5 file uses at most
one worker; zero workers processes input
in the main process. SG input assigns whole BAM files to workers. POD5 signal
goes directly to preprocessing without an intermediate SG conversion.
