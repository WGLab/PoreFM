# Supervised labels and training inputs

Use these JSON definitions to supply training labels or select inference targets.
Training requires SG-tagged signal BAMs. For direct BAM + POD5 inference, see
[inference inputs](inference-inputs.md).

## Supervised and inference BAM JSON

JSON files cannot contain comments or trailing commas. Use double quotes for
all keys and string values. A training config must define exactly one non-empty
sample list:

```json
{
  "config_type": "supervised",
  "schema_version": 1,
  "labelled_samples": [
    {
      "bam_path": "/path/to/modified.bam",
      "ref": "/path/to/reference.fa",
      "contigs": ["chr1", "chr2"],
      "motif": "CG 0;CH 0",
      "motif_label": "1;2"
    }
  ]
}
```

`bam_path` may name one BAM or a directory, which is recursively expanded to
naturally sorted `*.bam` files. Keep a directory free of both merged BAMs and
their component shards, or reads will be duplicated.

### Labelled sample modes

| Mode | Required fields | Forbidden or ignored fields |
|---|---|---|
| Read-based motif | `bam_path`, `read_based: true`, `motif`, `motif_label` | `pos_file` is forbidden; `ref` is unused for training labels; inference can retain reference coordinates |
| Reference motif | `bam_path`, `ref`, `motif`, `motif_label` | Omit `read_based` or set it to `false` |
| Explicit positions | `bam_path`, `ref`, `pos_file` | `read_based: true` is forbidden |

Common labelled-sample fields:

- `contigs`: JSON array of contig strings. Omit or use `[]` for all reference
  contigs. For SG input, supplying contigs in read-based mode selects indexed regional
  fetching. Direct POD5 inference uses its read-ID index instead.
- `motif`: one string containing semicolon-delimited motif specifications.
  Each specification is `IUPAC_PATTERN ZERO_BASED_OFFSET`; the offset selects
  the base inside the motif and is not a class index.
- `motif_label`: class indices separated with semicolons. Every index must be
  in `[0, num_classes)` and order must match `motif`; if fewer labels than
  motifs are supplied, the final label applies to the remaining motifs.
- `sample_labels`: probability of retaining each label occurrence; default `1`.
- `label_smoothing`: optional classification epsilon overriding the config/CLI
  default; explicit `0` disables smoothing for this sample.
- `flag`: additional SAM flag bits to exclude; default `0`.
- `polish`: boolean, default `false`. Polishing requires mapped records with
  CIGAR and MD information.
- `polish_indel_limit`: largest corrected unmasked indel; default `5`.
- `polish_vcf_path`: optional VCF/BCF whose ALT SNPs and indel loci are
  preserved. Prefer indexed input for large files.

Set the class count with `--num_classes`, not in the sample JSON.

## Reference-position files

Position files referenced by `pos_file` are tab-separated, have no header, use
zero-based reference coordinates, and contain:

```text
chromosome<TAB>position<TAB>strand<TAB>class_0<TAB>...<TAB>class_K-1
```

Strand may be `+`, `-`, `0`, or `1`; provide exactly one numeric target column
per classifier class.

### Tagged sample mode

Use `tagged_samples` instead of `labelled_samples`:

```json
{
  "config_type": "supervised",
  "schema_version": 1,
  "tagged_samples": [
    {
      "bam_path": "/path/to/tagged.bam",
      "read_label_dict_path": "/path/to/read_labels.pkl",
      "label_coordinates": "read",
      "label_smoothing": 0
    }
  ]
}
```

Every dictionary-labelled tagged sample must explicitly set `label_coordinates`
to `"read"` or `"ref"`; there is no default. Both systems are zero-based:

- `"read"` uses the original 5'-to-3' read sequence, including soft clips and
  insertions. Reverse alignments use offsets in `get_forward_sequence()`, not
  the stored reverse-complemented BAM sequence. No FASTA is consulted. Omit
  `contigs` to stream mapped or unmapped records without an index; explicit
  contigs still require indexed regional fetching.
- `"ref"` uses positions on the read's reference contig, including on reverse
  alignments, and requires mapped/indexed BAM plus matching indexed FASTA
  supplied as `ref`.

Choose one label source:

- Set `ML_tag: true` to decode MM/ML calls; it defaults to `false`. Calls with
  probability strictly greater than `ML_threshold` (default `0.5`) receive
  `label`. Optional `negative_label` assigns calls strictly below the same
  threshold a different class; omission or `null` leaves those calls ignored.
  Calls exactly at the threshold are always ignored. The fraction of decoded
  calls retained across both enabled classes must meet `min_read_frac`
  (default `0`); its denominator is decoded calls, not all read bases.
  This path still requires a mapped,
  indexed BAM and `ref`; omit the dictionary-only `label_coordinates` field.
- Set optional `MM_code` to one exact SAM modification code, such as `"a"`
  or numeric `17596` (also accepted as `"17596"`). It selects calls before
  thresholding: `label`, `negative_label`, and the `min_read_frac` denominator
  all apply only to that code. Reads without selected calls are skipped.
  Use the code actually present in MM, not a class name such as `"m6A"` or
  a full group header such as `"A+a"`. Letter/numeric aliases are not inferred.
  This option requires `ML_tag: true`.
- Otherwise provide `read_label_dict_path`, a pickle keyed by read name. Reads
  absent from the dictionary are skipped. The path has no default when
  `ML_tag` is `false` and is mutually exclusive with `ML_tag: true`.
- `label` defaults to class index `1`. `negative_label` is only supported with
  `ML_tag: true` and must be a valid class index different from `label`.
  `label_smoothing`, `flag`, and polishing fields use the labelled-sample
  defaults above.

For class order `A 6mA`, this sample uses both sides of the threshold:

```json
{
  "tagged_samples": [{
    "bam_path": "/path/to/data/signal.sorted.bam",
    "ref": "/path/to/references/genome.fa",
    "ML_tag": true,
    "MM_code": "a",
    "ML_threshold": 0.5,
    "label": 1,
    "negative_label": 0
  }]
}
```

The loader constructs one-hot targets, not ML probability vectors. The supervised
loss applies the resolved `label_smoothing` afterward.
Only explicitly decoded calls are eligible; unlisted bases are not inferred
to be negative. With `MM_code`, interleaved codes are decoded separately and
selected positions are sorted in original read orientation. Duplicate selected
positions raise an error rather than assigning ambiguous targets. Omitting
`MM_code` pools calls across codes and requires sorted coordinates. One sample
with `MM_code` selects at most one code and maps it
to one positive class; this is not a multi-code-to-multiclass mapping.

Dictionary entries have the form
`read_name: (target_values, target_positions)`.
Use `(N, num_classes)` probability arrays for per-position vectors; a shared
`(num_classes,)` vector or scalar class is also accepted. Position arrays must contain sorted,
unique, nonnegative integers; sort target rows together with positions. For
three-class C/5mC/5hmC targets, columns must follow exactly that model class
order. A one-dimensional array of length `num_classes` is interpreted as a
shared probability vector, so use explicit two-dimensional one-hot arrays
for per-site hard labels.

For dictionary labels, output `position` follows the selected coordinate
system. When polishing is enabled,
original read targets are remapped into polished coordinates and targets on
removed insertions are dropped. Polishing still requires aligned reads.
Read and reference dictionaries produce the same labels when they describe
the same aligned bases; insertion/soft-clip targets have no reference equivalent,
and a deleted reference base has no original read coordinate.

## Classifier modification metadata

Supply `--base` and `--mod_code` to `porefm finetune-modification` to save the
canonical base and modification codes in checkpoint `meta.base` and
`meta.mod_code`. For example, use `--num_classes 3 --class_labels C 5mC 5hmC
--base C --mod_code m h`. Class 0 is canonical; provide exactly one code for
each remaining class, in logit order. Codes are stored as strings, including
numeric codes. These options are optional for training.

With `--model_load_path`, omitted values inherit from the classifier checkpoint.
When extending its class count, supply the complete updated code list.
`porefm detect-modification` automatically uses the saved values. Explicit `--base` and
`--mod_code` override their respective metadata fields. BAM output (`--output_format bam` or `both`)
requires modification codes from metadata or the CLI; Parquet output does not.

### Classification label smoothing

Supervised fine-tuning resolves `label_smoothing` for each sample using
**sample > config > `--label_smoothing` > `0`**. Values must be finite numbers
in `[0, 1)`; omission inherits the next level, while explicit zero overrides it.
Every BAM expanded from a sample directory inherits that sample's value.

For example, native labels can inherit a config default while reliable IVT or
PCR controls opt out (paths below are illustrative):

```json
{
  "label_smoothing": 0.02,
  "labelled_samples": [
    {
      "bam_path": "/path/to/data/native.bam",
      "ref": "/path/to/references/genome.fa",
      "pos_file": "/path/to/labels/native.tsv"
    },
    {
      "bam_path": "/path/to/data/ivt.bam",
      "read_based": true,
      "motif": "A 0",
      "motif_label": 0,
      "label_smoothing": 0
    }
  ]
}
```

For target distribution `q`, class count `C`, and resolved epsilon `e`, the
loss uses `(1 - e) * q + e / C`. This applies once in float32 to retained
classification bases with either cross-entropy or focal loss. It works for
motifs, position files, per-read dictionaries, and MM/ML labels. Metrics use
the original targets. Zero preserves supplied soft target distributions; it
does not convert them to hard labels. The setting is training metadata and
does not alter classifier inference.

Startup logs list each sample's effective value and whether it came from the
sample, config, or CLI. On continuation, the supplied configs/CLI determine
the run's smoothing policy. Exclusion samples always use zero.
Knowing that one class is absent in a multiclass sample remains an exclusion
constraint, not a hard label choosing among the other classes.

### Top-level supervision

Omitting `supervision` means ordinary classification. To select it explicitly,
use `"supervision": {"type": "classification"}`;
include it alongside a complete `labelled_samples` or `tagged_samples` list.

Classification supervision accepts no fields other than `type`. The optional
`label_smoothing` setting belongs alongside `supervision` at the JSON top level
or inside a sample, not inside the `supervision` object.

Exclusion supervision applies to the entire JSON file and currently supports
only `labelled_samples`:

```json
{
  "config_type": "supervised",
  "schema_version": 1,
  "supervision": {
    "type": "exclusion",
    "excluded_class": "5mC",
    "expected_allowed_proportions": {"C": 0.8, "5hmC": 0.2},
    "proportion_tolerance": 0.05
  },
  "labelled_samples": [
    {
      "bam_path": "/path/to/no_5mc.bam",
      "ref": "/path/to/reference.fa",
      "motif": "CG 0",
      "motif_label": "1"
    }
  ]
}
```

Strict exclusion requirements:

- `excluded_class` must exactly match one unique `--class_labels` value.
- JSON `label_smoothing` must be zero or omitted at both config and sample
  levels. Exclusion loss always ignores the CLI smoothing fallback.
- Every motif label must equal the excluded class index.
- If expected proportions are present, they must name every allowed class
  exactly once, contain finite nonnegative values summing to one, and be paired
  with `proportion_tolerance` in `[0, 1)`.
- Put biologically different exclusions or expected mixtures in separate JSON
  files. Each file becomes one supervision and sampling group.
- Supply one `--group_weights` value per `--bam_config` file. These are source
  selection weights, not class weights.
