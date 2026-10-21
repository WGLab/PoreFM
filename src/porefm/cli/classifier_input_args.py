"""Direct supervised-inference inputs using the supervised sample contract."""

from __future__ import annotations


# None distinguishes omitted options from explicitly supplied JSON defaults.
_SAMPLE_FIELDS = (
    'pod5', 'bam_index', 'ref', 'contigs', 'motif', 'pos_file', 'motif_label',
    'flag', 'sample_labels', 'polish', 'polish_indel_limit', 'polish_vcf_path',
)


def add_classifier_input_arguments(parser):
    """Add alternative JSON/direct sources and shared per-sample CLI options.

    Direct inputs describe motif- or position-selected samples. Multiple SG
    BAMs share these settings; POD5 requires one BAM. JSON retains support for
    independently configured samples and tagged label sources. Imports for
    config validation are deferred so command help needs no scientific stack.
    """
    group = parser.add_argument_group('Sample input')
    source = group.add_mutually_exclusive_group()
    source.add_argument(
        '--bam_config', '--bam-config', nargs='+',
        help='Supervised JSON configs, as an alternative to direct sample options.',
    )
    source.add_argument(
        '--bam_path', '--bam-path', nargs='+',
        help=('BAM file(s) or recursively searched SG BAM directories. All inputs '
              'share the direct sample options; with --pod5, supply one BAM file.'),
    )
    group.add_argument('--pod5', help='Raw signal POD5 file or recursive directory; otherwise use BAM SG tags.')
    group.add_argument('--bam_index', '--bam-index', help='Optional saved BAM read-ID index (.pfmi); requires --pod5.')
    group.add_argument('--ref', help='Indexed reference FASTA; required for reference-based targets.')
    group.add_argument('--contigs', '--chrom', nargs='+', help='Restrict input to these contigs (default: all).')
    group.add_argument(
        '--read_based', '--read-based', action='store_true', default=False,
        help=('Select motifs in called read sequences instead of the reference. '
              'With --bam_config, applies the existing global read-based override.'),
    )
    targets = group.add_mutually_exclusive_group()
    targets.add_argument('--motif', help="Quoted motif/offset selector, e.g. 'A 0' or 'DRACH 2;CG 0'.")
    targets.add_argument('--pos_file', '--pos-file', help='Reference-position target table; incompatible with --read_based.')
    group.add_argument(
        '--motif_label', '--motif-label',
        help='Class index or semicolon-delimited indices assigned to motif targets (default: 0).',
    )
    group.add_argument('--flag', type=int, help='Additional SAM flag bits to exclude (default: 0).')
    group.add_argument(
        '--sample_labels', '--sample-labels', type=float,
        help='Per-chunk-occurrence target retention probability in [0, 1] (default: 1).',
    )
    group.add_argument(
        '--polish', '--use_polished_sequence', '--use-polished-sequence',
        action='store_true', default=None,
        help='Apply reference-based sequence polishing (default: disabled).',
    )
    group.add_argument(
        '--polish_indel_limit', '--polish-indel-limit', type=int,
        help='Maximum insertion/deletion length to polish (default: 5).',
    )
    group.add_argument(
        '--polish_vcf_path', '--polish-vcf-path',
        help='VCF of variants to preserve during polishing; requires --polish.',
    )


def resolve_classifier_input_config(args, *, class_labels=None, required=False):
    """Validate direct inputs and attach their normalized config to ``args``.

    Return the same normalized mapping as a supervised JSON labelled-sample
    config, or ``None`` for JSON input. Direct options cannot override JSON
    sample settings, except for the pre-existing global ``read_based`` flag.
    With ``required=True``, require either source before model resolution.
    ``args.resolved_input`` records direct settings and defaults in run manifests;
    no temporary config file is created. Paths retain JSON's cwd-relative rules.
    """
    sample = {
        name: getattr(args, name)
        for name in _SAMPLE_FIELDS if getattr(args, name, None) is not None
    }
    bam_paths = getattr(args, 'bam_path', None)
    configs = getattr(args, 'bam_config', None)
    if configs:
        if bam_paths or sample:
            raise ValueError('--bam_config cannot be combined with direct sample options; put per-sample settings in JSON')
        return None
    if not bam_paths:
        if sample or getattr(args, 'read_based', False) or required:
            raise ValueError('Supply --bam_path for direct sample options, or --bam_config for JSON input')
        return None

    from porefm.config.validation import validate_config_data

    sample['read_based'] = getattr(args, 'read_based', False)
    raw = {
        'config_type': 'supervised', 'schema_version': 1,
        'labelled_samples': [dict(sample, bam_path=path) for path in bam_paths],
    }
    config = validate_config_data(
        raw, expected_type='supervised', source='CLI', class_labels=class_labels,
    ).data
    if sample.get('pod5') and len(bam_paths) != 1:
        raise ValueError('POD5 inference requires exactly one BAM file')
    args.resolved_input = config
    return config
