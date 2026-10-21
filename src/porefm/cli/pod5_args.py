"""Lightweight POD5 input options shared by read-scoring commands."""


def add_pod5_arguments(parser):
    """Add raw-signal and optional saved-index paths without runtime imports."""
    parser.add_argument(
        '--pod5', default=None,
        help='POD5 file or recursively searched directory supplying signal; requires one BAM file.',
    )
    parser.add_argument(
        '--bam_index', default=None,
        help='Saved porefm utils index-bam index for --pod5. If omitted, reuse <bam_path>.pfmi when present; otherwise build <BAM filename>.pfmi in --output_dir before workers start.',
    )


def validate_pod5_arguments(parser, args):
    """Reject an orphan index option before preparing outputs or loading models."""
    if args.bam_index and not args.pod5:
        parser.error('--bam_index requires --pod5')
