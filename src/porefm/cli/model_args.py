"""Shared local-checkpoint and anonymous public-model CLI options."""

import argparse

from porefm.common.model_download import fetch_model, parse_model_reference


def model_reference_argument(value):
    """Validate the complete model/version contract before any download occurs."""
    try:
        parse_model_reference(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def add_model_cache_arguments(parser):
    """Expose one cache configuration contract for inference and prefetching."""
    parser.add_argument('--model-cache-dir', '--model_cache_dir', default=None,
                        help='Model cache directory; otherwise POREFM_MODEL_CACHE or ~/.cache/porefm/models (respects XDG_CACHE_HOME).')
    parser.add_argument('--offline', action='store_true',
                        help='Use cached model files only; never contact the download server.')


def add_model_arguments(parser, *, local_flags=('--model-path', '--model_path'),
                        dest='model_path', required=True, local_help='Local model checkpoint or inference bundle.'):
    """Add mutually exclusive local-file and versioned-model inputs without I/O."""
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument(*local_flags, dest=dest, default=None, help=local_help)
    group.add_argument('--model', type=model_reference_argument, metavar='OWNER/REPO@vVERSION',
                       help='Public release to download/cache, e.g. WGLab/PoreFM-RNA_prototype@v0.1.0.')
    add_model_cache_arguments(parser)


def resolve_model_arguments(args, *, roles, dest='model_path', parser=None):
    """Resolve a public release before existing preprocessing and checkpoint loaders.

    Local-file calls are unchanged. Public model provenance remains on ``args``
    so the existing run-manifest writers record the exact download identity.
    """
    if not getattr(args, 'model', None):
        return args
    try:
        downloaded = fetch_model(args.model, cache_dir=args.model_cache_dir,
                                 offline=args.offline, roles=roles)
    except (OSError, ValueError) as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    setattr(args, dest, str(downloaded.path))
    args.model_source = downloaded.provenance
    return args
