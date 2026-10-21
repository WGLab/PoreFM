"""Lightweight chunk-boundary options shared by data-consuming commands."""

import json


def add_chunking_arguments(parser):
    """Add an explicit mode override and an optional standalone JSON file."""
    parser.add_argument(
        '--chunking_mode', choices=('auto', 'mv', 'hierarchical', 'dtw', 'inferred'), default=None,
        help='Chunk boundaries: auto (default) uses moves when present, otherwise hierarchical for DNA or DTW for RNA; mv requires moves; hierarchical/dtw ignore moves; inferred aliases hierarchical.',
    )
    parser.add_argument(
        '--chunking_config', default=None,
        help='JSON file containing e.g. {"chunking_mode": "dtw"}. Explicit --chunking_mode takes precedence.',
    )


def resolve_chunking_mode(args):
    """Resolve CLI > standalone JSON > auto, rejecting unknown config keys.

    This run-wide loader setting is independent of model architecture and BAM
    manifests. The resolved mode is explicitly passed into worker datasets.
    """
    mode = 'auto'
    path = getattr(args, 'chunking_config', None)
    if path is not None:
        with open(path) as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or set(config) != {'chunking_mode'}:
            raise ValueError('Chunking JSON must contain exactly one field: chunking_mode')
        mode = config['chunking_mode']
        if mode not in ('auto', 'mv', 'hierarchical', 'dtw', 'inferred'):
            raise ValueError(f'Invalid chunking_mode in {path}: {mode!r}')
    override = getattr(args, 'chunking_mode', None)
    if override is not None:
        mode = override
    if mode not in ('auto', 'mv', 'hierarchical', 'dtw', 'inferred'):
        raise ValueError(f'Invalid chunking_mode: {mode!r}')
    return mode
