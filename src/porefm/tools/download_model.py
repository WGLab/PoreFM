"""Prefetch a public model release and print the verified local checkpoint path."""

from porefm.cli._parser import PoreFMArgumentParser
from porefm.cli.model_args import add_model_cache_arguments, model_reference_argument
from porefm.common.model_download import fetch_model


def main(argv=None):
    """Download one manifest-selected checkpoint; stdout contains only its path."""
    parser = PoreFMArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=model_reference_argument,
                        metavar='OWNER/REPO@vVERSION',
                        help='Public model and release tag, e.g. WGLab/PoreFM-DNA@v0.1.0.')
    parser.add_argument('--file', default=None,
                        help='Checkpoint filename from the release manifest. Base models default to backbone.pt; use pretraining.pt for pretraining heads and EMA.')
    add_model_cache_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = fetch_model(args.model, cache_dir=args.model_cache_dir,
                             offline=args.offline, filename=args.file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result.path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
