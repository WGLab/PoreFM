"""Build an unfiltered, reusable .pfmi BAM index for POD5 inference."""

import json

from porefm.cli._parser import PoreFMArgumentParser


def main(argv=None):
    """Parse CLI options and publish a new index without altering the BAM."""
    parser = PoreFMArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True, help="One input BAM file; no BAI required.")
    parser.add_argument(
        "--output", default=None,
        help="Index destination. Defaults to <bam_path>.pfmi alongside the BAM (reads.bam becomes reads.bam.pfmi); existing files are never overwritten.",
    )
    parser.add_argument("--threads", type=int, default=1, help="BAM decompression threads.")
    args = parser.parse_args(argv)
    from porefm.data.bam_index import build_bam_index
    try:
        result = build_bam_index(args.bam, args.output, threads=args.threads)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
