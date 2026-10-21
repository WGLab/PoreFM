"""Command-line validation and schema inspection for PoreFM configs."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from porefm.cli._parser import PoreFMArgumentParser
from porefm.config.schemas import CONFIG_TYPES, get_schema
from porefm.config.validation import ConfigValidationError, validate_config_file


def build_parser() -> PoreFMArgumentParser:
    """Build ``porefm utils validate-config`` validation and schema subcommands."""
    parser = PoreFMArgumentParser(
        description="Validate PoreFM JSON configs or print their versioned schemas."
    )
    commands = parser.add_subparsers(dest="config_command", required=True)

    validate = commands.add_parser(
        "validate",
        help="Validate a JSON config and report its detected type/version.",
    )
    validate.add_argument("config", help="JSON configuration file to validate.")
    validate.add_argument(
        "--type",
        choices=CONFIG_TYPES,
        default=None,
        help="Expected config type; by default use config_type or infer from structure.",
    )
    validate.add_argument(
        "--class-labels",
        nargs="+",
        default=None,
        help=(
            "Optional classifier labels in index order for additional supervised "
            "class-name and class-index validation."
        ),
    )
    validate.add_argument(
        "--show-normalized",
        action="store_true",
        help="Print normalized JSON with schema headers and defaults applied.",
    )

    schema = commands.add_parser(
        "schema",
        help="Print the JSON Schema for one config type.",
    )
    schema.add_argument("type", choices=CONFIG_TYPES, help="Config type to describe.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a config or emit a schema, returning a shell status code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config_command == "schema":
        print(json.dumps(get_schema(args.type), indent=2))
        return 0

    try:
        validated = validate_config_file(
            args.config,
            expected_type=args.type,
            class_labels=args.class_labels,
        )
    except ConfigValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(
        f"VALID: {args.config} ({validated.config_type} schema v{validated.schema_version})"
    )
    if args.show_normalized:
        print(json.dumps(validated.data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
