"""Shared argparse behavior for PoreFM command-line interfaces."""

from __future__ import annotations

import argparse
import os
import sys


_PROGRAM_NAME_ENV = "POREFM_PROGRAM_NAME"


class PoreFMHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show meaningful defaults while preserving deliberate line breaks."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        normalized_help = help_text.lower()
        already_explains_delimiter = any(
            word in normalized_help
            for word in ("whitespace", "space-separated", "comma-separated")
        )
        if action.nargs in ("+", "*") and not already_explains_delimiter:
            help_text += (
                " Supply multiple values as whitespace-separated arguments; "
                "commas are not list delimiters."
            )
        elif (
            isinstance(action.nargs, int)
            and action.nargs > 1
            and not already_explains_delimiter
        ):
            help_text += (
                f" Supply exactly {action.nargs} whitespace-separated values."
            )
        if (
            "%(default)" not in help_text
            and "(default:" not in normalized_help
            and action.option_strings
            and not action.required
            and action.default is not argparse.SUPPRESS
            and action.default is not None
        ):
            if action.default == "":
                help_text += " (default: empty string)"
            else:
                help_text += " (default: %(default)s)"
        return help_text


class PoreFMArgumentParser(argparse.ArgumentParser):
    """Argument parser with consistent defaults, list hints, and empty help.

    Parsers created through the unified ``porefm`` dispatcher obtain their
    displayed command name from an internal environment variable. Calling a
    leaf command without arguments prints help and exits successfully instead
    of reporting missing required arguments.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("formatter_class") is argparse.ArgumentDefaultsHelpFormatter:
            kwargs["formatter_class"] = PoreFMHelpFormatter
        kwargs.setdefault("formatter_class", PoreFMHelpFormatter)
        program_name = os.environ.get(_PROGRAM_NAME_ENV)
        if program_name and "prog" not in kwargs:
            kwargs["prog"] = program_name
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        """Add an option while retaining otherwise blank help entries."""
        if kwargs.get("help") is not argparse.SUPPRESS:
            kwargs["help"] = kwargs.get("help") or " "
        return super().add_argument(*args, **kwargs)

    def format_help(self):
        """Populate default-only help for options lacking a description."""
        for action in self._actions:
            if action.help is argparse.SUPPRESS or (action.help or "").strip():
                continue
            if (
                action.option_strings
                and not action.required
                and action.default is not argparse.SUPPRESS
                and action.default is not None
            ):
                action.help = (
                    "(default: empty string)"
                    if action.default == ""
                    else "(default: %(default)s)"
                )
            else:
                action.help = None
        return super().format_help()

    def parse_args(self, args=None, namespace=None):
        """Print help and exit successfully when no command arguments exist."""
        if args is None and not sys.argv[1:]:
            self.print_help()
            self.exit(0)
        return super().parse_args(args, namespace)
