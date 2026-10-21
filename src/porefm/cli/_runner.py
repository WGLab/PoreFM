"""Helpers for thin console and ``python -m`` entry points."""

import os
import runpy
import sys


def run_module(
    module_name: str,
    argv=None,
    *,
    program_name: str | None = None,
    help_if_empty: bool = True,
) -> int:
    """Execute a legacy module under the true ``__main__`` identity.

    ``alter_sys=True`` makes spawned workers see the same entry-point semantics
    as direct script execution, allowing target-defined datasets and collators
    to be pickled. ``argv`` replaces the target arguments when supplied.
    ``program_name`` controls the command displayed by shared PoreFM parsers,
    and ``help_if_empty`` converts a bare leaf command to ``--help``. Returns
    zero if the module finishes normally.
    """
    original_argv = sys.argv[:]
    original_program_name = os.environ.get("POREFM_PROGRAM_NAME")
    target_program = program_name or original_argv[0]
    target_args = list(original_argv[1:] if argv is None else argv)
    if help_if_empty and not target_args:
        target_args = ["--help"]

    sys.argv[:] = [target_program, *target_args]
    os.environ["POREFM_PROGRAM_NAME"] = target_program
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return 0
    finally:
        sys.argv[:] = original_argv
        if original_program_name is None:
            os.environ.pop("POREFM_PROGRAM_NAME", None)
        else:
            os.environ["POREFM_PROGRAM_NAME"] = original_program_name
