"""Lightweight command-group help and module dispatch."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from porefm.cli._runner import run_module


@dataclass(frozen=True)
class Command:
    """Describe a subcommand, its module, and any fixed backend argument prefix."""

    module: str
    description: str
    argument_prefix: tuple[str, ...] = ()


def format_command_help(
    commands: Mapping[str, Command], *, program_name: str, description: str
) -> str:
    """Describe a command group without importing its workflow modules."""
    width = max(len(name) for name in commands)
    command_lines = "\n".join(
        f"  {name:<{width}}  {command.description}"
        for name, command in commands.items()
    )
    return (
        f"usage: {program_name} <command> [options]\n\n"
        f"{description}\n\n"
        "commands:\n"
        f"{command_lines}\n\n"
        f"Run '{program_name} <command> --help' for command-specific options.\n"
        "Multi-value CLI options are whitespace-separated; commas are not list "
        "delimiters. Quote a single value that contains spaces.\n"
    )


def dispatch_commands(
    commands: Mapping[str, Command],
    argv: Sequence[str] | None = None,
    *,
    program_name: str,
    description: str,
    aliases: Mapping[str, str] | None = None,
) -> int:
    """Run a command or show group help, returning 2 for unknown commands.

    ``argv=None`` reads process arguments. A leading ``help`` forwards the
    remaining command path with ``--help``, including nested command groups.
    Leaf modules retain their normal ``__main__`` and exit-code semantics.
    Optional ``aliases`` map legacy names to registered commands without
    advertising them in group help. Leaf help retains the invoked name.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "help":
        args = [*args[1:], "--help"]

    if not args or args[0] in {"-h", "--help"}:
        print(format_command_help(
            commands, program_name=program_name, description=description
        ), end="")
        return 0

    command_name, *command_args = args
    command = commands.get((aliases or {}).get(command_name, command_name))
    if command is None:
        print(f"{program_name}: unknown command {command_name!r}\n", file=sys.stderr)
        print(format_command_help(
            commands, program_name=program_name, description=description
        ), file=sys.stderr, end="")
        return 2

    # Supply fixed workflow selectors without losing bare-command help.
    if command.argument_prefix:
        command_args = [*command.argument_prefix, *(command_args or ["--help"])]
    return run_module(
        command.module,
        argv=command_args,
        program_name=f"{program_name} {command_name}",
        help_if_empty=True,
    )
