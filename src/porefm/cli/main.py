"""Unified ``porefm`` console command and subcommand registry."""

from __future__ import annotations

from typing import Sequence

from porefm.cli._commands import Command, dispatch_commands, format_command_help


COMMANDS = {
    "models": Command("porefm.cli.models", "Download and cache versioned public model releases."),
    "pretrain": Command("porefm.training.pretrain", "Train the PoreFM foundation model on nanopore reads."),
    "finetune-modification": Command("porefm.training.supervised", "Finetune a per-base modification classifier"),
    "detect-modification": Command("porefm.inference.classifier", "Detect DNA and RNA modifications with a fine-tuned model."),
    "zero-shot": Command("porefm.cli.zero_shot", "Detect candidate modifications using canonical reference models."),
    "pairwise-comparison": Command("porefm.workflows.read_tagging.comparison", "Find differences between control and test samples."),
    "prototype": Command("porefm.cli.prototype", "Build reference prototypes and use them to classify modifications."),
    "utils": Command("porefm.cli.utils", "Prepare nanopore data and validate input configurations."),
}

ALIASES = {
    "infer": "detect-modification",
    "detect": "detect-modification",
    "supervised-finetune": "finetune-modification",
    "modification-finetune": "finetune-modification",
}

DESCRIPTION = "PoreFM: train models and detect DNA and RNA modifications from nanopore reads."


def format_help() -> str:
    """Return top-level help with an aligned, described subcommand list."""
    return format_command_help(
        COMMANDS, program_name="porefm", description=DESCRIPTION
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a unified PoreFM subcommand or display root help."""
    return dispatch_commands(
        COMMANDS, argv, program_name="porefm", description=DESCRIPTION, aliases=ALIASES
    )


if __name__ == "__main__":
    raise SystemExit(main())
