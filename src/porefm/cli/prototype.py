"""Prototype extraction, effect fitting, and classification commands."""

from __future__ import annotations

from collections.abc import Sequence

from porefm.cli._commands import Command, dispatch_commands


COMMANDS = {
    "extract-kmer-prototypes": Command("porefm.workflows.prototypes.extract", "Extract or merge per-k-mer embedding prototypes."),
    "extract-effect-neighborhoods": Command("porefm.workflows.prototypes.effects_neighborhoods", "Extract oligo neighborhood means for footprint fitting."),
    "fit-effect-prototypes": Command("porefm.workflows.prototypes.effects_fit", "Fit context-conditioned effects and canonical noise metrics."),
    "classify": Command("porefm.workflows.prototypes.classify", "Classify reads using prototype distances, fitted scorers, or effects."),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a prototype command or list available commands."""
    return dispatch_commands(
        COMMANDS, argv, program_name="porefm prototype",
        description="PoreFM prototype extraction, effect fitting, and classification.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
