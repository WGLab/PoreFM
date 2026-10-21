"""Canonical reference preparation and calibrated zero-shot detection."""

from __future__ import annotations

from collections.abc import Sequence

from porefm.cli._commands import Command, dispatch_commands
from porefm.cli.canonical_null_args import STAGE_DESCRIPTIONS


COMMANDS = {
    stage: Command("porefm.workflows.canonical_null.predictor", description, (stage,))
    for stage, description in STAGE_DESCRIPTIONS.items()
}
COMMANDS.update({
    "pipeline": Command("porefm.workflows.canonical_null.pipeline", "Run reference training and calibration stages from one configuration."),
    "detect": Command("porefm.workflows.canonical_null.compare", "Score reads against a canonical reference and write calibrated detections."),
})


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a zero-shot stage or list available stages."""
    return dispatch_commands(
        COMMANDS, argv, program_name="porefm zero-shot",
        description="Prepare canonical references and detect deviations without modification-labelled training data.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
