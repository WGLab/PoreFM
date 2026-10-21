"""Module entry point for DNA and RNA modification fine-tuning."""

from porefm.cli._runner import run_module


def main() -> int:
    """Run supervised modification fine-tuning with the process arguments."""
    return run_module("porefm.training.supervised")


if __name__ == "__main__":
    raise SystemExit(main())
