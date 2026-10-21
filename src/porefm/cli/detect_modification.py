"""Module entry point for checkpoint-based modification detection."""

from porefm.cli._runner import run_module


def main() -> int:
    """Run classifier inference with the supplied process arguments."""
    return run_module("porefm.inference.classifier")


if __name__ == "__main__":
    raise SystemExit(main())
