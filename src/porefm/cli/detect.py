"""Backward-compatible module alias for modification detection."""

from porefm.cli.detect_modification import main


if __name__ == "__main__":
    raise SystemExit(main())
