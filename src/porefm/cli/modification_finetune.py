"""Backward-compatible module alias for modification fine-tuning."""

from porefm.cli.finetune_modification import main


if __name__ == "__main__":
    raise SystemExit(main())
