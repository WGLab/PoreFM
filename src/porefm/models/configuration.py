"""Validation and extraction helpers for checkpointed backbone configuration."""

from collections.abc import Mapping


EMA_SIGNAL_DROPOUT_MODES = frozenset({"independent", "shared", "disabled"})


def validate_backbone_config_dict(config, *, source="backbone config"):
    """Return a copied backbone config after enforcing required policy fields.

    ``ema_signal_dropout_mode`` is deliberately required rather than inferred:
    checkpoints created before the field existed must be migrated explicitly
    so continuation behavior cannot change silently.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"{source} must be a mapping, got {type(config).__name__}.")

    normalized = dict(config)
    if "ema_signal_dropout_mode" not in normalized:
        raise ValueError(
            f"{source} is missing required field 'ema_signal_dropout_mode'. "
            "Migrate the checkpoint explicitly with one of: "
            f"{sorted(EMA_SIGNAL_DROPOUT_MODES)}."
        )

    mode = normalized["ema_signal_dropout_mode"]
    if mode not in EMA_SIGNAL_DROPOUT_MODES:
        raise ValueError(
            f"{source} has invalid ema_signal_dropout_mode={mode!r}; "
            f"expected one of {sorted(EMA_SIGNAL_DROPOUT_MODES)}."
        )
    return normalized


def get_backbone_config_from_checkpoint(checkpoint, *, source="checkpoint"):
    """Extract and validate a backbone config from supported checkpoint layouts.

    Pretraining, classifier, canonical-null, and versioned manifest layouts use
    different top-level keys. No missing policy field is inferred.
    """
    # Canonical-null checkpoints retain a legacy head-only ``config`` mapping,
    # so explicit backbone keys must take precedence when several are present.
    for key in ("pretrain_config", "backbone_config", "config"):
        if checkpoint.get(key) is not None:
            return validate_backbone_config_dict(
                checkpoint[key],
                source=f"{source}[{key!r}]",
            )

    manifest = checkpoint.get("model_manifest") or {}
    backbone = manifest.get("backbone") or {}
    if backbone.get("config") is not None:
        return validate_backbone_config_dict(
            backbone["config"],
            source=f"{source} model_manifest backbone config",
        )

    raise KeyError(
        f"{source} has no backbone configuration under 'config', "
        "'pretrain_config', 'backbone_config', or model_manifest['backbone']['config']."
    )
