"""Construct inference or training models from trusted PoreFM checkpoints.

The loader understands legacy flat and versioned modular checkpoints. It can
restore the declared complete model or intentionally construct a smaller
topology while checking that only parameters belonging to omitted components
are left unused.
"""

from __future__ import annotations

from porefm.preprocessing import preprocessing_from_checkpoint
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from porefm.models.backbone.model import (
    ONTBackbone,
    ONTModelForPreTraining,
    PretrainConfig,
)
from porefm.models.checkpointing import (
    get_classification_state_dict_from_checkpoint,
    get_pretraining_state_dict_from_checkpoint,
    normalize_state_dict_keys,
    split_classification_state_dict,
    split_pretraining_state_dict,
)
from porefm.models.classifier import ONTModelForClassification
from porefm.models.configuration import get_backbone_config_from_checkpoint


ModelType = Literal["auto", "backbone", "pretraining", "classifier"]
_MODEL_TYPES = frozenset({"auto", "backbone", "pretraining", "classifier"})
_MODEL_CLASS_TYPES = {
    "ONTBackbone": "backbone",
    "ONTModelForPreTraining": "pretraining",
    "PretrainModel": "pretraining",
    "ONTModelForClassification": "classifier",
    "ClassifierModel": "classifier",
}


def _classifier_config_from_checkpoint(checkpoint, *, source):
    """Return a copied classifier config from legacy or manifest storage.

    ``source`` identifies the checkpoint in errors. ``KeyError`` is raised
    when neither the legacy top-level field nor the versioned head manifest
    contains a mapping.
    """
    config = checkpoint.get("classifier_config")
    if config is None:
        manifest = checkpoint.get("model_manifest") or {}
        config = (manifest.get("head_configs") or {}).get("classifier_head")
    if not isinstance(config, Mapping):
        raise KeyError(f"{source} has no classifier-head configuration.")
    return dict(config)


def _has_classifier_state(checkpoint):
    """Return whether metadata or parameter namespaces identify a classifier.

    Configuration and manifest declarations are checked before falling back
    to legacy flat state-dict keys. This is detection only; compatibility is
    verified later when weights are loaded.
    """
    if checkpoint.get("classifier_config") is not None:
        return True
    manifest = checkpoint.get("model_manifest") or {}
    if "classifier_head" in (manifest.get("heads") or ()):
        return True
    if "classifier_head" in (checkpoint.get("head_state_dicts") or {}):
        return True
    state = checkpoint.get("model_state_dict") or {}
    state = normalize_state_dict_keys(state)
    return any(key.startswith("classifier_head.") for key in state)


def _detect_model_type(checkpoint, *, source):
    """Infer the complete model role represented by a checkpoint.

    Explicit class declarations take precedence over state inspection. An
    unsupported declared task model is not silently reduced to a backbone;
    callers must request that extraction explicitly.
    """
    manifest = checkpoint.get("model_manifest") or {}
    declared = checkpoint.get("model_class") or manifest.get("model_class")
    if declared is not None:
        detected = _MODEL_CLASS_TYPES.get(declared)
        if detected is None:
            raise ValueError(
                f"{source} declares unsupported model_class={declared!r}; "
                "request model_type='backbone' to extract only its PoreFM backbone."
            )
        return detected

    if _has_classifier_state(checkpoint):
        return "classifier"

    head_names = set(checkpoint.get("head_state_dicts") or {})
    state = normalize_state_dict_keys(checkpoint.get("model_state_dict") or {})
    pretraining_prefixes = (
        "sublm_head.",
        "masklm_head.",
        "ema_signal_encoder.",
        "latent_head.",
    )
    if head_names.intersection(
        {"sublm_head", "masklm_head", "ema_signal_encoder", "latent_head"}
    ) or any(key.startswith(pretraining_prefixes) for key in state):
        return "pretraining"
    return "backbone"


def _checkpoint_backbone_state(checkpoint):
    """Return normalized backbone state from legacy or modular layouts.

    Classifier flat states carry the backbone below a ``pretrain.`` namespace,
    whereas pretraining states store backbone keys directly. Modular version-2
    checkpoints already provide the partitioned state.
    """
    if checkpoint.get("backbone_state_dict") is not None:
        return normalize_state_dict_keys(checkpoint["backbone_state_dict"])
    if _has_classifier_state(checkpoint):
        flat_state = get_classification_state_dict_from_checkpoint(checkpoint)
        backbone_state, _ = split_classification_state_dict(flat_state)
    else:
        flat_state = get_pretraining_state_dict_from_checkpoint(checkpoint)
        backbone_state, _ = split_pretraining_state_dict(flat_state)
    return normalize_state_dict_keys(backbone_state)


def _load_state_checked(model, state_dict, *, allowed_unexpected=()):
    """Load parameters while permitting only deliberately omitted namespaces.

    ``allowed_unexpected`` contains prefixes belonging to components pruned by
    loader options, such as reverse cross-attention or EMA heads. Every model
    parameter must still be present, and any unrelated checkpoint parameter
    raises ``RuntimeError`` so ``strict=False`` cannot hide incompatibility.
    """
    incompatible = model.load_state_dict(
        normalize_state_dict_keys(state_dict),
        strict=False,
    )
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(tuple(allowed_unexpected))
    ]
    if incompatible.missing_keys or unexpected:
        details = []
        if incompatible.missing_keys:
            details.append(f"missing keys: {sorted(incompatible.missing_keys)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise RuntimeError("checkpoint is incompatible with the requested model (" + "; ".join(details) + ")")


def _resolve_backbone_config(
    checkpoint,
    *,
    source,
    add_signal_cross_attn,
    include_ema,
):
    """Return a validated config with the requested topology reductions.

    Components may be removed but cannot be enabled when the checkpoint does
    not declare trained support. The copied checkpoint config is modified;
    checkpoint metadata itself remains unchanged.
    """
    config = get_backbone_config_from_checkpoint(checkpoint, source=source)
    checkpoint_has_signal_cross_attn = bool(
        config.get("add_signal_cross_attn", True)
        and int(config.get("num_layers_signal_cross", 0)) > 0
    )
    if add_signal_cross_attn is True and not checkpoint_has_signal_cross_attn:
        raise ValueError(
            "cannot enable signal-to-base cross-attention because the checkpoint "
            "was not trained with that tower"
        )

    use_signal_cross_attn = (
        checkpoint_has_signal_cross_attn
        if add_signal_cross_attn is None
        else bool(add_signal_cross_attn)
    )
    config["add_signal_cross_attn"] = use_signal_cross_attn

    checkpoint_has_ema = bool(config.get("add_ema", False))
    if include_ema is True and not checkpoint_has_ema:
        raise ValueError(
            "cannot include the EMA encoder because the checkpoint configuration "
            "does not declare EMA weights"
        )
    if include_ema is True and not use_signal_cross_attn:
        raise ValueError(
            "include_ema=True requires signal-to-base cross-attention; the current "
            "pretraining wrapper couples its EMA/latent path to contextual signal"
        )

    use_ema = checkpoint_has_ema if include_ema is None else bool(include_ema)
    config["add_ema"] = use_ema and use_signal_cross_attn
    return PretrainConfig(**config)


def _configure_usage(model, *, for_training, device, checkpoint):
    """Move a model and apply its requested execution/gradient policy.

    Training loads retain constructor gradient settings so EMA parameters stay
    frozen. Inference loads disable gradients in addition to selecting eval
    mode; a later ``model.train()`` changes module mode but does not re-enable
    those gradients.
    """
    model.preprocessing = preprocessing_from_checkpoint(checkpoint, required=False)
    if hasattr(model, "pretrain"):
        model.pretrain.preprocessing = model.preprocessing
    model.to(torch.device(device))
    if for_training:
        # Constructor defaults retain gradients for trainable parameters while
        # leaving the EMA target frozen.
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def load_model(
    checkpoint_path: str | Path,
    *,
    model_type: ModelType = "auto",
    add_signal_cross_attn: bool | None = None,
    include_ema: bool | None = None,
    classifier_signal_source: str | None = None,
    for_training: bool = False,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Construct a PoreFM model directly from one trusted checkpoint path.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Pretraining, classifier, or backbone checkpoint. Checkpoints are loaded
        with pickle enabled and must therefore come from a trusted source.
    model_type : {"auto", "backbone", "pretraining", "classifier"}, default="auto"
        Model to construct. ``"auto"`` restores the complete declared model;
        ``"backbone"`` extracts only the shared backbone from any supported
        PoreFM checkpoint.
    add_signal_cross_attn : bool or None, default=None
        Preserve the checkpoint topology when ``None``. ``False`` omits the
        signal-query-to-base cross-attention tower and ``contextual_ts`` while
        retaining base-query-to-signal attention and contextual base outputs.
        A tower absent from the checkpoint cannot be enabled.
    include_ema : bool or None, default=None
        For a pretraining model, ``None`` preserves the checkpoint's EMA
        setting and ``False`` omits the EMA encoder and latent head. Classifier
        inference omits auxiliary pretraining/EMA modules by default; passing
        ``True`` restores them when compatible weights are present. Backbones
        never construct EMA modules.
    classifier_signal_source : {"auto", "contextual_ts", "fixed_sa_ts"} or None
        Optional classifier-only signal source override. A classifier that
        requires ``contextual_ts`` cannot disable signal-to-base attention.
    for_training : bool, default=False
        If false, return an evaluation-mode model with gradients disabled. If
        true, return training mode with constructor-defined gradient settings,
        including a frozen EMA target.
    device : str or torch.device, default="cpu"
        Device receiving the constructed model after CPU checkpoint loading.

    Returns
    -------
    torch.nn.Module
        An :class:`ONTBackbone`, :class:`ONTModelForPreTraining`, or
        :class:`ONTModelForClassification` according to ``model_type``.
        ``model.preprocessing`` holds the resolved signal contract for use
        with datasets/readers, or None for checkpoints without identifying
        metadata. Signal workflows require an identified contract.

    Raises
    ------
    ValueError
        If a requested component was not trained in the checkpoint or an
        option combination is incompatible with the requested model.
    RuntimeError
        If required checkpoint parameters are missing or unrelated unexpected
        parameters remain after applying an intentional topology reduction.
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {sorted(_MODEL_TYPES)}, got {model_type!r}"
        )
    for option_name, option_value in (
        ("add_signal_cross_attn", add_signal_cross_attn),
        ("include_ema", include_ema),
    ):
        if option_value is not None and not isinstance(option_value, bool):
            raise TypeError(f"{option_name} must be a bool or None")
    if not isinstance(for_training, bool):
        raise TypeError("for_training must be a bool")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"checkpoint payload must be a mapping, got {type(checkpoint).__name__}"
        )
    source = str(checkpoint_path)
    resolved_type = (
        _detect_model_type(checkpoint, source=source)
        if model_type == "auto"
        else model_type
    )

    if resolved_type == "backbone":
        if include_ema is True:
            raise ValueError("model_type='backbone' never includes an EMA encoder")
        if classifier_signal_source is not None:
            raise ValueError("classifier_signal_source applies only to classifier models")
        config = _resolve_backbone_config(
            checkpoint,
            source=source,
            add_signal_cross_attn=add_signal_cross_attn,
            include_ema=False,
        )
        model = ONTBackbone(config)
        allowed = ()
        if not config.add_signal_cross_attn:
            # These tensors remain in a full checkpoint but have no receiving
            # modules in the intentionally reduced backbone topology.
            allowed = (
                "cross_attn.signal_to_base_cross_tower.",
                "final_norm_ts.",
            )
        _load_state_checked(
            model,
            _checkpoint_backbone_state(checkpoint),
            allowed_unexpected=allowed,
        )
        return _configure_usage(model, for_training=for_training, device=device, checkpoint=checkpoint)

    if resolved_type == "pretraining":
        if classifier_signal_source is not None:
            raise ValueError("classifier_signal_source applies only to classifier models")
        config = _resolve_backbone_config(
            checkpoint,
            source=source,
            add_signal_cross_attn=add_signal_cross_attn,
            include_ema=include_ema,
        )
        model = ONTModelForPreTraining(config)
        allowed = []
        if not config.add_signal_cross_attn:
            allowed.extend(
                ["cross_attn.signal_to_base_cross_tower.", "final_norm_ts."]
            )
        if not config.add_ema:
            allowed.extend(["ema_signal_encoder.", "latent_head."])
        state = get_pretraining_state_dict_from_checkpoint(checkpoint)
        _load_state_checked(model, state, allowed_unexpected=allowed)
        return _configure_usage(model, for_training=for_training, device=device, checkpoint=checkpoint)

    classifier_config = _classifier_config_from_checkpoint(
        checkpoint,
        source=source,
    )
    saved_classifier_type = classifier_config.get("classifier_type")
    if saved_classifier_type != "transformer_cross":
        # Legacy CNN/self checkpoints may record ``auto`` even though those
        # heads consume no signal tensor. Migrate that metadata during load.
        classifier_config.pop("signal_source", None)
    # Ordinary classifier inference needs only the shared backbone and task
    # head. Opting into EMA restores the coupled auxiliary pretraining wrapper.
    use_classifier_ema = bool(include_ema) if include_ema is not None else False
    backbone_config = get_backbone_config_from_checkpoint(checkpoint, source=source)
    model = ONTModelForClassification(
        classifier_config_dict=classifier_config,
        pretrain_config=backbone_config,
        add_signal_cross_attn=add_signal_cross_attn,
        backbone_signal_cross_attn=(
            "inherit" if add_signal_cross_attn is None else None
        ),
        classifier_signal_source=classifier_signal_source,
        enable_mtr=use_classifier_ema,
        enable_pretraining_heads=use_classifier_ema,
    )
    state = get_classification_state_dict_from_checkpoint(checkpoint)
    allowed = []
    if not model.pretrain.config.add_signal_cross_attn:
        allowed.extend(
            [
                "pretrain.cross_attn.signal_to_base_cross_tower.",
                "pretrain.final_norm_ts.",
            ]
        )
    if not use_classifier_ema:
        allowed.extend(
            [
                "pretrain.sublm_head.",
                "pretrain.masklm_head.",
                "pretrain.ema_signal_encoder.",
                "pretrain.latent_head.",
            ]
        )
    _load_state_checked(model, state, allowed_unexpected=allowed)
    return _configure_usage(model, for_training=for_training, device=device, checkpoint=checkpoint)
