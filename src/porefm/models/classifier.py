"""Per-base classification heads built on PoreFM backbone embeddings."""

import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass, asdict

from typing import Optional, Dict, Any, Union, Tuple

from porefm.models.backbone.self_attention import SelfAttnBlock
from porefm.models.backbone.cross_attention import FlexCrossAttention
from porefm.models.backbone.utils import LM_Head, strip_module_prefix
from porefm.models.backbone.model import CrossAttnBlock, ONTBackbone, ONTModelForPreTraining, PretrainConfig
from porefm.models.checkpointing import get_classification_state_dict_from_checkpoint
from porefm.models.configuration import get_backbone_config_from_checkpoint


def _project_classifier_logits(projection: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Project ``(..., D)`` features to float32 logits, including under autocast.

    Projection parameters retain their normal float32 storage. Disabling
    autocast before the projection preserves small class-score differences
    that cannot be recovered by casting already-rounded logits afterward.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        return projection(features.float())


class CrossBlock_Head(nn.Module):
    """Cross-attention classifier head over base and signal embeddings.

    ``config`` must provide the dimensions/options consumed by
    ``CrossAttnBlock`` plus ``hidden_dim``, ``fc_dim_ff``, and ``out_dim``.
    Forward inputs have shapes ``(B, L_base, D)`` and ``(B, L_signal, D)`` with
    matching boolean masks. It returns contextual base embeddings and logits
    of shape ``(B, L_base, out_dim)``. The final two-layer projection and its
    logits are float32 during both training and inference. With
    ``center_source='backbone'``, forward also requires the final backbone
    ``u_center``; no head center predictor is
    constructed and the supplied centers are detached before use.
    """
    def __init__(self, config):
        
        super().__init__()
        
        query_modality='seq'
        self.center_source = _resolve_classifier_center_source(
            config.classifier_type, config.center_source
        )
        
        self.cross_attn = CrossAttnBlock(config, query_modality, 
                                         predict_center=(config.predict_center and self.center_source == 'head'),
                                         residual_control=config.residual_control)
        
        self.classifier_head = LM_Head(input_dim = config.hidden_dim,
                                  dim_ff=config.fc_dim_ff,
                                  out_dim=config.out_dim,
                                  dropout=0.0)
        
    def forward(self, base_seq: torch.Tensor, ts: torch.Tensor,
                mask_base_seq: torch.Tensor, mask_ts: torch.Tensor, *,
                u_center: Optional[torch.Tensor] = None, **kwargs):
        """Contextualize bases against signal and return embeddings plus logits.

        ``u_center`` is required only for backbone reuse and has shape
        ``(B, L_base)`` in signal-token coordinates, as returned by the final
        base-to-signal backbone layer. Both signal sources preserve that token
        axis. Reuse supplies the attention distance prior, while the head's
        attention projections, mixer, residual gate, and FFN remain trainable.
        """
        
        if self.center_source == 'backbone':
            if u_center is None:
                raise ValueError("classifier_center_source='backbone' requires backbone u_center.")
            if u_center.shape != base_seq.shape[:2]:
                raise ValueError("Backbone u_center must have shape (B, L_base) matching classifier queries.")
            if u_center.device != base_seq.device:
                raise ValueError("Backbone u_center and classifier queries must be on the same device.")

            # Use the block's existing modules with external centers. Keeping
            # this composition in the head preserves the backbone forward API.
            block = self.cross_attn
            x = block.pre_attention_mixer(base_seq, mask_base_seq)
            raw_qkv, _ = block.cross_attention(
                q_in=block.norm_q(x),
                kv_in=block.norm_kv(ts),
                q_mask=mask_base_seq,
                kv_mask=mask_ts,
                u_center=u_center.detach(),
            )
            ca_out = block.gate * x + raw_qkv
            base_seq = ca_out + block.MLP(block.norm_ffn(ca_out))
        else:
            base_seq, _, _, _, _ = self.cross_attn(base_seq, ts, mask_base_seq, mask_ts)
        logits = _project_classifier_logits(self.classifier_head, base_seq)
        
        return base_seq, logits
    

class SelfBlock_Head(nn.Module):
    """Self-attention classifier head over contextual base embeddings.

    Signal inputs are accepted for a common head interface but not used. The
    return value is ``(base_embeddings, logits)`` with logits shaped
    ``(B, L_base, out_dim)``. The final two-layer projection and its logits
    remain float32 under autocast during both training and inference.
    """
    def __init__(self, config):
        super().__init__()
        
        query_modality='seq'
               
        self.attn  = SelfAttnBlock(config, query_modality)
        
        self.classifier_head = LM_Head(input_dim = config.hidden_dim,
                                  dim_ff=config.fc_dim_ff,
                                  out_dim=config.out_dim,
                                  dropout=0.0)

    def forward(self, base_seq: torch.Tensor, ts: torch.Tensor,
                mask_base_seq: torch.Tensor, mask_ts: torch.Tensor, **kwargs):
        """Self-attend over valid bases and return embeddings plus logits."""
        base_seq = self.attn(base_seq, mask_base_seq)
        logits = _project_classifier_logits(self.classifier_head, base_seq)
        
        return base_seq, logits
    
class CNN_head(nn.Module):
    """Two-layer same-length convolutional per-base classifier.

    Parameters
    ----------
    input_dim : int
        Input embedding width.
    dim_ff : int
        Hidden channel width after the second convolution.
    out_dim : int
        Number of output classes per base token.
    dropout : float, default=0.0
        Retained for configuration compatibility; the current convolutional
        path does not apply dropout.
    kernel_size : int, default=3
        Kernel width for both same-padded convolutions.
    residual, base_emb_only : bool, default=False
        Stored compatibility options. The current forward path classifies
        ``base_emb`` directly and does not consume ``base_proj`` or the
        modality gate.
    """

    def __init__(self, input_dim, dim_ff, out_dim, dropout=0.0, kernel_size=3, residual=False, base_emb_only=False):
        super().__init__()        
        self.cnv1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim//2, kernel_size=kernel_size, padding='same', bias=True)
        self.cnv2 = nn.Conv1d(in_channels=input_dim//2, out_channels=dim_ff, kernel_size=kernel_size, padding='same', bias=True)

        self.act  = nn.GELU()
        
        self.base_emb_only=base_emb_only
        
        nn.init.xavier_uniform_(self.cnv1.weight)
        nn.init.xavier_uniform_(self.cnv2.weight)
        
        self.residual=residual
        
        if self.base_emb_only==False:
            self.modality_gate = nn.Sequential(
                nn.Linear(2 * input_dim + self.residual, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
            )
        
        self.fc = nn.Linear(dim_ff, out_dim)
        
    def forward(self, base_emb, base_proj=None, *args, **kwargs):
        """Return per-base logits for embeddings shaped ``(B, L, input_dim)``.

        Extra modality inputs are accepted for compatibility with the common
        classifier-head interface, but this head consumes only
        ``base_emb``. The final linear projection and returned logits are
        float32, including when the convolutional body runs under autocast.
        """
        x=base_emb

        x = x.transpose(1, 2)  # Conv1d uses (batch, channels, length).
        
        x = self.cnv1(x)
        x = self.act(x)
        
        x = self.cnv2(x)
        x = self.act(x).transpose(1, 2)
        
        x = _project_classifier_logits(self.fc, x)
        
        return x
    
@dataclass
class ClassifierConfig_CNN:
    """Configuration for the convolutional per-base classifier head."""
    hidden_dim: int                 # embedding dimensions
    out_dim: int                    # number of classes (per position)
    fc_dim_ff: int
    classifier_type: str = 'cnn'
    base_emb_only: bool = False
    signal_source: Optional[str] = None
        
@dataclass
class Transformer_Config:
    """Configuration shared by self- and cross-attention classifier heads.

    ``center_source`` applies only to cross heads: ``head`` uses the head's
    own center calculation and ``backbone`` reuses detached final backbone
    centers without a head center predictor. Missing values resolve to ``head``
    for cross heads and remain ``None`` for self-attention heads.
    """
    hidden_dim: int                 # embedding dimensions
    dim_feedforward: int
    out_dim: int                    # number of classes (per position)
    fc_dim_ff: int
        
    num_heads: int
    kv_num_heads: int
        
    classifier_type: str
        
    base_seq_window: tuple = (-1,-1)
    dropout: float = 0
    
    self_attn_temp: float = 1.0
        
    predict_center: bool = True
    pre_attention_mixer_type: Optional[str] = None
    residual_control: Optional[str] = None
    
    cross_attn_temp: float = 1.0
    expected_ratio: Optional[int] = None
    signal_source: Optional[str] = None
    center_source: Optional[str] = None


_CLASSIFIER_TYPES = frozenset({'cnn', 'transformer_self', 'transformer_cross'})
_SIGNAL_SOURCES = frozenset({'auto', 'contextual_ts', 'fixed_sa_ts'})
_SIGNAL_CROSS_ATTN_MODES = frozenset({'auto', 'inherit', 'on', 'off'})


def _resolve_classifier_center_source(
    classifier_type: str, center_source: Optional[str],
) -> Optional[str]:
    """Validate the cross-head center policy, preserving old heads by default."""
    if classifier_type != 'transformer_cross':
        if center_source is not None:
            raise ValueError("--classifier_center_source applies only to transformer_cross classifier heads.")
        return None
    if center_source is None:
        return 'head'
    if center_source not in {'head', 'backbone'}:
        raise ValueError("classifier_center_source must be 'head' or 'backbone'.")
    return center_source


def _supports_signal_cross_attn(pretrain_config_dict: dict) -> bool:
    """Return whether a checkpoint config declares a trained reverse tower."""
    return (
        bool(pretrain_config_dict.get('add_signal_cross_attn', True))
        and int(pretrain_config_dict.get('num_layers_signal_cross', 0)) > 0
    )


def checkpoint_has_signal_cross_attn_weights(
    state_dict: dict,
    *,
    prefix: str = '',
) -> bool:
    """Return whether state contains both reverse-tower and output-norm weights.

    Parameters
    ----------
    state_dict : dict
        Normalized pretraining or classifier state dictionary.
    prefix : str, default=''
        Namespace before backbone keys. Classifier states use ``"pretrain."``;
        standalone pretraining states use the empty prefix.
    """
    tower_prefix = f'{prefix}cross_attn.signal_to_base_cross_tower.'
    norm_prefix = f'{prefix}final_norm_ts.'
    return (
        any(key.startswith(tower_prefix) for key in state_dict)
        and any(key.startswith(norm_prefix) for key in state_dict)
    )


def _normalize_classifier_type(classifier_type: str) -> str:
    if classifier_type not in _CLASSIFIER_TYPES:
        raise ValueError(
            f"classifier_type must be one of {sorted(_CLASSIFIER_TYPES)}, got {classifier_type!r}"
        )
    return classifier_type


def _normalize_classifier_signal_source(signal_source: Optional[str]) -> str:
    signal_source = signal_source or 'auto'
    if signal_source not in _SIGNAL_SOURCES:
        raise ValueError(
            f"classifier_signal_source must be one of {sorted(_SIGNAL_SOURCES)}, got {signal_source!r}"
        )
    return signal_source


def _normalize_signal_cross_attn_mode(
    mode: Optional[str],
    add_signal_cross_attn: Optional[bool],
    *,
    default: Optional[str],
) -> Optional[str]:
    """Normalize the public mode and legacy boolean constructor option."""
    if add_signal_cross_attn is not None and not isinstance(add_signal_cross_attn, bool):
        raise TypeError("add_signal_cross_attn must be a bool or None")
    if mode is not None and mode not in _SIGNAL_CROSS_ATTN_MODES:
        raise ValueError(
            "backbone_signal_cross_attn must be one of "
            f"{sorted(_SIGNAL_CROSS_ATTN_MODES)}, got {mode!r}"
        )

    if add_signal_cross_attn is not None:
        legacy_mode = 'on' if add_signal_cross_attn else 'off'
        if mode is not None and mode != legacy_mode:
            raise ValueError(
                "Conflicting signal-to-base cross-attention controls: "
                f"backbone_signal_cross_attn={mode!r} and "
                f"add_signal_cross_attn={add_signal_cross_attn!r}."
            )
        return legacy_mode
    return default if mode is None else mode


def _resolve_auto_signal_source(checkpoint_supports_tower: bool) -> str:
    return 'contextual_ts' if checkpoint_supports_tower else 'fixed_sa_ts'


def _resolve_new_classifier_contract(
    pretrain_config_dict: dict,
    classifier_type: str,
    classifier_signal_source: Optional[str] = None,
    backbone_signal_cross_attn: Optional[str] = None,
    add_signal_cross_attn: Optional[bool] = None,
    enable_mtr: bool = False,
    checkpoint_supports_signal_cross_attn: Optional[bool] = None,
) -> Tuple[PretrainConfig, Optional[str]]:
    """Resolve execution for a newly initialized classifier head.

    ``auto`` is dependency-aware: a cross head uses ``contextual_ts`` and the
    signal-to-base tower when the pretrained checkpoint supports that tower,
    otherwise it falls back to ``fixed_sa_ts``. CNN and self-attention heads
    consume no signal tensor and omit the tower unless MTR or an explicit
    ``inherit``/``on`` mode requires it.
    """
    pretrain_config_dict = dict(pretrain_config_dict)
    classifier_type = _normalize_classifier_type(classifier_type)
    config_supports_signal_cross_attn = _supports_signal_cross_attn(pretrain_config_dict)
    checkpoint_has_signal_cross_attn = config_supports_signal_cross_attn and (
        True
        if checkpoint_supports_signal_cross_attn is None
        else bool(checkpoint_supports_signal_cross_attn)
    )
    mode = _normalize_signal_cross_attn_mode(
        backbone_signal_cross_attn,
        add_signal_cross_attn,
        default='auto',
    )

    if classifier_type == 'transformer_cross':
        requested_signal_source = _normalize_classifier_signal_source(classifier_signal_source)
        if requested_signal_source == 'auto':
            signal_source = (
                'fixed_sa_ts'
                if mode == 'off'
                else _resolve_auto_signal_source(checkpoint_has_signal_cross_attn)
            )
        else:
            signal_source = requested_signal_source
    else:
        if classifier_signal_source is not None:
            raise ValueError(
                "--classifier_signal_source applies only to transformer_cross classifier heads."
            )
        signal_source = None

    if signal_source == 'contextual_ts' and not checkpoint_has_signal_cross_attn:
        raise ValueError("classifier_signal_source='contextual_ts' requires a checkpoint with signal-to-base cross-attention.")
    if enable_mtr and not checkpoint_has_signal_cross_attn:
        raise ValueError("MTR auxiliary pretraining requires a checkpoint with signal-to-base cross-attention.")

    dependency_requires_tower = signal_source == 'contextual_ts' or enable_mtr
    if mode == 'auto':
        use_signal_cross_attn = checkpoint_has_signal_cross_attn and dependency_requires_tower
    elif mode == 'inherit':
        use_signal_cross_attn = checkpoint_has_signal_cross_attn
    elif mode == 'on':
        if not checkpoint_has_signal_cross_attn:
            raise ValueError(
                "Cannot enable signal-to-base cross-attention because the loaded checkpoint was not trained with it."
            )
        use_signal_cross_attn = True
    else:
        use_signal_cross_attn = False

    if not use_signal_cross_attn and dependency_requires_tower:
        raise ValueError(
            "Cannot disable signal-to-base cross-attention when contextual_ts or MTR is requested."
        )
    if use_signal_cross_attn and signal_source == 'fixed_sa_ts' and not enable_mtr:
        warnings.warn(
            "signal-to-base cross-attention is enabled but the classifier uses fixed_sa_ts; "
            "the tower is retained but unused by the classifier.",
            UserWarning,
            stacklevel=2,
        )

    pretrain_config_dict['add_signal_cross_attn'] = use_signal_cross_attn
    pretrain_config_dict['add_ema'] = bool(enable_mtr)

    return PretrainConfig(**pretrain_config_dict), signal_source


def _migrate_saved_signal_source(
    classifier_config_dict: dict,
    *,
    saved_add_signal_cross_attn: bool,
) -> Optional[str]:
    """Resolve legacy checkpoint ``auto``/missing signal-source metadata."""
    classifier_type = _normalize_classifier_type(classifier_config_dict.get('classifier_type'))
    if classifier_type != 'transformer_cross':
        return None

    saved_source = _normalize_classifier_signal_source(
        classifier_config_dict.get('signal_source', 'auto')
    )
    if saved_source == 'auto':
        return _resolve_auto_signal_source(saved_add_signal_cross_attn)
    if saved_source == 'contextual_ts' and not saved_add_signal_cross_attn:
        raise ValueError(
            "Classifier checkpoint requests contextual_ts but records "
            "add_signal_cross_attn=False."
        )
    return saved_source


def _resolve_continuation_classifier_contract(
    pretrain_config_dict: dict,
    classifier_config_dict: dict,
    *,
    classifier_signal_source: Optional[str] = None,
    backbone_signal_cross_attn: Optional[str] = None,
    add_signal_cross_attn: Optional[bool] = None,
    classifier_type: Optional[str] = None,
    enable_mtr: bool = False,
    checkpoint_contains_signal_cross_attn_weights: Optional[bool] = None,
) -> Tuple[PretrainConfig, dict]:
    """Migrate legacy metadata and enforce exact classifier continuation.

    The saved classifier type, resolved signal tensor, and effective
    signal-to-base topology are authoritative. Explicit requests are accepted
    only when they resolve to that exact saved contract.
    """
    pretrain_config_dict = dict(pretrain_config_dict)
    classifier_config_dict = dict(classifier_config_dict)
    saved_classifier_type = _normalize_classifier_type(
        classifier_config_dict.get('classifier_type')
    )
    saved_add_signal_cross_attn = bool(
        pretrain_config_dict.get('add_signal_cross_attn', True)
    )

    if saved_add_signal_cross_attn and int(
        pretrain_config_dict.get('num_layers_signal_cross', 0)
    ) <= 0:
        raise ValueError(
            "Classifier checkpoint records add_signal_cross_attn=True but has no "
            "signal-to-base cross-attention layers."
        )
    if (
        saved_add_signal_cross_attn
        and checkpoint_contains_signal_cross_attn_weights is False
    ):
        raise ValueError(
            "Classifier checkpoint records add_signal_cross_attn=True but is missing "
            "signal-to-base cross-attention weights."
        )

    saved_signal_source = _migrate_saved_signal_source(
        classifier_config_dict,
        saved_add_signal_cross_attn=saved_add_signal_cross_attn,
    )

    if classifier_type is not None:
        requested_classifier_type = _normalize_classifier_type(classifier_type)
        if requested_classifier_type != saved_classifier_type:
            raise ValueError(
                "--classifier_type cannot change when continuing a classifier checkpoint: "
                f"requested {requested_classifier_type!r}, saved {saved_classifier_type!r}."
            )

    if classifier_signal_source is not None:
        if saved_classifier_type != 'transformer_cross':
            raise ValueError(
                "--classifier_signal_source applies only to transformer_cross classifier heads."
            )
        requested_source = _normalize_classifier_signal_source(classifier_signal_source)
        if requested_source == 'auto':
            requested_source = _resolve_auto_signal_source(saved_add_signal_cross_attn)
        if requested_source != saved_signal_source:
            raise ValueError(
                "--classifier_signal_source cannot change when continuing a classifier checkpoint: "
                f"requested {requested_source!r}, saved {saved_signal_source!r}."
            )

    mode = _normalize_signal_cross_attn_mode(
        backbone_signal_cross_attn,
        add_signal_cross_attn,
        default=None,
    )
    if mode is not None:
        if mode == 'inherit':
            requested_add_signal_cross_attn = saved_add_signal_cross_attn
        elif mode == 'auto':
            dependency_requires_tower = saved_signal_source == 'contextual_ts' or enable_mtr
            requested_add_signal_cross_attn = (
                _supports_signal_cross_attn(pretrain_config_dict)
                and dependency_requires_tower
            )
        else:
            requested_add_signal_cross_attn = mode == 'on'
        if requested_add_signal_cross_attn != saved_add_signal_cross_attn:
            raise ValueError(
                "--backbone_signal_cross_attn cannot change when continuing a classifier checkpoint: "
                f"requested effective value {requested_add_signal_cross_attn}, "
                f"saved {saved_add_signal_cross_attn}."
            )

    if saved_signal_source == 'contextual_ts' and not saved_add_signal_cross_attn:
        raise ValueError(
            "Saved classifier signal source contextual_ts requires "
            "add_signal_cross_attn=True."
        )
    if enable_mtr:
        if not saved_add_signal_cross_attn:
            raise ValueError(
                "MTR auxiliary pretraining requires the continued checkpoint's "
                "signal-to-base cross-attention tower."
            )
        if not bool(pretrain_config_dict.get('add_ema', False)):
            raise ValueError(
                "MTR auxiliary pretraining requires the continued checkpoint's EMA/latent architecture."
            )

    classifier_config_dict['signal_source'] = saved_signal_source
    return PretrainConfig(**pretrain_config_dict), classifier_config_dict


class ONTModelForClassification(nn.Module):
    """Attach a configurable per-base classifier to a pretrained PoreFM model.

    Parameters
    ----------
    classifier_config_dict : dict
        Head configuration. ``classifier_type`` is ``"cnn"``,
        ``"transformer_self"``, or ``"transformer_cross"`` and ``hidden_dim``
        must equal the backbone width.
    pretrain_config : PretrainConfig or dict, optional
        Backbone configuration when constructing without a classifier
        checkpoint.
    model_load_path : str or None
        Classifier checkpoint. Saved backbone/head configuration and weights
        are loaded when provided.
    add_signal_cross_attn : bool or None
        Legacy boolean spelling of ``backbone_signal_cross_attn=on/off``.
    backbone_signal_cross_attn : {"auto", "inherit", "on", "off"}, optional
        Scratch-only execution policy. Continuation accepts an explicit value
        only when it matches the saved effective topology.
    classifier_signal_source : {"auto", "contextual_ts", "fixed_sa_ts"}, optional
        Signal embedding consumed by a transformer-cross head. ``auto``
        chooses contextual signal when the checkpoint supports it and fixed
        self-attention signal otherwise. Invalid for CNN/self heads.
    classifier_type : str, optional
        Validation-only requested type for classifier-checkpoint continuation.
    enable_mtr : bool, default=False
        Enable EMA/latent components required by the auxiliary masked-trace
        reconstruction task. Requires checkpoint signal cross-attention.
    enable_pretraining_heads : bool, default=False
        Construct the full pretraining wrapper instead of a bare backbone so
        auxiliary pretraining heads remain callable.
    checkpoint_supports_signal_cross_attn : bool or None, default=None
        Scratch-construction capability derived from checkpoint weights. When
        omitted, declared configuration capability is trusted.
    classifier_center_source : {"head", "backbone"}, optional
        Cross-head alignment policy. New heads default to their own centers;
        ``backbone`` uses detached final backbone centers without learning a
        new center predictor. Continuation preserves the saved policy and
        rejects explicit changes. Does not freeze other backbone parameters.

    Attributes
    ----------
    pretrain : ONTBackbone or ONTModelForPreTraining
        Backbone or auxiliary-head wrapper.
    classifier_head : torch.nn.Module
        Selected per-base head.
    signal_key : str or None
        Resolved backbone output used by a cross-attention head. CNN and
        self-attention heads consume no signal tensor and store ``None``.
    """

    def __init__(
        self,
        classifier_config_dict: Optional[dict] = None,
        pretrain_config=None,
        model_load_path = None,
        add_signal_cross_attn: Optional[bool] = None,
        backbone_signal_cross_attn: Optional[str] = None,
        classifier_signal_source: Optional[str] = None,
        classifier_type: Optional[str] = None,
        enable_mtr: bool = False,
        enable_pretraining_heads: bool = False,
        checkpoint_supports_signal_cross_attn: Optional[bool] = None,
        classifier_center_source: Optional[str] = None,
    ):
        super().__init__()
        classifier_config_dict = dict(classifier_config_dict or {})
        self.classifier_initialization = (
            'classifier_checkpoint_continuation'
            if model_load_path
            else 'new_head_from_pretrained_backbone'
        )
        
        if model_load_path:
            pretrain_config, classifier_config_dict = self.load_configs(
                model_load_path,
                add_signal_cross_attn=add_signal_cross_attn,
                backbone_signal_cross_attn=backbone_signal_cross_attn,
                classifier_signal_source=classifier_signal_source,
                classifier_type=classifier_type,
                enable_mtr=enable_mtr,
            )
            self.checkpoint_supports_signal_cross_attn = _supports_signal_cross_attn(
                asdict(pretrain_config)
            )
        else:
            configured_classifier_type = _normalize_classifier_type(
                classifier_config_dict.get('classifier_type')
            )
            if classifier_type is not None and classifier_type != configured_classifier_type:
                raise ValueError(
                    f"classifier_type={classifier_type!r} does not match classifier_config_dict "
                    f"type {configured_classifier_type!r}."
                )
            source_pretrain_config = (
                asdict(pretrain_config)
                if isinstance(pretrain_config, PretrainConfig)
                else dict(pretrain_config)
            )
            self.checkpoint_supports_signal_cross_attn = (
                _supports_signal_cross_attn(source_pretrain_config)
                and (
                    True
                    if checkpoint_supports_signal_cross_attn is None
                    else bool(checkpoint_supports_signal_cross_attn)
                )
            )
            pretrain_config, resolved_signal_source = _resolve_new_classifier_contract(
                source_pretrain_config,
                classifier_type=configured_classifier_type,
                classifier_signal_source=(
                    classifier_signal_source
                    if classifier_signal_source is not None
                    else classifier_config_dict.get('signal_source')
                ),
                backbone_signal_cross_attn=backbone_signal_cross_attn,
                add_signal_cross_attn=add_signal_cross_attn,
                enable_mtr=enable_mtr,
                checkpoint_supports_signal_cross_attn=checkpoint_supports_signal_cross_attn,
            )
            classifier_config_dict['signal_source'] = resolved_signal_source

        configured_type = classifier_config_dict['classifier_type']
        center_source = _resolve_classifier_center_source(
            configured_type, classifier_config_dict.get('center_source')
        )
        if classifier_center_source is not None:
            requested_center_source = _resolve_classifier_center_source(
                configured_type, classifier_center_source
            )
            if model_load_path and requested_center_source != center_source:
                raise ValueError(
                    "--classifier_center_source cannot change when continuing a classifier checkpoint: "
                    f"requested {requested_center_source!r}, saved {center_source!r}."
                )
            center_source = requested_center_source
        if configured_type != 'cnn':
            classifier_config_dict['center_source'] = center_source
        if center_source == 'backbone':
            if pretrain_config.num_layers_base_cross < 1:
                raise ValueError("Backbone center reuse requires at least one base-to-signal layer.")
            classifier_config_dict['predict_center'] = False

        backbone_cls = ONTModelForPreTraining if enable_pretraining_heads else ONTBackbone
        self.pretrain = backbone_cls(pretrain_config)
        
        assert classifier_config_dict['hidden_dim']==self.pretrain.config.hidden_dim, 'Classifier and Embedding models should have the same hidden_dim'
                
        if classifier_config_dict['classifier_type']=='transformer_cross':
            self.classifier_config = Transformer_Config(**classifier_config_dict)
            self.classifier_head = CrossBlock_Head(self.classifier_config)
            
        elif classifier_config_dict['classifier_type']=='transformer_self':
            self.classifier_config = Transformer_Config(**classifier_config_dict)
            self.classifier_head = SelfBlock_Head(self.classifier_config)
            
        elif classifier_config_dict['classifier_type']=='cnn':
            self.classifier_config = ClassifierConfig_CNN(**classifier_config_dict)
            self.classifier_head = CNN_head(input_dim = self.classifier_config.hidden_dim,
                                  dim_ff=self.classifier_config.fc_dim_ff,
                                  out_dim=self.classifier_config.out_dim, base_emb_only=self.classifier_config.base_emb_only,
                                  dropout=0.0)
        
        if model_load_path:
            model_ckpt = torch.load(model_load_path, map_location="cpu")
            state_dict=get_classification_state_dict_from_checkpoint(model_ckpt)
            state_dict=strip_module_prefix(state_dict)
            if enable_pretraining_heads:
                required_prefixes = ['pretrain.sublm_head.', 'pretrain.masklm_head.']
                if enable_mtr:
                    required_prefixes.extend(['pretrain.ema_signal_encoder.', 'pretrain.latent_head.'])
                missing_prefixes = [
                    prefix for prefix in required_prefixes
                    if not any(key.startswith(prefix) for key in state_dict)
                ]
                if missing_prefixes:
                    raise ValueError(
                        "Cannot enable auxiliary pretraining from this classifier checkpoint because "
                        f"it is missing weights with prefixes: {missing_prefixes}"
                    )
            self.load_state_dict(state_dict, strict=False)
        
        self.signal_key = self._resolve_signal_key(
            self.classifier_config.classifier_type,
            self.classifier_config.signal_source,
        )

    @property
    def backbone(self):
        """Return the pretrained module retained under the legacy ``pretrain`` name."""
        return self.pretrain

    def execution_contract(self) -> Dict[str, Any]:
        """Return the resolved classifier/backbone execution design.

        The mapping is suitable for startup logging and checkpoint metadata.
        ``contextual_base_seq`` is always the classifier's base input. Only a
        transformer-cross head additionally consumes ``signal_key``.
        """
        return {
            'initialization': self.classifier_initialization,
            'classifier_type': self.classifier_config.classifier_type,
            'classifier_base_source': 'contextual_base_seq',
            'classifier_signal_source': self.classifier_config.signal_source,
            'resolved_signal_key': self.signal_key,
            'checkpoint_supports_signal_cross_attn': bool(
                self.checkpoint_supports_signal_cross_attn
            ),
            'effective_add_signal_cross_attn': bool(
                self.pretrain.config.add_signal_cross_attn
            ),
            'configured_signal_cross_layers': int(
                self.pretrain.config.num_layers_signal_cross
            ),
            'effective_signal_cross_layers': (
                int(self.pretrain.config.num_layers_signal_cross)
                if self.pretrain.config.add_signal_cross_attn
                else 0
            ),
            'base_to_signal_cross_attn': 'enabled',
            'classifier_center_source': getattr(self.classifier_config, 'center_source', None),
        }

    def format_execution_contract(self) -> str:
        """Format the resolved model design as a one-line startup check."""
        contract = self.execution_contract()
        signal_description = (
            contract['resolved_signal_key']
            if contract['resolved_signal_key'] is not None
            else 'none (head does not consume a signal tensor)'
        )
        return (
            "Classifier execution contract: "
            f"initialization={contract['initialization']}; "
            f"head={contract['classifier_type']}; "
            f"base_key={contract['classifier_base_source']}; "
            f"classifier_signal_source={contract['classifier_signal_source']}; "
            f"classifier_center_source={contract['classifier_center_source']}; "
            f"resolved_signal_key={signal_description}; "
            "pretrained_signal_to_base_capability="
            f"{contract['checkpoint_supports_signal_cross_attn']}; "
            "effective_add_signal_cross_attn="
            f"{contract['effective_add_signal_cross_attn']}; "
            "configured_signal_cross_layers="
            f"{contract['configured_signal_cross_layers']}; "
            "effective_signal_cross_layers="
            f"{contract['effective_signal_cross_layers']}; "
            "base_to_signal_cross_attn=enabled."
        )
            
    def _resolve_signal_key(
        self,
        classifier_type: str,
        signal_source: Optional[str],
    ) -> Optional[str]:
        if classifier_type != 'transformer_cross':
            if signal_source is not None:
                raise ValueError(
                    "Non-cross classifier checkpoints must record signal_source=None."
                )
            return None
        signal_source = _normalize_classifier_signal_source(signal_source)
        if signal_source == 'contextual_ts':
            if not self.pretrain.config.add_signal_cross_attn:
                raise ValueError("classifier_signal_source='contextual_ts' requires signal-to-base cross-attention to be enabled.")
            return 'contextual_ts'
        if signal_source == 'fixed_sa_ts':
            return 'fixed_sa_ts'
        return 'contextual_ts' if self.pretrain.config.add_signal_cross_attn else 'fixed_sa_ts'

    def load_configs(
        self,
        model_load_path,
        add_signal_cross_attn: Optional[bool] = None,
        backbone_signal_cross_attn: Optional[str] = None,
        classifier_signal_source: Optional[str] = None,
        classifier_type: Optional[str] = None,
        enable_mtr: bool = False,
    ):
        """Load and reconcile classifier/backbone configs from a checkpoint.

        Returns the exact saved topology and a classifier config with legacy
        signal-source metadata migrated to a resolved value. Explicit
        architecture requests are validation-only and cannot fork the saved
        execution contract. Model weights are loaded separately by
        ``__init__``.
        """
        model_ckpt = torch.load(model_load_path, map_location="cpu")
        
        classifier_config_dict=dict(model_ckpt['classifier_config'])

        pretrain_config_dict = get_backbone_config_from_checkpoint(
            model_ckpt,
            source=str(model_load_path),
        )
        checkpoint_state = strip_module_prefix(
            get_classification_state_dict_from_checkpoint(model_ckpt)
        )
        pretrain_config, classifier_config_dict = _resolve_continuation_classifier_contract(
            pretrain_config_dict,
            classifier_config_dict,
            classifier_signal_source=classifier_signal_source,
            backbone_signal_cross_attn=backbone_signal_cross_attn,
            add_signal_cross_attn=add_signal_cross_attn,
            classifier_type=classifier_type,
            enable_mtr=enable_mtr,
            checkpoint_contains_signal_cross_attn_weights=(
                checkpoint_has_signal_cross_attn_weights(
                    checkpoint_state,
                    prefix='pretrain.',
                )
            ),
        )

        return pretrain_config, classifier_config_dict
  
    def freeze_pretrained_model(self):
        """Disable gradients for every parameter in the pretrained module."""
        for p in self.pretrain.parameters():
            p.requires_grad = False
                
    def forward(
        self, base_seq_input, ts_input, percentiles_input=None,
        pad_base_seq = None, pad_ts = None, mask_ts = None, embeddings_only=True,
        return_pretrain=False, get_signal=False, *, signal_dropout_prob=0.0,
        **kwargs
    ) -> torch.Tensor:
        """Run auxiliary pretraining or per-base classification.

        ``base_seq_input`` has shape ``(B, L_base)`` and ``ts_input`` shape
        ``(B, L_signal, channels)``; masks are boolean tensors on their
        respective sequence axes. With ``return_pretrain=True``, arguments are
        forwarded to the pretraining module with ``embeddings_only=False`` and
        its dictionary is returned directly.

        Otherwise every head consumes ``contextual_base_seq``; only a
        transformer-cross head additionally consumes the resolved signal
        output and, when configured, the returned backbone ``u_center`` as a
        detached alignment prior. The method adds float32 ``classifier_logits``
        of shape ``(B, L_base, out_dim)`` and ``classifier_base_embedding`` to the
        backbone output dictionary and returns it.
        """
        if return_pretrain:
            return self.pretrain(
                base_seq_input=base_seq_input,
                ts_input=ts_input,
                percentiles_input=percentiles_input,
                pad_base_seq=pad_base_seq,
                pad_ts=pad_ts,
                mask_ts=mask_ts,
                embeddings_only=False,
                get_signal=get_signal,
                signal_dropout_prob=signal_dropout_prob,
            )

        if signal_dropout_prob != 0:
            raise ValueError(
                "signal_dropout_prob must be 0 for classification forwards; "
                "nonzero values are only valid with return_pretrain=True."
            )

        output = self.pretrain(
            base_seq_input=base_seq_input,
            ts_input=ts_input,
            percentiles_input=percentiles_input,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
            mask_ts=mask_ts,
            embeddings_only=embeddings_only,
            get_signal=get_signal,
            signal_dropout_prob=0.0,
        )
        contextual_base_seq = output["contextual_base_seq"]  # (B, L, D_pretrain)
        if self.classifier_config.classifier_type=='cnn':
            logits=self.classifier_head(contextual_base_seq)
            base_embedding= None, None
        elif self.classifier_config.classifier_type=='transformer_self':
            base_embedding, logits=self.classifier_head(
                contextual_base_seq,
                None,
                pad_base_seq,
                pad_ts,
                **kwargs,
            )
        else:
            contextual_ts_seq = output[self.signal_key]  # (B, L_signal, D_pretrain)
            base_embedding, logits=self.classifier_head(contextual_base_seq, contextual_ts_seq,
                                   pad_base_seq, pad_ts,
                                   u_center=output.get('u_center'),
                                   **kwargs)
                
        output['classifier_logits']=logits
        output['classifier_base_embedding']=base_embedding
        
        return output

ClassifierModel = ONTModelForClassification
