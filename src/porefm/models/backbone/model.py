"""Protected PoreFM backbone and auxiliary pretraining heads."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass, asdict
import math
from numbers import Real

from typing import Optional, Tuple

from porefm.models.backbone.utils import LM_Head
from porefm.models.backbone.projection import (
    BaseSeq_Projector,
    SignalPercentile_Projector,
    Signal_Projector,
)
from porefm.models.backbone.self_attention import SelfAttnBlock
from porefm.models.backbone.cross_attention import CrossAttnBlock
from porefm.models.configuration import EMA_SIGNAL_DROPOUT_MODES


def _validate_signal_dropout_probability(signal_dropout_prob, *, add_percentile, training):
    """Validate and normalize one forward's signal-modality dropout policy."""
    if isinstance(signal_dropout_prob, bool) or not isinstance(signal_dropout_prob, Real):
        raise TypeError("signal_dropout_prob must be a real number in [0, 1].")

    probability = float(signal_dropout_prob)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("signal_dropout_prob must be finite and in [0, 1].")
    if probability > 0 and not add_percentile:
        raise ValueError("signal_dropout_prob must be 0 when percentile fusion is disabled.")
    if probability > 0 and not training:
        raise ValueError("signal_dropout_prob must be 0 in evaluation mode.")
    return probability


def _sample_signal_dropout_choices(ts_input, probability):
    """Sample one broadcastable modality-dropout choice per batch item.

    Choice zero keeps both modalities, one drops percentiles, and two drops raw
    signal. A zero probability returns ``None`` without consuming RNG state.
    """
    if probability == 0:
        return None

    rand_val = torch.rand(ts_input.size(0), 1, 1, device=ts_input.device)
    choices = torch.zeros_like(rand_val, dtype=torch.int8)
    choices = torch.where(rand_val < probability / 2.0, 1, choices)
    choices = torch.where(
        (rand_val >= probability / 2.0) & (rand_val < probability),
        2,
        choices,
    )
    return choices


@torch.no_grad()
def _update_ema_parameters(online_module: nn.Module, ema_module: nn.Module, momentum: float):
    """EMA-update parameters shared by the online and target modules.

    Parameters are matched by name instead of iterator position so online-only
    parameters, such as LoRA adapters, do not shift the correspondence with the
    structurally equivalent EMA module.
    """
    online_parameters = dict(online_module.named_parameters())
    ema_parameters = dict(ema_module.named_parameters())

    missing_online = sorted(set(ema_parameters) - set(online_parameters))
    if missing_online:
        raise RuntimeError(
            "EMA target parameters are missing from the online module: "
            + ", ".join(missing_online)
        )

    for name, ema_param in ema_parameters.items():
        online_param = online_parameters[name]
        if ema_param.shape != online_param.shape:
            raise RuntimeError(
                f"EMA parameter shape mismatch for {name!r}: "
                f"target={tuple(ema_param.shape)} online={tuple(online_param.shape)}"
            )
        ema_param.mul_(momentum).add_(online_param, alpha=1 - momentum)

class EncoderTower(nn.Module):
    """Encode bases and signal through self- and cross-attention towers.

    Base and signal inputs first pass through separate self-attention stacks.
    Base queries then attend to the fixed self-attended signal representation;
    optional signal queries attend back to the final base representation. The
    last base-to-signal layer can also aggregate raw signal values through its
    first attention head when ``get_signal`` is true.
    """
    def __init__(self, config):
        super().__init__()

        self.num_layers_base_cross=config.num_layers_base_cross
        # base self-attention tower
        self.base_seq_self_tower = nn.ModuleList([
            SelfAttnBlock(config, query_modality='seq')
            for _ in range(config.num_layers_base_seq_self)
        ])

        # signal self-attention tower
        self.ts_self_tower = nn.ModuleList([
            SelfAttnBlock(config, query_modality='signal')
            for _ in range(config.num_layers_ts_self)
        ])

        # cross-modality attention tower
        self.base_to_signal_cross_tower = nn.ModuleList([
            CrossAttnBlock(config, query_modality='seq', predict_center=n>0,
                           residual_control = config.residual_control if (n==config.num_layers_base_cross-1) else None
                          )
            for n in range(config.num_layers_base_cross)
        ])

        self.add_signal_cross_attn=config.add_signal_cross_attn

        if self.add_signal_cross_attn:
            self.signal_to_base_cross_tower = nn.ModuleList([
                CrossAttnBlock(config, query_modality='signal',
                               residual_control = config.residual_control if (n==config.num_layers_signal_cross-1) else None
                              )
                for n in range(config.num_layers_signal_cross)
            ])

    def forward(self, base_seq, ts, pad_base_seq, pad_ts, ts_input, get_signal=False):
        """Return fixed and contextual embeddings plus alignment diagnostics.

        ``base_seq`` and ``ts`` have shapes ``(B, L_base, D)`` and
        ``(B, L_signal, D)``; masks use ``True`` for valid tokens. The nine-item
        return contains fixed self-attention outputs, evolved base/signal
        outputs, final base-to-signal attention context and head-0 context,
        center positions/increments, and optional raw-signal context.
        """
        for layer in self.base_seq_self_tower:
            base_seq = layer(base_seq, pad_base_seq)

        for layer in self.ts_self_tower:
            ts = layer(ts, pad_ts)

        fixed_sa_base = base_seq
        fixed_sa_ts   = ts

        base_seq_evolved = base_seq


        for num, layer in enumerate(self.base_to_signal_cross_tower):
            if num == self.num_layers_base_cross - 1:
                (
                    base_seq_evolved,
                    base_to_signal_qkv,
                    u_center,
                    delta,
                    s_ctx,
                    base_to_signal_head0_ctx,
                ) = layer.forward_with_head0_ctx(
                    query=base_seq_evolved,
                    kv=fixed_sa_ts,
                    pad_query=pad_base_seq,
                    pad_kv=pad_ts,
                    s=ts_input if get_signal else None,
                )
            else:
                base_seq_evolved, base_to_signal_qkv, u_center, delta, s_ctx = layer(
                    query=base_seq_evolved,
                    kv=fixed_sa_ts,
                    pad_query=pad_base_seq,
                    pad_kv=pad_ts,
                )

        fixed_ca_base = base_seq_evolved
        ts_evolved = ts

        if self.add_signal_cross_attn>0:
            for layer in self.signal_to_base_cross_tower:
                res = layer(
                    query=ts_evolved,
                    kv=fixed_ca_base,
                    pad_query=pad_ts,
                    pad_kv=pad_base_seq,
                )
                ts_evolved=res[0]

        return (
            fixed_sa_base,
            fixed_sa_ts,
            base_seq_evolved,
            ts_evolved,
            base_to_signal_qkv,
            base_to_signal_head0_ctx,
            u_center,
            delta,
            s_ctx,
        )

class EMASignalEncoder(nn.Module):
    """Momentum target encoder for signal-projector and self-attention states.

    Its architecture mirrors the online signal path through projection and
    signal-only self-attention. Parameters are initialized and updated by
    :class:`ONTModelForPreTraining` and do not receive gradients.
    """

    def __init__(self, config):
        super().__init__()
        self.add_percentile = config.add_percentile

        if self.add_percentile:
            self.signal_projector = SignalPercentile_Projector(
                config.signal_strides, config.signal_kernels,
                config.hidden_dim
            )
        else:
            self.signal_projector = Signal_Projector(
                config.signal_strides, config.signal_kernels, config.hidden_dim
            )

        self.ts_self_tower = nn.ModuleList([
            SelfAttnBlock(config, query_modality='signal')
            for _ in range(config.num_layers_ts_self)
        ])

    def forward(
        self,
        ts_input,
        percentiles_input=None,
        pad_ts=None,
        *,
        signal_dropout_choices=None,
    ):
        """Encode target signal tokens, applying supplied batch-item choices."""
        if self.add_percentile:
            x = self.signal_projector(
                ts_input,
                percentiles_input,
                pad_ts,
                None,
                signal_dropout_choices=signal_dropout_choices,
            )
        else:
            x = self.signal_projector(ts_input, pad_ts, None)

        for layer in self.ts_self_tower:
            x = layer(x, pad_ts)

        return x

@dataclass
class PretrainConfig:
    """Architecture configuration for the PoreFM pretrained backbone.

    Sequence/signal vocabulary and maximum lengths define input contracts;
    ``hidden_dim`` and ``dim_feedforward`` define transformer widths.
    ``signal_kernels``/``signal_strides`` configure the four-stage signal
    projector, while layer counts select the base self-attention, signal
    self-attention, and two cross-attention directions. Attention head counts,
    local windows, temperatures, expected signal/base ratio, optional
    pre-attention mixer/residual control, percentile fusion, dropout, and EMA
    target support complete the forward configuration. ``kv_num_heads=0`` is
    normalized to ``num_heads`` after initialization.

    ``signal_dropout_prob`` records the training policy used to create or
    continue a checkpoint; model forwards do not apply it implicitly.
    ``ema_signal_dropout_mode`` is required and selects independent student/EMA
    choices, shared choices, or disabled EMA modality dropout.
    """
    base_seq_vocab:int
    ts_dim: int

    max_base_seq_len: int
    max_ts_len:int

    hidden_dim: int
    dim_feedforward: int

    signal_kernels: int
    signal_strides: int

    move_downsample_factor: int

    num_layers_base_seq_self: int
    num_layers_ts_self: int

    num_layers_base_cross: int
    num_layers_signal_cross: int

    num_heads: int
    kv_num_heads: int
    ema_signal_dropout_mode: str

    add_signal_cross_attn: bool = True

    pre_attention_mixer_type: Optional[str] = None
    residual_control: Optional[str] = None

    add_percentile: bool = True

    base_seq_window: tuple = (-1,-1)
    ts_window: tuple = (-1,-1)

    self_attn_temp: float = 1.0
    cross_attn_temp: float = 1.0
    expected_ratio: Optional[int] = None

    signal_dropout_prob: float = 0
    dropout: float = 0.0
    add_ema: bool =True

    def __post_init__(self):
        if self.ema_signal_dropout_mode not in EMA_SIGNAL_DROPOUT_MODES:
            raise ValueError(
                "ema_signal_dropout_mode must be one of "
                f"{sorted(EMA_SIGNAL_DROPOUT_MODES)}, got {self.ema_signal_dropout_mode!r}."
            )
        _validate_signal_dropout_probability(
            self.signal_dropout_prob,
            add_percentile=self.add_percentile,
            training=True,
        )
        if self.kv_num_heads == 0:
            self.kv_num_heads = self.num_heads

ONTBackboneConfig = PretrainConfig

class ONTBackbone(nn.Module):
    """Project and jointly contextualize base tokens and nanopore signal.

    Parameters
    ----------
    config : PretrainConfig or dict
        Backbone architecture and modality settings.
    Notes
    -----
    Base inputs are one-based token IDs with zero padding. Signal inputs have
    shape ``(B, L_ts, ts_dim)`` after loader move downsampling. This class owns
    no task heads; its output dictionary reserves their keys as ``None`` for a
    stable downstream contract.
    """

    def __init__(self, config: PretrainConfig):
        super().__init__()

        self.config = PretrainConfig(**config) if type(config) == dict else config

        assert self.config.hidden_dim % self.config.num_heads == 0

        if self.config.add_percentile:
            self.signal_projector = SignalPercentile_Projector(
                self.config.signal_strides,
                self.config.signal_kernels,
                self.config.hidden_dim,
            )

        else:
            self.signal_projector=Signal_Projector(self.config.signal_strides, self.config.signal_kernels, self.config.hidden_dim)

        self.base_seq_projector = BaseSeq_Projector(self.config.base_seq_vocab, self.config.hidden_dim)

        self.cross_attn = EncoderTower(self.config)

        self.final_norm_base_seq = nn.LayerNorm(self.config.hidden_dim)

        if self.config.add_signal_cross_attn:
            self.final_norm_ts  = nn.LayerNorm(self.config.hidden_dim)

    def encode(self, base_seq_input, ts_input, percentiles_input=None,
                pad_base_seq = None, pad_ts = None, mask_ts = None,
                get_signal=False, signal_dropout_choices=None):
        """Encode both modalities and return embeddings/attention diagnostics.

        Parameters
        ----------
        base_seq_input : torch.Tensor
            Integer tokens with shape ``(B, L_base)``.
        ts_input : torch.Tensor
            Signal features with shape ``(B, L_ts, ts_dim)``.
        percentiles_input : torch.Tensor or None
            Rank features matching ``ts_input`` when percentile fusion is
            enabled.
        pad_base_seq, pad_ts : torch.Tensor or None
            Boolean valid-token masks. Omitted masks default to all valid.
        mask_ts : torch.Tensor or None
            Boolean signal positions replaced by learned masking embeddings.
        get_signal : bool, default=False
            Also aggregate raw ``ts_input`` values and source indices through
            the final base-to-signal attention head.

        Returns
        -------
        dict
            Contextual/fixed base and signal embeddings, pre-normalized bases,
            center predictions, cross-attention contexts, and reserved task-head
            outputs. ``contextual_ts`` is ``None`` when reverse cross-attention
            is disabled.
        """
        B, L_base_seq = base_seq_input.shape
        B, L_ts, _ = ts_input.shape

        pad_base_seq = torch.ones(B, L_base_seq, dtype=torch.bool, device=base_seq_input.device) if pad_base_seq is None else pad_base_seq
        pad_ts = torch.ones(B, L_ts, dtype=torch.bool, device=ts_input.device) if pad_ts is None else pad_ts

        if self.config.add_percentile:
            ts_in = self.signal_projector(
                ts_input,
                percentiles_input,
                pad_ts,
                mask_ts,
                signal_dropout_choices=signal_dropout_choices,
            )
        else:
            ts_in = self.signal_projector(ts_input, pad_ts, mask_ts)

        base_seq_in = self.base_seq_projector(base_seq_input, pad_base_seq)


        (
            fixed_sa_base,
            fixed_sa_ts,
            base_seq_evolved,
            ts_evolved,
            base_to_signal_qkv,
            base_to_signal_head0_ctx,
            u_center,
            delta,
            s_ctx,
        ) = self.cross_attn(
            base_seq_in,
            ts_in,
            pad_base_seq,
            pad_ts,
            ts_input,
            get_signal=get_signal,
        )

        base_seq_out = self.final_norm_base_seq(base_seq_evolved)
        ts_out = self.final_norm_ts(ts_evolved) if self.config.add_signal_cross_attn else None

        return {'masklm_logits': None,
                'sublm_logits': None,
                'latent_pred': None,
                'latent_target': None,
                'contextual_base_seq': base_seq_out,
                'contextual_ts': ts_out,
                'u_center': u_center,
                'delta': delta,
                'base_to_signal_qkv': base_to_signal_qkv,
                'base_to_signal_head0_ctx': base_to_signal_head0_ctx,
                'prenorm_contextual_base_seq': base_seq_evolved,
                'fixed_sa_base': fixed_sa_base,
                'fixed_sa_ts': fixed_sa_ts,
                's_ctx': s_ctx,
                }

    def forward(self, base_seq_input, ts_input, percentiles_input=None,
                pad_base_seq = None, pad_ts = None, mask_mlm = None, mask_ts = None,
                embeddings_only=True, get_signal=False, *, signal_dropout_prob=0.0):
        """Forward inputs to :meth:`encode` using the backbone-only contract.

        ``mask_mlm`` and ``embeddings_only`` are accepted for interface
        compatibility with the pretraining wrapper and do not alter backbone
        execution. ``signal_dropout_prob`` is a keyword-only, per-forward
        policy. It defaults to zero and is rejected outside training mode or
        when percentile fusion is disabled.
        """
        probability = _validate_signal_dropout_probability(
            signal_dropout_prob,
            add_percentile=self.config.add_percentile,
            training=self.training,
        )
        signal_dropout_choices = _sample_signal_dropout_choices(ts_input, probability)
        return self.encode(
            base_seq_input=base_seq_input,
            ts_input=ts_input,
            percentiles_input=percentiles_input,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
            mask_ts=mask_ts,
            get_signal=get_signal,
            signal_dropout_choices=signal_dropout_choices,
        )

class ONTModelForPreTraining(ONTBackbone):
    """Backbone plus base-language and optional EMA latent-prediction heads.

    SUBLM emits two logits and MASKLM four logits per base token. When both
    signal cross-attention and EMA are enabled, an online latent head predicts
    the momentum signal encoder's targets for JEPA/MTR-style training.
    """

    def __init__(self, config: PretrainConfig):
        super().__init__(config)

        self.sublm_head = LM_Head(self.config.hidden_dim, 32, 2)      # Sub MLM
        self.masklm_head = LM_Head(self.config.hidden_dim, 32, 4)     # maskMLM

        if self.config.add_signal_cross_attn and self.config.add_ema==True:
            self.ema_signal_encoder = EMASignalEncoder(self.config)
            self._init_ema_weights()
            self.latent_head = LM_Head(self.config.hidden_dim, self.config.hidden_dim, self.config.hidden_dim)     # JEPA

    def _init_ema_weights(self):
        """Copy online signal-path weights and disable EMA gradients."""
        for param in self.ema_signal_encoder.parameters():
            param.requires_grad = False

        self.ema_signal_encoder.signal_projector.load_state_dict(self.signal_projector.state_dict())
        self.ema_signal_encoder.ts_self_tower.load_state_dict(self.cross_attn.ts_self_tower.state_dict())

    @torch.no_grad()
    def update_ema(self, momentum=0.996):
        """Momentum-update the target signal encoder from online parameters.

        ``momentum`` weights the existing target; values nearer one update more
        slowly. The operation is a no-op when ``config.add_ema`` is false and
        runs without autograd.
        """
        if not getattr(self.config, 'add_ema', False):
            return

        _update_ema_parameters(
            self.signal_projector,
            self.ema_signal_encoder.signal_projector,
            momentum,
        )
        _update_ema_parameters(
            self.cross_attn.ts_self_tower,
            self.ema_signal_encoder.ts_self_tower,
            momentum,
        )

    def forward(self, base_seq_input, ts_input, percentiles_input=None,
                pad_base_seq = None, pad_ts = None, mask_mlm = None, mask_ts = None,
                embeddings_only=False, get_signal=False, *, signal_dropout_prob=0.0):
        """Return backbone embeddings and, unless requested otherwise, task heads.

        With ``embeddings_only=True`` the backbone dictionary is returned
        unchanged. Otherwise SUBLM/MASKLM logits are populated. EMA latent
        predictions are produced when configured; the target is computed only
        when ``mask_ts`` is supplied and remains detached from autograd.
        ``signal_dropout_prob`` is applied per batch item to the student; the
        config's required EMA mode determines whether target choices are
        independently sampled, shared, or disabled.
        """
        probability = _validate_signal_dropout_probability(
            signal_dropout_prob,
            add_percentile=self.config.add_percentile,
            training=self.training,
        )
        signal_dropout_choices = _sample_signal_dropout_choices(ts_input, probability)
        output = self.encode(
            base_seq_input=base_seq_input,
            ts_input=ts_input,
            percentiles_input=percentiles_input,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
            mask_ts=mask_ts,
            get_signal=get_signal,
            signal_dropout_choices=signal_dropout_choices,
        )

        if embeddings_only:
            return output

        output['sublm_logits'] = self.sublm_head(output['contextual_base_seq'])
        output['masklm_logits'] = self.masklm_head(output['contextual_base_seq'])

        if self.config.add_signal_cross_attn and self.config.add_ema:
            if mask_ts is not None:
                if self.config.ema_signal_dropout_mode == "independent":
                    ema_dropout_choices = _sample_signal_dropout_choices(ts_input, probability)
                elif self.config.ema_signal_dropout_mode == "shared":
                    ema_dropout_choices = signal_dropout_choices
                else:
                    ema_dropout_choices = None
                target_percentiles = percentiles_input.clone() if percentiles_input is not None else None
                target_pad_ts = pad_ts.clone() if pad_ts is not None else None
                with torch.no_grad():
                    output['latent_target'] = self.ema_signal_encoder(
                        ts_input.clone(),
                        percentiles_input=target_percentiles,
                        pad_ts=target_pad_ts,
                        signal_dropout_choices=ema_dropout_choices,
                    )
            output['latent_pred'] = self.latent_head(output['contextual_ts'])

        return output

PretrainModel = ONTModelForPreTraining
