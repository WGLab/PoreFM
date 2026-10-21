"""Dynamic-center cross-attention blocks for the protected PoreFM backbone."""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from porefm.models.backbone.utils import MLP
from porefm.models.backbone.self_attention import SelfAttention

from flash_attn.modules.mha import get_alibi_slopes
from einops import rearrange
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func,   # self-attn
    flash_attn_varlen_kvpacked_func     # cross-attn
)

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _FLEX_OK = True
except Exception:
    _FLEX_OK = False

    
def alibi_lambdas_from_half_lives(
    n_heads: int,
    half_life_min: float,
    half_life_max: float,
    largest_first: bool = True,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Convert geometric attention half-lives to ALiBi decay coefficients.

    ``half_life_min`` and ``half_life_max`` are token distances at which the
    distance penalty reaches ``ln(2)``. With ``largest_first=True``, head zero
    is most local (smallest half-life/largest coefficient). The result has
    shape ``(n_heads,)`` on the requested device and dtype.
    """
    if n_heads < 1:
        raise ValueError("n_heads must be >= 1")

    if n_heads == 1:
        half_lives = torch.tensor([half_life_min], dtype=dtype, device=device)
    else:
        i = torch.linspace(0, 1, steps=n_heads, dtype=dtype, device=device)
        ratio = float(half_life_max) / float(half_life_min)
        half_lives = torch.as_tensor(half_life_min, dtype=dtype, device=device) * (ratio ** i)

    lambdas = (math.log(2.0) / half_lives).to(dtype)
    if largest_first:
        lambdas = lambdas.sort(descending=True).values
    return lambdas


class FlexCrossAttention(nn.Module):
    """Grouped-query cross-attention with per-query monotonic center priors.

    Queries have ``num_heads`` while keys/values use ``kv_num_heads``. Each
    head's score receives a linear ALiBi penalty for distance from the supplied
    center and invalid key/value positions are set to negative infinity. An
    optional raw-signal value tensor can be aggregated through head zero using
    the identical score function.
    """
    def __init__(
        self, config):
        
        super().__init__()
        if not _FLEX_OK:
            raise RuntimeError("FlexAttention is not available in this PyTorch build.")

        assert config.hidden_dim % config.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert config.num_heads % config.kv_num_heads == 0, "num_heads must be divisible by kv_num_heads"
        
        self.h = config.num_heads
        self.hk = config.kv_num_heads
        self.dh = config.hidden_dim // config.num_heads
        self.dropout_p = float(config.dropout)

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_dim, 2 * self.hk * self.dh, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_dropout = nn.Dropout(self.dropout_p)

        self.temperature = config.cross_attn_temp
        self.attn_scale = (1.0 / math.sqrt(self.dh)) / self.temperature

        alibi_bias_fp32 = alibi_lambdas_from_half_lives(
            n_heads=config.num_heads,
            half_life_min=16,
            half_life_max=64,
            largest_first=True,
            device=None,
            dtype=torch.float32,
        )
        self.register_buffer("alibi_bias", alibi_bias_fp32, persistent=False)
    
    def _flex_attention_forward(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
        q_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        u_center: torch.Tensor = None,
        s: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply flex attention to head-major projections.

        ``q_heads`` is ``(B, H, Lq, Dh)``; key/value tensors use
        ``(B, Hkv, Lkv, Dh)``. Centers have shape ``(B, Lq)``. The optional
        ``s`` has shape ``(B, Lkv, channels)`` and produces context with its
        source index appended as one extra channel.
        """

        alibi = self.alibi_bias

        def _dynamic_score_mod(score, b, h, q_idx, kv_idx):
            tgt = u_center[b, q_idx] 
            strength = alibi[h]                
            delta = abs(kv_idx - tgt) 
            score = score - (strength * delta)
            
            is_valid = kv_mask[b, kv_idx] == 1 
            score = torch.where(is_valid, score, float('-inf'))
            return score
        
        out = flex_attention(
            q_heads, k_heads, v_heads,
            block_mask=None,
            score_mod=_dynamic_score_mod,
            scale=self.attn_scale,
            enable_gqa=True,
        )
        
        if s is None:
            return out, None
        
        indices = torch.arange(s.size(1), device=s.device)
        indices = indices.view(1, -1, 1).expand(s.size(0), -1, -1)
        s_extended = torch.cat((s, indices), dim=2)
        
        s_val = s_extended.to(device=q_heads.device, dtype=q_heads.dtype).unsqueeze(1).contiguous()
        
        s_out = flex_attention(
            q_heads[:, :1],
            k_heads[:, :1],
            s_val,
            block_mask=None,
            score_mod=_dynamic_score_mod,
            scale=self.attn_scale,
            enable_gqa=False,
        ).squeeze(1)
         
        return out, s_out
    
    def forward(
        self,
        q_in: torch.Tensor,     
        kv_in: torch.Tensor,    
        q_mask: torch.Tensor,   
        kv_mask: torch.Tensor,  
        u_center: torch.Tensor = None,
        s: torch.Tensor = None,
        return_head0_ctx: bool = False,
    ) -> torch.Tensor:
        """Project modalities and return masked cross-attention contexts.

        ``q_in`` and ``kv_in`` have shapes ``(B, Lq, D)`` and ``(B, Lkv, D)``;
        masks are boolean valid-token matrices. Returns ``(output, s_ctx)`` and,
        when ``return_head0_ctx`` is true, the pre-output-projection head-zero
        context ``(B, Lq, Dh)``. ``s_ctx`` is ``None`` unless raw signal ``s``
        is provided.
        """
        B, Lq, D = q_in.shape
        _, Lkv, _ = kv_in.shape

        q = self.q_proj(q_in).view(B, Lq, self.h, self.dh).contiguous()
        kv = self.kv_proj(kv_in).view(B, Lkv, 2, self.hk, self.dh).contiguous()

        k, v = kv[:, :, 0], kv[:, :, 1]
                
        q_heads = q.permute(0, 2, 1, 3).contiguous()
        k_heads = k.permute(0, 2, 1, 3).contiguous()
        v_heads = v.permute(0, 2, 1, 3).contiguous()

        ctx, s_ctx = self._flex_attention_forward(
            q_heads, k_heads, v_heads, q_mask, kv_mask, u_center, s
        )

        out = rearrange(ctx, "b h l d -> b l (h d)")
        out = self.proj(out)
        out = self.out_dropout(out)
        out = out * q_mask.unsqueeze(-1).to(out.dtype)
        
        if s is None:
            outputs = (out, None)
        else:
            s_ctx = s_ctx * q_mask.unsqueeze(-1).to(s_ctx.dtype)
            outputs = (out, s_ctx)

        if return_head0_ctx:
            head0_ctx = ctx[:, 0]
            head0_ctx = head0_ctx * q_mask.unsqueeze(-1).to(head0_ctx.dtype)
            return (*outputs, head0_ctx)
        return outputs
    
    
class CNN_mixer(nn.Module):
    """Residual depthwise/pointwise convolutional query pre-mixer.

    The same four-layer CNN is applied twice before masking and residual
    addition. Inputs and outputs have shape ``(B, L_query, hidden_dim)``.
    """

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=1, padding='same', groups=config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1, stride=1, padding='same', bias=True),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=1, padding='same', groups=config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1, stride=1, padding='same', bias=False)
        )

        for m in self.cnn:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                 
    def forward(self, query: torch.Tensor, pad_query: torch.Tensor):
        """Contextualize valid query positions and add the input residual."""
        x = self.norm(query)
        
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2) 

        if pad_query is not None:
            x = x * pad_query.unsqueeze(-1).to(x.dtype)

        x = query + x
        return x
                                                                    
class Self_Attn_mixer(nn.Module):
    """Residual self-attention query pre-mixer with pre-normalization."""

    def __init__(self, config, query_modality):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.attn = SelfAttention(config, query_modality)
        
    def forward(self, query: torch.Tensor, pad_query: torch.Tensor):
        """Return self-attended queries with their original residual."""
        x = self.norm(query)
        
        x = self.attn(x, pad_query)
        
        x = query + x
        
        return x
    
class PassThrough(nn.Module):
    """Identity module that accepts multiple arguments and returns the first."""
    def forward(self, x, *args):
        """Return ``x`` and ignore compatibility arguments."""
        return x
    
class _CrossAttnBlockBase(nn.Module):
    """Shared cross-attention normalization, mixer, center, and FFN modules.

    ``query_modality`` selects the pre-mixer attention window. Centers are
    static length-ratio positions or cumulative positive increments predicted
    from queries. ``residual_control`` is validated here while the public
    factory selects fixed or trainable topology from ``residual_control``.
    """
    def __init__(
        self,
        config,
        query_modality,
        predict_center=False,
        residual_control=None,
        gate_as_parameter=False,
        gate_value=1.0,
    ):
        
        super().__init__()
        self.query_modality=query_modality
        self.residual_control = residual_control
        self.expected_ratio=config.expected_ratio
        self.predict_center= predict_center
        
        self.pre_attention_mixer_type = config.pre_attention_mixer_type
        
        valid_mixer_types = {None, "cnn", "self_attn"}
        if self.pre_attention_mixer_type not in valid_mixer_types:
            raise ValueError(
                f"Unsupported pre_attention_mixer_type={self.pre_attention_mixer_type!r}; "
                f"expected one of {sorted(x for x in valid_mixer_types if x is not None)} or None."
            )

        valid_residual_controls = {None, "skip", "gate"}
        if self.residual_control not in valid_residual_controls:
            raise ValueError(
                f"Unsupported residual_control={self.residual_control!r}; "
                f"expected one of {sorted(x for x in valid_residual_controls if x is not None)} or None."
            )

            
        self.norm_ffn = nn.LayerNorm(config.hidden_dim)
        self.norm_q = nn.LayerNorm(config.hidden_dim)
        self.norm_kv = nn.LayerNorm(config.hidden_dim)
                
        if self.pre_attention_mixer_type=='cnn':
            self.pre_attention_mixer = CNN_mixer(config)
        
        elif self.pre_attention_mixer_type=='self_attn':
            self.pre_attention_mixer = Self_Attn_mixer(config, query_modality)
            
        else:
            self.pre_attention_mixer = PassThrough()
            
        if self.predict_center:
            # Softplus-transformed increments start near the expected
            # signal/base ratio and accumulate into monotonic centers.
            self.center_predictor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1)
            )

            init_val = math.log(math.exp(self.expected_ratio) - 1.0)

            nn.init.constant_(self.center_predictor[-1].bias, init_val)
            nn.init.normal_(self.center_predictor[-1].weight, mean=0.0, std=1e-3)
            

        self.cross_attention = FlexCrossAttention(config)

        self.MLP = MLP(hidden_dim=config.hidden_dim, dim_ff=config.dim_feedforward, dropout=config.dropout)

        if gate_as_parameter:
            self.gate = nn.Parameter(torch.tensor(gate_value))
        else:
            self.register_buffer("gate", torch.tensor(gate_value))

    def forward_with_head0_ctx(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        pad_query: torch.Tensor,
        pad_kv: torch.Tensor,
        s: torch.Tensor = None,
    ):
        """Run a block and additionally return pre-projection head-zero context.

        The return is ``(output, attention_output, centers, increments,
        signal_context, head0_context)``. ``increments`` is ``None`` for static
        centers and ``signal_context`` is ``None`` unless ``s`` is supplied.
        """
        x = self.pre_attention_mixer(query, pad_query)
        q_normalized = self.norm_q(x)

        if self.predict_center:
            u_center, delta = self._get_predicted_center(q_normalized, pad_query)
        else:
            u_center, delta = self._get_static_center(pad_query, pad_kv)

        raw_qkv, s_ctx, head0_ctx = self.cross_attention(
            q_in=q_normalized,
            kv_in=self.norm_kv(kv),
            q_mask=pad_query,
            kv_mask=pad_kv,
            u_center=u_center,
            s=s,
            return_head0_ctx=True,
        )

        ca_out = self.gate * x + raw_qkv
        out = ca_out + self.MLP(self.norm_ffn(ca_out))
        return out, raw_qkv, u_center, delta, s_ctx, head0_ctx

class _StaticCenterMixin:
    def _get_static_center(self, pad_query: torch.Tensor, pad_kv: torch.Tensor):
        length_ratio = pad_kv.sum(1, keepdim=True) / pad_query.sum(1, keepdim=True)
        steps = torch.arange(1, 1 + pad_query.size(1), device=pad_query.device)
        return steps * length_ratio, None


class _PredictedCenterMixin:
    def _get_predicted_center(self, q_normalized: torch.Tensor, pad_query: torch.Tensor):
        delta = self.center_predictor(q_normalized).to(torch.float32).squeeze(-1)
        delta = F.softplus(delta)

        if pad_query is not None:
            delta = delta * pad_query.to(delta.dtype)

        return torch.cumsum(delta, dim=1), delta


class _TrainableGateStaticCenterCrossAttnBlock(_StaticCenterMixin, _CrossAttnBlockBase):
    def __init__(self, config, query_modality, predict_center=False, residual_control=None):
        super().__init__(
            config,
            query_modality,
            predict_center=False,
            residual_control=residual_control,
            gate_as_parameter=True,
            gate_value=0.0,
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor, pad_query: torch.Tensor, pad_kv: torch.Tensor, s: torch.Tensor = None):
        """Cross-attend using static centers and a trainable zero-initialized gate.

        Returns contextual output, raw attention output, centers, center deltas,
        and optional raw-signal context, all following the block contract.
        """
        x = self.pre_attention_mixer(query, pad_query)
        q_normalized = self.norm_q(x)
        u_center, delta = self._get_static_center(pad_query, pad_kv)
        raw_qkv, s_ctx = self.cross_attention(
            q_in=q_normalized,
            kv_in=self.norm_kv(kv),
            q_mask=pad_query,
            kv_mask=pad_kv,
            u_center=u_center,
            s=s,
        )
        ca_out = self.gate * x + raw_qkv
        out = ca_out + self.MLP(self.norm_ffn(ca_out))
        return out, raw_qkv, u_center, delta, s_ctx


class _TrainableGatePredictedCenterCrossAttnBlock(_PredictedCenterMixin, _CrossAttnBlockBase):
    def __init__(self, config, query_modality, predict_center=False, residual_control=None):
        super().__init__(
            config,
            query_modality,
            predict_center=True,
            residual_control=residual_control,
            gate_as_parameter=True,
            gate_value=0.0,
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor, pad_query: torch.Tensor, pad_kv: torch.Tensor, s: torch.Tensor = None):
        """Cross-attend using predicted centers and a trainable zero-initialized gate.

        Returns contextual output, raw attention output, centers, center deltas,
        and optional raw-signal context, all following the block contract.
        """
        x = self.pre_attention_mixer(query, pad_query)
        q_normalized = self.norm_q(x)
        u_center, delta = self._get_predicted_center(q_normalized, pad_query)
        raw_qkv, s_ctx = self.cross_attention(
            q_in=q_normalized,
            kv_in=self.norm_kv(kv),
            q_mask=pad_query,
            kv_mask=pad_kv,
            u_center=u_center,
            s=s,
        )
        ca_out = self.gate * x + raw_qkv
        out = ca_out + self.MLP(self.norm_ffn(ca_out))
        return out, raw_qkv, u_center, delta, s_ctx


class _FixedGateStaticCenterCrossAttnBlock(_StaticCenterMixin, _CrossAttnBlockBase):
    def __init__(
        self,
        config,
        query_modality,
        predict_center=False,
        residual_control=None,
        gate_value=1.0,
    ):
        super().__init__(
            config,
            query_modality,
            predict_center=False,
            residual_control=residual_control,
            gate_as_parameter=False,
            gate_value=gate_value,
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor, pad_query: torch.Tensor, pad_kv: torch.Tensor, s: torch.Tensor = None):
        """Cross-attend using static centers and a fixed residual gate.

        Returns contextual output, raw attention output, centers, center deltas,
        and optional raw-signal context, all following the block contract.
        """
        x = self.pre_attention_mixer(query, pad_query)
        q_normalized = self.norm_q(x)
        u_center, delta = self._get_static_center(pad_query, pad_kv)
        raw_qkv, s_ctx = self.cross_attention(
            q_in=q_normalized,
            kv_in=self.norm_kv(kv),
            q_mask=pad_query,
            kv_mask=pad_kv,
            u_center=u_center,
            s=s,
        )
        ca_out = self.gate * x + raw_qkv
        out = ca_out + self.MLP(self.norm_ffn(ca_out))
        return out, raw_qkv, u_center, delta, s_ctx


class _FixedGatePredictedCenterCrossAttnBlock(_PredictedCenterMixin, _CrossAttnBlockBase):
    def __init__(
        self,
        config,
        query_modality,
        predict_center=False,
        residual_control=None,
        gate_value=1.0,
    ):
        super().__init__(
            config,
            query_modality,
            predict_center=True,
            residual_control=residual_control,
            gate_as_parameter=False,
            gate_value=gate_value,
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor, pad_query: torch.Tensor, pad_kv: torch.Tensor, s: torch.Tensor = None):
        """Cross-attend using predicted centers and a fixed residual gate.

        Returns contextual output, raw attention output, centers, center deltas,
        and optional raw-signal context, all following the block contract.
        """
        x = self.pre_attention_mixer(query, pad_query)
        q_normalized = self.norm_q(x)
        u_center, delta = self._get_predicted_center(q_normalized, pad_query)
        raw_qkv, s_ctx = self.cross_attention(
            q_in=q_normalized,
            kv_in=self.norm_kv(kv),
            q_mask=pad_query,
            kv_mask=pad_kv,
            u_center=u_center,
            s=s,
        )
        ca_out = self.gate * x + raw_qkv
        out = ca_out + self.MLP(self.norm_ffn(ca_out))
        return out, raw_qkv, u_center, delta, s_ctx


def CrossAttnBlock(config, query_modality, predict_center=False, residual_control=None):
    """Construct the cross-attention block variant for residual control.

    Parameters
    ----------
    config : PretrainConfig or compatible object
        Attention, mixer, width, and expected-ratio settings.
    query_modality : {"seq", "signal"}
        Query stream used to choose modality-specific mixer settings.
    predict_center : bool, default=False
        Predict positive center increments from queries; otherwise use the
        valid key/query length ratio.
    residual_control : {None, "skip", "gate"}, optional
        ``None`` fixes the query residual multiplier at one. ``"skip"`` fixes
        it at zero. ``"gate"`` creates a trainable parameter initialized at
        zero; workflows freeze it through ordinary parameter policy when needed.

    Returns
    -------
    torch.nn.Module
        Concrete block with a five-value forward contract.
    """
    fixed_cls = _FixedGatePredictedCenterCrossAttnBlock if predict_center else _FixedGateStaticCenterCrossAttnBlock
    trainable_cls = _TrainableGatePredictedCenterCrossAttnBlock if predict_center else _TrainableGateStaticCenterCrossAttnBlock

    if residual_control == "skip":
        return fixed_cls(
            config,
            query_modality,
            predict_center=predict_center,
            residual_control=residual_control,
            gate_value=0.0,
        )

    if residual_control == "gate":
        return trainable_cls(
            config,
            query_modality,
            predict_center=predict_center,
            residual_control=residual_control,
        )

    return fixed_cls(
        config,
        query_modality,
        predict_center=predict_center,
        residual_control=residual_control,
        gate_value=1.0,
    )
