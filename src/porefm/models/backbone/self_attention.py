"""FlashAttention self-attention blocks for PoreFM modality towers."""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from porefm.models.backbone.utils import MLP

from flash_attn.modules.mha import get_alibi_slopes
from einops import rearrange
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func,   # self-attn
    flash_attn_varlen_kvpacked_func     # cross-attn
)

from porefm.models.backbone.utils import MLP

def alibi_lambdas_from_half_lives(
    n_heads: int,
    half_life_min: float,
    half_life_max: float,
    largest_first: bool = True,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Convert geometric attention half-lives to ALiBi decay coefficients.

    Half-lives are token distances and are spaced geometrically between the
    supplied endpoints. With ``largest_first=True``, head zero receives the
    most local (largest) decay. The result has shape ``(n_heads,)``.
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

class SelfAttention(nn.Module):
    """
    Self-attention with fused QKV projection + alibi + FlashAttention (qkvpacked) + output proj.
    Applies dropout to attention probabilities (inside FlashAttention via argument)
    and residual dropout to the projected attention output.
    Inputs are assumed pre-normalized by the caller. ``query_modality`` selects
    ``base_seq_window`` or ``ts_window`` from the config. Padding is removed
    before the variable-length FlashAttention call and restored afterward.
    """
    def __init__(self, config, query_modality):
        super().__init__()
        self.query_modality=query_modality
        assert config.hidden_dim % config.num_heads == 0
        self.h = config.num_heads
        self.dh = config.hidden_dim // config.num_heads
        assert self.dh % 2 == 0
        
        self.dropout = float(config.dropout)

        self.qkv  = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_dropout = nn.Dropout(self.dropout)  # residual dropout after attention
        
        self.window=config.base_seq_window  if self.query_modality == 'seq' else config.ts_window
        
        self.temperature=config.self_attn_temp
        self.attn_scale=(1.0 / math.sqrt(self.dh)) / self.temperature
        half_life_min = getattr(config, "self_attn_alibi_half_life_min", None)
        half_life_max = getattr(config, "self_attn_alibi_half_life_max", None)
        if half_life_min is not None and half_life_max is not None:
            slopes = alibi_lambdas_from_half_lives(
                n_heads=self.h,
                half_life_min=half_life_min,
                half_life_max=half_life_max,
                largest_first=True,
                device=None,
                dtype=torch.float32,
            )
        else:
            slopes = torch.Tensor(get_alibi_slopes(self.h)).to(torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)
    
    @torch._dynamo.disable()
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to valid tokens and restore padded layout.

        Parameters
        ----------
        x : torch.Tensor
            Pre-normalized embeddings with shape ``(B, L, D)``.
        mask : torch.Tensor
            Boolean matrix ``(B, L)`` with true for valid tokens.

        Returns
        -------
        torch.Tensor
            Projected attention output with shape ``(B, L, D)``. Residual
            addition is performed by :class:`SelfAttnBlock`.
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).contiguous()  # (B,L,3,H,Dh)

        qkv_flat = rearrange(qkv, "b l three h d -> b l (three h d)")
        qkv_unpad, idx, cu, max_s, _ = unpad_input(qkv_flat, mask)
        qkv_unpad = rearrange(qkv_unpad, "nnz (three h d) -> nnz three h d", three=3, h=self.h)

        out_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad, cu, max_s,
            self.dropout if self.training else 0.0,
            softmax_scale=self.attn_scale, causal=False,
            window_size=self.window,
            alibi_slopes=self.alibi_slopes)  # (nnz, H, Dh)

        out = pad_input(rearrange(out_unpad, "nnz h d -> nnz (h d)"), idx, B, L)
        out = self.proj(out)
        out = self.out_dropout(out)  # residual dropout before residual add (done by caller)
        return out
    
    
class SelfAttnBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""
    def __init__(self, config, query_modality):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        
        self.attn  = SelfAttention(config, query_modality)
        
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        self.MLP = MLP(hidden_dim=config.hidden_dim, dim_ff=config.dim_feedforward, dropout=config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Contextualize ``(B, L, D)`` embeddings at true mask positions."""
        x = x + self.attn(self.norm1(x), mask)

        x = x+ self.MLP(self.norm2(x))
        return x
