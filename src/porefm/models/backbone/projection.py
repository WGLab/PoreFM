"""Project raw signal, percentile ranks, and base tokens into hidden space."""

import torch
import torch.nn as nn
from porefm.models.backbone.extraction import ConvProjector, SignalConvProjector

class SignalPercentile_Projector(nn.Module):
    """Fuse convolutional signal and percentile-rank embeddings.

    ``stride_list`` and ``kernel_list`` configure parallel four-stage signal
    projectors. Optional per-batch-item modality choices are sampled by the
    owning backbone and applied here by replacing the selected modality with a
    learned missing-modality embedding.
    """

    def __init__(self, stride_list, kernel_list, hidden_dim: int):
        super().__init__()

        self.missing_ts = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.missing_percentile = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.mask_token_ts = nn.Parameter(torch.zeros(hidden_dim))
        self.mask_token_percentile = nn.Parameter(torch.zeros(hidden_dim))
        
        self.ts_conv = SignalConvProjector(2, stride_list, kernel_list, hidden_dim)
        self.percentile_conv = SignalConvProjector(2, stride_list, kernel_list, hidden_dim)

        self.layernorm = nn.LayerNorm(hidden_dim)

        self.modality_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
    
    def _prepare_cnn_input(self, input_seq, mask):
        """Flatten signal rows and add a masked-value indicator channel.

        Input ``(B, L, R)`` becomes ``(B, 2, L*R)``. A true ``(B, L)`` mask
        is repeated over the row width, zeroes masked values, and appears as
        channel one so the convolution can distinguish masking from real zero.
        """
        B, L, R = input_seq.shape
        
        flat_input = input_seq.view(B, -1)
        
        if mask is not None:
            flat_mask = mask.repeat_interleave(R, dim=1) 
            
            indicator = flat_mask.float()
            
            signal = flat_input.clone()
            signal[flat_mask] = 0.0
            
            cnn_input = torch.stack([signal, indicator], dim=1)
        else:
            indicator = torch.zeros_like(flat_input)
            cnn_input = torch.stack([flat_input, indicator], dim=1)
            
        return cnn_input
    
    def forward(
        self,
        ts_input: torch.Tensor,
        percentiles_input: torch.Tensor,
        pad_ts: torch.Tensor,
        mask_ts=None,
        *,
        signal_dropout_choices: torch.Tensor | None = None,
    ):
        """Return fused signal-token embeddings with shape ``(B, L, D)``.

        ``ts_input`` and ``percentiles_input`` share shape ``(B, L, R)``;
        ``pad_ts`` is true for valid rows and ``mask_ts`` is true for masked
        rows. Learned mask embeddings overwrite convolutional features after
        projection, while padded output tokens are zeroed before layer norm.
        ``signal_dropout_choices`` has shape ``(B, 1, 1)`` and uses zero to
        keep both modalities, one to drop percentiles, and two to drop raw
        signal. Choices are broadcast across each batch item's sequence.
        """
        if percentiles_input is None:
            raise ValueError("percentiles_input is required when percentile fusion is enabled.")
        if signal_dropout_choices is not None:
            expected_shape = (ts_input.size(0), 1, 1)
            if tuple(signal_dropout_choices.shape) != expected_shape:
                raise ValueError(
                    "signal_dropout_choices must have shape "
                    f"{expected_shape}, got {tuple(signal_dropout_choices.shape)}."
                )
            if signal_dropout_choices.device != ts_input.device:
                raise ValueError("signal_dropout_choices must be on the same device as ts_input.")

        ts_cnn_in = self._prepare_cnn_input(ts_input, mask_ts)
        perc_cnn_in = self._prepare_cnn_input(percentiles_input, mask_ts)
        
        
        ts_emb = self.ts_conv(ts_cnn_in)
        percentiles_emb = self.percentile_conv(perc_cnn_in)
        
        if mask_ts is not None:
            mask_expanded = mask_ts.unsqueeze(-1) # (B, L, 1)
            
            ts_emb = torch.where(mask_expanded, 
                                 self.mask_token_ts.to(ts_emb.dtype), 
                                 ts_emb)
            
            percentiles_emb = torch.where(mask_expanded, 
                                          self.mask_token_percentile.to(percentiles_emb.dtype), 
                                          percentiles_emb)

        if signal_dropout_choices is not None:
            drop_perc = signal_dropout_choices == 1
            drop_ts = signal_dropout_choices == 2
            percentiles_emb = torch.where(drop_perc, 
                                          self.missing_percentile.to(percentiles_emb.dtype), 
                                          percentiles_emb)
            ts_emb = torch.where(drop_ts, 
                                 self.missing_ts.to(ts_emb.dtype), 
                                 ts_emb)

        combined = torch.cat([ts_emb, percentiles_emb], dim=-1)
        ts_emb_fused = self.modality_gate(combined)
        
        ts_emb_fused = ts_emb_fused * pad_ts.unsqueeze(-1).to(ts_emb_fused.dtype)
        
        return self.layernorm(ts_emb_fused)

class Signal_Projector(nn.Module):
    """Project raw signal with mask indicators but without percentile fusion."""

    def __init__(self, stride_list, kernel_list, hidden_dim: int):
        super().__init__()
        
        self.mask_token_ts = nn.Parameter(torch.zeros(hidden_dim))
        self.ts_conv = SignalConvProjector(2, stride_list, kernel_list, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def _prepare_cnn_input(self, ts_input, mask):
        B, L, R = ts_input.shape
        flat_input = ts_input.view(B, -1)

        if mask is not None:
            flat_mask = mask.repeat_interleave(R, dim=1)
            indicator = flat_mask.float()
            signal = flat_input.clone()
            signal[flat_mask] = 0.0
            return torch.stack([signal, indicator], dim=1)

        indicator = torch.zeros_like(flat_input)
        return torch.stack([flat_input, indicator], dim=1)
        
    def forward(self, ts_input: torch.Tensor, pad_ts: torch.Tensor, mask_ts = None):
        """Return masked, normalized signal embeddings shaped ``(B, L, D)``."""
        ts_cnn_in = self._prepare_cnn_input(ts_input, mask_ts)
        ts_emb = self.ts_conv(ts_cnn_in)

        if mask_ts is not None:
            ts_emb = torch.where(
                mask_ts.unsqueeze(-1),
                self.mask_token_ts.to(ts_emb.dtype),
                ts_emb,
            )
        else:
            ts_emb = ts_emb + 0 * self.mask_token_ts.sum()

        ts_emb = ts_emb * pad_ts.unsqueeze(-1).to(ts_emb.dtype)
        return self.layernorm(ts_emb)

class BaseSeq_Projector(nn.Module):
    """Embed base tokens and add a gated convolutional residual.

    Token zero is padding. Embeddings use width ``hidden_dim/8`` before a
    linear skip projection and three-layer convolutional path; the convolution
    starts disabled because ``base_seq_gate`` is initialized to zero.
    """

    def __init__(self, base_seq_vocab: int, hidden_dim: int):
        super().__init__()
        self.base_seq_embed = nn.Embedding(base_seq_vocab, hidden_dim//8, padding_idx=0)
        
        self.skip_proj = nn.Linear(hidden_dim//8, hidden_dim, bias=False)
        
        self.base_seq_conv = ConvProjector(hidden_dim)
        
        self.base_seq_gate = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, base_seq_input: torch.Tensor, pad_base_seq: torch.Tensor):
        """Project ``(B, L)`` tokens to masked embeddings ``(B, L, D)``."""
        base_seq_emb = self.base_seq_embed(base_seq_input)
        base_seq_emb = base_seq_emb*pad_base_seq.unsqueeze(-1).to(base_seq_emb.dtype)

        base_seq_conv = self.base_seq_conv(base_seq_emb)
        
        skip_identity = self.skip_proj(base_seq_emb)
        
        base_seq_in = skip_identity + self.base_seq_gate*base_seq_conv

        return base_seq_in
