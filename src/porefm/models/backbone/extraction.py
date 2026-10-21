"""Convolutional feature extractors used by backbone input projectors."""

import torch
import torch.nn as nn

class SignalConvProjector(nn.Module):
    """Four-stage strided Conv1d projector for flattened signal rows.

    ``stride_list`` and ``kernel_list`` must each describe four stages. Input
    has shape ``(B, in_channels, flattened_signal_length)``; configured strides
    are expected to reduce it to the model token length. Output is layer-
    normalized and transposed to ``(B, L, hidden_dim)``.
    """

    def __init__(self, in_channels, stride_list, kernel_list, hidden_dim, kernel_size=7):
        super().__init__()
        
        assert len(stride_list) == 4, "Must provide exactly 4 strides."
        s1, s2, s3, s4 = stride_list
        
        k1, k2, k3, k4 = kernel_list
        
        c1 = hidden_dim // 8
        c2 = hidden_dim // 4
        c3 = hidden_dim // 2

        pad = kernel_size // 2
        
        self.cnv1 = nn.Conv1d(in_channels, c1, kernel_size=k1, stride=s1, padding=pad, bias=False)
        self.cnv2 = nn.Conv1d(c1, c2, kernel_size=k2, stride=s2, padding=pad, bias=False)
        self.cnv3 = nn.Conv1d(c2, c3, kernel_size=k3, stride=s3, padding=pad, bias=False)
        self.cnv4 = nn.Conv1d(c3, hidden_dim, kernel_size=k4, stride=s4, padding=pad, bias=False)
        

        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d) and hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """Project channel-first flattened signal to token-first embeddings."""
        x = self.act(self.cnv1(x))
        x = self.act(self.cnv2(x))
        x = self.act(self.cnv3(x))
        x = self.cnv4(x)

        x = x.transpose(1, 2)
        x = self.ln(x)
        return x
    
class ConvProjector(nn.Module):
    """Three-stage same-length Conv1d projector for base embeddings."""

    def __init__(self, hidden_dim, kernel_size=7, dropout=0.1):
        super().__init__()   
        
        c1 = hidden_dim // 8
        c2 = hidden_dim // 4
        c3 = hidden_dim // 2
        
        pad = kernel_size // 2
        
        self.cnv1 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.cnv2 = nn.Conv1d(in_channels=c2, out_channels=c3, kernel_size=kernel_size, padding=pad, bias=False)
        self.cnv3 = nn.Conv1d(in_channels=c3, out_channels=hidden_dim, kernel_size=kernel_size, padding=pad, bias=False)
        
        self.ln   = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d) and hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        """Project ``(B, L, hidden_dim/8)`` to ``(B, L, hidden_dim)``."""
        x = x.transpose(1, 2)
        
        x = self.act(self.cnv1(x))
        x = self.act(self.cnv2(x))
        
        x = self.cnv3(x)
        
        x = x.transpose(1, 2)
        x = self.ln(x)
        
        return x
