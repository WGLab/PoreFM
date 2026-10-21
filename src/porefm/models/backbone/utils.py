"""Small neural-network and state-dictionary helpers for the backbone."""

import torch
import torch.nn as nn

def strip_module_prefix(state_dict):
    """Strip repeated DDP and ``torch.compile`` prefixes from state keys.

    The original mapping is returned when no key changes; otherwise a new
    mapping shares the same tensor values.
    """
    normalized = {}
    changed = False
    for key, value in state_dict.items():
        original_key = key
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        changed = changed or key != original_key
        normalized[key] = value
    return normalized if changed else state_dict
    
    
class MLP(nn.Module):
    """Transformer feed-forward sublayer with hidden and residual dropout."""

    def __init__(self, hidden_dim: int, dim_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ff_lin1 = nn.Linear(hidden_dim, dim_ff)
        self.ff_act  = nn.GELU()
        self.ff_drop_hidden = nn.Dropout(dropout)   # between the two FFN linears
        self.ff_lin2 = nn.Linear(dim_ff, hidden_dim)
        self.ff_drop_out = nn.Dropout(dropout)      # residual dropout after FFN
        
    def forward(self, x: torch.Tensor):
        """Map ``(..., hidden_dim)`` through GELU back to ``hidden_dim``."""
        y = self.ff_lin1(x)
        y = self.ff_act(y)
        y = self.ff_drop_hidden(y)
        y = self.ff_lin2(y)
        y = self.ff_drop_out(y) 
        return y
    
class LM_Head(nn.Module):
    """Two-layer GELU projection from embeddings to per-token outputs."""

    def __init__(self, input_dim: int, dim_ff: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim_ff)
        self.act  = nn.GELU()
        self.fc2 = nn.Linear(dim_ff, out_dim)
        
    def forward(self, x: torch.Tensor):
        """Return logits with the input leading dimensions and ``out_dim`` last."""
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        return y
