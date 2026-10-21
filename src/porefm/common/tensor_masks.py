"""Shared tensor-mask operations."""

import torch


def ignore_edges(x, num_trim):
    """Clear the first and last valid tokens in each left-aligned mask row.

    ``x`` is a boolean ``(batch, length)`` mask whose true values are assumed
    contiguous from index zero. ``num_trim`` valid positions are removed from
    both ends; short rows may become entirely false. The input is not mutated.
    """
    lens = x.sum(dim=1, keepdim=True)
    indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    valid_region = (indices >= num_trim) & (indices < (lens - num_trim))
    trimmed_x = x & valid_region

    return trimmed_x
