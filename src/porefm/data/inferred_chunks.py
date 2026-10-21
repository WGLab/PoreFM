"""Slice inferred chunks and all token metadata through common base bounds."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def has_inferred_boundaries(read):
    """Return whether this read carries an inferred chunk-boundary provider."""
    return getattr(read, "chunk_boundaries", None) is not None


def chunk_bounds(read, size, step, polished_read=None):
    """Return base intervals, optionally mapped into polished coordinates."""
    bounds = read.chunk_boundaries.bounds(size, step)
    if polished_read is not None:
        bounds = np.array([polished_read.get_coordinates(int(a), int(b))
                           for a, b in bounds], dtype=np.int64)
        bounds[np.any(bounds < 0, axis=1)] = 0
    return bounds


def slice_base_values(read, values, size, step):
    """Slice any original-read per-base array using the shared chunk bounds."""
    return [torch.as_tensor(np.asarray(values)[a:b].copy())
            for a, b in chunk_bounds(read, size, step)]


def inferred_chunks(read, size, step, polished_read=None):
    """Return signal, tokens, reference positions and read positions.

    Signal has shape ``(chunks, channels, size)`` and retains the established
    clipping/padding rules. The other three outputs are ragged tensor lists
    in original read coordinates, or polished coordinates when requested.
    Reference gaps retain their existing sentinels; unaligned positions use
    -1. No move masks or per-base dwell times are synthesized.
    """
    bounds = chunk_bounds(read, size, step, polished_read)
    signal = torch.from_numpy(read.signal).clamp(-5, 5)
    count = max(1, math.ceil((len(signal) - size) / step + 1))
    padding = (count - 1) * step + size - len(signal)
    signal = F.pad(signal, (0, 0, 0, padding), value=-10)
    signals = signal.unfold(0, size, step)
    if polished_read is not None:
        sequence, reference = polished_read.sequence, polished_read.p_to_ref
    else:
        sequence = read.get_base_seq()
        reference = np.full(read.length, -1, dtype=np.int64)
        if hasattr(read, "aligned_pairs"):
            pairs = read.aligned_pairs
            valid = pairs[:, 0] >= 0
            reference[pairs[valid, 0]] = pairs[valid, 1]
    sequences = [torch.from_numpy(sequence[a:b].copy() + 1) for a, b in bounds]
    references = [torch.as_tensor(reference[a:b].copy()).long() for a, b in bounds]
    positions = [torch.arange(a, b, dtype=torch.long) for a, b in bounds]
    return signals, sequences, references, positions


def inferred_labelled_chunks(read, size, step, polished_read=None):
    """Adapt boundary slices to the existing labelled chunk output contract.

    The move-mask result is None because it has no meaning for inferred maps.
    """
    signal, sequences, references, positions = inferred_chunks(read, size, step, polished_read)
    tokens = sequences if polished_read is not None else pad_sequence(
        sequences, batch_first=True, padding_value=0)
    return (signal, tokens, None,
            pad_sequence(references, batch_first=True, padding_value=-2),
            pad_sequence(positions, batch_first=True, padding_value=-1))
