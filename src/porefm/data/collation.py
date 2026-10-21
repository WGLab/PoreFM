"""Batch collation for aligned reads."""

from typing import List

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def aligned_collate_fn(batch):
    """Collate aligned dataset records into the model batch contract.

    Each input record is a seven-item tuple containing sequence tokens, a
    fixed-size signal window, signal percentiles, read name, reference
    positions, the buffer's minimum reference start, and strand. The returned
    tensors are delegated to :func:`collate_modal_aligned_batch`.

    Parameters
    ----------
    batch : sequence of tuple
        Records emitted by ``AlignedBamIterableDataset`` or a compatible
        iterable.

    Returns
    -------
    tuple
        Padded sequence and position tensors, stacked signal features, masks,
        metadata, strand indices, and the minimum reference start.
    """
    seqs_batch, signals_batch, percentiles_batch = [], [], []
    read_name_batch, positions_batch, strand_batch = [], [], []
    min_start = 1e12

    for batch_item in batch:
        item_seqs, item_signals, item_percentiles, item_read_names, item_positions, item_min_start, item_strand = batch_item

        min_start = min(min_start, item_min_start)
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)
        read_name_batch.append(item_read_names)
        positions_batch.append(item_positions)
        strand_batch.append(item_strand)

    return collate_modal_aligned_batch(seqs_batch, signals_batch, percentiles_batch, read_name_batch, positions_batch, strand_batch, min_start)

def collate_modal_aligned_batch(
    seqs_batch: List[torch.Tensor],
    signals_batch: List[torch.Tensor],
    percentiles_batch: List[torch.Tensor],
    read_name_batch: List,
    positions_batch: List[torch.Tensor],
    strand_batch: List[int],
    min_start
):
    """Pad and stack aligned read windows while preserving model masks.

    Parameters
    ----------
    seqs_batch : list of torch.Tensor
        Variable-length, one-based base-token vectors. Zero is reserved for
        sequence padding.
    signals_batch : list of torch.Tensor
        Signal windows with a common ``(channels, time)`` shape. Values of
        ``-10`` across all channels mark padded time steps.
    percentiles_batch : list of torch.Tensor
        Per-window signal percentile features with the same leading dimensions
        as ``signals_batch``.
    read_name_batch : list
        Read identifiers, one per window.
    positions_batch : list of torch.Tensor
        Variable-length zero-based reference coordinates aligned to sequence
        tokens. ``-1`` is used for padding or positions without a reference
        coordinate.
    strand_batch : list of int
        Strand indices, with ``0`` for forward and ``1`` for reverse.
    min_start : int or float
        Minimum valid reference coordinate observed in the source buffer, or
        the dataset's large sentinel when no coordinate is available.

    Returns
    -------
    tuple
        ``(sequences, signals, percentiles, base_mask, signal_mask,
        read_names, positions, strands, min_start)``. Sequence and position
        lengths are rounded up to a multiple of 128 for downstream attention.
    """
    signals_batch = torch.stack(signals_batch, dim=0)
    percentiles_batch = torch.stack(percentiles_batch, dim=0)

    pad_ts = torch.any(signals_batch != -10, dim=-1)
    signals_batch = torch.where(pad_ts.unsqueeze(-1), signals_batch, 0)
    percentiles_batch = torch.where(pad_ts.unsqueeze(-1), percentiles_batch, 0)

    seqs_batch_padded = pad_sequence(seqs_batch, batch_first=True, padding_value=0)

    current_max_len = seqs_batch_padded.shape[1]
    target_len = ((current_max_len + 127) // 128) * 128
    pad_needed = target_len - current_max_len

    positions_batch_padded = pad_sequence(positions_batch, batch_first=True, padding_value=-1)

    if pad_needed > 0:
        seqs_batch_padded = F.pad(seqs_batch_padded, (0, pad_needed), value=0)
        positions_batch_padded = F.pad(positions_batch_padded, (0, pad_needed), value=-1)

    pad_base_seq = seqs_batch_padded != 0

    strand = torch.tensor(strand_batch, dtype=torch.long)

    return (
        seqs_batch_padded.to(torch.long),
        signals_batch.to(torch.float32),
        percentiles_batch.to(torch.float32),
        pad_base_seq,
        pad_ts, read_name_batch, positions_batch_padded, strand, min_start
    )
