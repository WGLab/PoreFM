"""Read-to-model chunk construction."""

from porefm.data.inferred_chunks import (has_inferred_boundaries, inferred_chunks,
    inferred_labelled_chunks, slice_base_values, chunk_bounds)
import torch
from torch.nn import functional as F

from porefm.data.reader import chunk_padded_components, pad_read_components


def get_read_chunks_aligned(read, polished_read, size, step):
    """Split an aligned read into overlapping model windows.

    Parameters
    ----------
    read : porefm.data.bam.Read
        Read whose signal has already been normalized. Signal rows and move
        table entries must share the same time-step axis.
    polished_read : porefm.data.polishing.PolishedRead or None
        Optional reference-polished sequence and coordinate maps. When absent,
        sequence tokens and reference positions come from ``read`` directly.
    size : int
        Number of signal time steps per window.
    step : int
        Offset, in signal time steps, between consecutive windows.

    Returns
    -------
    tuple
        ``(signal_chunks, sequence_chunks, initial_move_tables,
        position_chunks)``. Signal chunks have shape
        ``(n_chunks, signal_channels, size)``; sequence and position chunks
        contain one entry per emitted base token and may have variable length.
        Initial move tables retain the unfiltered window-level move masks.
    """
    if has_inferred_boundaries(read):
        signal, sequences, references, _ = inferred_chunks(read, size, step, polished_read)
        return signal.transpose(1, 2), sequences, None, references
    try:
        signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
        signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks = chunk_padded_components(
            signal, base_seq, move_table, size, step
        )
    except RuntimeError:
        print(read.read_name, size, step, signal.shape)

    if polished_read!=None:
        base_seq_idx=torch.arange(read.length)[read.indexes]
        base_seq_idx = F.pad(
            base_seq_idx,
            (0,  padding_needed),
            mode='constant',
            value=read.length-1
        )

        base_seq_idx_chunks=base_seq_idx.unfold(dimension=0, size=size, step=step)

        base_starts=base_seq_idx_chunks[:,0]
        base_ends=base_seq_idx_chunks[:,-1]

        final_base_seq_chunks=[]
        aligned_pos_chunks=[]

        for m,n in zip(base_starts, base_ends):
            p_start, p_end=polished_read.get_coordinates(int(m), int(n) + 1)
            final_base_seq_chunks.append(torch.from_numpy(polished_read.sequence[p_start:p_end]+1))
            aligned_pos_chunks.append(torch.from_numpy(polished_read.p_to_ref[p_start:p_end].copy()))

    else:
        aligned_pairs=read.aligned_pairs
        read_wise_aligned_pos=aligned_pairs[aligned_pairs[:,0]>=0][:,1]
        read_wise_aligned_pos=torch.from_numpy(read_wise_aligned_pos[read.indexes])

        read_wise_aligned_pos=F.pad(
            read_wise_aligned_pos,
            (0,  padding_needed),
            mode='constant',
            value=0
        )

        read_wise_aligned_pos_chunks = read_wise_aligned_pos.unfold(dimension=0, size=size, step=step)
        aligned_pos_chunks=[read_wise_aligned_pos_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]

        final_base_seq_chunks=base_seq_chunks

    return signal_chunks.transpose(1, 2), final_base_seq_chunks, init_move_table_chunks, aligned_pos_chunks
