"""Optional RNALfold features in read-oriented, unpadded chunk coordinates.

The optional native extension requires ViennaRNA. Importing this module does
not load the extension; only enabling the structure features does.
"""

from importlib import import_module

import numpy as np
import torch


RNA_STRUCTURE_SPAN = 30
RNA_STRUCTURE_RADIUS = 30
RNA_STRUCTURE_OFFSETS = tuple(range(-30, -3)) + tuple(range(4, 31))
RNA_STRUCTURE_FEATURE_DIM = 2 * len(RNA_STRUCTURE_OFFSETS) + 1
_RNA_ALPHABET = np.array(list('NACGUN'), dtype='S1')
_RNA_OFFSETS = np.asarray(RNA_STRUCTURE_OFFSETS)


def require_rnalocal_extension():
    """Load the optional RNALfold extension."""
    try:
        return import_module('porefm.workflows.canonical_null.rnalocal_ext')
    except ModuleNotFoundError as exc:
        if exc.name != 'porefm.workflows.canonical_null.rnalocal_ext':
            raise
    try:
        return import_module('rnalocal_ext')
    except ImportError as exc:
        raise ImportError(
            'RNA structure features require the optional rnalocal_ext extension. '
            'Make rnalocal_ext available on PYTHONPATH, or use a model '
            'without RNA structure features.'
        ) from exc


def _sequence_features(sequence, extension):
    """Fold one contiguous sequence and retain the fixed 109-column layout.

    The prototype requires length > span. For short chunks, reduce span and
    radius to length - 1, then scatter into the same offset columns. Fewer
    than five bases cannot form a state with minimum pair distance four.
    """
    if len(sequence) <= 4:
        return np.zeros((len(sequence), RNA_STRUCTURE_FEATURE_DIM), dtype=np.float32)
    span = min(RNA_STRUCTURE_SPAN, len(sequence) - 1)
    radius = min(RNA_STRUCTURE_RADIUS, span)
    values, offsets = extension.local_structure_features(
        sequence, span=span, radius=radius, noLP=True,
    )
    values, offsets = np.asarray(values), np.asarray(offsets)
    expected_offsets = (_RNA_OFFSETS if radius == RNA_STRUCTURE_RADIUS else
                        np.array(tuple(range(-radius, -3)) + tuple(range(4, radius + 1))))
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError('rnalocal_ext returned unexpected pairing offsets')
    if values.shape != (len(sequence), 2 * len(offsets) + 1) or not np.isfinite(values).all():
        raise ValueError('rnalocal_ext returned invalid structure features')
    # Ordinary chunks already have the final column layout. Avoid allocating
    # and scattering a second dense matrix for every native folding result.
    if radius == RNA_STRUCTURE_RADIUS:
        return values
    result = np.zeros((len(sequence), RNA_STRUCTURE_FEATURE_DIM), dtype=np.float32)
    columns = np.searchsorted(RNA_STRUCTURE_OFFSETS, offsets)
    k = len(offsets)
    result[:, columns] = values[:, :k]
    result[:, len(RNA_STRUCTURE_OFFSETS) + columns] = values[:, k:2 * k]
    result[:, -1] = values[:, -1]
    return result


@torch.compiler.disable
def build_rna_structure_features(base_seq_input, pad_base_seq):
    """Return detached float32 ``(B,L,109)`` features on the input device.

    IDs 1/2/3/4 encode A/C/G/U and 5 encodes unknown N. The existing RNA
    loader supplies biological read orientation; do not reverse these IDs.
    False mask positions split sequences and remain zero, so neither padding
    nor other batch members influence folding. Unknown bases retain their
    positions as N. Energies and delta-f3 remain in kcal/mol; validity masks
    remain binary. Each contiguous chunk receives one native DP scan.

    Canonical workflows call this on CPU tensors during DataLoader collation.
    Direct CUDA inputs incur a device synchronization. Results depend on
    available chunk context, including exterior DP effects beyond the maximum
    pair span.
    """
    if base_seq_input.ndim != 2 or pad_base_seq.shape != base_seq_input.shape:
        raise ValueError('RNA structure inputs must both have shape (B,L)')
    extension = require_rnalocal_extension()
    ids = base_seq_input.detach().cpu().numpy()
    mask = pad_base_seq.detach().bool().cpu().numpy()
    if np.any(mask & ((ids < 1) | (ids > 5))):
        raise ValueError('RNA structure features require base IDs 1..5 at valid positions')
    result = np.zeros((*ids.shape, RNA_STRUCTURE_FEATURE_DIM), dtype=np.float32)
    for row in range(len(ids)):
        # Split rather than compact internal mask gaps, preserving distances.
        boundaries = np.diff(np.pad(mask[row].astype(np.int8), (1, 1)))
        for start, end in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            sequence = _RNA_ALPHABET[ids[row, start:end].astype(np.int64)].tobytes().decode('ascii')
            result[row, start:end] = _sequence_features(sequence, extension)
    return torch.from_numpy(result).to(base_seq_input.device)


def append_rna_structure_features(batch, *, enabled=False):
    """Append CPU ``(B,L,109)`` float32 features, or None when disabled.

    ``batch`` is already collated; indices 0 and 3 hold the final base IDs
    and padding mask. Used directly as the canonical inference DataLoader's
    collator (automatic batching is disabled), and after scale collation.
    Computing here preserves exact chunk context and lets loader workers
    prefetch folding while the main process runs the GPU. No CUDA tensors or
    model objects are sent to workers. DataLoader subsequently pins the result.
    """
    # Preserve DataLoader's default conversion of NumPy metadata (notably
    # complete-read strands) when replacing its unbatched collator.
    batch = torch.utils.data.default_convert(batch)
    features = None
    if enabled:
        if batch[0].device.type != 'cpu' or batch[3].device.type != 'cpu':
            raise ValueError('RNA structure batch preparation requires CPU inputs')
        features = build_rna_structure_features(batch[0], batch[3])
    return (*batch, features)
