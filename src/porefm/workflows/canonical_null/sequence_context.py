"""Lossless model-input sequence context with optional compact DNA encoding."""

import operator

import numpy as np
from numba import njit


PACKED_ENCODING = 'dna2bit-nmask-v1'
_BASE_TOKEN_TRANSLATION = bytes.maketrans(b'\x01\x02\x03\x04\x05', b'ACGTN')


def encode_sequence_contexts(base_tokens, positions, radius, output_format='packed'):
    """Encode centered windows only at retained zero-based read positions.

    Parameters
    ----------
    base_tokens : numpy.ndarray of uint8, shape (L,)
        Stitched model-input tokens: 1..5 encode A/C/G/T/N; 0 is unavailable.
        Coordinates and orientation follow model input, including polishing.
    positions : numpy.ndarray of int64, shape (K,)
        Retained positions in the stitched sequence.
    radius : int
        Positive flank length; each window contains 2R+1 bases.
    output_format : {'packed', 'text'}
        Packed output uses dna2bit-nmask-v1 (see decode_sequence_context).

    Returns
    -------
    tuple[list[bytes] or list[str], list[bool]]
        Contexts and completeness flags. Unavailable positions become N and
        make the window incomplete. Observed N bases count as available.

    Notes
    -----
    Packing uses a cached Numba kernel, O(K * (2R+1)) work and packed scratch
    storage. It never builds strings or windows for unselected positions.
    """
    if radius <= 0:
        raise ValueError('Sequence context radius must be positive')
    if output_format == 'packed':
        packed, sizes, complete = _pack_contexts(base_tokens, positions, radius)
        return [row[:size].tobytes() for row, size in zip(packed, sizes)], complete.tolist()
    if output_format != 'text':
        raise ValueError("Sequence context format must be 'packed' or 'text'")
    # Translate once in C, preserving unavailable positions as NUL until each
    # retained window is checked. Actual N is distinct from this sentinel.
    sequence = base_tokens.tobytes().translate(_BASE_TOKEN_TRANSLATION).decode('ascii')
    length = len(sequence)
    contexts, complete = [], []
    for position in positions.tolist():
        start, stop = position - radius, position + radius + 1
        window = sequence[max(0, start):min(length, stop)]
        complete.append(start >= 0 and stop <= length and '\x00' not in window)
        contexts.append(
            'N' * max(0, -start) + window.replace('\x00', 'N')
            + 'N' * max(0, stop - length)
        )
    return contexts, complete


@njit(cache=True, nogil=True)
def _pack_contexts(base_tokens, positions, radius):
    """Pack selected windows directly; append an N mask only when required."""
    width = 2 * radius + 1
    base_bytes = (width + 3) // 4
    mask_bytes = (width + 7) // 8
    packed = np.zeros((positions.size, base_bytes + mask_bytes), dtype=np.uint8)
    sizes = np.full(positions.size, base_bytes, dtype=np.int64)
    complete = np.ones(positions.size, dtype=np.bool_)
    for row in range(positions.size):
        start = positions[row] - radius
        for offset in range(width):
            pos = start + offset
            token = base_tokens[pos] if 0 <= pos < base_tokens.size else 0
            if token == 0:
                complete[row] = False
            if token == 0 or token == 5:
                packed[row, base_bytes + offset // 8] |= 1 << (7 - offset % 8)
                sizes[row] = base_bytes + mask_bytes
            else:
                packed[row, offset // 4] |= (token - 1) << (6 - 2 * (offset % 4))
    return packed, sizes, complete


def decode_sequence_context(values, context_length):
    """Decode a batch of dna2bit-nmask-v1 contexts with NumPy and PyArrow.

    Parameters
    ----------
    values : sequence of bytes or 1-D object array
        Non-null packed contexts, for example a pandas Series or its
        ``to_numpy()`` result. Pass a batch, including for one context.
    context_length : int
        Positive decoded width, from the sequence_context field's ``length``
        metadata. All values must use this width.

    Returns
    -------
    numpy.ndarray of object, shape (N,)
        Python A/C/G/T/N strings in input order, ready for pandas assignment.

    Raises
    ------
    ValueError
        For nonpositive width, non-1-D input, nulls, or invalid byte counts.

    Notes
    -----
    The first ceil(length/4) bytes encode A=00, C=01, G=10, T=11, most-significant
    pair first. Optional ceil(length/8) N-mask bytes use most-significant bits
    first. N has zero base bits; unused trailing bits are zero. Missing flanks
    decode to N; the separate context_complete column records availability.

    Decoding and string conversion use array operations with no Python loop
    over rows. Temporary storage scales with N * context_length; process large
    datasets one Parquet batch at a time. Inputs are not modified.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    context_length = operator.index(context_length)
    if context_length <= 0:
        raise ValueError('context_length must be positive')
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError('values must be a one-dimensional batch of packed bytes')
    base_bytes = (context_length + 3) // 4
    width = base_bytes + (context_length + 7) // 8

    # Validate before fixed-width conversion: it would otherwise silently pad
    # short values or truncate long ones, potentially decoding the wrong width.
    encoded = pa.array(values, type=pa.binary())
    if encoded.null_count:
        raise ValueError('Packed contexts must not contain null values')
    sizes = pc.binary_length(encoded).to_numpy()
    if np.any((sizes != base_bytes) & (sizes != width)):
        raise ValueError('Packed context byte count does not match context_length')
    del encoded, sizes

    # Zero-fill absent N masks while preserving embedded/trailing zero bytes.
    raw = np.asarray(values, dtype=f'S{width}').view(np.uint8).reshape(-1, width)
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
    codes = (raw[:, :base_bytes, None] >> shifts) & 3
    codes = codes.reshape(len(raw), base_bytes * 4)[:, :context_length]
    letters = np.frombuffer(b'ACGT', dtype=np.uint8)[codes]
    n_mask = np.unpackbits(raw[:, base_bytes:], axis=1, bitorder='big')[:, :context_length]
    letters[n_mask.astype(bool)] = ord('N')

    # Arrow converts the contiguous ASCII buffer to Python strings in compiled
    # code, avoiding NumPy's slower conversion to fixed-width Unicode strings.
    offsets = np.arange(len(raw) + 1, dtype=np.int64) * context_length
    strings = pa.Array.from_buffers(
        pa.large_string(), len(raw),
        [None, pa.py_buffer(offsets), pa.py_buffer(letters)],
    )
    return strings.to_numpy(zero_copy_only=False)


def warm_context_encoder():
    """Compile/load the packed encoder once before read processing."""
    _pack_contexts(np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int64), 1)
