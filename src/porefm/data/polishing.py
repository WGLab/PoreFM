"""Reference-guided read polishing and coordinate mapping."""

import re

import numpy as np

from porefm.data.bam import revcomp, translate_table
from porefm.data.polishing_options import DEFAULT_POLISH_INDEL_LIMIT
CANONICAL_BASES = frozenset("ACGT")


class PolishedRead:
    """Reference-polished sequence with read and reference coordinate maps.

    Parameters
    ----------
    name : str
        Source read identifier.
    sequence : numpy.ndarray
        Polished base tokens in read 5'-to-3' orientation.
    p_from_read : numpy.ndarray
        Map from original read indices to polished indices. Removed insertions
        are represented by ``-1``.
    p_to_ref : numpy.ndarray
        Map from polished indices to zero-based reference coordinates. Soft
        clips and retained insertions are represented by ``-1``.
    ins_left, ins_right : numpy.ndarray
        Boundary indices used to map intervals whose endpoints fall inside a
        removed insertion.
    del_left, del_right : numpy.ndarray
        Boundary indices used to map intervals adjacent to a repaired
        deletion.

    Notes
    -----
    All arrays are oriented with the read, including for reverse alignments.
    """

    def __init__(self, name, sequence, p_from_read, p_to_ref,
                 ins_left, ins_right, del_left, del_right):
        self.name = name
        self.sequence = sequence
        self.len = len(sequence)

        self.p_from_read = p_from_read
        self.p_to_ref = p_to_ref
        self.ins_snap_left = ins_left
        self.ins_snap_right = ins_right
        self.del_mid_left = del_left
        self.del_mid_right = del_right

    def get_coordinates(self, start_idx, end_idx):
        """Map a half-open original-read interval into polished coordinates.

        Endpoints removed by indel correction are snapped to the appropriate
        polished boundary. Requested bounds are clipped to the original read.

        Parameters
        ----------
        start_idx, end_idx : int
            Half-open interval ``[start_idx, end_idx)`` in read-oriented base
            coordinates.

        Returns
        -------
        tuple of int
            Half-open polished interval. ``(-1, -1)`` indicates an empty or
            invalid mapped interval.
        """
        R = len(self.p_from_read)
        K = self.len

        if start_idx < 0: start_idx = 0
        if end_idx > R: end_idx = R
        if start_idx >= end_idx: return -1, -1

        # Map Start
        m = start_idx
        p_start = -1
        if self.p_from_read[m] != -1:
            p_start = self.p_from_read[m]
        elif self.ins_snap_right[m] != -1:
            p_start = self.ins_snap_right[m]
        elif self.del_mid_right[m] != -1:
            p_start = self.del_mid_right[m]
        else:
            for rr in range(m + 1, R):
                if self.p_from_read[rr] != -1:
                    p_start = self.p_from_read[rr]
                    break
            if p_start == -1: p_start = K

        # Map End
        n = end_idx - 1
        p_end_inclusive = -1
        if self.p_from_read[n] != -1:
            p_end_inclusive = self.p_from_read[n]
        elif self.ins_snap_left[n] != -1:
            p_end_inclusive = self.ins_snap_left[n]
        elif self.del_mid_left[n] != -1:
            p_end_inclusive = self.del_mid_left[n]
        else:
            for rr in range(n - 1, -1, -1):
                if self.p_from_read[rr] != -1:
                    p_end_inclusive = self.p_from_read[rr]
                    break
            if p_end_inclusive == -1: p_end_inclusive = -1

        p_end = p_end_inclusive + 1
        if p_start >= p_end: return -1, -1
        return p_start, p_end

def parse_md_tag(md_str):
    """Parse a SAM MD tag into match, deletion, and mismatch operations.

    Returns a list of ``("M", length)``, ``("D", reference_bases)``, and
    ``("X", reference_base)`` tuples in alignment order.
    """
    tokens = re.findall(r"(\d+)|(\^[A-Za-z]+)|([A-Za-z])", md_str)
    ops = []
    for match_len, del_seq, snp_ref in tokens:
        if match_len: ops.append(('M', int(match_len)))
        elif del_seq: ops.append(('D', del_seq[1:]))
        elif snp_ref: ops.append(('X', snp_ref))
    return ops

def polish_read_with_maps(
    read,
    variant_mask=None,
    indel_lim=DEFAULT_POLISH_INDEL_LIMIT,
):
    """Correct likely alignment errors and retain coordinate correspondence.

    Short insertions and deletions up to ``indel_lim`` and mismatching bases
    are corrected toward the reference unless ``variant_mask`` marks them as
    biological variants. Larger indels, soft clips, and listed variants are
    preserved. The returned sequence and all coordinate maps follow the
    original read's 5'-to-3' orientation.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Mapped primary alignment with query sequence, CIGAR, and MD tag.
    variant_mask : object, optional
        Object providing ``is_valid_snp(chrom, position, read_base)`` and
        ``has_indel(chrom, position)``. Coordinates passed to it are
        zero-based reference positions.
    indel_lim : int, default=5
        Largest unmasked insertion or deletion to correct. Indels longer than
        this threshold are retained.

    Returns
    -------
    PolishedRead
        Polished integer base tokens plus read-to-polished,
        polished-to-reference, and interval-boundary maps.

    Raises
    ------
    ValueError
        If the read is unmapped or lacks a CIGAR or MD tag.
    """
    if read.is_unmapped or read.reference_name is None or read.reference_start is None:
        raise ValueError(
            f"Read {read.query_name!r} must be mapped to use reference-guided polishing"
        )

    chrom = read.reference_name
    read_seq = read.query_sequence
    read_len = len(read_seq)

    if variant_mask:
        check_snp = variant_mask.is_valid_snp
        check_indel = variant_mask.has_indel
    else:
        # Without a variant mask, every eligible discrepancy is treated as an
        # alignment/basecalling error rather than a biological allele.
        check_snp = lambda c, p, b: False
        check_indel = lambda c, p: False

    # A read base may disappear during correction, so initialize all mappings
    # as unmapped and fill only positions represented in the polished result.
    p_from_read = np.full(read_len, -1, dtype=np.int64)
    ins_snap_left = np.full(read_len, -1, dtype=np.int64)
    ins_snap_right = np.full(read_len, -1, dtype=np.int64)
    del_mid_left = np.full(read_len, -1, dtype=np.int64)
    del_mid_right = np.full(read_len, -1, dtype=np.int64)

    new_seq_parts = []
    ref_coords_parts = []

    try:
        md_ops = parse_md_tag(read.get_tag("MD"))
    except KeyError as exc:
        raise ValueError(
            f"Read {read.query_name!r} lacks the MD tag required for polishing"
        ) from exc

    q_idx = 0  # Original-read index.
    k_idx = 0  # Polished-sequence index.
    r_idx = read.reference_start  # Zero-based reference index.

    md_ptr = 0
    current_md = md_ops[0] if md_ops else None
    md_offset = 0

    cigar_ops = read.cigartuples
    if not cigar_ops:
        raise ValueError(f"Read {read.query_name!r} lacks a CIGAR required for polishing")

    for c_op, c_len in cigar_ops:

        # Soft clips and retained insertions remain in the sequence but have no
        # reference coordinate.
        if c_op == 4:
            chunk = read_seq[q_idx : q_idx + c_len]
            new_seq_parts.append(chunk)
            p_from_read[q_idx : q_idx + c_len] = np.arange(k_idx, k_idx + c_len)

            ref_coords_parts.append(np.full(c_len, -1, dtype=np.int64))

            q_idx += c_len
            k_idx += c_len

        elif c_op == 1:
            is_valid = False
            if c_len > indel_lim: is_valid = True
            elif check_indel(chrom, r_idx): is_valid = True

            if is_valid:
                chunk = read_seq[q_idx : q_idx + c_len]
                new_seq_parts.append(chunk)
                p_from_read[q_idx : q_idx + c_len] = np.arange(k_idx, k_idx + c_len)

                ref_coords_parts.append(np.full(c_len, -1, dtype=np.int64))

                q_idx += c_len
                k_idx += c_len
            else:
                # Removed insertion bases retain left/right snap points so a
                # read-space window can still map to a polished interval.
                snap_l = k_idx - 1
                snap_r = k_idx
                ins_snap_left[q_idx : q_idx + c_len] = snap_l
                ins_snap_right[q_idx : q_idx + c_len] = snap_r
                q_idx += c_len

        elif c_op == 3:
            r_idx += c_len

        # CIGAR match/deletion spans are reconciled with the finer-grained MD
        # stream, which identifies mismatches and deleted reference bases.
        elif c_op in [0, 2, 7, 8]:
            bases_covered = 0
            while bases_covered < c_len:
                if not current_md: break
                m_type, m_val = current_md

                if m_type == 'M':
                    rem_md = m_val - md_offset
                    step = min(c_len - bases_covered, rem_md)

                    if c_op != 2:
                        chunk = read_seq[q_idx : q_idx + step]
                        new_seq_parts.append(chunk)

                        p_from_read[q_idx : q_idx + step] = np.arange(k_idx, k_idx + step)

                        ref_coords_parts.append(np.arange(r_idx, r_idx + step, dtype=np.int64))

                        q_idx += step
                        k_idx += step

                    r_idx += step
                    bases_covered += step
                    md_offset += step
                    if md_offset >= m_val:
                        md_ptr += 1
                        current_md = md_ops[md_ptr] if md_ptr < len(md_ops) else None
                        md_offset = 0

                elif m_type == 'X':
                    if c_op != 2:
                        read_base = read_seq[q_idx]

                        # An ambiguous MD reference base cannot support a
                        # reference-guided correction. Keep the observed read
                        # base rather than introducing an N/ambiguity token.
                        if m_val.upper() not in CANONICAL_BASES:
                            new_seq_parts.append(read_base)
                        elif check_snp(chrom, r_idx, read_base):
                            new_seq_parts.append(read_base)
                        else:
                            new_seq_parts.append(m_val)

                        p_from_read[q_idx] = k_idx

                        ref_coords_parts.append(np.array([r_idx], dtype=np.int64))

                        q_idx += 1
                        k_idx += 1

                    r_idx += 1
                    bases_covered += 1
                    md_ptr += 1
                    current_md = md_ops[md_ptr] if md_ptr < len(md_ops) else None
                    md_offset = 0

                elif m_type == 'D':
                    d_len = len(m_val)
                    rem_del = d_len - md_offset
                    step = min(c_len - bases_covered, rem_del)

                    fix_deletion = False
                    deletion_is_canonical = all(
                        base.upper() in CANONICAL_BASES for base in m_val
                    )
                    if d_len <= indel_lim and deletion_is_canonical:
                        del_start_pos = r_idx - md_offset
                        if not check_indel(chrom, del_start_pos):
                            fix_deletion = True

                    if fix_deletion:
                        bases_to_add = m_val[md_offset : md_offset + step]
                        new_seq_parts.append(bases_to_add)

                        ref_coords_parts.append(np.arange(r_idx, r_idx + step, dtype=np.int64))

                        # A repaired deletion has no original read indices.
                        # Split it at the midpoint to define deterministic
                        # boundaries for windows on either side.
                        abs_p_start = k_idx - md_offset
                        mid_left = abs_p_start + (d_len - 1) // 2
                        mid_right = abs_p_start + (d_len) // 2
                        if q_idx > 0: del_mid_left[q_idx - 1] = mid_left
                        if q_idx < read_len: del_mid_right[q_idx] = mid_right

                        k_idx += step

                    r_idx += step
                    bases_covered += step
                    md_offset += step
                    if md_offset >= d_len:
                        md_ptr += 1
                        current_md = md_ops[md_ptr] if md_ptr < len(md_ops) else None
                        md_offset = 0

    polished_sequence = "".join(new_seq_parts)

    if ref_coords_parts:
        p_to_ref = np.concatenate(ref_coords_parts)
        p_to_ref = p_to_ref if read.is_forward else p_to_ref[::-1]
    else:
        p_to_ref = np.array([], dtype=np.int64)

    polished_sequence=polished_sequence if read.is_forward else revcomp(polished_sequence)
    polished_sequence=translate_table[np.frombuffer(polished_sequence.encode('ascii'), dtype=np.int8)]

    if not read.is_forward:
        def reverse_index_map(values):
            """Reflect valid original-read indices while reversing polished order."""
            values = values[::-1].copy()
            valid = values >= 0
            values[valid] = len(polished_sequence) - values[valid] - 1
            return values

        p_from_read = reverse_index_map(p_from_read)
        ins_snap_left, ins_snap_right = (
            reverse_index_map(ins_snap_right),
            reverse_index_map(ins_snap_left),
        )
        del_mid_left, del_mid_right = (
            reverse_index_map(del_mid_right),
            reverse_index_map(del_mid_left),
        )

    return PolishedRead(read.query_name, polished_sequence, p_from_read, p_to_ref,
                        ins_snap_left, ins_snap_right, del_mid_left, del_mid_right)
