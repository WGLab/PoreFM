"""Decode signal BAM records and construct strand-aware sequence labels."""

import numpy as np
import pod5 as p5
import blosc2
import pysam
import os, re, argparse, math
import numpy as np
from pathlib import Path
from numba import jit, njit
from itertools import repeat
from porefm.common.utils import quantile_normalize
from porefm.data.chunk_boundaries import uses_inferred_chunks, decode_inferred_signal
from porefm.preprocessing import build_lookup_table, get_kmer_size_from_lookup_table, resolve_preprocessing

# Allow blosc2 to fan out across available CPU cores for decompression-heavy
# workers.
blosc2.set_nthreads(os.cpu_count() or 1)


translate_table = np.zeros(128, dtype=np.int8)
translate_table[ord('A')] = 0
translate_table[ord('C')] = 1
translate_table[ord('G')] = 2
translate_table[ord('T')] = 3
translate_table[ord('N')] = 4

num_to_base_map={0:'A', 1:'C', 2:'G', 3:'T', 4:'N'}

cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':3,'H':3, 'P':4, 'B':4, 'N':5}
cigar_pattern = r'\d+[A-Za-z=]'

_revcomp_trans = str.maketrans("ATCG[]atcg", "TAGC][tagc")


def revcomp(s):
    """Return the reverse complement of a nucleotide or regex-like string."""
    return s.translate(_revcomp_trans)[::-1]


def validate_move_table_stride(stride, expected_ts_dim=None, read_name=None):
    """Check that a BAM move-table stride matches the model signal width.

    Validation is disabled when ``expected_ts_dim`` is ``None``. ``read_name``
    is used only to make a mismatch error easier to diagnose.
    """
    if expected_ts_dim is None:
        return

    stride = int(stride)
    expected_ts_dim = int(expected_ts_dim)
    if stride != expected_ts_dim:
        name = f" for read {read_name}" if read_name is not None else ""
        raise ValueError(
            f"BAM mv tag stride{name} is {stride}, but configured ts_dim is {expected_ts_dim}."
        )


def passes_min_qscore(bam_read, min_qscore):
    """Return whether a BAM record passes an optional mean-qscore filter.

    Records without the optional ``qs`` tag pass; records with the tag must
    have a value greater than or equal to ``min_qscore``.
    """
    return not bam_read.has_tag('qs') or bam_read.get_tag('qs') >= min_qscore

ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


@njit(nogil=True, fastmath=True)
def _fast_seq_to_signal_9mer(seq_ints, lookup_table):
    """Preserve the established R10/RNA004 9-mer extraction fast path."""
    n = len(seq_ints)
    exp = np.zeros(n, dtype=np.float32)
    if n < 9:
        return exp

    curr_kmer = 0
    for i in range(8):
        curr_kmer = (curr_kmer << 2) | seq_ints[i]

    for i in range(8, n):
        curr_kmer = ((curr_kmer << 2) | seq_ints[i]) & 262143
        exp[i - 3] = lookup_table[curr_kmer]

    return exp


@njit(nogil=True, fastmath=True)
def _fast_seq_to_signal_5mer(seq_ints, lookup_table):
    """Extract centered expected levels from an R9/RNA002 5-mer table."""
    n = len(seq_ints)
    exp = np.zeros(n, dtype=np.float32)
    if n < 5:
        return exp

    curr_kmer = 0
    valid_bases = 0
    for i in range(n):
        base = seq_ints[i]
        if base < 0 or base > 3:
            curr_kmer = 0
            valid_bases = 0
            continue

        curr_kmer = ((curr_kmer << 2) | base) & 1023
        valid_bases += 1
        if valid_bases >= 5:
            exp[i - 2] = lookup_table[curr_kmer]

    return exp


@njit(nogil=True, fastmath=True)
def _fast_seq_to_signal_generic(seq_ints, lookup_table, kmer_size):
    n = len(seq_ints)
    exp = np.zeros(n, dtype=np.float32)
    if n < kmer_size:
        return exp

    mask = len(lookup_table) - 1
    center_offset = kmer_size // 2
    curr_kmer = 0
    valid_bases = 0

    for i in range(n):
        base = seq_ints[i]
        if base < 0 or base > 3:
            curr_kmer = 0
            valid_bases = 0
            continue

        curr_kmer = ((curr_kmer << 2) | base) & mask
        valid_bases += 1
        if valid_bases >= kmer_size:
            exp[i - center_offset] = lookup_table[curr_kmer]

    return exp


def fast_seq_to_signal(seq_ints, lookup_table, kmer_size=None):
    """Assign an expected-current level to each complete centered k-mer.

    Parameters
    ----------
    seq_ints : array-like
        Base integers encoded as ``A=0, C=1, G=2, T=3``. Generic and 5-mer
        paths reset across other values.
    lookup_table : numpy.ndarray
        Dense base-4 expected-current table with length ``4**k``.
    kmer_size : int, optional
        Odd k-mer length. Inferred from ``lookup_table`` when omitted.

    Returns
    -------
    numpy.ndarray
        ``float32`` levels with the same length as ``seq_ints``. Positions
        lacking a complete valid centered k-mer remain zero.
    """
    if kmer_size is None:
        kmer_size = get_kmer_size_from_lookup_table(lookup_table)
    if kmer_size == 9:
        return _fast_seq_to_signal_9mer(seq_ints, lookup_table)
    if kmer_size == 5:
        return _fast_seq_to_signal_5mer(seq_ints, lookup_table)
    return _fast_seq_to_signal_generic(seq_ints, lookup_table, kmer_size)


def get_expected_signal_for_normalization(read):
    """Return the interior expected levels used for signal normalization.

    ``read`` must expose ``exp_signal``, ``kmer_size``, and ``read_name``.
    Half a k-mer is removed from each edge so every returned value represents
    a complete centered k-mer.
    """
    kmer_size = int(read.kmer_size)
    if kmer_size < 1 or kmer_size % 2 == 0:
        raise ValueError(f"Expected an odd read.kmer_size, got {kmer_size}.")
    if len(read.exp_signal) < kmer_size:
        raise ValueError(
            f"Read {read.read_name} has length {len(read.exp_signal)}, shorter than "
            f"the {kmer_size}-mer signal model."
        )

    edge_size = kmer_size // 2
    if edge_size == 0:
        return read.exp_signal
    return read.exp_signal[edge_size:-edge_size]
        
def get_candidates(read_seq, align_data, aligned_pairs, ref_pos_dict, ):    
    """Map labelled reference candidates onto aligned read coordinates.

    ``align_data`` is the tuple ``(is_mapped, is_forward, ref_name,
    reference_start, reference_end, read_length)`` and ``ref_pos_dict`` maps a
    contig to forward/reverse sorted position arrays. The function returns
    reference-filtered alignment pairs and all non-gap read-wise pairs. Reverse
    read coordinates are converted to read-oriented indices for candidate
    filtering.

    Notes
    -----
    This legacy helper is currently not called by the package. Its ``read_seq``
    and ``is_mapped`` inputs are retained for the historical contract but are
    not consulted by the implementation.
    """
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=align_data

    ref_motif_pos=ref_pos_dict[ref_name][0] if is_forward else ref_pos_dict[ref_name][1]

    common_pos=ref_motif_pos[(ref_motif_pos>=reference_start)&(ref_motif_pos<reference_end)]
    aligned_pairs_ref_wise=aligned_pairs[aligned_pairs[:,1]>=0][common_pos-reference_start]

    aligned_pairs_ref_wise=aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0]>=0]
    aligned_pairs_read_wise_original=aligned_pairs[aligned_pairs[:,0]>=0]
    aligned_pairs_read_wise=np.copy(aligned_pairs_read_wise_original)

    if not is_forward:
        aligned_pairs_ref_wise=aligned_pairs_ref_wise[::-1]
        aligned_pairs_read_wise=aligned_pairs_read_wise[::-1]
        aligned_pairs_ref_wise[:,0]=read_length-aligned_pairs_ref_wise[:,0]-1
        aligned_pairs_read_wise[:,0]=read_length-aligned_pairs_read_wise[:,0]-1

    return aligned_pairs_ref_wise, aligned_pairs_read_wise_original


@jit(nopython=True)
def get_aligned_pairs(cigar_tuples, ref_start):
    """Expand compact CIGAR operations into read/reference coordinate pairs.

    Parameters
    ----------
    cigar_tuples : numpy.ndarray
        Integer array of shape ``(n_operations, 2)`` with operation lengths in
        column 0 and the internal ``cigar_map`` code in column 1.
    ref_start : int
        Zero-based reference start of the alignment.

    Returns
    -------
    numpy.ndarray
        ``int32`` array of shape ``(sum(operation_lengths), 2)`` containing
        read and reference coordinates. ``-1`` denotes an indel gap; ``-2``
        denotes clipping on the read side or a reference skip on the reference
        side.
    """
    alen = np.sum(cigar_tuples[:, 0])
    pairs = np.zeros((alen, 2), dtype=np.int32)

    ref_cord = ref_start - 1
    read_cord = -1
    pair_cord = 0
    
    for i in range(len(cigar_tuples)):
        len_op = cigar_tuples[i, 0]
        op = cigar_tuples[i, 1]
        
        if op == 0:  # Match
            for k in range(len_op):            
                ref_cord += 1
                read_cord += 1
                pairs[pair_cord, 0] = read_cord
                pairs[pair_cord, 1] = ref_cord
                pair_cord += 1

        elif op == 2:  # Insert
            for k in range(len_op):            
                read_cord += 1            
                pairs[pair_cord, 0] = read_cord
                pairs[pair_cord, 1] = -1
                pair_cord += 1
        
        elif op == 3:  # Soft Clip
            for k in range(len_op):            
                read_cord += 1            
                pairs[pair_cord, 0] = read_cord
                pairs[pair_cord, 1] = -2
                pair_cord += 1
                
        elif op == 1:  # Deletion
            for k in range(len_op):            
                ref_cord += 1            
                pairs[pair_cord, 0] = -1
                pairs[pair_cord, 1] = ref_cord
                pair_cord += 1
                
        elif op == 5:  # Skip
            for k in range(len_op):            
                ref_cord += 1            
                pairs[pair_cord, 0] = -2
                pairs[pair_cord, 1] = ref_cord
                pair_cord += 1
                
    return pairs

@jit(nopython=True)
def get_ref_to_num(x):
    """Encode a reference contig for forward and reverse-strand lookup.

    Returns an ``int8`` array of shape ``(len(x) + 2, 2)``. Column 0 contains
    ``A/C/G/T/N = 0/1/2/3/4`` and column 1 contains the complementary base.
    Two trailing ``N`` rows intentionally make alignment sentinels ``-2`` and
    ``-1`` resolve to the unknown-base token through NumPy negative indexing.
    """
    b=np.full((len(x)+2,2),fill_value=0,dtype=np.int8)
    
    for i,l in enumerate(x):
        if l=='A':
            b[i,0]=0
            b[i,1]=3
            
        elif l=='T':
            b[i,0]=3
            b[i,1]=0
            
        elif l=='C':
            b[i,0]=1
            b[i,1]=2
            
        elif l=='G':
            b[i,0]=2
            b[i,1]=1
            
        else:
            b[i,0]=4
            b[i,1]=4
    
    b[-1,0]=4
    b[-1,1]=4
    b[-2,0]=4
    b[-2,1]=4
            
    return b

def get_ref_info(args):
    """Load and encode one FASTA contig for multiprocessing-style mapping.

    ``args`` is a ``(reference_path, contig_name)`` pair. The return value is
    ``(contig_name, encoded_reference)`` as produced by
    :func:`get_ref_to_num`.
    """
    ref_path, chrom = args
    ref_fasta = pysam.FastaFile(ref_path)
    seq = ref_fasta.fetch(chrom).upper()
    seq_array = get_ref_to_num(seq)
    return chrom, seq_array

def get_pos(path):
    """Load a strand-aware per-position target table.

    The headerless tab-separated format is ``chrom, position, strand,
    target_0, ..., target_K``. Positions are used as zero-based reference
    coordinates without conversion; strands accept ``+``, ``-``, ``0``, and
    ``1``. Rows with fewer than four fields are ignored.

    Returns
    -------
    dict
        Nested ``contig -> strand_index -> position -> list[float]`` mapping,
        where strand indices are ``0`` for forward and ``1`` for reverse.
    """
    labelled_pos_list={}
    strand_map={'+':0, '-':1, '0':0, '1': 1}
    
    with open(path) as file:
            for line in file:
                line=line.rstrip('\n').split('\t')
                if len(line) < 4:
                    continue
                if line[0] not in labelled_pos_list:
                    labelled_pos_list[line[0]]={0:{}, 1:{}}

                # Position files now store one label column per class after
                # chrom, position, and strand. Keep the class vector intact so
                # downstream code can consume probabilities/regression-style
                # float labels without interpreting them as class indices.
                labelled_pos_list[line[0]][strand_map[line[2]]][int(line[1])]=[float(x) for x in line[3:]]
    return labelled_pos_list

class Read():
    """Decoded aligned nanopore read in model and reference orientation.

    Parameters
    ----------
    move_table : numpy.ndarray
        Vector of length ``T`` whose nonzero entries begin new base events.
    signal : numpy.ndarray
        Raw signal values reshapeable to ``(T, stride)``.
    base_qual : numpy.ndarray
        Per-base error probabilities with length ``N``.
    full_seq : numpy.ndarray
        Integer array of shape ``(R, 2)`` containing aligned read and reference
        tokens; ``4`` represents an unknown base or alignment gap.
    aligned_pairs : numpy.ndarray
        Array of shape ``(R, 2)`` containing read and zero-based reference
        coordinates with negative gap/sentinel values.
    read_name, ref_name : str
        Read identifier and aligned reference contig.
    seq_type : str
        Sequence type, conventionally ``"dna"`` or ``"rna"``.
    div : tuple
        ``(mismatches, indel_bases, alignment_length)`` counts.
    flag : int
        SAM flag; bit 16 determines ``is_forward`` and ``is_reverse``.
    reference_start, reference_end : int
        Zero-based half-open reference span.
    cigarstring : str
        Original SAM CIGAR text.
    exp_signal : numpy.ndarray
        Expected current per read base.
    kmer_size : int
        K-mer size used to compute ``exp_signal``.
    chunk_boundaries : InferredBoundaries, DtwBoundaries or None
        Optional chunk-boundary provider. When present, ``signal`` is already
        cropped and shaped, and ``move_table``, ``indexes`` and ``segments``
        are None. Chunk consumers slice sequence metadata using this map.

    Attributes
    ----------
    indexes : numpy.ndarray
        Length-``T`` mapping from signal rows to zero-based base indices.
    segments : numpy.ndarray
        ``(N, 2)`` half-open signal-row bounds for base events.
    """

    def __init__(self, move_table, signal, base_qual, full_seq, aligned_pairs, read_name, ref_name, seq_type, div, flag, reference_start, reference_end, cigarstring, exp_signal, kmer_size, chunk_boundaries=None):
        self.chunk_boundaries = chunk_boundaries
        self.move_table=move_table
        self.signal = signal if chunk_boundaries is not None else signal.reshape(len(move_table), -1)
        self.base_qual=base_qual
        self.full_seq=full_seq
        self.aligned_pairs=aligned_pairs
        self.length=len(self.base_qual)
        
        if chunk_boundaries is None:
            self.indexes=np.cumsum(move_table)-1

            # Event starts define the half-open signal span assigned to each base.
            move_indices = np.flatnonzero(move_table)
            starts = move_indices
            ends = np.empty_like(starts)
            ends[:-1] = move_indices[1:]
            ends[-1] = len(move_table)
            self.segments = np.stack((starts, ends), axis=1)
        else:
            self.indexes = None
            self.segments = None
        
        
        self.read_name=read_name
        self.ref_name=ref_name
        
        self.seq_type=seq_type
        self.div=div
        
        self.flag=flag
        self.is_forward=flag&16==0
        self.is_reverse=flag&16>0
        
        self.ref_start = reference_start
        self.ref_end = reference_end
        
        self.cigarstring=cigarstring
        
        self.exp_signal=exp_signal
        self.kmer_size=kmer_size
    
    def get_base_seq(self, string=False, expanded=False):
        """Return the gap-free basecall sequence in read orientation.

        ``string=True`` converts integer tokens to nucleotide strings.
        ``expanded=True`` repeats base values over move-table time steps and
        therefore returns length ``T`` rather than read length ``N``.
        """
                    
        seq=self.full_seq[self.aligned_pairs[:,0]>=0][:,0]
        if string:
            seq=np.vectorize(num_to_base_map.get)(seq)
        
        if expanded:
            if self.indexes is None:
                raise ValueError("Inferred chunks do not define per-base signal alignment")
            return seq[self.indexes]
        else:
            return seq
        
            
    def get_ref_seq(self, string=False, expanded=False):
        """Return reference tokens in the read's 5'-to-3' orientation.

        With ``expanded=False``, deletion-free reference positions are
        returned. With ``expanded=True``, reference bases corresponding to
        read bases are repeated over move-table time steps; insertions map to
        the unknown token. ``string=True`` affects only the non-expanded path,
        matching the current API contract.
        """
        
        if expanded:
            if self.indexes is None:
                raise ValueError("Inferred chunks do not define per-base signal alignment")
            seq=self.full_seq[self.aligned_pairs[:,0]>=0][:,1]
            return seq[self.indexes]
        
        else:
            seq=self.full_seq[self.aligned_pairs[:,1]>=0][:,1]
        
        if string:
            seq=np.vectorize(num_to_base_map.get)(seq)
        
        return seq
    
    def get_alignment(self, string=False):
        """Return the full read/reference alignment including gap tokens.

        The result has shape ``(R, 2)`` with read tokens in column 0 and
        reference tokens in column 1. ``string=True`` converts integer tokens
        to base characters.
        """
        
        if string:
            return np.vectorize(num_to_base_map.get)(self.full_seq) 
        else:
            return self.full_seq
    
    def get_divergence(self, include_mismatches=False):
        """Compute alignment divergence from stored mismatch and indel counts.

        By default only indel bases contribute because mismatches may reflect
        modified bases. Set ``include_mismatches=True`` to include both. The
        denominator is the stored alignment length excluding splicing.
        """
        mismatches, indels, alen=self.div
        return (indels+mismatches)/alen if include_mismatches else indels/alen
    
    def get_labels(self, labelled_arrays):
        """Map sorted reference labels onto aligned, non-gap read positions.

        ``labelled_arrays`` must use the structure returned by
        :func:`get_labelled_arrays`. Only labels within this read's half-open
        reference span and with a corresponding read base are retained. The
        legacy record-array return contains ``read_pos``, ``ref_pos``,
        ``align_idx``, and scalar ``label`` fields.

        Notes
        -----
        Current training loaders map vector targets through newer chunk-level
        paths rather than this legacy scalar-label method.
        """
        positions, labels_arr = labelled_arrays[self.ref_name][self.is_reverse]
        
        if len(positions) == 0:
            return np.array([], dtype=[('read_pos', 'i4'), ('ref_pos', 'i4'),('align_idx', 'i4'), ('label', 'f4')])

        # First restrict the sorted labels by interval, then use the alignment
        # pairs to discard reference positions deleted from the read.
        start_idx = np.searchsorted(positions, self.ref_start)
        end_idx = np.searchsorted(positions, self.ref_end)
        
        relevant_positions = positions[start_idx:end_idx]
        relevant_labels = labels_arr[start_idx:end_idx]
        
        if len(relevant_positions) == 0:
             return np.array([], dtype=[('read_pos', 'i4'), ('ref_pos', 'i4'),('align_idx', 'i4'), ('label', 'f4')])

        # Retain the alignment-row index because consumers may need to recover
        # the corresponding row in ``full_seq``.
        tmp_aligned_pairs = np.hstack([self.aligned_pairs, np.arange(len(self.aligned_pairs))[:,np.newaxis]])
        
        if self.is_forward:
            aligned_pairs_subset = tmp_aligned_pairs
        else:
            aligned_pairs_subset = tmp_aligned_pairs[::-1]

        mask = np.isin(aligned_pairs_subset[:, 1], relevant_positions)
        aligned_pairs_ref_wise = aligned_pairs_subset[mask]
        
        if not self.is_forward:
            aligned_pairs_ref_wise = aligned_pairs_ref_wise[::-1]

        aligned_pairs_ref_wise = aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0] >= 0]
        
        query_pos = aligned_pairs_ref_wise[:, 1]
        
        local_idx = np.searchsorted(relevant_positions, query_pos)
        final_labels = relevant_labels[local_idx]

        rec_arr = np.core.records.fromarrays(
            np.hstack([aligned_pairs_ref_wise, final_labels[:,np.newaxis]]).T, 
            dtype=[('read_pos', 'i4'), ('ref_pos', 'i4'),('align_idx', 'i4'), ('label', 'f4')]
        )
        
        return rec_arr

class Read_Lite():
    """Decoded signal read without reference-alignment metadata.

    This lightweight counterpart to :class:`Read` stores signal, read-oriented
    integer base tokens and expected current. ``chunk_boundaries=None`` uses
    the original move-derived indexes/events; an inferred boundary provider
    keeps ``move_table``, ``indexes`` and ``segments`` as None. It is used for
    pretraining and read-based labelling without reference coordinates.
    """

    def __init__(self, move_table, signal, base_seq, read_name, seq_type, exp_signal, kmer_size, chunk_boundaries=None):
        self.chunk_boundaries = chunk_boundaries
        self.move_table=move_table
        self.signal = signal if chunk_boundaries is not None else signal.reshape(len(move_table), -1)
        self.base_seq=base_seq
        self.length=len(self.base_seq)
        if chunk_boundaries is None:
            self.indexes=np.cumsum(move_table)-1

            # Event starts define the half-open signal span assigned to each base.
            move_indices = np.flatnonzero(move_table)
            starts = move_indices
            ends = np.empty_like(starts)
            ends[:-1] = move_indices[1:]
            ends[-1] = len(move_table)
            self.segments = np.stack((starts, ends), axis=1)
        else:
            self.indexes = None
            self.segments = None
        
        self.read_name=read_name

        self.seq_type=seq_type
        
        self.exp_signal=exp_signal
        self.kmer_size=kmer_size
    
    def get_base_seq(self, string=False, expanded=False):
        """Return read-oriented base tokens or move-table-expanded values.

        ``string=True`` converts integer tokens to nucleotide characters;
        ``expanded=True`` repeats each base across its signal time steps.
        """
                    
        seq=self.base_seq
        if string:
            seq=np.vectorize(num_to_base_map.get)(seq)
        
        if expanded:
            if self.indexes is None:
                raise ValueError("Inferred chunks do not define per-base signal alignment")
            return seq[self.indexes]
        else:
            return seq
        
def get_read_info_lite(
    bam_read, seq_type=None, kmer_table=None, expected_ts_dim=None, oriented_sequence=None,
    *, raw_signal=None, chunking_mode="auto", preprocessing=None, chemistry=None,
):
    """Decode sequence and signal with move-based or inferred chunk boundaries.

    Parameters
    ----------
    bam_read : pysam.AlignedSegment
        Record with sequence and a blosc2-compressed ``SG`` tag, unless
        ``raw_signal`` is supplied. Move-based decoding requires ts and mv;
        auto mode infers boundaries when mv is absent.
    seq_type : str, optional
        Sequence type. ``"rna"`` reverses the signal and reconstructs the move
        table so time follows the read-oriented sequence. Inferred from profile.
    kmer_table : numpy.ndarray, optional
        Dense expected-current lookup table; inferred from profile.
    preprocessing : PreprocessingSpec or mapping, optional
        Resolved profile supplying sequence type and current table.
    chemistry : str, optional
        Built-in chemistry when no checkpoint/profile is available.
    expected_ts_dim : int or None
        Expected stride stored as the first value of the ``mv`` tag.
    oriented_sequence : str or None
        Optional sequence already in desired read orientation. Otherwise the
        BAM sequence is reverse-complemented for reverse alignments.
    raw_signal : numpy.ndarray or None
        Optional one-dimensional int16 ADC signal anchored at this read's
        untrimmed origin. Replaces SG decoding for POD5 inference. Dorado
        parent-to-child ``sp`` slicing must already have been applied; integer
        ``ts``, move-table trimming and RNA orientation are applied here.
    chunking_mode : {"auto", "mv", "hierarchical", "dtw", "inferred"}, default="auto"
        Auto uses moves when present, otherwise hierarchical for DNA or DTW
        for RNA. Explicit mv preserves the original decoder, including ts=0.
        Hierarchical/dtw ignore moves; inferred aliases hierarchical. Both
        inference methods use ts>0/ns as endpoint constraints; ts=0 estimates
        the start. Inference requires primary/unmapped DNA/RNA with A/C/G/T
        bases and a 9-mer current model. expected_ts_dim sets the signal row
        width (six if omitted); inference averaging is always six samples.

    Returns
    -------
    tuple
        Constructor arguments for :class:`Read_Lite`: move table, signal of
        shape ``(T, stride)``, integer base sequence, read name, sequence type,
        expected current, k-mer size, and optional inferred boundary map.
        The move table is None for inferred reads; the map is None in mv mode.

    Raises
    ------
    KeyError
        If a required signal tag is absent.
    ValueError
        If ``expected_ts_dim`` is set and does not match the move-table stride.
    """
    if preprocessing is not None or chemistry is not None or seq_type is None or kmer_table is None:
        profile = resolve_preprocessing(preprocessing=preprocessing, chemistry=chemistry,
                                        seq_type=seq_type, kmer_table=kmer_table)
        seq_type, kmer_table = profile.seq_type, profile.kmer_table
    is_mapped, is_forward=bam_read.is_mapped, bam_read.is_forward

    fq = oriented_sequence
    if fq is None:
        fq = bam_read.seq if is_forward else revcomp(bam_read.seq)

    base_seq=translate_table[np.frombuffer(fq.encode('ascii'), dtype=np.int8)]
    kmer_size=get_kmer_size_from_lookup_table(kmer_table)
    exp_signal=fast_seq_to_signal(base_seq, kmer_table, kmer_size)
    
    tags={x[0]:x[1] for x in bam_read.get_tags(with_value_type=True) if not (x[2]=='A' and x[0]=='ts')}

    if uses_inferred_chunks(bam_read, chunking_mode):
        raw = (np.frombuffer(blosc2.decompress2(tags['SG']), dtype=np.int16)
               if raw_signal is None else np.asarray(raw_signal))
        mat, boundary = decode_inferred_signal(
            bam_read, tags, raw, base_seq, exp_signal, kmer_size, seq_type, expected_ts_dim, chunking_mode,
        )
        return None, mat, base_seq, bam_read.qname, seq_type, exp_signal, kmer_size, boundary

    start=tags['ts']
    mv=tags['mv']

    stride=mv[0]
    validate_move_table_stride(stride, expected_ts_dim, bam_read.qname)
    move_table=np.array(mv[1:])
    
    if raw_signal is None:
        signal = np.frombuffer(blosc2.decompress2(tags['SG']), dtype=np.int16)
    else:
        signal = raw_signal

    max_idx=start+len(move_table)*stride
    mat=signal[start:max_idx].reshape(-1, stride)

    if seq_type=='rna':
        mat=np.flip(mat)
        new_move_table=np.full(len(move_table),0)
        new_move_table[np.where(move_table)[0][1:]-1]=1
        new_move_table[-1]=1
        new_move_table=new_move_table[::-1]
        move_table=new_move_table.copy()

    return move_table, mat, base_seq, bam_read.qname, seq_type, exp_signal, kmer_size, None

def get_read_info(bam_read, ref_seq_dict, seq_type=None, kmer_table=None, expected_ts_dim=None,
                  *, raw_signal=None, chunking_mode="auto", preprocessing=None, chemistry=None):
    """Decode an aligned signal BAM record into the full read contract.

    Sequence, qualities, alignment rows, move table, and signal are oriented
    5'-to-3' with the read. Reference coordinates remain zero-based genomic
    coordinates. RNA records additionally reverse signal time and rebuild the
    move table.

    Parameters
    ----------
    bam_read : pysam.AlignedSegment
        Mapped record with sequence, qualities, CIGAR, ``ts`` and ``mv``. A
        blosc2-compressed ``SG`` tag is required unless ``raw_signal`` is supplied.
    ref_seq_dict : dict
        Encoded references from :func:`get_ref_seq_dict`.
    seq_type : str, optional
        Sequence type; ``"rna"`` activates RNA signal orientation handling.
        Inferred from profile when omitted.
    kmer_table : numpy.ndarray, optional
        Dense expected-current lookup table; inferred from profile.
    preprocessing : PreprocessingSpec or mapping, optional
        Resolved profile supplying sequence type and current table.
    chemistry : str, optional
        Built-in chemistry when no checkpoint/profile is available.
    expected_ts_dim : int or None
        Expected move-table stride, or ``None`` to skip stride validation.
    raw_signal : numpy.ndarray or None
        Optional one-dimensional int16 ADC signal at this read's untrimmed
        origin, replacing SG decoding. POD5 child offsets must already have
        been applied. All subsequent trimming and orientation match SG mode.
    chunking_mode : {"auto", "mv", "hierarchical", "dtw", "inferred"}, default="auto"
        Same selection and endpoint policy as :func:`get_read_info_lite`.
        Inference applies only to primary DNA/RNA records with a 9-mer model.

    Returns
    -------
    tuple
        Sixteen constructor arguments for :class:`Read`, including signal with
        shape ``(T, stride)``, alignment with shape ``(R, 2)``, divergence
        counts, reference span, expected current and optional inferred map.
    """
    if preprocessing is not None or chemistry is not None or seq_type is None or kmer_table is None:
        profile = resolve_preprocessing(preprocessing=preprocessing, chemistry=chemistry,
                                        seq_type=seq_type, kmer_table=kmer_table)
        seq_type, kmer_table = profile.seq_type, profile.kmer_table
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=bam_read.is_mapped, bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length

    fq=bam_read.seq
    qual=bam_read.qual
    sequence_length=len(fq)
    reverse= not is_forward
    fq=revcomp(fq) if reverse else fq
    qual=qual[::-1] if reverse else qual
    
    base_qual=10**((33-np.array([ord(x) for x in qual]))/10)
    mean_qscore=-10*np.log10(np.mean(base_qual))
    base_qual=base_qual
            
    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, bam_read.cigarstring)])
    ref_start=bam_read.reference_start
    aligned_pairs=get_aligned_pairs(cigar_tuples, ref_start)

    if reverse:
        # Alignment rows follow read orientation, while reference coordinates
        # retain their genomic coordinate system.
        aligned_pairs=aligned_pairs[::-1]
        aligned_pairs[aligned_pairs[:,0]>=0,0]=read_length-aligned_pairs[aligned_pairs[:,0]>=0,0]-1

    base_seq=translate_table[np.frombuffer(fq.encode('ascii'), dtype=np.int8)]
    kmer_size=get_kmer_size_from_lookup_table(kmer_table)
    exp_signal=fast_seq_to_signal(base_seq, kmer_table, kmer_size)
    
    ref_seq=ref_seq_dict[ref_name][:,1][aligned_pairs[:, 1]] if reverse else ref_seq_dict[ref_name][:,0][aligned_pairs[:, 1]]

    exp_base_seq=np.full(len(aligned_pairs),4).astype(np.int8)
    exp_base_seq[aligned_pairs[:,0]>=0]=base_seq
    full_seq=np.stack([exp_base_seq, ref_seq]).T

    tags={x[0]:x[1] for x in bam_read.get_tags(with_value_type=True) if not (x[2]=='A' and x[0]=='ts')}
    
    boundary = None
    if uses_inferred_chunks(bam_read, chunking_mode):
        raw = (np.frombuffer(blosc2.decompress2(tags['SG']), dtype=np.int16)
               if raw_signal is None else np.asarray(raw_signal))
        mat, boundary = decode_inferred_signal(
            bam_read, tags, raw, base_seq, exp_signal, kmer_size, seq_type, expected_ts_dim, chunking_mode,
        )
        move_table = None
    else:
        start=tags['ts']
        mv=tags['mv']
    
        stride=mv[0]
        validate_move_table_stride(stride, expected_ts_dim, bam_read.qname)
        move_table=np.array(mv[1:])

        if raw_signal is None:
            signal = np.frombuffer(blosc2.decompress2(tags['SG']), dtype=np.int16)
        else:
            signal = raw_signal
    
        max_idx=start+len(move_table)*stride
        mat=signal[start:max_idx].reshape(-1, stride)

        if seq_type=='rna':
            mat=np.flip(mat)
            new_move_table=np.full(len(move_table),0)
            new_move_table[np.where(move_table)[0][1:]-1]=1
            new_move_table[-1]=1
            new_move_table=new_move_table[::-1]
            move_table=new_move_table.copy()


    indels=np.sum(cigar_tuples[:,0][(cigar_tuples[:,1]==1)|(cigar_tuples[:,1]==2)])
    mismatches=np.sum((full_seq[:,0]!=full_seq[:,1])[np.all(aligned_pairs>=0,axis=1)])
    alen=np.sum(cigar_tuples[:,0][cigar_tuples[:,1]<=2])
    div=(mismatches, indels, alen)
    
    return move_table, mat, base_qual, full_seq, aligned_pairs, bam_read.qname, ref_name, seq_type, div, bam_read.flag, bam_read.reference_start, bam_read.reference_end, bam_read.cigarstring, exp_signal, kmer_size, boundary


def get_ref_seq_dict(ref: str, chrom_list: list = [], bam: str = None):
    """Load encoded forward/complement reference arrays by contig.

    Parameters
    ----------
    ref : str
        Indexed reference FASTA path.
    chrom_list : list, optional
        Contigs to load. When empty, contigs are derived from ``bam`` if
        supplied and otherwise from the FASTA index.
    bam : str, optional
        BAM path used only to obtain the reference-name list.

    Returns
    -------
    dict
        Mapping from contig name to the ``(contig_length + 2, 2)`` encoded
        array returned by :func:`get_ref_to_num`.
    """
    chroms = chrom_list
    if not chroms:
        if bam:
            chroms = pysam.Samfile(bam).references
        else:
            chroms = pysam.FastaFile(ref).references

    _ = get_ref_to_num('ACGT')

    ref_seq_dict = {}
    for chrom in chroms:
        chrom_name, seq_array = get_ref_info((ref, chrom))
        ref_seq_dict[chrom_name] = seq_array

    return ref_seq_dict

def motif_check(motif):
    """Validate one IUPAC motif and expand it to a regular expression.

    ``motif`` uses the form ``"SEQUENCE INDEX [INDEX ...]"``. Indices are
    zero-based offsets into the motif and must all select the same literal
    ``A``, ``C``, ``G``, or ``T`` base. IUPAC ambiguity codes are allowed at
    other positions.

    Returns
    -------
    tuple
        ``(literal_motif, regex_motif, indices, valid)``. Invalid user input is
        reported to standard output and returns ``valid=False`` rather than
        raising a validation exception.
    """
    nt_dict={'R': 'GA', 'Y': 'CT', 'K': 'GT', 'M': 'AC', 'S': 'GC', 'W': 'AT',
             'B': 'GTC', 'D': 'GAT', 'H': 'ACT', 'V': 'GCA', 'N': 'AGCT'}
    
    valid_alphabet=set(nt_dict.keys()).union({'A', 'C', 'G', 'T'})
    
    exp_motif_seq, final_motif_ind, valid = None, None, False
    
    motif=motif.split()
    if len(motif)<2:
        print('--motif not specified correctly. You need to specify a motif and at least one index',flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    elif len(set(motif[0])-valid_alphabet)>0:
        print('--motif not specified correctly. Motif should only consist of the following extended nucleotide letters: {}'.format(','.join(valid_alphabet)),flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    elif all([a.isnumeric() for a in motif[1:]])==False:
        print('--motif not specified correctly. Motif indices should be integers separated by whitespace and shoud come after the motif sequence.',flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    else:
        motif_seq=motif[0]
        motif_ind=[int(x) for x in motif[1:]]
        
        if len(set(motif_seq[x] for x in motif_ind))!=1 or len(set(motif_seq[x] for x in motif_ind)-set('ACGT'))>0:
            print('Base of interest should be same for all indices and must be one of A, C, G, or T.', flush=True)
            return motif_seq, exp_motif_seq, final_motif_ind, valid
        
        else:
            exp_motif_seq=motif_seq
            for nt in nt_dict:
                if nt in exp_motif_seq:
                    exp_motif_seq=exp_motif_seq.replace(nt, '[{}]'.format(nt_dict[nt]))
            return motif_seq, exp_motif_seq, motif_ind, True
        



def parse_motif_specs(motif, motif_label="0", num_classes=None):
    """Parse semicolon-delimited motif and class-label specifications.

    A shorter label list reuses its final label for remaining motifs. Invalid
    motifs are skipped after :func:`motif_check` reports them. Each returned
    mapping includes public motif fields plus cached arrays used by read-based
    matching.

    Parameters
    ----------
    motif : str or None
        Semicolon-delimited motif specifications.
    motif_label : str or int, default="0"
        Semicolon-delimited integer class labels.
    num_classes : int or None
        When supplied, require every label to be in ``[0, num_classes)``.
    """
    parsed_motifs = []

    if motif and len(motif) > 0:
        motif_list = motif.split(';')
        label_list = str(motif_label).split(';')

        for i, motif_entry in enumerate(motif_list):
            motif_seq, exp_motif_seq, motif_ind, valid = motif_check(motif_entry.strip())
            if not valid:
                continue

            label_val = int(label_list[i]) if i < len(label_list) else int(label_list[-1])
            if num_classes is not None and (label_val < 0 or label_val >= num_classes):
                raise ValueError(f'motif_label {label_val} is outside num_classes={num_classes}')

            parsed_motifs.append({
                'seq': motif_seq,
                'exp_seq': exp_motif_seq,
                'ind': motif_ind,
                'label': label_val,
                '_read_indices': np.asarray(motif_ind, dtype=np.int32),
                '_read_is_literal': set(motif_seq).issubset(ALPHABET),
            })

    return parsed_motifs


def get_read_motif_targets(read_sequence, parsed_motifs, num_classes):
    """Return sorted read indices and one-hot class targets for motif hits.

    Parameters
    ----------
    read_sequence : str
        Basecall sequence in the desired read orientation.
    parsed_motifs : sequence of dict
        Specifications returned by :func:`parse_motif_specs`.
    num_classes : int
        Width of the one-hot target matrix.

    Returns
    -------
    tuple of numpy.ndarray
        Sorted ``int32`` read positions with shape ``(N,)`` and ``float32``
        targets with shape ``(N, num_classes)``.

    Literal A/C/G/T motifs use str.find so matching runs in optimized C
    without constructing a regex Match object per hit. IUPAC motifs retain the
    same look-ahead regex semantics used by reference-based motif discovery.
    Later motif entries take priority at overlapping target indices, matching
    get_labelled_info's default behavior.
    """
    if num_classes is None or num_classes < 1:
        raise ValueError(f'num_classes must be a positive integer, got {num_classes!r}')

    if not read_sequence or not parsed_motifs:
        return (
            np.array([], dtype=np.int32),
            np.empty((0, num_classes), dtype=np.float32),
        )

    sequence = read_sequence.upper()
    labels_by_position = np.full(len(sequence), -1, dtype=np.int32)

    for motif_spec in parsed_motifs:
        motif_seq = motif_spec['seq']
        motif_indices = motif_spec['_read_indices']

        if motif_spec['_read_is_literal']:
            starts = []
            start = sequence.find(motif_seq)
            while start >= 0:
                starts.append(start)
                start = sequence.find(motif_seq, start + 1)
        else:
            pattern = motif_spec.get('_read_pattern')
            if pattern is None:
                pattern = re.compile(r'(?={})'.format(motif_spec['exp_seq']))
                motif_spec['_read_pattern'] = pattern
            starts = [match.start(0) for match in pattern.finditer(sequence)]

        if starts:
            anchors = np.asarray(starts, dtype=np.int32)
            target_positions = (anchors[:, None] + motif_indices).ravel()
            labels_by_position[target_positions] = motif_spec['label']

    target_positions = np.flatnonzero(labels_by_position >= 0).astype(np.int32, copy=False)
    target_labels = np.zeros((len(target_positions), num_classes), dtype=np.float32)
    if len(target_positions) > 0:
        target_labels[
            np.arange(len(target_positions)),
            labels_by_position[target_positions],
        ] = 1.0

    return target_positions, target_labels


def get_labelled_info(args):
    """Find reference-motif targets on both strands of one contig.

    Parameters
    ----------
    args : tuple
        ``(params, contig)`` pair suitable for a mapping worker. ``params``
        contains ``ref`` and parsed motif specifications plus overlap policy.

    Returns
    -------
    tuple
        ``(contig, forward_targets, reverse_targets)`` where each target
        mapping is ``zero_based_position -> class_index``. Reverse motif
        offsets are mirrored within the motif. When positions overlap, the
        first motif wins unless ``prioritize_latest`` is true.
    """
    params, chrom = args
    ref_fasta = pysam.FastaFile(params['ref'])
    seq = ref_fasta.fetch(chrom).upper()

    fwd_pos_dict = {}
    rev_pos_dict = {}

    if not params.get('parsed_motifs'):
        return chrom, fwd_pos_dict, rev_pos_dict

    # Hoist loop-invariant options and compiled motif metadata because this
    # scan runs once over every selected reference contig.
    warn_overlap = params.get('warn_overlap', False)
    prioritize_latest = params.get('prioritize_latest', False)

    for m_dict in params['parsed_motifs']:
        motif_seq = m_dict['seq']
        exp_motif_seq = m_dict['exp_seq']
        motif_ind = np.array(m_dict['ind'], dtype=np.int32) 
        label = m_dict['label']

        fwd_pattern = re.compile(r'(?={})'.format(exp_motif_seq))
        rev_pattern = re.compile(r'(?={})'.format(revcomp(exp_motif_seq)))

        fwd_matches = [m.start(0) for m in fwd_pattern.finditer(seq)]
        if fwd_matches:
            anchors = np.array(fwd_matches, dtype=np.int32)
            
            # Broadcasting generates all selected motif offsets at once;
            # np.unique both deduplicates multi-offset hits and sorts them.
            pos_array = np.unique((anchors[:, None] + motif_ind).ravel())
            pos_list = pos_array.tolist()

            if warn_overlap:
                for pos in pos_list:
                    if pos in fwd_pos_dict:
                        print(f"Warning: Overlap at {chrom} fwd:{pos}. Prioritizing latest: {prioritize_latest}", flush=True)
                        if prioritize_latest:
                            fwd_pos_dict[pos] = label
                    else:
                        fwd_pos_dict[pos] = label
            else:
                if prioritize_latest:
                    fwd_pos_dict.update(dict.fromkeys(pos_list, label))
                else:
                    for pos in pos_list:
                        if pos not in fwd_pos_dict:
                            fwd_pos_dict[pos] = label

        rev_matches = [m.start(0) for m in rev_pattern.finditer(seq)]
        if rev_matches:
            anchors = np.array(rev_matches, dtype=np.int32)
            
            rev_offsets = len(motif_seq) - 1 - motif_ind
            pos_array = np.unique((anchors[:, None] + rev_offsets).ravel())
            pos_list = pos_array.tolist()

            if warn_overlap:
                for pos in pos_list:
                    if pos in rev_pos_dict:
                        print(f"Warning: Overlap at {chrom} rev:{pos}. Prioritizing latest: {prioritize_latest}", flush=True)
                        if prioritize_latest:
                            rev_pos_dict[pos] = label
                    else:
                        rev_pos_dict[pos] = label
            else:
                if prioritize_latest:
                    rev_pos_dict.update(dict.fromkeys(pos_list, label))
                else:
                    for pos in pos_list:
                        if pos not in rev_pos_dict:
                            rev_pos_dict[pos] = label

    return chrom, fwd_pos_dict, rev_pos_dict

def get_labelled_arrays(ref: str, seq_type: str, pos_list: str, motif: str, motif_label: str = "0", chrom_list: list = [], bam: str = None, prioritize_latest: bool = True, warn_overlap: bool = False, num_classes: int = None):
    """Build sorted strand-specific reference targets for selected contigs.

    Explicit position files take precedence over motif discovery when both are
    supplied. Position-file rows retain their floating-point target vectors;
    motif hits are converted to one-hot targets. Contigs absent from an
    explicit position file are omitted.

    Parameters
    ----------
    ref : str
        Indexed reference FASTA path.
    seq_type : str
        Sequence type retained in preprocessing parameters for caller
        compatibility.
    pos_list : str or None
        Headerless target table accepted by :func:`get_pos`.
    motif : str or None
        Semicolon-delimited reference motif specifications.
    motif_label : str, default="0"
        Semicolon-delimited class indices corresponding to ``motif``.
    chrom_list : list, optional
        Contigs to process. Derived from ``bam`` or ``ref`` when empty.
    bam : str or None
        Optional BAM used to derive contig names.
    prioritize_latest : bool, default=True
        Let later motif specifications overwrite earlier labels at overlaps.
    warn_overlap : bool, default=False
        Print every motif-label overlap while applying the priority policy.
    num_classes : int or None
        Required target width. When omitted, motif widths are inferred per
        populated strand; empty targets have width zero.

    Returns
    -------
    dict
        ``contig -> strand -> (positions, targets)``. Positions are sorted
        ``int32`` zero-based coordinates and targets are ``float32`` arrays of
        shape ``(N, num_classes)`` when the class count is known.
    """
    parsed_motifs = parse_motif_specs(
        motif=motif, motif_label=motif_label, num_classes=num_classes,
    )

    if not chrom_list:
        if bam:
            chrom_list=pysam.Samfile(bam).references 
        else:
            chrom_list=pysam.FastaFile(ref).references
    
    params = dict(bam=bam, 
                  seq_type=seq_type,
                  pos_list=pos_list, 
                  ref=ref,
                  parsed_motifs=parsed_motifs,
                  chrom=chrom_list,
                  prioritize_latest=prioritize_latest,
                  warn_overlap=warn_overlap)
    
    raw_pos_dict = {}
    if params['pos_list']:
        raw_pos_dict = get_pos(params['pos_list'])
        params['chrom'] = [x for x in params['chrom'] if x in raw_pos_dict.keys()]

    labelled_arrays = {}

    for c in params['chrom']:
        chrom, fwd_pos_dict_motif, rev_pos_dict_motif = get_labelled_info((params, c))
        
        labelled_arrays[chrom] = {}
        
        for strand, pos_dict_motif in enumerate([fwd_pos_dict_motif, rev_pos_dict_motif]):
            
            # Explicit vectors and motif-derived class indices share a single
            # sorted ``(positions, targets)`` representation downstream.
            if params['pos_list']:
                if chrom in raw_pos_dict and raw_pos_dict[chrom][strand]:
                    data = raw_pos_dict[chrom][strand]
                    sorted_pos = np.array(sorted(data.keys()), dtype=np.int32)
                    sorted_labels = np.array([data[k] for k in sorted_pos], dtype=np.float32)
                    if sorted_labels.ndim != 2:
                        raise ValueError(f'Position file labels for {chrom} strand {strand} must be N x K')
                    if num_classes is not None and sorted_labels.shape[1] != num_classes:
                        raise ValueError(
                            f'Position file labels for {chrom} strand {strand} have '
                            f'{sorted_labels.shape[1]} class columns, expected {num_classes}'
                        )
                else:
                    sorted_pos = np.array([], dtype=np.int32)
                    sorted_labels = np.empty((0, num_classes or 0), dtype=np.float32)

            elif params['parsed_motifs']:
                if pos_dict_motif:
                    sorted_pos = np.array(sorted(pos_dict_motif.keys()), dtype=np.int32)
                    label_indices = np.array([pos_dict_motif[k] for k in sorted_pos], dtype=np.int64)
                    if num_classes is None:
                        inferred_classes = int(label_indices.max()) + 1 if label_indices.size else 0
                    else:
                        inferred_classes = num_classes
                    sorted_labels = np.zeros((len(sorted_pos), inferred_classes), dtype=np.float32)
                    if len(sorted_pos) > 0:
                        sorted_labels[np.arange(len(sorted_pos)), label_indices] = 1.0
                else:
                    sorted_pos = np.array([], dtype=np.int32)
                    sorted_labels = np.empty((0, num_classes or 0), dtype=np.float32)
                    
            else:
                sorted_pos = np.array([], dtype=np.int32)
                sorted_labels = np.empty((0, num_classes or 0), dtype=np.float32)
            
            labelled_arrays[chrom][strand] = (sorted_pos, sorted_labels)

    return labelled_arrays
def pre_process(ref: str, seq_type: str, pos_list:str, motif: str, motif_label:int=0, chrom_list: list=[],bam: str=None):
    """Load encoded references and strand-specific labels together.

    This convenience wrapper keeps the reference and label contig selection in
    sync. It returns ``(ref_seq_dict, labelled_arrays)`` from
    :func:`get_ref_seq_dict` and :func:`get_labelled_arrays`, respectively.
    The class count is intentionally left unspecified for legacy callers.
    """
    ref_seq_dict = get_ref_seq_dict(ref=ref, chrom_list=chrom_list, bam=bam)
    labelled_arrays = get_labelled_arrays(
        ref=ref,
        seq_type=seq_type,
        pos_list=pos_list,
        motif=motif,
        motif_label=motif_label,
        chrom_list=chrom_list,
        bam=bam,
        num_classes=None,
    )
    return ref_seq_dict, labelled_arrays
