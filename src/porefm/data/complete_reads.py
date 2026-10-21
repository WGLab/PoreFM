"""Complete-read datasets, overlap stitching, and MM/ML BAM output helpers."""

from porefm.preprocessing import resolve_preprocessing
from porefm.data.chunk_boundaries import validate_chunking_mode
from porefm.data.inferred_chunks import (has_inferred_boundaries, inferred_chunks,
    inferred_labelled_chunks, slice_base_values, chunk_bounds)
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
from porefm.data.datasets.aligned import AlignedBamIterableDataset, AlignedBamSample
from porefm.data.collation import aligned_collate_fn, collate_modal_aligned_batch
from porefm.data.reader import (
    LabelledBamSample,
    align_chunk_fields_to_sequences,
    chunk_padded_components,
    format_iterable_bam_log,
    get_mapped_chroms,
    get_percentiles,
    get_polished_read_for_sample,
    get_read_chunks_labelled,
    normalize_read_signal,
    normalize_read_signal_with_stats,
    pad_read_components,
)
from porefm.data.bam import (
    Read,
    Read_Lite,
    get_labelled_arrays,
    get_read_info,
    get_read_info_lite,
    get_read_motif_targets,
    get_ref_seq_dict,
    num_to_base_map,
    parse_motif_specs,
    passes_min_qscore,
    pre_process,
    revcomp,
)
from porefm.common.utils import DegenerateQuantileNormalizationError
import array
from contextlib import ExitStack, closing
import datetime
import random
import re
from typing import List

import numpy as np
import pysam
from torch.nn.utils.rnn import pad_sequence
from porefm.data.bam_headers import add_pg_to_header
from porefm.data.excluded_regions import get_excluded_regions
from porefm.data.pod5 import iter_pod5_bam_records, validate_pod5_samples

def base_error_prob_from_bam_read(bam_read):
    """Return read-oriented Phred error probabilities from a BAM record.

    Missing qualities produce an all-ones vector. Reverse-alignment qualities
    are reversed so the output follows the oriented sequence used by the
    model.
    """
    if bam_read.qual is None:
        return np.ones(len(bam_read.seq), dtype=np.float32)
    qual = bam_read.qual if bam_read.is_forward else bam_read.qual[::-1]
    qual_ord = np.fromiter((ord(x) for x in qual), dtype=np.float32, count=len(qual))
    return np.power(10.0, (33.0 - qual_ord) / 10.0).astype(np.float32)

def read_scale_metadata(norm_stats, base_error_prob):
    """Combine signal-normalization statistics with normalized median quality.

    Parameters
    ----------
    norm_stats : numpy.ndarray
        Five values returned by :func:`normalize_read_signal_with_stats`.
    base_error_prob : numpy.ndarray
        Per-base error probabilities.

    Returns
    -------
    torch.Tensor
        Six ``float32`` features: the five normalization statistics followed
        by median Phred quality divided by 40.
    """
    base_error_prob = np.clip(base_error_prob.astype(np.float32), 1e-12, 1.0)
    phred = -10.0 * np.log10(base_error_prob)
    median_q = float(np.median(phred) / 40.0)
    return torch.tensor(
        np.concatenate([norm_stats.astype(np.float32), np.array([median_q], dtype=np.float32)]),
        dtype=torch.float32,
    )

def get_base_quality_chunks(read, size, step):
    """Align per-base error probabilities to each signal window's base tokens.

    Returns a list of variable-length float tensors, one per chunk. Forced
    chunk-start move events follow the same convention as sequence chunking;
    padded signal time steps use error probability one.
    """
    if has_inferred_boundaries(read):
        return [x.float() for x in slice_base_values(read, read.base_qual, size, step)]
    _, _, move_table, padding_needed = pad_read_components(read, size, step)
    base_quality = torch.from_numpy(read.base_qual[read.indexes]).float()
    base_quality = F.pad(base_quality, (0, padding_needed), mode='constant', value=1.0)
    base_quality_chunks = base_quality.unfold(dimension=0, size=size, step=step)
    init_move_table_chunks = move_table.unfold(dimension=0, size=size, step=step)
    move_table_chunks = init_move_table_chunks.clone()
    move_table_chunks[:, 0] += 1
    move_table_chunks = move_table_chunks.bool()
    return [base_quality_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]


def get_polished_base_quality_chunks(read, polished_read, base_seq_chunks, read_pos_chunks):
    """Project original error probabilities onto polished sequence chunks.

    Corrected bases with no source read base retain error probability one.
    ``read_pos_chunks`` contains polished-sequence indices aligned to the
    variable-length ``base_seq_chunks``.
    """
    polished_quality = np.ones(len(polished_read.sequence), dtype=np.float32)
    for read_idx, polished_idx in enumerate(polished_read.p_from_read):
        if polished_idx >= 0:
            polished_quality[int(polished_idx)] = read.base_qual[read_idx]

    quality = torch.from_numpy(polished_quality)
    return [
        quality[read_pos_chunks[idx, :len(seq_chunk)]]
        for idx, seq_chunk in enumerate(base_seq_chunks)
    ]

def get_read_pos_chunks_aligned(read, polished_read, size, step):
    """Chunk an aligned read with both reference and read-space positions.

    Returns signal windows in ``(n_chunks, channels, size)`` order, variable-
    length sequence chunks, raw move-table windows, zero-based reference
    coordinate chunks, and zero-based read/polished coordinate chunks. When a
    ``polished_read`` is supplied, sequence and both coordinate fields are
    sliced through its maps.
    """
    if has_inferred_boundaries(read):
        signal, sequences, references, positions = inferred_chunks(read, size, step, polished_read)
        return signal.transpose(1, 2), sequences, None, references, positions
    try:
        signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
        signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks = chunk_padded_components(
            signal, base_seq, move_table, size, step
        )
    except RuntimeError:
        print(read.read_name, size, step, signal.shape)

    base_seq_idx=torch.arange(read.length)[read.indexes]
    base_seq_idx = F.pad(
        base_seq_idx, 
        (0,  padding_needed), 
        mode='constant', 
        value=read.length-1
    )

    base_seq_idx_chunks=base_seq_idx.unfold(dimension=0, size=size, step=step)
        
    if polished_read!=None:
        
        base_starts=base_seq_idx_chunks[:,0]
        base_ends=base_seq_idx_chunks[:,-1]

        final_base_seq_chunks=[]
        aligned_pos_chunks=[]
        
        read_pos_chunks=[]
        
        for m,n in zip(base_starts, base_ends):
            p_start, p_end=polished_read.get_coordinates(int(m), int(n) + 1)
            final_base_seq_chunks.append(torch.from_numpy(polished_read.sequence[p_start:p_end]+1))
            aligned_pos_chunks.append(torch.from_numpy(polished_read.p_to_ref[p_start:p_end].copy()))
            read_pos_chunks.append(torch.arange(p_start, p_end, dtype=torch.long))
        
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
        
        read_pos_chunks = [base_seq_idx_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
        
    return signal_chunks.transpose(1, 2), final_base_seq_chunks, init_move_table_chunks, aligned_pos_chunks, read_pos_chunks

def _polished_sequence_to_string(polished_read):
    """Decode integer tokens from a polished read to an A/C/G/T/N string."""
    return "".join(num_to_base_map.get(int(base), "N") for base in polished_read.sequence)

def _append_cigar_op(ops, op, length):
    """Append a positive-length CIGAR operation, coalescing adjacent types."""
    if length <= 0:
        return
    if ops and ops[-1][1] == op:
        ops[-1] = (ops[-1][0] + length, op)
    else:
        ops.append((length, op))

def _reference_gap_intervals(cigartuples, ref_start):
    """Extract deletion and reference-skip intervals from an original CIGAR."""
    intervals = []
    ref_cursor = int(ref_start)

    for cigar_op, length in cigartuples or []:
        if cigar_op in (0, 7, 8):
            ref_cursor += length
        elif cigar_op == 2:
            intervals.append((ref_cursor, ref_cursor + length, "D"))
            ref_cursor += length
        elif cigar_op == 3:
            intervals.append((ref_cursor, ref_cursor + length, "N"))
            ref_cursor += length

    return intervals

def _append_reference_gap(ops, gap_start, gap_end, gap_intervals):
    """Represent a reference gap while preserving original ``D`` versus ``N``."""
    cursor = int(gap_start)
    gap_end = int(gap_end)

    for interval_start, interval_end, interval_op in gap_intervals:
        if interval_end <= cursor:
            continue
        if interval_start >= gap_end:
            break

        if cursor < interval_start:
            _append_cigar_op(ops, "D", min(interval_start, gap_end) - cursor)
            cursor = min(interval_start, gap_end)

        op_start = max(cursor, interval_start)
        op_end = min(gap_end, interval_end)
        if op_end > op_start:
            _append_cigar_op(ops, interval_op, op_end - op_start)
            cursor = op_end

    if cursor < gap_end:
        _append_cigar_op(ops, "D", gap_end - cursor)

def _polished_cigar_and_start(polished_read, is_reverse, fallback_ref_start, original_cigartuples):
    """Reconstruct a CIGAR and start position from polished reference maps.

    Unmapped polished bases outside the mapped span become soft clips; those
    inside become insertions. Missing reference coordinates are emitted as
    deletions or original reference skips, and original terminal hard clips
    are retained. A fully unmapped polished sequence uses ``fallback_ref_start``.
    """
    ref_coords = polished_read.p_to_ref[::-1] if is_reverse else polished_read.p_to_ref
    mapped = np.flatnonzero(ref_coords >= 0)
    leading_hard_clip = (
        original_cigartuples[0][1]
        if original_cigartuples and original_cigartuples[0][0] == 5
        else 0
    )
    trailing_hard_clip = (
        original_cigartuples[-1][1]
        if original_cigartuples and original_cigartuples[-1][0] == 5
        else 0
    )

    if len(mapped) == 0:
        cigar_ops = []
        _append_cigar_op(cigar_ops, "H", leading_hard_clip)
        _append_cigar_op(cigar_ops, "S", len(ref_coords))
        _append_cigar_op(cigar_ops, "H", trailing_hard_clip)
        return "".join(f"{length}{op}" for length, op in cigar_ops), fallback_ref_start

    first_mapped = int(mapped[0])
    last_mapped = int(mapped[-1])
    ref_start = int(ref_coords[first_mapped])
    ref_cursor = ref_start
    cigar_ops = []
    gap_intervals = _reference_gap_intervals(original_cigartuples, fallback_ref_start)

    _append_cigar_op(cigar_ops, "H", leading_hard_clip)

    for idx, ref_coord in enumerate(ref_coords):
        ref_coord = int(ref_coord)

        if ref_coord < 0:
            if idx < first_mapped or idx > last_mapped:
                _append_cigar_op(cigar_ops, "S", 1)
            else:
                _append_cigar_op(cigar_ops, "I", 1)
            continue

        if ref_coord > ref_cursor:
            _append_reference_gap(cigar_ops, ref_cursor, ref_coord, gap_intervals)

        _append_cigar_op(cigar_ops, "M", 1)
        ref_cursor = ref_coord + 1

    _append_cigar_op(cigar_ops, "H", trailing_hard_clip)
    return "".join(f"{length}{op}" for length, op in cigar_ops), ref_start

def make_polished_read_dict(bam_read, polished_read, tags_to_drop):
    """Create a pysam read dictionary consistent with a polished sequence.

    The stored sequence is restored to BAM alignment orientation, qualities
    are cleared, and CIGAR/reference start are rebuilt from polished maps.
    Caller-requested tags plus ``MM``, ``ML``, ``MN``, ``MD``, and ``NM`` are
    removed because they no longer describe the rewritten record. The input
    ``bam_read`` is not mutated.
    """
    read_dict = bam_read.to_dict()
    polished_seq = _polished_sequence_to_string(polished_read)
    is_reverse = bool(int(read_dict["flag"]) & 16)
    stored_seq = revcomp(polished_seq) if is_reverse else polished_seq
    cigar, ref_start = _polished_cigar_and_start(
        polished_read,
        is_reverse,
        bam_read.reference_start,
        bam_read.cigartuples,
    )

    tags_to_drop = set(tags_to_drop) | {"MM", "ML", "MN", "MD", "NM"}
    read_dict["seq"] = stored_seq
    read_dict["qual"] = "*"
    read_dict["cigar"] = cigar
    read_dict["ref_pos"] = str(ref_start + 1)
    read_dict["tags"] = [
        tag for tag in read_dict["tags"]
        if tag.split(":", 1)[0] not in tags_to_drop
    ]
    return read_dict

def collate_complete_batch(
    seqs_batch: List[torch.Tensor],
    signals_batch: List[torch.Tensor],
    percentiles_batch: List[torch.Tensor],
):
    """Pad complete-read modal tensors and build attention masks.

    This has the same six-value return contract as
    :func:`porefm.data.reader.collate_modal_batch`: padded sequences, stacked
    signal/percentiles, base and signal masks, and sequence padding count.
    Signal rows filled with ``-10`` are padding; sequence length is rounded to
    a multiple of 128.
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
    
    if pad_needed > 0:
        seqs_batch_padded = F.pad(seqs_batch_padded, (0, pad_needed), value=0)
        
    pad_base_seq = seqs_batch_padded != 0

    return (
        seqs_batch_padded.to(torch.long),
        signals_batch.to(torch.float32),
        percentiles_batch.to(torch.float32),
        pad_base_seq,
        pad_ts, pad_needed
    )

class CompleteAlignedReadIterableDataset(IterableDataset):
    """Yield batches containing every chunk and reconstructable BAM metadata.

    Unlike :class:`~porefm.data.datasets.aligned.AlignedBamIterableDataset`,
    this dataset batches all windows from eligible aligned reads and retains
    read-space positions plus a dictionary representation of each source BAM
    record. This supports stitching per-chunk predictions back into complete
    reads and writing updated BAM records.

    ``bam_list`` contains ``AlignedBamSample`` objects. ``size`` and ``step``
    are signal-time units; length filters apply to query bases and aligned
    reference span. Files are sharded whole across ranks/workers. ``tags_to_drop``
    removes invalidated tags from returned record dictionaries, and
    ``expected_ts_dim`` optionally validates move-table stride.

    Supply ``preprocessing=model.preprocessing`` or a standalone ``chemistry``
    (``dna_r10.4.1_e8.2_400bps``, ``rna004``, ``rna002``) to infer ``seq_type``,
    ``pore_type``, and ``kmer_table``. Explicit values must match that profile.

    Yields
    ------
    tuple
        One pre-collated batch containing padded sequences, signal,
        percentiles, masks, read names, reference and read positions, minimum
        alignment-start frontier, strands, and a read-name-to-pysam-dictionary
        mapping. The frontier is derived from source BAM alignment starts, not
        retained token coordinates, so it remains monotonic for coordinate-
        sorted inputs.
    """

    def __init__(
        self,
        bam_list,
        size,
        step,
        seq_type=None,
        kmer_table=None,
        reads_per_chunk=1000,
        min_read_length=100,
        max_read_length=5e5,
        max_alignment_length=1e6,
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        tags_to_drop=['ML', 'MM'],
        expected_ts_dim=None,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
        self.bam_list = bam_list
        self.size = size
        self.step = step
        self.preprocessing = resolve_preprocessing(
            preprocessing=preprocessing, chemistry=chemistry, seq_type=seq_type,
            pore_type=pore_type, kmer_table=kmer_table,
        )
        seq_type, pore_type, kmer_table = (
            self.preprocessing.seq_type, self.preprocessing.pore_type, self.preprocessing.kmer_table
        )
        self.seq_type = seq_type
        self.reads_per_chunk = reads_per_chunk
        self.min_read_length = min_read_length
        self.max_read_length = max_read_length
        self.max_alignment_length = max_alignment_length
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.tags_to_drop=tags_to_drop
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        
    def _yield_buffer(self, data, polish):
        """Pad a complete-read buffer and yield its single batch tuple."""
        seqs_batch = data['seq']
        signals_batch = data['signal']
        alignment_frontier = torch.tensor(
            min(data['alignment_start']), dtype=torch.long
        )
        
        read_name_batch = data['read_name']
        strand_batch = torch.cat(data['strand'])
        positions_batch = data['position']
        
        read_pos = data['read_position']

        signals_batch = torch.cat(signals_batch, dim=0)
        percentiles_batch = get_percentiles(signals_batch)
        
        pad_ts = torch.any(signals_batch != -10, dim=-1)
        signals_batch = torch.where(pad_ts.unsqueeze(-1), signals_batch, 0)
        percentiles_batch = torch.where(pad_ts.unsqueeze(-1), percentiles_batch, 0)

        seqs_batch_padded = pad_sequence(seqs_batch, batch_first=True, padding_value=0)

        current_max_len = seqs_batch_padded.shape[1]
        target_len = ((current_max_len + 127) // 128) * 128
        pad_needed = target_len - current_max_len

        positions_batch_padded =  pad_sequence(positions_batch, batch_first=True, padding_value=-1)
        
        read_pos_chunks_padded =  pad_sequence(read_pos, batch_first=True, padding_value=-1)

        if pad_needed > 0:
            seqs_batch_padded = F.pad(seqs_batch_padded, (0, pad_needed), value=0)
            positions_batch_padded = F.pad(positions_batch_padded, (0, pad_needed), value=-1)
            read_pos_chunks_padded = F.pad(read_pos_chunks_padded, (0, pad_needed), value=-1)
            
        pad_base_seq = seqs_batch_padded != 0
    
        yield (
                seqs_batch_padded.to(torch.long),
                signals_batch.to(torch.float32),
                percentiles_batch.to(torch.float32),
                pad_base_seq,
                pad_ts, read_name_batch, positions_batch_padded,
                read_pos_chunks_padded, alignment_frontier, strand_batch,
                data['pysam_data']
            )

    def __iter__(self):
        """Stream this worker's aligned reads and all of their model windows."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
                
        global_rank = self.rank 
        global_world_size = self.world_size

        global_worker_id = global_rank * num_workers + worker_id
        global_worker_count = max(global_world_size * num_workers, 1)
        
        seed = self.base_seed + 1000003 * (global_worker_id + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))  
        torch.manual_seed(seed)
        
        my_files=[]
        for bam_idx in range(global_worker_id, len(self.bam_list), global_worker_count):
            my_files.append(self.bam_list[bam_idx])
        
        if len(my_files) == 0:
            return
        
        preprocess_cache={}
        polish_variant_mask_cache = {}
        
        data = {
            'signal': [],
            'seq': [],
            'read_name': [],
            'position': [],
            'read_position': [],
            'strand': [],
            'pysam_data': {},
            'alignment_start': [],
        }
            
        for bam_sample in my_files:
            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info=bam_sample.flag
            polish=bam_sample.polish
            
            chroms=(x[0] for x in chrom_info)
            
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path)}',
                    flush=True,
                )
                
                preprocess_args=(bam_sample.ref,
                    self.seq_type,
                    bam_sample.pos_file,
                    bam_sample.motif,
                    bam_sample.motif_label,
                    chroms)
                
                if preprocess_args in preprocess_cache:
                    ref_seq_dict = preprocess_cache[preprocess_args]
                else:
                    ref_seq_dict, _ = pre_process(*preprocess_args)
                    preprocess_cache[preprocess_args] = ref_seq_dict
                
                for contig_info in chrom_info:
                    contig, start, end = contig_info
                    for bam_read in bam.fetch(contig, start=start, end=end, until_eof=True):
                        if (
                            bam_read.flag & (0x900|flag_info) == 0
                            and passes_min_qscore(bam_read, self.min_qscore)
                            and len(bam_read.seq) >= self.min_read_length
                            and len(bam_read.seq) <= self.max_read_length
                            and bam_read.reference_end-bam_read.reference_start<=self.max_alignment_length
                        ):
                            read_info = get_read_info(
                                bam_read, ref_seq_dict, self.seq_type, self.kmer_table,
                                expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                            )
                            read = Read(*read_info)

                            try:
                                read.signal = normalize_read_signal(read, self.pore_type)
                            except DegenerateQuantileNormalizationError:
                                continue
                            
                            read_name_str = read.read_name
                            strand_val = 0 if read.is_forward else 1
                            
                            polished_read = (
                                get_polished_read_for_sample(
                                    bam_read, bam_sample, polish_variant_mask_cache,
                                )
                                if polish
                                else None
                            )
                            
                            signal_chunks, base_seq_chunks, _, pos_chunks, read_pos_chunks = get_read_pos_chunks_aligned(read, polished_read, self.size, self.step)
                            
                            
                            if polished_read is None:
                                read_dict=bam_read.to_dict()
                                read_dict['tags']=[x for x in read_dict['tags'] if x[:2] not in self.tags_to_drop]
                            else:
                                read_dict = make_polished_read_dict(
                                    bam_read,
                                    polished_read,
                                    self.tags_to_drop,
                                )

                            data['signal'].append(signal_chunks)

                            data['seq'].extend(base_seq_chunks)
                            data['read_name'].extend([read_name_str] * len(base_seq_chunks))
                            data['position'].extend(pos_chunks)
                            data['read_position'].extend(read_pos_chunks)
                            data['strand'].append(torch.full((len(base_seq_chunks),), strand_val))
                            data['pysam_data'][read_name_str]=read_dict
                            data['alignment_start'].append(int(bam_read.reference_start))

                                    
                            if len(data['signal']) == self.reads_per_chunk:
                                yield from self._yield_buffer(data, polish)
                                data = {
                                    'signal': [],
                                    'seq': [],
                                    'read_name': [],
                                    'position': [],
                                    'read_position': [],
                                    'strand': [],
                                    'pysam_data': {},
                                    'alignment_start': [],
                                }
        if len(data['signal']) > 0:
            yield from self._yield_buffer(data, polish)

def prepare_complete_labelled_chunks(
    read,
    labelled_arrays,
    size,
    step,
    pore_type=None,
    discard_empty_labels = False,
    return_scale_features=False,
    polished_read=None,
    label_all_aligned_bases=False,
):
    """Prepare all reference-labelled chunks needed for read reconstruction.

    ``read`` is normalized in place, chunked, and matched against sorted
    ``labelled_arrays`` for its contig and strand. With
    ``label_all_aligned_bases=True``, every non-negative reference coordinate
    is selected directly and ``labelled_arrays`` is not consulted. Returned
    labels are a boolean mask only; target values are not copied.
    ``discard_empty_labels`` may return ``None``, while false retains every
    chunk when the read has no hits.

    When ``return_scale_features`` is true, the result also contains per-token
    base error probabilities and repeated six-value read-level scale metadata.
    A supplied ``polished_read`` replaces sequence/coordinate chunks and maps
    qualities into polished coordinates.

    Returns
    -------
    dict or None
        Signal in ``(n_chunks, channels, size)`` order plus sequence,
        ``labels_idx``, zero-based reference positions, read positions, and
        optional scale features.
    """
    norm_stats = None
    if return_scale_features:
        read.signal, norm_stats = normalize_read_signal_with_stats(read, pore_type)
    else:
        read.signal = normalize_read_signal(read, pore_type)

    signal_chunks, base_seq_chunks,  _, pos_chunks_padded, read_pos_chunks = get_read_chunks_labelled(
        read, size, step, polished_read=polished_read,
    )
    if return_scale_features:
        base_quality_chunks = (
            get_base_quality_chunks(read, size, step)
            if polished_read is None
            else get_polished_base_quality_chunks(
                read, polished_read, base_seq_chunks, read_pos_chunks,
            )
        )
    else:
        base_quality_chunks = None

    if label_all_aligned_bases:
        labels_idx = pos_chunks_padded >= 0
    else:
        target_pos, _ = labelled_arrays[read.ref_name][read.is_reverse]

        if len(target_pos) == 0:
            labels_idx = torch.zeros_like(pos_chunks_padded, dtype=torch.bool)
        else:
            query_pos = pos_chunks_padded.numpy()

            idx = np.searchsorted(target_pos, query_pos)

            # Clip insertion points before equality validation; this also
            # ensures negative padding coordinates cannot be mistaken for
            # target hits.
            idx = np.clip(idx, 0, len(target_pos) - 1)
            valid_mask_np = (target_pos[idx] == query_pos)
            labels_idx = torch.from_numpy(valid_mask_np)


    keep = labels_idx.sum(1) > 0
    
    if not keep.any():
        if discard_empty_labels:
            return None
        
        else:
            keep=torch.ones_like(keep, dtype=bool)
    
    keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
    signals = signal_chunks[keep].transpose(1, 2)
    seqs, aligned_labels_idx, positions, read_pos = align_chunk_fields_to_sequences(
        [base_seq_chunks[i] for i in keep_idx],
        labels_idx[keep],
        pos_chunks_padded[keep],
        read_pos_chunks[keep],
    )
    
    out = {
        'signal': signals,
        'seq': seqs,
        'labels_idx': aligned_labels_idx,
        'position': positions,
        'read_pos': read_pos,
    }
    if return_scale_features:
        global_features = read_scale_metadata(norm_stats, read.base_qual)
        out['base_quality'] = [base_quality_chunks[i] for i in keep.nonzero(as_tuple=True)[0].tolist()]
        out['global_features'] = [global_features] * len(out['seq'])
    return out

def parse_read_based_motifs(motif):
    """Parse a read-motif string for position-only complete-read labelling.

    Motifs are assigned the sole class zero because this path needs only hit
    positions. ``ValueError`` is raised when no valid motif remains.
    """
    parsed_motifs = parse_motif_specs(
        motif=motif,
        motif_label="0",
        num_classes=1,
    )

    if not parsed_motifs:
        raise ValueError(f"No valid read-based motifs parsed from motif string: {motif}")

    return parsed_motifs

def get_read_based_motif_positions(read, parsed_motifs, polished_read=None):
    """Return sorted motif-hit positions in read or polished-read coordinates."""
    if not parsed_motifs:
        return np.array([], dtype=np.int64)

    if polished_read is None:
        read_seq = ''.join(read.get_base_seq(string=True)).upper()
    else:
        read_seq = ''.join(
            num_to_base_map.get(int(base), 'N')
            for base in polished_read.sequence
        ).upper()
    target_positions, _ = get_read_motif_targets(
        read_sequence=read_seq,
        parsed_motifs=parsed_motifs,
        num_classes=1,
    )
    return target_positions.astype(np.int64, copy=False)

def get_unaligned_read_chunks_labelled(read, size, step):
    """Chunk a reference-free read with read positions and placeholder loci.

    Returns signal windows, padded sequence chunks, an all-``-1`` reference
    position matrix, and padded zero-based read positions.
    """
    if has_inferred_boundaries(read):
        signal, sequences, _, references, positions = inferred_labelled_chunks(read, size, step)
        return signal, sequences, references, positions
    signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
    signal_chunks, base_seq_chunks, _, move_table_chunks = chunk_padded_components(
        signal, base_seq, move_table, size, step
    )

    seqs_batch_padded = pad_sequence(base_seq_chunks, batch_first=True, padding_value=0)

    base_seq_idx = torch.arange(read.length)[read.indexes]
    base_seq_idx = F.pad(
        base_seq_idx,
        (0, padding_needed),
        mode='constant',
        value=read.length - 1
    )

    base_seq_idx_chunks = base_seq_idx.unfold(dimension=0, size=size, step=step)
    read_pos_chunks = [base_seq_idx_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
    read_pos_chunks_padded = pad_sequence(read_pos_chunks, batch_first=True, padding_value=-1)
    ref_pos_chunks_padded = torch.full_like(read_pos_chunks_padded, -1)

    return signal_chunks, seqs_batch_padded, ref_pos_chunks_padded, read_pos_chunks_padded

def prepare_complete_read_based_chunks(
    read,
    parsed_motifs,
    size,
    step,
    pore_type=None,
    discard_empty_labels=False,
    return_scale_features=False,
    polished_read=None,
):
    """Prepare complete-read chunks supervised by motifs in the read sequence.

    The function accepts aligned :class:`Read` or unaligned
    :class:`Read_Lite` instances. Reference coordinates are retained when
    available and otherwise filled with ``-1``. ``read`` is normalized in
    place; filtering, optional scale features, polishing, and the returned
    dictionary follow :func:`prepare_complete_labelled_chunks`.
    """
    norm_stats = None
    if return_scale_features:
        read.signal, norm_stats = normalize_read_signal_with_stats(read, pore_type)
    else:
        read.signal = normalize_read_signal(read, pore_type)

    if polished_read is not None:
        signal_chunks, base_seq_chunks, _, pos_chunks_padded, read_pos_chunks = get_read_chunks_labelled(
            read, size, step, polished_read=polished_read,
        )
    elif hasattr(read, 'aligned_pairs'):
        signal_chunks, base_seq_chunks, _, pos_chunks_padded, read_pos_chunks = get_read_chunks_labelled(read, size, step)
    else:
        signal_chunks, base_seq_chunks, pos_chunks_padded, read_pos_chunks = get_unaligned_read_chunks_labelled(read, size, step)
    if return_scale_features:
        base_quality_chunks = (
            get_base_quality_chunks(read, size, step)
            if polished_read is None
            else get_polished_base_quality_chunks(
                read, polished_read, base_seq_chunks, read_pos_chunks,
            )
        )
    else:
        base_quality_chunks = None

    target_pos = get_read_based_motif_positions(
        read, parsed_motifs, polished_read=polished_read,
    )

    if len(target_pos) == 0:
        labels_idx = torch.zeros_like(read_pos_chunks, dtype=torch.bool)
    else:
        query_pos = read_pos_chunks.numpy()
        idx = np.searchsorted(target_pos, query_pos)
        idx = np.clip(idx, 0, len(target_pos) - 1)
        valid_mask_np = (target_pos[idx] == query_pos)
        labels_idx = torch.from_numpy(valid_mask_np)

    keep = labels_idx.sum(1) > 0

    if not keep.any():
        if discard_empty_labels:
            return None
        keep = torch.ones_like(keep, dtype=bool)

    keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
    signals = signal_chunks[keep].transpose(1, 2)
    seqs, aligned_labels_idx, positions, read_pos = align_chunk_fields_to_sequences(
        [base_seq_chunks[i] for i in keep_idx],
        labels_idx[keep],
        pos_chunks_padded[keep],
        read_pos_chunks[keep],
    )

    out = {
        'signal': signals,
        'seq': seqs,
        'labels_idx': aligned_labels_idx,
        'position': positions,
        'read_pos': read_pos,
    }
    if return_scale_features:
        global_features = read_scale_metadata(norm_stats, read.base_qual)
        out['base_quality'] = [base_quality_chunks[i] for i in keep.nonzero(as_tuple=True)[0].tolist()]
        out['global_features'] = [global_features] * len(out['seq'])
    return out

class CompleteLabelledReadIterableDataset(IterableDataset):
    """Yield reconstructable complete-read batches around labelled positions.

    This dataset underlies full-read inference and canonical-null workflows.
    It supports reference-position labels and per-sample read-based motifs,
    including unaligned read-based BAM streams. Every accepted read is split
    into overlapping windows while preserving read names, strands, reference
    and read positions, and optionally source BAM dictionaries and signal-scale
    features.

    Parameters
    ----------
    bam_list : list of LabelledBamSample
        SG samples are sharded whole across ranks and DataLoader workers. POD5
        inference accepts one BAM sample and shards its whole POD5 files.
    size, step : int
        Model-level window length and offset; multiplied by
        ``move_downsample_factor`` for move-table chunking.
    move_downsample_factor : int
        Adjacent signal rows packed into the final feature dimension.
    seq_type : str, optional
        ``"rna"`` or ``"dna"`` orientation; inferred from the profile.
    kmer_table : numpy.ndarray, optional
        Dense expected-current lookup; inferred from the profile.
    preprocessing : PreprocessingSpec or mapping, optional
        Shared contract, e.g. ``model.preprocessing``. Explicit sequence,
        pore, and table values must agree with it.
    chemistry : str, optional
        Standalone selection: ``dna_r10.4.1_e8.2_400bps``, ``rna004``, or
        ``rna002``; no model or external table is needed.
    reads_per_chunk : int, default=1000
        Read-level chunk tensors buffered per yielded batch.
    min_read_length, max_read_length : int or float
        Inclusive query-sequence length bounds in bases.
    max_alignment_length : int or float
        Maximum reference span for reference-based reads; not applied to
        read-based inputs.
    min_qscore : float, default=10
        Minimum optional ``qs`` tag.
    pore_type : str or None
        Optional chemistry assertion; inferred from the resolved profile.
    base_seed : int, default=1234
        Base for deterministic per-worker random seeds.
    rank, world_size : int
        Distributed whole-file sharding coordinates.
    tags_to_drop : sequence of str
        Tags removed from returned pysam dictionaries.
    discard_empty_labels : bool, default=False
        Drop reads with no target-bearing chunk.
    return_pysam : bool, default=False
        Include reconstructable source-read dictionaries in each batch.
    read_based : bool, default=False
        Legacy global override; new callers should set ``read_based`` on each
        sample so modes can be mixed.
    return_scale_features : bool, default=False
        Append padded base error probabilities and six read-level scale
        features to each batch.
    label_all_aligned_bases : bool, default=False
        Select every read base carrying a non-negative reference coordinate.
        This bypasses reference-wide motif/position materialization and is
        safe to serialize into spawned DataLoader workers.
    exclude_bed : str or None
        Optional ``chrom start end [strand]`` file for all input
        BAMs. Skip same-strand reference-span overlaps before signal decoding.
        Coordinates are zero-based, half-open; strand is ``+`` or ``-``, or
        omitted to match both strands.
    expected_ts_dim : int or None
        Required move-table stride when set.
    chunking_mode : {"auto", "mv", "hierarchical", "dtw", "inferred"}, default="auto"
        Boundary selection passed to the shared read decoder. Inferred chunks
        slice sequence metadata without synthesizing per-base move events.
    allow_pod5 : bool, default=False
        Opt in to inference-only samples with ``pod5`` signal sources. These
        stream in POD5 order and have no monotonic coordinate frontier. Only
        inference/scoring entry points enable this mode.

    Yields
    ------
    tuple
        Pre-collated complete-read batch with modal tensors, target mask,
        read/reference metadata and positions, alignment-start frontier,
        strand, pysam data, and optional scale features. The scale-feature
        suffix has base qualities, global features, and full read lengths
        (three tensors); lengths use polished coordinates when polishing.
        The frontier is the
        minimum source BAM ``reference_start`` among accepted reads in the
        batch. It intentionally does not depend on motif/position filtering.
    """

    def __init__(
        self,
        bam_list: List[LabelledBamSample],
        size,
        step,
        move_downsample_factor,
        seq_type=None,
        kmer_table=None,
        reads_per_chunk=1000,
        min_read_length=100,
        max_read_length=5e5,
        max_alignment_length=1e6,
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        tags_to_drop=['ML', 'MM'],
        discard_empty_labels = False,
        return_pysam = False,
        read_based = False,
        return_scale_features = False,
        label_all_aligned_bases = False,
        expected_ts_dim=None,
        exclude_bed=None,
        allow_pod5=False,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
        if not allow_pod5 and any(getattr(sample, 'pod5', None) for sample in bam_list):
            raise ValueError("POD5 inputs are supported only by inference/scoring workflows")
        self.pod5_files = validate_pod5_samples(bam_list)
        if not isinstance(label_all_aligned_bases, bool):
            raise TypeError(
                "label_all_aligned_bases must be true or false, got "
                f"{label_all_aligned_bases!r}"
            )
        if label_all_aligned_bases and (
            read_based or any(getattr(sample, 'read_based', False) for sample in bam_list)
        ):
            raise ValueError(
                "label_all_aligned_bases is a reference-coordinate mode and "
                "cannot be combined with read_based inputs"
            )
        self.bam_list = bam_list
        self.size = size*move_downsample_factor
        self.move_downsample_factor=move_downsample_factor
        self.step = step*move_downsample_factor
        
        self.preprocessing = resolve_preprocessing(
            preprocessing=preprocessing, chemistry=chemistry, seq_type=seq_type,
            pore_type=pore_type, kmer_table=kmer_table,
        )
        seq_type, pore_type, kmer_table = (
            self.preprocessing.seq_type, self.preprocessing.pore_type, self.preprocessing.kmer_table
        )
        self.seq_type = seq_type
        self.reads_per_chunk = reads_per_chunk
        self.min_read_length = min_read_length
        self.max_read_length = max_read_length
        self.max_alignment_length = max_alignment_length
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.tags_to_drop=tags_to_drop
        self.discard_empty_labels = discard_empty_labels
        self.return_pysam = return_pysam
        # Legacy global override. New callers should set read_based on each
        # LabelledBamSample so reference- and read-based inputs can be mixed.
        self.read_based = read_based
        self.return_scale_features = return_scale_features
        self.label_all_aligned_bases = label_all_aligned_bases
        self.exclude_bed = exclude_bed
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        
    def _accept_pod5_record(self, bam_read, bam_sample, sample_read_based, excluded_regions):
        """Apply inference QC and half-open region filters without indexed fetch."""
        if bam_read.flag & (0x900 | bam_sample.flag):
            return False
        if not passes_min_qscore(bam_read, self.min_qscore) or bam_read.seq is None:
            return False
        if not self.min_read_length <= len(bam_read.seq) <= self.max_read_length:
            return False
        if not sample_read_based and (
            not bam_read.is_mapped
            or bam_read.reference_end - bam_read.reference_start > self.max_alignment_length
        ):
            return False
        if excluded_regions is not None and excluded_regions.overlaps_read(bam_read):
            return False
        if bam_sample.chrom_info:
            return bam_read.is_mapped and any(
                bam_read.reference_name == contig
                and bam_read.reference_end > start
                and (end is None or bam_read.reference_start < end)
                for contig, start, end in bam_sample.chrom_info
            )
        return True

    def _yield_buffer(self, data):
        """Pack, pad, and yield one complete-read buffer as a model batch."""
        seqs_batch = data['seq']
        signals_batch = data['signal']
        alignment_frontier = torch.tensor(
            min(data['alignment_start']), dtype=torch.long
        )
        
        read_name_batch = np.array(data['read_name'])
        ref_name_batch = np.array(data['ref_name'])
        strand_batch = np.concatenate(data['strand'])
        positions_batch = data['position']
        
        read_pos_chunks = data['read_position']
        labels_idx_batch = data['labels_idx']
        
        signals_batch = torch.cat(signals_batch, dim=0)
        percentiles_batch = get_percentiles(signals_batch)

        signals_batch = signals_batch.reshape(signals_batch.size(0), signals_batch.size(1)//self.move_downsample_factor, signals_batch.size(2)*self.move_downsample_factor)
        percentiles_batch = percentiles_batch.reshape(percentiles_batch.size(0), percentiles_batch.size(1)//self.move_downsample_factor, percentiles_batch.size(2)*self.move_downsample_factor)
        
        pad_ts = torch.any(signals_batch != -10, dim=-1)
        signals_batch = torch.where(pad_ts.unsqueeze(-1), signals_batch, 0)
        percentiles_batch = torch.where(pad_ts.unsqueeze(-1), percentiles_batch, 0)
        
        seqs_batch_padded = pad_sequence(seqs_batch, batch_first=True, padding_value=0)
        positions_batch_padded =  pad_sequence(positions_batch, batch_first=True, padding_value=-1)
        read_pos_chunks_padded =  pad_sequence(read_pos_chunks, batch_first=True, padding_value=-1)
        labels_idx_batch_padded = pad_sequence(labels_idx_batch, batch_first=True, padding_value=False)
        
        current_max_len = seqs_batch_padded.shape[1]
        target_len = ((current_max_len + 127) // 128) * 128
        pad_needed = target_len - current_max_len
        
        if pad_needed > 0:
            seqs_batch_padded = F.pad(seqs_batch_padded, (0, pad_needed), value=0)
            positions_batch_padded = F.pad(positions_batch_padded, (0, pad_needed), value=-1)
            read_pos_chunks_padded = F.pad(read_pos_chunks_padded, (0, pad_needed), value=-1)
            labels_idx_batch_padded = F.pad(labels_idx_batch_padded, (0, pad_needed), value=False)
            
        pad_base_seq = seqs_batch_padded != 0
    
        batch = (
                seqs_batch_padded.to(torch.long),
                signals_batch.to(torch.float32),
                percentiles_batch.to(torch.float32),
                pad_base_seq,
                pad_ts,
                labels_idx_batch_padded,
                read_name_batch, ref_name_batch, positions_batch_padded,
                read_pos_chunks_padded, alignment_frontier, strand_batch,
                data['pysam_data']
            )
        if self.return_scale_features:
            base_quality_padded = pad_sequence(data['base_quality'], batch_first=True, padding_value=1.0)
            base_quality_len = base_quality_padded.shape[1]
            if base_quality_len < target_len:
                base_quality_padded = F.pad(base_quality_padded, (0, target_len - base_quality_len), value=1.0)
            elif base_quality_len > target_len:
                base_quality_padded = base_quality_padded[:, :target_len]
            base_quality_padded = torch.where(
                pad_base_seq,
                base_quality_padded.float(),
                torch.ones_like(base_quality_padded, dtype=torch.float32),
            )
            batch = batch + (
                base_quality_padded,
                torch.stack(data['global_features'], dim=0).float(),
                torch.tensor(data['read_lengths'], dtype=torch.long),
            )

        yield batch

    def __iter__(self):
        """Stream this worker's reference- and read-labelled BAM sources."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
                
        global_rank = self.rank 
        global_world_size = self.world_size

        global_worker_id = global_rank * num_workers + worker_id
        global_worker_count = max(global_world_size * num_workers, 1)
        
        seed = self.base_seed + 1000003 * (global_worker_id + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))  
        torch.manual_seed(seed)
        
        assigned_pod5_files = None
        if self.pod5_files:
            # Split by rank first, allowing each rank to cap its worker count
            # independently when there are fewer POD5 files than processes.
            assigned_pod5_files = self.pod5_files[global_rank::global_world_size][worker_id::num_workers]
            my_files = self.bam_list if assigned_pod5_files else []
        else:
            my_files = self.bam_list[global_worker_id::global_worker_count]
        
        if len(my_files) == 0:
            return
        
        ref_seq_cache = {}
        labelled_arrays_cache = {}
        read_motif_cache = {}
        polish_variant_mask_cache = {}
        
        data = {
            'signal': [],
            'seq': [],
            'read_name': [],
            'ref_name': [],
            'position': [],
            'read_position': [],
            'strand': [],
            'pysam_data': {},
            'labels_idx': [],
            'base_quality': [],
            'global_features': [],
            'read_lengths': [],
            'alignment_start': [],
        }
            
        for bam_sample in my_files:
            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info=bam_sample.flag
            polish=bam_sample.polish
            sample_read_based = self.read_based or getattr(bam_sample, 'read_based', False)
            pod5_source = getattr(bam_sample, 'pod5', None)
            excluded_regions = get_excluded_regions(self.exclude_bed)
            
            chroms=tuple(x[0] for x in chrom_info)

            active_chroms = None
            if pod5_source:
                # POD5 iteration filters alignments directly; no coordinate
                # index, coordinate sort, or mapped-contig counts are needed.
                active_chroms = set(chroms)
            elif sample_read_based and chroms:
                active_chroms = get_mapped_chroms(bam_path) & set(chroms)
                if len(active_chroms) == 0:
                    continue
            elif not sample_read_based:
                active_chroms = get_mapped_chroms(bam_path)&set(chroms)
                if len(active_chroms)==0:
                    continue
            
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam, ExitStack() as resources:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path)}',
                    flush=True,
                )


                if sample_read_based:
                    ref_seq_dict = None
                    if bam_sample.ref:
                        if bam_sample.ref in ref_seq_cache:
                            ref_seq_dict = ref_seq_cache[bam_sample.ref]
                        else:
                            ref_seq_dict = get_ref_seq_dict(ref=bam_sample.ref, chrom_list=chroms)
                            ref_seq_cache[bam_sample.ref] = ref_seq_dict

                    motif = getattr(bam_sample, 'motif', None)
                    if motif in read_motif_cache:
                        parsed_read_motifs = read_motif_cache[motif]
                    else:
                        parsed_read_motifs = parse_read_based_motifs(motif)
                        read_motif_cache[motif] = parsed_read_motifs
                else:
                    if bam_sample.ref in ref_seq_cache:
                        ref_seq_dict = ref_seq_cache[bam_sample.ref]
                    else:
                        ref_seq_dict = get_ref_seq_dict(ref=bam_sample.ref, chrom_list=chroms)
                        ref_seq_cache[bam_sample.ref] = ref_seq_dict
                
                if sample_read_based:
                    labelled_arrays = None
                elif self.label_all_aligned_bases:
                    # Targets are derived from each read's aligned reference
                    # coordinates below; no chromosome-wide target dictionary
                    # is needed in this worker.
                    labelled_arrays = None
                else:
                    labelled_cache_key = (
                        bam_sample.ref,
                        self.seq_type,
                        bam_sample.pos_file,
                        bam_sample.motif,
                        bam_sample.motif_label,
                        chroms,
                    )
                    if labelled_cache_key in labelled_arrays_cache:
                        labelled_arrays = labelled_arrays_cache[labelled_cache_key]
                    else:
                        labelled_arrays = get_labelled_arrays(
                            ref=bam_sample.ref,
                            seq_type=self.seq_type,
                            pos_list=bam_sample.pos_file,
                            motif=bam_sample.motif,
                            motif_label=bam_sample.motif_label,
                            chrom_list=list(chroms),
                        )
                        labelled_arrays_cache[labelled_cache_key] = labelled_arrays

                if pod5_source:
                    read_iter = resources.enter_context(closing(iter_pod5_bam_records(
                        bam, pod5_source, chunking_mode=self.chunking_mode,
                        index_path=getattr(bam_sample, 'bam_index', None),
                        assigned_files=assigned_pod5_files,
                        accept_record=lambda record: self._accept_pod5_record(
                            record, bam_sample, sample_read_based, excluded_regions,
                        ),
                    )))
                elif sample_read_based and active_chroms:
                    def _iter_read_based_reference_reads():
                        for contig_info in chrom_info:
                            contig, start, end = contig_info
                            if contig not in active_chroms:
                                continue
                            if end is None:
                                yield from bam.fetch(contig, start=start)
                            else:
                                yield from bam.fetch(contig, start=start, end=end)
                    read_iter = _iter_read_based_reference_reads()
                elif sample_read_based:
                    read_iter = bam.fetch(until_eof=True)
                else:
                    def _iter_reference_reads():
                        for contig_info in chrom_info:
                            contig, start, end = contig_info
                            if contig not in active_chroms:
                                continue
                            yield from bam.fetch(contig, start=start, end=end, until_eof=True)
                    read_iter = _iter_reference_reads()

                if not pod5_source:
                    read_iter = ((record, None) for record in read_iter)
                chrom_filter = set(chroms)
                for bam_read, raw_signal in read_iter:
                    if excluded_regions is not None and excluded_regions.overlaps_read(bam_read):
                        continue
                    if sample_read_based and chrom_filter and bam_read.is_mapped and bam_read.reference_name not in chrom_filter:
                        continue
                    if (
                        bam_read.flag & (0x900|flag_info) == 0
                        and passes_min_qscore(bam_read, self.min_qscore)
                        and len(bam_read.seq) >= self.min_read_length
                        and len(bam_read.seq) <= self.max_read_length
                        and (
                            sample_read_based
                            or bam_read.reference_end-bam_read.reference_start<=self.max_alignment_length
                        )
                    ):
                        if sample_read_based and (not bam_read.is_mapped or ref_seq_dict is None):
                            read_info = get_read_info_lite(
                                bam_read, self.seq_type, self.kmer_table,
                                expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                raw_signal=raw_signal,
                            )
                            read = Read_Lite(*read_info)
                            read_name_str = read.read_name
                            ref_name_str = bam_read.reference_name if bam_read.is_mapped and bam_read.reference_name else '*'
                            strand_val = 0 if bam_read.is_forward else 1
                            if not bam_read.is_mapped:
                                strand_val = -1
                        else:
                            read_info = get_read_info(
                                bam_read, ref_seq_dict, self.seq_type, self.kmer_table,
                                expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                raw_signal=raw_signal,
                            )
                            read = Read(*read_info)
                            read_name_str = read.read_name
                            ref_name_str = read.ref_name
                            strand_val = 0 if read.is_forward else 1
                        if self.return_scale_features and not hasattr(read, 'base_qual'):
                            read.base_qual = base_error_prob_from_bam_read(bam_read)

                        polished_read = (
                            get_polished_read_for_sample(
                                bam_read, bam_sample, polish_variant_mask_cache,
                            )
                            if polish
                            else None
                        )

                        try:
                            if sample_read_based:
                                processed = prepare_complete_read_based_chunks(
                                    read,
                                    parsed_read_motifs,
                                    self.size,
                                    self.step,
                                    pore_type=self.pore_type,
                                    discard_empty_labels=self.discard_empty_labels,
                                    return_scale_features=self.return_scale_features,
                                    polished_read=polished_read,
                                )
                            else:
                                processed = prepare_complete_labelled_chunks(
                                    read,
                                    labelled_arrays,
                                    self.size,
                                    self.step,
                                    pore_type=self.pore_type,
                                    discard_empty_labels=self.discard_empty_labels,
                                    return_scale_features=self.return_scale_features,
                                    polished_read=polished_read,
                                    label_all_aligned_bases=self.label_all_aligned_bases,
                                )
                        except DegenerateQuantileNormalizationError:
                            continue

                        if processed is not None:
                            if self.return_pysam:
                                if polished_read is None:
                                    read_dict=bam_read.to_dict()
                                    read_dict['tags']=[x for x in read_dict['tags'] if x[:2] not in self.tags_to_drop]
                                else:
                                    read_dict = make_polished_read_dict(
                                        bam_read, polished_read, self.tags_to_drop,
                                    )
                                data['pysam_data'][read_name_str]=read_dict

                            data['signal'].append(processed['signal'])

                            data['seq'].extend(processed['seq'])
                            data['labels_idx'].extend(processed['labels_idx'])

                            data['read_name'].extend([read_name_str] * len(processed['position']))
                            data['ref_name'].extend([ref_name_str] * len(processed['position']))
                            data['position'].extend(processed['position'])
                            data['read_position'].extend(processed['read_pos'])
                            data['strand'].append(np.full((len(processed['position']),), strand_val, dtype=np.int8))
                            data['alignment_start'].append(
                                int(bam_read.reference_start)
                                if bam_read.is_mapped
                                else 0
                            )
                            if self.return_scale_features:
                                full_length = read.length if polished_read is None else len(polished_read.sequence)
                                data['read_lengths'].extend([full_length] * len(processed['seq']))
                                if 'base_quality' in processed:
                                    data['base_quality'].extend(processed['base_quality'])
                                    data['global_features'].extend(processed['global_features'])
                                else:
                                    data['base_quality'].extend([torch.ones_like(seq, dtype=torch.float32) for seq in processed['seq']])
                                    data['global_features'].extend([
                                        torch.zeros(6, dtype=torch.float32)
                                        for _ in processed['seq']
                                    ])

                        if len(data['signal']) == self.reads_per_chunk:
                            yield from self._yield_buffer(data)
                            data = {
                                'signal': [],
                                'seq': [],
                                'labels_idx': [],
                                'read_name': [],
                                'ref_name': [],
                                'position': [],
                                'read_position': [],
                                'strand': [],
                                'pysam_data': {},
                                'base_quality': [],
                                'global_features': [],
                                'read_lengths': [],
                                'alignment_start': [],
                            }
                                
        if len(data['signal']) > 0:
            yield from self._yield_buffer(data)

def get_unique_sorted(arr):
    """Remove adjacent duplicates from an already sorted NumPy array."""
    if arr.size == 0:
        return arr
    
    mask = np.empty(arr.shape, dtype=bool)
    mask[0] = True
    mask[1:] = arr[1:] != arr[:-1]
    
    return arr[mask]            

def stitch_1d_distances_ultra_fast(metric_array, read_positions, ref_positions, pad_base_seq, read_names):
    """Stitch overlapping per-token chunk metrics into complete-read vectors.

    Parameters
    ----------
    metric_array : torch.Tensor
        Per-chunk scalar metrics with shape ``(n_chunks, padded_length)``.
    read_positions, ref_positions : torch.Tensor
        Read and reference coordinate matrices aligned to ``metric_array``.
    pad_base_seq : torch.Tensor
        Boolean mask selecting real base tokens in each chunk.
    read_names : numpy.ndarray
        Read identifier per chunk. Chunks for each read must be contiguous and
        ordered by increasing read position.

    Returns
    -------
    tuple
        Adjacent-deduplicated read names, one stitched metric tensor per read,
        and corresponding stitched reference-position tensors. Overlaps are
        divided at their midpoint; unwritten metric positions remain ``NaN``.
    """
    read_wise_metric = []
    read_wise_refpos = []
    
    boundaries = np.where(read_names[:-1] != read_names[1:])[0] + 1
    chunks_per_read = np.split(np.arange(len(read_names)), boundaries)
    
    for read_indices in chunks_per_read:
        read_length = int(read_positions[read_indices].max().item()) + 1
        
        # NaNs distinguish positions not covered after padding/overlap handling
        # from genuine zero-valued metrics.
        accum_metric = torch.full((read_length,), float('nan'), dtype=metric_array.dtype)
        accum_ref = torch.empty((read_length,), dtype=ref_positions.dtype)
        
        last_written_pos = -1
        
        for idx in read_indices:
            mask = pad_base_seq[idx]
            
            curr_metric = metric_array[idx][mask]
            curr_pos = read_positions[idx][mask]
            curr_ref = ref_positions[idx][mask]
            
            if curr_pos.numel() == 0:
                continue
                
            start_pos = curr_pos[0].item()
            
            # Divide overlapping chunks at the midpoint so each read position
            # is contributed by exactly one window.
            if start_pos <= last_written_pos:
                overlap_len = last_written_pos - start_pos + 1
                split_offset = (overlap_len + 1) // 2 
            else:
                split_offset = 0
                
            write_metric = curr_metric[split_offset:]
            write_pos = curr_pos[split_offset:]
            write_ref = curr_ref[split_offset:]
            
            if write_pos.numel() > 0:
                write_start = write_pos[0].item()
                write_end = write_pos[-1].item() + 1
                
                accum_metric[write_start:write_end] = write_metric
                accum_ref[write_start:write_end] = write_ref
                
                last_written_pos = write_end - 1
                
        read_wise_metric.append(accum_metric)
        read_wise_refpos.append(accum_ref)
        
    return get_unique_sorted(read_names), read_wise_metric, read_wise_refpos


def make_mm_ml_tags(seq, base, coords, probs, seq_type, mod_code="m", strand="+", skipped_flag="?"):
    """
    Build SAM MM/ML tags for one modification class.

    Coordinates outside the sequence or not matching the requested
    fundamental base are skipped.

    Parameters
    ----------
    seq : str
        Read sequence in original/as-sequenced 5'->3' orientation.
    base : str
        Fundamental base to use for candidate-base counting.
        One of A,C,G,T,U,N.
    coords : array-like of int
        0-based read coordinates of called sites, ideally in 5'->3' order.
    probs : array-like of float
        Probabilities in [0, 1], same order as coords.
    seq_type : str
        ``"dna"`` or ``"rna"``. The value is validated for compatibility
        with callers; base matching and the MM fundamental base are taken
        directly from ``base``.
    mod_code : str
        SAM modification code, e.g. 'm' for 5mC.
    strand : str
        '+' or '-'.
    skipped_flag : str or None
        '?', '.', or None.

    Returns
    -------
    mm : str
        MM tag payload, e.g. 'C+m?,5,12,0;'
    ml : list[int]
        ML tag payload as integers 0..255
    """
    seq = seq.upper()
    base = base.upper()
    seq_type = seq_type.lower()
    
    seq_match_base=base
    mm_base=base

    if base not in {"A", "C", "G", "T", "U", "N"}:
        raise ValueError("base must be one of A,C,G,T,U,N")
    if seq_type not in {"dna", "rna"}:
        raise ValueError("seq_type must be 'dna' or 'rna'")
    if strand not in {"+", "-"}:
        raise ValueError("strand must be '+' or '-'")
    if skipped_flag not in {"?", ".", None, ""}:
        raise ValueError("skipped_flag must be '?', '.', or None")

    coords = np.asarray(coords, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    if coords.ndim != 1 or probs.ndim != 1 or len(coords) != len(probs):
        raise ValueError("coords and probs must be 1D arrays of equal length")

    flag = "" if skipped_flag in (None, "") else skipped_flag

    if len(coords) == 0:
        return f"{mm_base}{strand}{mod_code}{flag};", []

    in_range = (coords >= 0) & (coords < len(seq))
    coords = coords[in_range]
    probs = probs[in_range]

    if len(coords) == 0:
        return f"{mm_base}{strand}{mod_code}{flag};", []

    order = np.argsort(coords, kind="stable")
    coords = coords[order]
    probs = probs[order]

    # SAM MM deltas count occurrences of the fundamental base, not absolute
    # read positions. ``N`` is the special case where every base is a candidate.
    if base == "N":
        base_pos = np.arange(len(seq), dtype=np.int64)
        valid = np.ones(len(coords), dtype=bool)
        ranks = coords.copy()
    else:
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        base_pos = np.flatnonzero(seq_bytes == ord(seq_match_base))

        if len(base_pos) == 0:
            return f"{mm_base}{strand}{mod_code}{flag};", []

        ranks_all = np.searchsorted(base_pos, coords)

        in_bounds = ranks_all < len(base_pos)
        valid = np.zeros(len(coords), dtype=bool)

        if np.any(in_bounds):
            valid[in_bounds] = (base_pos[ranks_all[in_bounds]] == coords[in_bounds])

        coords = coords[valid]
        probs = probs[valid]
        ranks = ranks_all[valid]

        if len(coords) == 0:
            return f"{mm_base}{strand}{mod_code}{flag};", []

    deltas = np.diff(np.concatenate(([-1], ranks))) - 1

    # ML stores floor(probability * 256), saturated to the uint8 maximum.
    ml = np.minimum(
        (np.clip(probs, 0.0, 1.0) * 256.0).astype(np.int64),
        255
    ).astype(np.uint8)

    mm = f"{mm_base}{strand}{mod_code}{flag}," + ",".join(map(str, deltas.tolist())) + ";"
    return mm, ml.tolist()


def bam_writer_worker(write_queue, bam_output, bam_threads, out_header, chrom_labelled_arrays, base, mod_code, seq_type):
    """Consume stitched predictions and write MM/ML-annotated BAM records.

    Parameters
    ----------
    write_queue : multiprocessing.JoinableQueue
        Items are ``(strand_dict, metrics, reference_positions, pysam_data)``;
        ``None`` terminates the worker. Every item is acknowledged with
        ``task_done``.
    bam_output : str
        Output BAM path, overwritten by pysam in ``wb`` mode.
    bam_threads : int
        Compression threads passed to ``pysam.AlignmentFile``.
    out_header : pysam.AlignmentHeader or compatible object
        Source header. A new ``@PG`` record is appended.
    chrom_labelled_arrays : dict
        Strand-indexed sorted reference target positions.
    base, mod_code, seq_type : str
        Fundamental base, SAM modification code, and DNA/RNA mode passed to
        :func:`make_mm_ml_tags`.

    Notes
    -----
    Metrics are matched back to labelled reference positions, converted to
    read-coordinate MM/ML calls, and written with an ``MN`` sequence-length
    tag. The function is intended as a dedicated CPU/I/O worker.
    """
    out_header = add_pg_to_header(out_header)
    with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=out_header) as outf:
        while True:
            item = write_queue.get()
            
            if item is None:
                write_queue.task_done()
                break
                
            strand_dict, read_wise_metric, read_wise_refpos, pysam_data = item
            
            for i, (rname, r_strand) in enumerate(strand_dict.items()):
                r_metric = read_wise_metric[i].numpy()
                r_pos = read_wise_refpos[i].numpy()
                positions = chrom_labelled_arrays[r_strand][0]

                match_idx = np.searchsorted(positions, r_pos)
                match_idx = np.clip(match_idx, 0, len(positions) - 1)
                valid_searches_mask = positions[match_idx] == r_pos
                r_pos_indices = np.where(valid_searches_mask & (~np.isnan(r_metric)))[0]
                                
                read_dict = pysam_data[rname]
                seq=read_dict['seq']
                seq = revcomp(seq) if int(read_dict['flag']) & 16 else seq
                    
                mm_string, ml_array = make_mm_ml_tags(seq, base, r_pos_indices,  r_metric[r_pos_indices], seq_type, mod_code=mod_code, strand="+", skipped_flag="?")

                new_read = pysam.AlignedSegment.from_dict(read_dict, out_header)
                new_read.set_tag("MM", mm_string, value_type="Z")
                new_read.set_tag("ML", array.array('B', ml_array))
                new_read.set_tag("MN", len(seq))

                outf.write(new_read)
            
            write_queue.task_done()
