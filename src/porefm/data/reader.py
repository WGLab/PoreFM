"""Iterable signal-BAM datasets, chunk preparation, and batch collation."""

from porefm.preprocessing import resolve_preprocessing
from porefm.data.chunk_boundaries import validate_chunking_mode
from porefm.data.inferred_chunks import (has_inferred_boundaries, inferred_chunks,
    inferred_labelled_chunks, slice_base_values, chunk_bounds)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
import random, time, os, sys, argparse, datetime, csv
import numpy as np
import pysam
import math, pickle
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, asdict

from typing import Optional, Tuple, List
from numba import njit
from porefm.data.excluded_regions import get_excluded_regions
from porefm.data.bam import (
    Read,
    Read_Lite,
    get_expected_signal_for_normalization,
    get_labelled_arrays,
    get_read_info,
    get_read_info_lite,
    get_read_motif_targets,
    get_ref_seq_dict,
    num_to_base_map,
    parse_motif_specs,
    passes_min_qscore,
    revcomp,
)
from porefm.common.utils import (
    DegenerateQuantileNormalizationError,
    mm_ml_coords,
    normalize_mm_code,
    quantile_normalize,
)
from porefm.data.polishing_options import (
    DEFAULT_POLISH_INDEL_LIMIT,
    get_polishing_variant_mask,
    validate_polishing_options,
)
import multiprocessing as mp
from dataclasses import dataclass, asdict
import torch.distributed as dist


def iter_file_cycles(my_files, cycle):
    """Yield worker-assigned files with a one-based pass counter.

    Parameters
    ----------
    my_files : iterable
        File descriptors already assigned to the current worker.
    cycle : bool
        Repeat the full iterable indefinitely when true; otherwise stop after
        one pass.

    Yields
    ------
    tuple
        ``(cycle_number, item)`` in the original order. An empty input yields
        nothing even when cycling is enabled.
    """
    if not my_files:
        return

    cycle_counter = 1
    while True:
        for item in my_files:
            yield cycle_counter, item
        if not cycle:
            return
        cycle_counter += 1


def ddp_rank_file_counts(num_files, world_size, dataloader_workers):
    """Return whole-file assignments per DDP rank.

    Iterable BAM datasets shard files across the Cartesian product of DDP
    ranks and DataLoader workers. A BAM is never split between worker slots,
    so a rank can receive no input even when the overall file list is nonempty.

    Returns
    -------
    tuple of int
        Number of complete files assigned to each rank, in rank order. A
        ``dataloader_workers`` value of zero is treated as one in-process
        worker per rank.
    """
    num_files = int(num_files)
    world_size = int(world_size)
    dataloader_workers = int(dataloader_workers)
    if num_files < 0:
        raise ValueError("num_files must be nonnegative.")
    if world_size < 1:
        raise ValueError("world_size must be at least 1.")
    if dataloader_workers < 0:
        raise ValueError("dataloader_workers must be nonnegative.")

    workers_per_rank = max(dataloader_workers, 1)
    global_worker_count = world_size * workers_per_rank
    counts = [0] * world_size
    for file_index in range(num_files):
        global_worker_id = file_index % global_worker_count
        counts[global_worker_id // workers_per_rank] += 1
    return tuple(counts)


def validate_ddp_bam_sharding(
    num_files,
    world_size,
    dataloader_workers,
    source_name,
):
    """Reject a whole-file DDP layout that leaves any rank without a BAM.

    Parameters are passed through to :func:`ddp_rank_file_counts`;
    ``source_name`` is included in diagnostics. The per-rank counts are
    returned when all ranks receive input, otherwise ``ValueError`` explains
    the minimum number of files needed for the current layout.
    """
    counts = ddp_rank_file_counts(num_files, world_size, dataloader_workers)
    missing_ranks = [rank for rank, count in enumerate(counts) if count == 0]
    if not missing_ranks:
        return counts

    workers_per_rank = max(int(dataloader_workers), 1)
    minimum_files = (int(world_size) - 1) * workers_per_rank + 1
    raise ValueError(
        f"{source_name}: DDP whole-file sharding would assign no BAM files to "
        f"rank(s) {missing_ranks}. Got {int(num_files)} BAM files for "
        f"world_size={int(world_size)} and dataloader_workers={int(dataloader_workers)}. "
        f"Provide at least {minimum_files} BAM files/shards, reduce the number of "
        "DDP ranks or DataLoader workers, or shard a merged BAM before training; "
        "one BAM file is not split by reads."
    )


def format_iterable_bam_log(
    dataset,
    global_rank,
    worker_id,
    bam_path,
    cycle_counter=None,
    contig=None,
):
    """Format consistent BAM progress context for an iterable dataset.

    The message identifies the dataset type, distributed rank, worker, BAM,
    and optional cycle/contig without reading or mutating the dataset.
    """
    fields = [f'{type(dataset).__name__}:', f'global_rank={global_rank}']
    if cycle_counter is not None:
        fields.append(f'cycle={cycle_counter}')
    fields.extend([f'worker_id={worker_id}.', f'BAM={bam_path}'])
    if contig is not None:
        fields.append(f'Contig={contig}')
    return ' '.join(fields)


def normalize_read_signal_with_stats(read, pore_type: Optional[str] = None):
    """Standardize raw signal and calibrate it to expected k-mer current.

    Parameters
    ----------
    read : Read or Read_Lite
        Decoded read exposing a two-dimensional ``signal`` matrix and expected
        current metadata. The input signal is not modified.
    pore_type : str or None
        Reserved chemistry selector retained for API compatibility; current
        normalization derives the k-mer model from ``read``.

    Returns
    -------
    tuple of numpy.ndarray
        Calibrated signal with the original shape and five ``float32`` global
        features: scaled raw shift, log raw scale, quantile-fit scale,
        quantile-fit shift, and fit correlation. Non-finite feature values are
        replaced by zero.

    Raises
    ------
    DegenerateQuantileNormalizationError
        If the expected or observed quantiles cannot support calibration.
    """
    signal = read.signal
    shift = float(np.mean(signal))
    scale = float(np.std(signal) + 1e-10)
    scaled_signal = (signal - shift) / scale
    expected_signal = get_expected_signal_for_normalization(read)
    norm_signal, q_scale, q_shift, q_corr = quantile_normalize(
        scaled_signal.reshape(-1),
        expected_signal,
    )
    norm_signal = norm_signal.reshape(scaled_signal.shape)

    stats = np.array(
        [
            shift / 200.0,
            np.log(scale + 1e-6),
            q_scale,
            q_shift,
            q_corr,
        ],
        dtype=np.float32,
    )
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    return norm_signal, stats


def normalize_read_signal(read, pore_type: Optional[str] = None):
    """Normalize a read's signal against its centered k-mer expectations.

    The raw matrix is first standardized globally, then affine-calibrated by
    quantile matching. The returned array has the same shape as ``read.signal``
    and the input object is not mutated. ``pore_type`` is currently retained
    for API compatibility.
    """
    signal = read.signal
    scaled_signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    expected_signal = get_expected_signal_for_normalization(read)
    norm_signal = quantile_normalize(
        scaled_signal.reshape(-1),
        expected_signal,
    )[0]
    return norm_signal.reshape(scaled_signal.shape)

    
def _new_labelled_buffer():
    return {
        'signal': [],
        'seq': [],
        'labels': [],
        'labels_idx': [],
        'label_smoothing': [],
        'position': [],
        'strand': [],
        'read_name': [],
        'ref_name': [],
        'frontier_chrom': 'unknown',
        'frontier_start': 1e12,
        'frontier_chrom_idx': float('inf'),
    }


def _update_labelled_buffer_frontier(data, chrom, chrom_idx, alignment_start):
    """Update a buffer frontier from a source BAM alignment start.

    Retained motif/label coordinates are not safe frontiers because their
    minima can regress between batches even when the BAM itself is sorted.
    """
    if alignment_start is None:
        return
    start = int(alignment_start)
    if (chrom_idx, start) < (data['frontier_chrom_idx'], data['frontier_start']):
        data['frontier_chrom'] = chrom
        data['frontier_start'] = start
        data['frontier_chrom_idx'] = chrom_idx


def _frontier_from_items(item_frontiers):
    if not item_frontiers:
        return 'unknown', 1e12, float('inf')
    return min(item_frontiers, key=lambda item: (item[2], item[1]))


def light_collate_fn(batch):
    """Collect labelled records without padding or stacking modal tensors.

    This lightweight contract preserves per-item lists for workflows that
    perform their own batching. It returns the same metadata order as
    :func:`labelled_collate_fn`, with both attention masks set to ``None`` and
    the earliest coordinate frontier across input items.
    """
    seqs_batch, signals_batch, percentiles_batch, labels_batch, labels_idx_batch = [], [], [], [], []
    positions_batch, strands_batch = [], []
    readnames_batch, refnames_batch = [], []
    item_frontiers = []
    
    for item_seqs, item_signals, item_percentiles, item_labels, items_labels_idx, item_positions, item_min_chrom, item_min_start, item_min_chrom_idx, item_rank, item_strand, item_readname, item_refname in batch:
        
        item_frontiers.append((item_min_chrom, item_min_start, item_min_chrom_idx))
        rank = item_rank
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)
        labels_batch.append(item_labels)
        labels_idx_batch.append(items_labels_idx)
        positions_batch.append(item_positions)
        strands_batch.append(item_strand)
        readnames_batch.append(item_readname)
        refnames_batch.append(item_refname)

    min_chrom, min_start, _ = _frontier_from_items(item_frontiers)
    
    return (
        seqs_batch,
        signals_batch,
        percentiles_batch,
        None,  # (B, Lseq) bool
        None,  # (B, Lts)  bool
        labels_batch,
        labels_idx_batch, positions_batch, strands_batch, readnames_batch, refnames_batch, min_chrom, min_start, rank
    )

def labelled_collate_fn(batch):
    """Pad labelled read windows and assemble the supervised batch contract.

    Input records are the 13-item tuples emitted by labelled or tagged
    iterable datasets. Base tokens use ``1..4`` for ``A/C/G/T``, ``0`` for
    padding, and ``5`` for masking. Per-base targets have shape
    ``(sequence_length, num_classes)`` and ``labels_idx`` identifies positions
    that contribute supervision.

    Returns
    -------
    tuple
        Padded modal tensors, boolean attention masks, padded targets and label
        masks, zero-based reference positions, strand/read/reference metadata,
        the earliest coordinate frontier, and originating worker rank.
    """
    seqs_batch, signals_batch, percentiles_batch, labels_batch, labels_idx_batch = [], [], [], [], []
    positions_batch, strands_batch = [], []
    readnames_batch, refnames_batch = [], []
    item_frontiers = []
    
    for item_seqs, item_signals, item_percentiles, item_labels, items_labels_idx, item_positions, item_min_chrom, item_min_start, item_min_chrom_idx, item_rank, item_strand, item_readname, item_refname in batch:
        
        item_frontiers.append((item_min_chrom, item_min_start, item_min_chrom_idx))
        rank = item_rank
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)
        labels_batch.append(item_labels)
        labels_idx_batch.append(items_labels_idx)
        positions_batch.append(item_positions)
        strands_batch.append(item_strand)
        readnames_batch.append(item_readname)
        refnames_batch.append(item_refname)

    seqs_batch_padded, signals_batch, percentiles_batch, pad_base_seq, pad_ts, pad_needed = collate_modal_batch(
        seqs_batch, signals_batch, percentiles_batch)

    # Soft labels are floats, padding with 0.0 is appropriate
    labels_batch_padded = pad_sequence(labels_batch, batch_first=True, padding_value=0.0)
    labels_idx_batch_padded = pad_sequence(labels_idx_batch, batch_first=True, padding_value=False)
    positions_batch_padded = pad_sequence(positions_batch, batch_first=True, padding_value=-1)
    
    strands = torch.tensor(strands_batch, dtype=torch.int8)
    readnames = np.array(readnames_batch)
    refnames = np.array(refnames_batch)
    min_chrom, min_start, _ = _frontier_from_items(item_frontiers)
    
    if pad_needed > 0:
        # Targets are three-dimensional, so pad the penultimate (sequence)
        # dimension; masks and positions pad along their final dimension.
        labels_batch_padded = F.pad(labels_batch_padded, (0, 0, 0, pad_needed), value=0.0)
        labels_idx_batch_padded = F.pad(labels_idx_batch_padded, (0, pad_needed), value=False)
        positions_batch_padded = F.pad(positions_batch_padded, (0, pad_needed), value=-1)
        
    return (
        seqs_batch_padded,
        signals_batch,
        percentiles_batch,
        pad_base_seq,  # (B, Lseq) bool
        pad_ts,  # (B, Lts)  bool
        labels_batch_padded,
        labels_idx_batch_padded, positions_batch_padded, strands, readnames, refnames, min_chrom, min_start, rank
    )

def collate_modal_batch(
    seqs_batch: List[torch.Tensor],
    signals_batch: List[torch.Tensor],
    percentiles_batch: List[torch.Tensor],
):
    """Pad and stack sequence/signal modalities and build attention masks.

    Signal and percentile inputs must share a fixed ``(time, channels)``
    shape; rows filled with ``-10`` are signal padding and are replaced by
    zero after ``pad_ts`` is derived. Variable-length one-based base-token
    vectors are padded with zero and rounded up to a multiple of 128.

    Returns
    -------
    tuple
        Long sequence tokens, float signal and percentile tensors, base and
        signal boolean masks, and the extra sequence padding added to reach a
        multiple of 128.
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

def pad_read_components(read, size: int, step: int):
    """Pad one decoded read so sliding windows cover its full signal.

    Base tokens are expanded over signal time steps and shifted to the
    model's one-based vocabulary. Signal padding uses ``-10``, expanded
    sequence padding ``-1``, and move-table padding zero.

    Returns
    -------
    tuple
        Padded signal, expanded sequence, move table, and the number of added
        time steps. At least one window of length ``size`` is produced.
    """
    if has_inferred_boundaries(read):
        raise ValueError("Inferred chunks must use boundary slicing, not expanded moves")
    base_seq = torch.from_numpy(read.get_base_seq(expanded=True) + 1)
    signal = torch.from_numpy(read.signal).clamp(-5, 5)
    move_table = torch.from_numpy(read.move_table)
    
    signal_len = signal.shape[0]
    N = max(1, math.ceil((signal_len - size) / step + 1))
    padding_needed = (N - 1) * step + size - signal_len

    signal = F.pad(signal, (0, 0, 0, padding_needed), mode='constant', value=-10)
    base_seq = F.pad(base_seq, (0, padding_needed), mode='constant', value=-1)
    move_table = F.pad(move_table, (0, padding_needed), mode='constant', value=0)

    return signal, base_seq, move_table, padding_needed

def chunk_padded_components(signal: torch.Tensor, base_seq: torch.Tensor, move_table: torch.Tensor, size: int, step: int):
    """Create sliding windows and collapse expanded tokens to base events.

    The first time step of every window is forced to act as an event boundary,
    ensuring each chunk begins with a base token even when it starts mid-event.

    Returns
    -------
    tuple
        Signal windows, a list of variable-length base-token vectors, original
        move-table windows, and boolean event masks.
    """
    signal_chunks = signal.unfold(dimension=0, size=size, step=step)
    base_seq_chunks = base_seq.unfold(dimension=0, size=size, step=step)
    init_move_table_chunks = move_table.unfold(dimension=0, size=size, step=step)

    move_table_chunks = init_move_table_chunks.clone()
    move_table_chunks[:, 0] += 1
    move_table_chunks = move_table_chunks.bool()

    base_seq_chunks = [base_seq_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
    return signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks


class BamIterableDataset(IterableDataset):
    """Stream unlabelled nanopore windows for foundation-model pretraining.

    Each ``bam_list`` entry is ``(bam_path, sampling_probability)``. Complete
    BAMs are assigned in a strided partition over distributed ranks and
    DataLoader workers, and records are streamed with ``fetch(until_eof=True)``
    so coordinate sorting and indexes are unnecessary. Signal windows are
    buffered, shuffled, and downsampled by packing adjacent move-table rows
    into channels.

    Parameters
    ----------
    bam_list : sequence of tuple
        ``(path, probability)`` inputs; the probability independently retains
        each eligible record.
    size, step : int
        Model-level window length and step. Both are multiplied internally by
        ``move_downsample_factor`` before raw move-table chunking.
    move_downsample_factor : int
        Number of adjacent signal rows packed into the final feature width.
    seq_type : str, optional
        ``"rna"`` enables RNA orientation and repeat-artifact filtering.
        Inferred from ``preprocessing`` or ``chemistry`` when omitted.
    kmer_table : numpy.ndarray, optional
        Dense expected-current lookup; inferred from the resolved profile.
    preprocessing : PreprocessingSpec or mapping, optional
        Shared contract, e.g. ``model.preprocessing``. Explicit sequence,
        pore, and table arguments must agree with it.
    chemistry : str, optional
        Standalone selection: ``dna_r10.4.1_e8.2_400bps``, ``rna004``, or
        ``rna002``. No model or external table is required.
    reads_per_chunk : int, default=1000
        Number of read-level chunk tensors buffered before percentile features
        are computed.
    min_read_length : int, default=100
        Minimum query sequence length in bases.
    min_qscore : float, default=10
        Minimum ``qs`` tag when the optional tag exists.
    pore_type : str or None
        Optional chemistry assertion; inferred from the resolved profile.
    base_seed : int, default=1234
        Base for deterministic per-worker random seeds.
    rank, world_size : int
        Distributed file-sharding coordinates.
    expected_ts_dim : int or None
        Required BAM move-table stride when set.
    chunking_mode : {"auto", "mv", "hierarchical", "dtw", "inferred"}, default="auto"
        Boundary selection passed to the shared read decoder. Inferred chunks
        slice sequence metadata without synthesizing per-base move events.
    resume_start_bam_by_worker : mapping, optional
        Global worker ID to BAM basename. The worker rotates its assigned list
        to start at that uniquely matching file.
    use_polished_sequence : bool, default=False
        Replace base tokens with reference-polished tokens. Unmapped records
        are skipped in this mode.
    polish_indel_limit : int, default=5
        Largest unmasked indel corrected during polishing.
    polish_vcf_path : str or None
        Variant file whose alleles polishing preserves.
    cycle : bool, default=True
        Repeat assigned BAMs indefinitely.

    Yields
    ------
    tuple
        Variable-length base tokens, ``float16`` signal, and ``float16`` rank
        percentile features. Sequences shorter than 64 tokens are omitted.
    """

    def __init__(
        self,
        bam_list,
        size,
        step,
        move_downsample_factor,
        seq_type=None,
        kmer_table=None,
        reads_per_chunk=1000,
        min_read_length=100,
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        expected_ts_dim=None,
        resume_start_bam_by_worker=None,
        use_polished_sequence=False,
        polish_indel_limit=DEFAULT_POLISH_INDEL_LIMIT,
        polish_vcf_path=None,
        cycle=True,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
        use_polished_sequence, polish_indel_limit, polish_vcf_path = validate_polishing_options(
            use_polished_sequence, polish_indel_limit, polish_vcf_path,
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
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        self.resume_start_bam_by_worker = dict(resume_start_bam_by_worker or {})
        self.use_polished_sequence = use_polished_sequence
        self.polish = use_polished_sequence
        self.polish_indel_limit = polish_indel_limit
        self.polish_vcf_path = polish_vcf_path
        self.cycle = bool(cycle)
        
        
    def _yield_buffer(self, data):
        """Pack the move-downsample dimension and yield shuffled valid chunks."""
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)
        
        signals = signals.reshape(signals.size(0), signals.size(1)//self.move_downsample_factor, signals.size(2)*self.move_downsample_factor)
        percentiles = percentiles.reshape(percentiles.size(0), percentiles.size(1)//self.move_downsample_factor, percentiles.size(2)*self.move_downsample_factor)
        
        idxs = list(range(len(data['seq'])))
        random.shuffle(idxs)
        
        for i in idxs:
            if len(data['seq'][i])>=64:
                yield data['seq'][i], signals[i].to(dtype=torch.float16), percentiles[i].to(dtype=torch.float16)
                

    def __iter__(self):
        """Iterate the current rank/worker's deterministic whole-file shard."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        
        poly_gaa='GAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAA'
        poly_tgg='TGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGG'
        
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

        resume_start_bam = self.resume_start_bam_by_worker.get(global_worker_id)
        if resume_start_bam is not None:
            matching_idxs = [
                idx
                for idx, (bam_path, _) in enumerate(my_files)
                if os.path.basename(bam_path) == resume_start_bam
            ]
            if len(matching_idxs) != 1:
                raise ValueError(
                    f"Interrupted resume expected exactly one match for BAM={resume_start_bam} "
                    f"in global_worker_id={global_worker_id}'s assigned files, found {len(matching_idxs)}."
                )
            start_idx = matching_idxs[0]
            my_files = my_files[start_idx:] + my_files[:start_idx]
        
        data = {'signal': [], 'seq': []}
        polish_variant_mask_cache = {} if self.use_polished_sequence else None
            
        for cycle_counter, (bam_path, sample) in iter_file_cycles(my_files, cycle=self.cycle):
            
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter)}',
                    flush=True,
                )

                for bam_read in bam.fetch(until_eof=True):
                    if (
                        np.random.binomial(n=1, p=sample)
                        and bam_read.flag & 0x900 == 0
                        and passes_min_qscore(bam_read, self.min_qscore)
                        and len(bam_read.seq) >= self.min_read_length
                    ):
                        if self.seq_type=='rna' and (poly_gaa in bam_read.seq or poly_tgg in bam_read.seq): continue
                        if self.use_polished_sequence and bam_read.is_unmapped:
                            continue
                            
                        read_info = get_read_info_lite(
                            bam_read, self.seq_type, self.kmer_table,
                            expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                        )
                        read = Read_Lite(*read_info)

                        try:
                            read.signal = normalize_read_signal(read, self.pore_type)
                        except DegenerateQuantileNormalizationError:
                            continue

                        if self.use_polished_sequence:
                            polished_read = get_polished_read_for_sample(
                                bam_read, self, polish_variant_mask_cache,
                            )
                            if np.any(
                                (polished_read.sequence < 0)
                                | (polished_read.sequence > 3)
                            ):
                                continue
                            (
                                signal_chunks,
                                base_seq_chunks,
                                _,
                                _,
                                _,
                            ) = get_read_chunks_labelled(
                                read,
                                self.size,
                                self.step,
                                polished_read=polished_read,
                            )
                            signal_chunks = signal_chunks.transpose(1, 2)
                        else:
                            signal_chunks, base_seq_chunks, _ = get_read_chunks(read, self.size, self.step)
                        data['signal'].append(signal_chunks)
                        data['seq'].extend(base_seq_chunks)

                        if len(data['signal']) == self.reads_per_chunk:
                            yield from self._yield_buffer(data)
                            data = {'signal': [], 'seq': []}
        if len(data['signal']) > 0:
            yield from self._yield_buffer(data)


def get_percentiles(signal: torch.Tensor, fill_value: int = -1) -> torch.Tensor:
    """Rank signal values within each window while ignoring padded rows.

    Parameters
    ----------
    signal : torch.Tensor
        Tensor with shape ``(batch, time, channels)``. A time row whose values
        are all ``-10`` is padding; the implementation checks the first column
        under this invariant.
    fill_value : int or float, default=-1
        Output assigned to padded elements.

    Returns
    -------
    torch.Tensor
        ``float32`` tensor of the same shape. Valid ranks are scaled to
        approximately ``[-0.5, 0.5]`` independently for each batch item.
    """
    B, M, N = signal.shape
    P = M * N

    # Padding is row-wise, so one fixed column is sufficient to count valid
    # time steps without reducing over the channel axis.
    P_rows = (signal[:, :, 0] != -10).sum(dim=1)        # (B,)
    n_valid = (P_rows * N)                               # (B,)
    n_invalid = (M * N) - n_valid                        # (B,)

    x = signal.view(B, P)
    idx = x.argsort(dim=1, stable=False)  # -10 padding sorts before valid values.

    pos = torch.arange(P, device=x.device, dtype=torch.float32).view(1, P)
    # Shift rank zero to the first valid element after the sorted padding block.
    pos_shift = (pos - n_invalid.view(B, 1)).clamp_min_(0)
    bins_sorted = pos_shift / n_valid.view(B, 1)  # 0..1 on the valid block
    
    bins_sorted=bins_sorted-0.5
    
    out = torch.full((B, P), fill_value, dtype=torch.float32, device=x.device)
    out.scatter_(1, idx, bins_sorted)
    return out.view(B, M, N)

def get_read_chunks(read, size, step):
    """Return unlabelled signal windows, base tokens, and raw move masks.

    Signal windows are transposed to ``(n_chunks, channels, size)``; base-token
    vectors contain one token per event and may differ in length.
    """
    if has_inferred_boundaries(read):
        signal, sequences, _, _ = inferred_chunks(read, size, step)
        return signal.transpose(1, 2), sequences, None
    signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
    signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks, = chunk_padded_components(
        signal, base_seq, move_table,  size, step
    )

    return signal_chunks.transpose(1, 2), base_seq_chunks, init_move_table_chunks

def get_polished_chunk_bounds(read, polished_read, size, step):
    """Map every signal window's original-read span into polished coordinates.

    Returns one half-open ``(start, end)`` pair per signal chunk. A window with
    no valid mapped span is represented by ``(0, 0)``.
    """
    if has_inferred_boundaries(read):
        return chunk_bounds(read, size, step, polished_read).tolist()
    _, _, _, padding_needed = pad_read_components(read, size, step)
    base_seq_idx = torch.arange(read.length)[read.indexes]
    base_seq_idx = F.pad(
        base_seq_idx,
        (0, padding_needed),
        mode='constant',
        value=read.length - 1,
    )
    base_seq_idx_chunks = base_seq_idx.unfold(dimension=0, size=size, step=step)

    bounds = []
    for chunk in base_seq_idx_chunks:
        p_start, p_end = polished_read.get_coordinates(
            int(chunk[0]), int(chunk[-1]) + 1,
        )
        if p_start < 0 or p_end < 0:
            p_start = p_end = 0
        bounds.append((p_start, p_end))
    return bounds


def get_read_chunks_labelled(read, size, step, polished_read=None):
    """Chunk an aligned read and retain reference and read coordinates.

    Parameters
    ----------
    read : porefm.data.bam.Read or compatible object
        Decoded read with signal, move table, base sequence, and alignment.
    size, step : int
        Sliding-window length and offset in signal time steps.
    polished_read : PolishedRead or None
        Optional replacement sequence and coordinate maps.

    Returns
    -------
    tuple
        Signal windows, sequence chunks, original move-table windows, padded
        zero-based reference positions, and padded read/polished positions.
        Reference padding is ``-2`` and read-position padding is ``-1``. In the
        unpolished path sequence chunks are returned as one padded tensor; in
        the polished path they remain a list of variable-length tensors.
    """
    if has_inferred_boundaries(read):
        return inferred_labelled_chunks(read, size, step, polished_read)
    signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
    signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks = chunk_padded_components(
        signal, base_seq, move_table, size, step
    )

    if polished_read is not None:
        bounds = get_polished_chunk_bounds(read, polished_read, size, step)
        polished_seq_chunks = [
            torch.from_numpy(polished_read.sequence[p_start:p_end].copy() + 1)
            for p_start, p_end in bounds
        ]
        polished_ref_chunks = [
            torch.from_numpy(polished_read.p_to_ref[p_start:p_end].copy())
            for p_start, p_end in bounds
        ]
        polished_pos_chunks = [
            torch.arange(p_start, p_end, dtype=torch.long)
            for p_start, p_end in bounds
        ]
        return (
            signal_chunks,
            polished_seq_chunks,
            init_move_table_chunks,
            pad_sequence(polished_ref_chunks, batch_first=True, padding_value=-2),
            pad_sequence(polished_pos_chunks, batch_first=True, padding_value=-1),
        )


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
    read_wise_aligned_pos_chunks=[read_wise_aligned_pos_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
    read_wise_aligned_pos_chunks_padded = pad_sequence(read_wise_aligned_pos_chunks, batch_first=True, padding_value=-2).type(torch.LongTensor)
    
    seqs_batch_padded = pad_sequence(base_seq_chunks, batch_first=True, padding_value=0)

    base_seq_idx=torch.arange(read.length)[read.indexes]
    base_seq_idx = F.pad(
        base_seq_idx, 
        (0,  padding_needed), 
        mode='constant', 
        value=read.length-1
    )

    base_seq_idx_chunks=base_seq_idx.unfold(dimension=0, size=size, step=step)
    read_pos_chunks = [base_seq_idx_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
    read_pos_chunks_padded =  pad_sequence(read_pos_chunks, batch_first=True, padding_value=-1)
    
    return signal_chunks, seqs_batch_padded, init_move_table_chunks, read_wise_aligned_pos_chunks_padded, read_pos_chunks_padded



def get_read_chunks_read_based(read, size, step):
    """Chunk a read and retain read-base indices instead of reference indices.

    Returns signal windows, padded sequence chunks, raw move-table windows,
    and a padded matrix of zero-based read positions. Position padding is
    ``-1``.
    """
    if has_inferred_boundaries(read):
        signal, sequences, _, _, positions = inferred_labelled_chunks(read, size, step)
        return signal, sequences, None, positions
    signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
    signal_chunks, base_seq_chunks, init_move_table_chunks, move_table_chunks = chunk_padded_components(
        signal, base_seq, move_table, size, step
    )

    seqs_batch_padded = pad_sequence(base_seq_chunks, batch_first=True, padding_value=0)

    base_seq_idx = torch.arange(read.length)[read.indexes]
    base_seq_idx = F.pad(
        base_seq_idx,
        (0, padding_needed),
        mode='constant',
        value=read.length - 1,
    )
    base_seq_idx_chunks = base_seq_idx.unfold(dimension=0, size=size, step=step)
    read_pos_chunks = [
        base_seq_idx_chunks[i][move_table_chunks[i]]
        for i in range(len(move_table_chunks))
    ]
    read_pos_chunks_padded = pad_sequence(
        read_pos_chunks, batch_first=True, padding_value=-1,
    ).type(torch.LongTensor)

    return (
        signal_chunks,
        seqs_batch_padded,
        init_move_table_chunks,
        read_pos_chunks_padded,
    )


def _labels_for_target_positions(
    query_positions,
    target_positions,
    target_values,
    num_classes,
):
    """Match supplied targets without smoothing; non-hits remain masked out."""

    if len(target_positions) == 0:
        labels = torch.full(
            (*query_positions.shape, num_classes),
            0.0,
            dtype=torch.float32,
            device=query_positions.device,
        )
        labels[..., 0] = 1.0
        labels_idx = torch.zeros_like(query_positions, dtype=torch.bool)
        return labels, labels_idx

    query_pos_np = query_positions.numpy()
    target_idx = np.searchsorted(target_positions, query_pos_np)
    target_idx = np.clip(target_idx, 0, len(target_positions) - 1)
    valid_mask_np = target_positions[target_idx] == query_pos_np

    labels_np = np.full(
        (*query_pos_np.shape, num_classes), 0.0, dtype=np.float32,
    )
    labels_np[..., 0] = 1.0

    if len(target_values) > 0 and np.any(valid_mask_np):
        matched_targets = target_values[target_idx[valid_mask_np]]
        if matched_targets.ndim != 2 or matched_targets.shape[1] != num_classes:
            raise ValueError(
                f"Expected target values with shape N x {num_classes}; "
                f"got {matched_targets.shape}. Check pos_file label columns "
                "or motif_label/num_classes configuration."
            )
        labels_np[valid_mask_np] = matched_targets

    return torch.from_numpy(labels_np), torch.from_numpy(valid_mask_np)


def align_chunk_fields_to_sequences(sequences, *fields):
    """Trim padded per-base fields to the matching sequence lengths.

    Each field must have one row per sequence and every row must be at least as
    long as its sequence. The return value is ``(sequences, *aligned_fields)``
    with each field represented as a list of trimmed rows. ``ValueError``
    identifies inconsistent row counts or lengths.
    """
    sequences = list(sequences)
    aligned_fields = []

    for field_index, field in enumerate(fields):
        rows = list(field)
        if len(rows) != len(sequences):
            raise ValueError(
                f"Chunk field {field_index} has {len(rows)} rows for "
                f"{len(sequences)} sequences."
            )

        aligned_rows = []
        for chunk_index, (sequence, row) in enumerate(zip(sequences, rows)):
            sequence_length = len(sequence)
            if len(row) < sequence_length:
                raise ValueError(
                    f"Chunk field {field_index} row {chunk_index} has length "
                    f"{len(row)}, shorter than its sequence length {sequence_length}."
                )
            aligned_rows.append(row[:sequence_length])
        aligned_fields.append(aligned_rows)

    return (sequences, *aligned_fields)


def prepare_labelled_chunks(
    read,
    labelled_arrays,
    size,
    step,
    num_classes,
    pore_type=None,
    discard_empty_labels = False,
    sample_labels = 1,
    keep_all=False,
    polished_read=None,
):
    """Normalize, chunk, and attach reference-position targets to a read.

    Parameters
    ----------
    read : porefm.data.bam.Read
        Aligned decoded read. Its ``signal`` attribute is replaced by the
        normalized signal as a side effect.
    labelled_arrays : dict
        Sorted targets from :func:`porefm.data.bam.get_labelled_arrays`.
    size, step : int
        Window length and offset in signal time steps.
    num_classes : int
        Width of every per-base target vector.
    pore_type : str or None
        Chemistry selector passed to signal normalization.
    discard_empty_labels : bool, default=False
        Return ``None`` when no chunk contains a retained labelled position.
    sample_labels : float, default=1
        Independent probability of retaining each label occurrence after
        chunking. Overlapping chunks sample repeated occurrences separately.
    keep_all : bool, default=False
        Keep all chunks and preserve untransposed signal windows, regardless of
        whether they contain a retained label.
    polished_read : PolishedRead or None
        Optional polished sequence and coordinate maps.

    Returns
    -------
    dict or None
        Signal, variable-length sequence, target, supervision-mask, reference-
        position, and read-position chunks. In normal mode signal has shape
        ``(n_chunks, channels, size)``; ``None`` is possible only when empty
        chunks are discarded.
    """
    read.signal = normalize_read_signal(read, pore_type)

    signal_chunks, base_seq_chunks,  _, pos_chunks_padded, read_pos_chunks = get_read_chunks_labelled(
        read, size, step, polished_read=polished_read,
    )

    target_pos, target_vals = labelled_arrays[read.ref_name][read.is_reverse]
    
    if len(target_pos) == 0:
        labels_torch = torch.full(
            (*pos_chunks_padded.shape, num_classes), 
            0.0,
            dtype=torch.float32,
            device=pos_chunks_padded.device
        )
        
        labels_torch[..., 0] = 1.0
        labels_idx = torch.zeros_like(pos_chunks_padded, dtype=torch.bool)
        
    else:
        query_pos = pos_chunks_padded.numpy()
        
        idx = np.searchsorted(target_pos, query_pos)

        # searchsorted may return the insertion point after the final target;
        # clipping permits vectorized equality validation below.
        idx = np.clip(idx, 0, len(target_pos) - 1)
        valid_mask_np = (target_pos[idx] == query_pos)

        labels_np = np.zeros((*query_pos.shape, num_classes), dtype=np.float32)

        # Unlabelled positions carry a background-shaped value but do not
        # contribute loss because ``labels_idx`` remains false there.
        labels_np[..., 0] = 1.0
        
        # Fill only the valid hits. target_vals is already shaped N x K
        # regardless of whether labels came from a multi-column position file
        # or from motif labels expanded by get_labelled_arrays.
        if len(target_vals) > 0 and np.any(valid_mask_np):
            matched_targets = target_vals[idx[valid_mask_np]]
            if matched_targets.ndim != 2 or matched_targets.shape[1] != num_classes:
                raise ValueError(
                    f"Expected target values with shape N x {num_classes}; "
                    f"got {matched_targets.shape}. Check pos_file label columns "
                    "or motif_label/num_classes configuration."
                )
            labels_np[valid_mask_np] = matched_targets
            
        labels_torch = torch.from_numpy(labels_np)
        labels_idx = torch.from_numpy(valid_mask_np)


    if sample_labels<1:
        labels_idx=labels_idx & (torch.rand(labels_idx.shape) <= sample_labels)

    if keep_all:
        seqs, labels, aligned_labels_idx, positions, aligned_read_pos = (
            align_chunk_fields_to_sequences(
                base_seq_chunks,
                labels_torch.float(),
                labels_idx,
                pos_chunks_padded,
                read_pos_chunks,
            )
        )
        return {
            'signal': signal_chunks,
            'seq': seqs,
            'labels': labels,
            'labels_idx': aligned_labels_idx,
            'position': positions,
            'read_pos': aligned_read_pos,
        }
        
    keep = labels_idx.sum(1) > 0
    
    if not keep.any():
        if discard_empty_labels:
            return None
        
        else:
            keep=torch.ones_like(keep, dtype=bool)
    
    keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
    signals = signal_chunks[keep].transpose(1, 2)
    seqs, labels, labels_idx, positions, read_pos = align_chunk_fields_to_sequences(
        [base_seq_chunks[i] for i in keep_idx],
        labels_torch[keep].float(),
        labels_idx[keep],
        pos_chunks_padded[keep],
        read_pos_chunks[keep],
    )
    
    return {
        'signal': signals,
        'seq': seqs,
        'labels': labels,
        'labels_idx': labels_idx,
        'position': positions,
        'read_pos': read_pos,
        }



def prepare_read_based_labelled_chunks(
    read,
    read_sequence,
    parsed_motifs,
    size,
    step,
    num_classes,
    pore_type=None,
    discard_empty_labels=False,
    sample_labels=1,
    keep_all=False,
    polished_read=None,
):
    """Prepare chunks using motif targets found in read coordinates.

    This is the reference-free counterpart to :func:`prepare_labelled_chunks`.
    ``read_sequence`` must follow the same 5'-to-3' orientation as ``read``;
    when ``polished_read`` is supplied it should be the polished sequence used
    to derive that string. Remaining options and the returned dictionary have
    the same semantics as the reference-based function, except ``position``
    and ``read_pos`` both contain read or polished-read coordinates.
    """
    read.signal = normalize_read_signal(read, pore_type)

    if polished_read is None:
        signal_chunks, base_seq_chunks, _, read_pos_chunks = get_read_chunks_read_based(
            read, size, step,
        )
    else:
        signal_chunks, base_seq_chunks, _, _, read_pos_chunks = get_read_chunks_labelled(
            read, size, step, polished_read=polished_read,
        )
    target_pos, target_vals = get_read_motif_targets(
        read_sequence=read_sequence,
        parsed_motifs=parsed_motifs,
        num_classes=num_classes,
    )
    labels_torch, labels_idx = _labels_for_target_positions(
        query_positions=read_pos_chunks,
        target_positions=target_pos,
        target_values=target_vals,
        num_classes=num_classes,
    )

    if sample_labels < 1:
        labels_idx = labels_idx & (torch.rand(labels_idx.shape) <= sample_labels)

    if keep_all:
        seqs, labels, aligned_labels_idx, positions, aligned_read_pos = (
            align_chunk_fields_to_sequences(
                base_seq_chunks,
                labels_torch.float(),
                labels_idx,
                read_pos_chunks,
                read_pos_chunks,
            )
        )
        return {
            'signal': signal_chunks,
            'seq': seqs,
            'labels': labels,
            'labels_idx': aligned_labels_idx,
            'position': positions,
            'read_pos': aligned_read_pos,
        }

    keep = labels_idx.sum(1) > 0

    if not keep.any():
        if discard_empty_labels:
            return None
        keep = torch.ones_like(keep, dtype=bool)

    keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
    seqs, labels, aligned_labels_idx, positions, aligned_read_pos = (
        align_chunk_fields_to_sequences(
            [base_seq_chunks[i] for i in keep_idx],
            labels_torch[keep].float(),
            labels_idx[keep],
            read_pos_chunks[keep],
            read_pos_chunks[keep],
        )
    )

    return {
        'signal': signal_chunks[keep].transpose(1, 2),
        'seq': seqs,
        'labels': labels,
        'labels_idx': aligned_labels_idx,
        'position': positions,
        'read_pos': aligned_read_pos,
    }


def get_mapped_chroms(bam_path):
    """Return indexed BAM contigs whose mapped-record count is nonzero."""
    with pysam.AlignmentFile(bam_path, "rb") as samfile:
        stats = samfile.get_index_statistics()
        active_chroms = set([s.contig for s in stats if s.mapped > 0])
    return active_chroms


def get_polished_read_for_sample(bam_read, bam_sample, variant_mask_cache):
    """Build a polished read from sample settings and a worker-local VCF cache.

    Returns ``None`` when ``bam_sample.polish`` is false. Otherwise the sample
    must provide ``polish_indel_limit`` and may provide ``polish_vcf_path``.
    """
    if not getattr(bam_sample, 'polish', False):
        return None

    from porefm.data.polishing import polish_read_with_maps

    variant_mask = get_polishing_variant_mask(bam_sample, variant_mask_cache)
    return polish_read_with_maps(
        bam_read,
        variant_mask=variant_mask,
        indel_lim=bam_sample.polish_indel_limit,
    )
    
class LabelledBamSample:
    """Configuration for one motif- or position-labelled BAM source.

    Parameters
    ----------
    bam_path : str
        BAM file path. Reference-based SG modes require mapped, indexed input;
        unscoped read-based mode may stream unmapped or unindexed records.
    chrom_info : list
        Optional ``(contig, start, end)`` regions in zero-based half-open
        coordinates. Regions require indexed fetching even in read-based mode.
    ref : str or None
        Indexed reference FASTA for reference-based labels. It is not consulted
        by unscoped read-based motif discovery.
    num_classes : int
        Target-vector width.
    pos_file : str or None
        Explicit zero-based reference-position table. Incompatible with
        ``read_based=True``.
    motif : str or None
        Semicolon-delimited motif specifications.
    motif_label : str or int, default=0
        Class indices assigned to motifs.
    label_smoothing : float, default=0
        Classification smoothing epsilon, carried as training metadata only.
        Targets are left unchanged until the supervised loss is computed.
    flag : int, default=0
        Additional SAM flag bits to exclude.
    sample_labels : float, default=1
        Per-chunk-occurrence probability of retaining each target.
    polish : bool, default=False
        Enable reference-guided sequence polishing.
    read_based : bool, default=False
        Discover motifs in each oriented read rather than the reference.
    polish_indel_limit : int, default=5
        Largest unmasked indel corrected during polishing.
    polish_vcf_path : str or None
        Variant file whose alleles polishing preserves.
    pod5 : str or None
        Inference-only raw-signal file or recursive directory. When provided,
        inference traverses POD5 order and does not need a coordinate BAM index.
    bam_index : str or None
        Optional unfiltered read/parent-ID index produced by ``porefm utils index-bam``
        for POD5 inference. Filtering remains an inference-time operation.
    """

    def __init__(self, bam_path: str, chrom_info: List[str], ref: str, num_classes: int,
                 pos_file: Optional[str] = None, motif: Optional[str] = None, motif_label: Optional[str] = 0, label_smoothing=0,
                 flag=0, sample_labels=1, polish=False, read_based=False,
                 polish_indel_limit=DEFAULT_POLISH_INDEL_LIMIT,
                 polish_vcf_path=None, pod5=None, bam_index=None):
        polish, polish_indel_limit, polish_vcf_path = validate_polishing_options(
            polish, polish_indel_limit, polish_vcf_path,
        )
        if not isinstance(read_based, bool):
            raise TypeError(f"read_based must be true or false, got {read_based!r}")
        if read_based and pos_file:
            raise ValueError("read_based samples use motif/motif_label and cannot also specify pos_file")
        if read_based and not motif:
            raise ValueError("read_based samples must specify motif")

        self.bam_path = bam_path
        self.chrom_info = tuple(sorted(chrom_info))
        self.ref = ref
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.pos_file = pos_file
        self.motif = motif
        self.motif_label = motif_label
        self.flag = flag
        self.sample_labels = sample_labels
        self.polish = polish
        self.polish_indel_limit = polish_indel_limit
        self.polish_vcf_path = polish_vcf_path
        self.read_based = read_based
        self.pod5 = pod5
        self.bam_index = bam_index

class LabelledBamIterableDataset(IterableDataset):
    """Stream supervised chunks from labelled nanopore BAM samples.

    The dataset supports reference position files, reference motif discovery,
    and reference-free read motif discovery. BAM files are sharded whole across
    distributed ranks and DataLoader workers. Reference preprocessing and VCF
    masks are cached per worker, normalized read chunks are buffered, and only
    sequences of at least 64 tokens are emitted.

    ``size`` and ``step`` are model-level units multiplied by
    ``move_downsample_factor`` before chunking. Other common loader parameters
    match :class:`BamIterableDataset`; ``shuffle`` controls within-buffer order,
    ``discard_empty_labels`` drops reads without supervised chunks, and
    ``cycle`` repeats assigned samples. Yielded 13-item records contain sequence,
    signal, percentiles, targets, target mask, positions, coordinate frontier,
    worker ID, strand, read name, and reference name.

    ``include_label_smoothing=True`` wraps each record as ``(record, epsilon)``
    for supervised training. The default preserves the shared record contract.
    Epsilon follows its chunk through buffers that may span multiple BAMs.

    ``exclude_bed`` optionally supplies a ``chrom start end [strand]``
    file for all input BAMs. Same-strand reference-span overlaps are skipped
    before signal decoding. Coordinates are zero-based, half-open; strand is
    ``+`` or ``-``, or omitted to match both. The default ``None`` disables this filter.
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
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        shuffle=True,
        discard_empty_labels=True,
        cycle=True,
        expected_ts_dim=None,
        exclude_bed=None,
        include_label_smoothing=False,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
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
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.shuffle=shuffle
        self.cycle=cycle
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        self.exclude_bed = exclude_bed
        self.include_label_smoothing = include_label_smoothing
        
        self.discard_empty_labels = discard_empty_labels
        
    def _yield_buffer(self, data):
        """Compute percentiles, optionally shuffle, and emit buffered chunks."""
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)
            
        signals = signals.reshape(signals.size(0), signals.size(1)//self.move_downsample_factor, signals.size(2)*self.move_downsample_factor)
        percentiles = percentiles.reshape(percentiles.size(0), percentiles.size(1)//self.move_downsample_factor, percentiles.size(2)*self.move_downsample_factor)
        
        idxs = list(range(len(data['seq'])))
        
        if self.shuffle:
            random.shuffle(idxs)
        
        for i in idxs:
            if len(data['seq'][i])>=64:
                item = (
                    data['seq'][i],
                    signals[i].to(dtype=torch.float16),
                    percentiles[i].to(dtype=torch.float16),
                    data['labels'][i],
                    data['labels_idx'][i],
                    data['position'][i],
                    data['frontier_chrom'],
                    data['frontier_start'],
                    data['frontier_chrom_idx'],
                    data['rank'],
                    data['strand'][i],
                    data['read_name'][i],
                    data['ref_name'][i],
                )
                if self.include_label_smoothing:
                    yield item, data['label_smoothing'][i]
                else:
                    yield item

                


    def _iter_read_based_sample(
        self,
        bam_sample,
        parsed_motifs,
        global_rank,
        cycle_counter,
        worker_id,
        poly_gaa,
        poly_tgg,
        polish_variant_mask_cache,
    ):
        """Yield read-motif chunks, avoiding a FASTA for unscoped streams.

        Explicit ``chrom_info`` switches to indexed regional fetching and adds
        unmapped records to the excluded flags. Without regions, records are
        streamed through ``fetch(until_eof=True)``.
        """
        bam_path = bam_sample.bam_path
        chrom_info = bam_sample.chrom_info
        flag_info = bam_sample.flag
        excluded_regions = get_excluded_regions(self.exclude_bed)

        with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
            print(
                f'{str(datetime.datetime.now())}: '
                f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter)}',
                flush=True,
            )

            if chrom_info:
                active_chroms = set(bam.references) & {item[0] for item in chrom_info}
                region_specs = [
                    (contig_idx, contig, start, end)
                    for contig_idx, (contig, start, end) in enumerate(chrom_info)
                    if contig in active_chroms
                ]
                excluded_flags = 0x904 | flag_info
            else:
                region_specs = [(None, None, None, None)]
                excluded_flags = 0x900 | flag_info

            for contig_idx, contig, start, end in region_specs:
                if contig is None:
                    bam_reads = bam.fetch(until_eof=True)
                else:
                    print(
                        f'{str(datetime.datetime.now())}: '
                        f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter, contig=contig)}',
                        flush=True,
                    )
                    bam_reads = bam.fetch(
                        contig, start=start, end=end, until_eof=True,
                    )

                for bam_read in bam_reads:
                    if excluded_regions is not None and excluded_regions.overlaps_read(bam_read):
                        continue
                    if (
                        bam_read.flag & excluded_flags != 0
                        or not passes_min_qscore(bam_read, self.min_qscore)
                        or len(bam_read.seq) < self.min_read_length
                    ):
                        continue
                    if self.seq_type == 'rna' and (
                        poly_gaa in bam_read.seq or poly_tgg in bam_read.seq
                    ):
                        continue

                    read_sequence = (
                        bam_read.seq
                        if bam_read.is_forward
                        else revcomp(bam_read.seq)
                    )
                    read_info = get_read_info_lite(
                        bam_read,
                        self.seq_type,
                        self.kmer_table,
                        expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                        oriented_sequence=read_sequence,
                    )
                    read = Read_Lite(*read_info)

                    polished_read = (
                        get_polished_read_for_sample(
                            bam_read, bam_sample, polish_variant_mask_cache,
                        )
                        if bam_sample.polish
                        else None
                    )
                    if polished_read is not None:
                        read_sequence = ''.join(
                            num_to_base_map.get(int(base), 'N')
                            for base in polished_read.sequence
                        )

                    try:
                        processed = prepare_read_based_labelled_chunks(
                            read=read,
                            read_sequence=read_sequence,
                            parsed_motifs=parsed_motifs,
                            size=self.size,
                            step=self.step,
                            num_classes=bam_sample.num_classes,
                            pore_type=self.pore_type,
                            discard_empty_labels=self.discard_empty_labels,
                            sample_labels=bam_sample.sample_labels,
                            keep_all=False,
                            polished_read=polished_read,
                        )
                    except DegenerateQuantileNormalizationError:
                        continue
                    if processed is None:
                        continue

                    yield (
                        processed,
                        0 if bam_read.is_forward else 1,
                        read.read_name,
                        bam_read.reference_name or 'unknown',
                    )

    def __iter__(self):
        """Yield this worker's labelled chunks across repeated BAM cycles."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
                
        poly_gaa='GAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAA'
        poly_tgg='TGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGG'
        
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
        
        ref_seq_cache = {}
        labelled_arrays_cache = {}
        polish_variant_mask_cache = {}
        
        read_motifs_cache = {}
        data = _new_labelled_buffer()
        
        for cycle_counter, bam_sample in iter_file_cycles(my_files, self.cycle):
            
            if bam_sample.read_based:
                read_motif_key = (
                    bam_sample.motif,
                    bam_sample.motif_label,
                    bam_sample.num_classes,
                )
                if read_motif_key not in read_motifs_cache:
                    read_motifs_cache[read_motif_key] = parse_motif_specs(
                        motif=bam_sample.motif,
                        motif_label=bam_sample.motif_label,
                        num_classes=bam_sample.num_classes,
                    )
                parsed_motifs = read_motifs_cache[read_motif_key]

                for processed, strand_val, read_name_str, ref_name_str in self._iter_read_based_sample(
                    bam_sample=bam_sample,
                    parsed_motifs=parsed_motifs,
                    global_rank=global_rank,
                    cycle_counter=cycle_counter,
                    worker_id=worker_id,
                    poly_gaa=poly_gaa,
                    poly_tgg=poly_tgg,
                    polish_variant_mask_cache=polish_variant_mask_cache,
                ):
                    data['signal'].append(processed['signal'])
                    data['seq'].extend(processed['seq'])
                    data['labels'].extend(processed['labels'])
                    data['labels_idx'].extend(processed['labels_idx'])
                    data['label_smoothing'].extend([bam_sample.label_smoothing] * len(processed['seq']))
                    data['position'].extend(processed['position'])
                    data['strand'].extend([strand_val] * len(processed['position']))
                    data['read_name'].extend([read_name_str] * len(processed['position']))
                    data['ref_name'].extend([ref_name_str] * len(processed['position']))

                    if len(data['signal']) == self.reads_per_chunk:
                        data['rank'] = global_worker_id
                        yield from self._yield_buffer(data)
                        data = _new_labelled_buffer()
                continue


            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info, sample_labels = bam_sample.flag, bam_sample.sample_labels
            excluded_regions = get_excluded_regions(self.exclude_bed)
            chroms=tuple(x[0] for x in chrom_info)

            num_classes = bam_sample.num_classes
            active_chroms = get_mapped_chroms(bam_path)&set(chroms)
            if len(active_chroms)==0:
                continue
                
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter)}',
                    flush=True,
                )
                
                if bam_sample.ref in ref_seq_cache:
                    ref_seq_dict = ref_seq_cache[bam_sample.ref]
                else:
                    ref_seq_dict = get_ref_seq_dict(ref=bam_sample.ref, chrom_list=chroms)
                    ref_seq_cache[bam_sample.ref] = ref_seq_dict
                labelled_cache_key = (
                    bam_sample.ref,
                    self.seq_type,
                    bam_sample.pos_file,
                    bam_sample.motif,
                    bam_sample.motif_label,
                    num_classes,
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
                        num_classes=num_classes
                    )
                    labelled_arrays_cache[labelled_cache_key] = labelled_arrays
                for contig_idx, contig_info in enumerate(chrom_info):
                    contig, start, end = contig_info
                    if contig not in active_chroms:
                        continue
                    
                    print(
                        f'{str(datetime.datetime.now())}: '
                        f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter, contig=contig)}',
                        flush=True,
                    )

                    for bam_read in bam.fetch(contig, start=start, end=end, until_eof=True):
                        if excluded_regions is not None and excluded_regions.overlaps_read(bam_read):
                            continue
                        if (bam_read.flag & (0x904|flag_info) == 0
                            and passes_min_qscore(bam_read, self.min_qscore)
                            and len(bam_read.seq) >= self.min_read_length
                        ):
                            if self.seq_type=='rna' and (poly_gaa in bam_read.seq or poly_tgg in bam_read.seq): continue
                            
                            read_info = get_read_info(
                                bam_read, ref_seq_dict, self.seq_type, self.kmer_table,
                                expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                            )
                            read = Read(*read_info)
                            polished_read = (
                                get_polished_read_for_sample(
                                    bam_read, bam_sample, polish_variant_mask_cache,
                                )
                                if bam_sample.polish
                                else None
                            )
                            
                            strand_val = 0 if read.is_forward else 1 
                            read_name_str = read.read_name
                            ref_name_str = read.ref_name

                            try:
                                processed = prepare_labelled_chunks(
                                    read,
                                    labelled_arrays,
                                    self.size,
                                    self.step,
                                    num_classes,
                                    pore_type=self.pore_type,
                                    discard_empty_labels=self.discard_empty_labels,
                                    sample_labels=sample_labels, keep_all=False,
                                    polished_read=polished_read,
                                )
                            except DegenerateQuantileNormalizationError:
                                continue

                            if processed is not None:
                                data['signal'].append(processed['signal'])
                                data['seq'].extend(processed['seq'])
                                data['labels'].extend(processed['labels'])
                                data['labels_idx'].extend(processed['labels_idx'])
                                data['label_smoothing'].extend([bam_sample.label_smoothing] * len(processed['seq']))
                                data['position'].extend(processed['position'])
                                data['strand'].extend([strand_val] * len(processed['position']))
                                data['read_name'].extend([read_name_str] * len(processed['position']))
                                data['ref_name'].extend([ref_name_str] * len(processed['position']))
                                _update_labelled_buffer_frontier(
                                    data,
                                    contig,
                                    contig_idx,
                                    bam_read.reference_start,
                                )
                                
                            if len(data['signal']) == self.reads_per_chunk:
                                data['rank']=global_worker_id
                                yield from self._yield_buffer(data)
                                data = _new_labelled_buffer()
        if len(data['signal']) > 0:
            data['rank']=global_worker_id
            yield from self._yield_buffer(data)

class WeightedLabelledBamIterableDataset(IterableDataset):
    """Randomly multiplex labelled or tagged dataset groups by source weight.

    Parameters
    ----------
    bam_lists : list of list
        Homogeneous sample groups. Each non-empty group must contain either
        :class:`LabelledBamSample` or :class:`TaggedBamSample` objects.
    group_weights : list of float
        Nonnegative relative probabilities, one per group. At least one must be
        positive; values are normalized internally.
    **kwargs
        Loader options forwarded to every underlying dataset constructor.

    Notes
    -----
    Weighting chooses among already processed dataset streams. It does not
    directly balance BAMs, reads, labels, or classes. A finite stream that is
    exhausted is removed and the remaining probabilities are renormalized.
    """
    def __init__(
        self,
        bam_lists: list[list['LabelledBamSample'] | list['TaggedBamSample']],
        group_weights: list[float],
        **kwargs
    ):
        super().__init__()
        if len(bam_lists) == 0:
            raise ValueError("bam_lists must contain at least one BAM list")
        if len(bam_lists) != len(group_weights):
            raise ValueError("bam_lists and group_weights must have the same length")
        if any(w < 0 for w in group_weights):
            raise ValueError("group_weights cannot contain negative values")
        if sum(group_weights) <= 0:
            raise ValueError("group_weights must contain at least one positive value")

        self.bam_lists = bam_lists
        
        total_weight = sum(group_weights)
        self.group_probs = [float(w) / total_weight for w in group_weights]
        
        self.dataset_kwargs = kwargs

    def _iter_grouped(self):
        """Yield ``(group_id, item)`` from randomly selected active streams."""

        # Sub-datasets inspect worker_info themselves, preserving their normal
        # rank/worker whole-file sharding under composition.
        DATASET_MAP = {
            LabelledBamSample: LabelledBamIterableDataset,
            TaggedBamSample: TaggedBamIterableDataset,
        }
        
        datasets=[]
        dataset_group_ids=[]
        dataset_probs=[]
        for group_id, bl in enumerate(self.bam_lists):
            if bl:
                sample_type = type(bl[0])
                ds_class = DATASET_MAP.get(sample_type)
                if ds_class:
                    datasets.append(ds_class(bam_list=bl, **self.dataset_kwargs))
                    dataset_group_ids.append(group_id)
                    dataset_probs.append(self.group_probs[group_id])

        iters = [iter(ds) for ds in datasets]
        
        active_indices = list(range(len(iters)))
        probs = list(dataset_probs)
        
        while active_indices:
            # Repeated weighted selection approaches the configured mixture
            # over time without requiring equal-sized source streams.
            idx_idx = random.choices(range(len(active_indices)), weights=probs, k=1)[0]
            idx = active_indices[idx_idx]
            
            try:
                yield dataset_group_ids[idx], next(iters[idx])
            except StopIteration:
                active_indices.pop(idx_idx)
                probs.pop(idx_idx)

                if sum(probs) > 0:
                    total = sum(probs)
                    probs = [p / total for p in probs]
                else:
                    break

    def __iter__(self):
        """Yield mixed records while hiding their source-group identifiers."""
        for _group_id, item in self._iter_grouped():
            yield item


class SupervisedWeightedLabelledBamIterableDataset(WeightedLabelledBamIterableDataset):
    """Mix source groups while retaining IDs for supervision-aware training.

    Each value is ``(dataset_item, group_id, label_smoothing)``. Metadata stays
    outside the shared 13-item payload; targets remain unsmoothed for metrics.
    """

    def __init__(self, bam_lists, group_weights, **kwargs):
        super().__init__(
            bam_lists, group_weights, include_label_smoothing=True, **kwargs,
        )

    def __iter__(self):
        for group_id, (item, label_smoothing) in self._iter_grouped():
            yield item, group_id, label_smoothing

import torch

def prepare_tagged_chunks(
    read,
    target_info,
    size,
    step,
    use_ML_tag,
    num_classes,  
    pore_type=None,
    discard_empty_labels = False,
    polished_read=None,
    *,
    label_coordinates=None,
):
    """Normalize and label chunks from per-read targets or decoded MM/ML calls.

    Parameters
    ----------
    read : porefm.data.bam.Read or Read_Lite
        Decoded read. Reference coordinates require ``Read``; read coordinates
        also support ``Read_Lite``. Signal is normalized in place.
    target_info : tuple
        ``(target_values, target_positions)``. Dictionary positions must be
        sorted, unique, nonnegative integer offsets in the selected coordinate
        system. Prefer
        ``(N, num_classes)`` target arrays; a one-dimensional array of length
        ``num_classes`` is a shared vector, not per-position class indices.
    size, step : int
        Window length and offset in signal time steps.
    use_ML_tag : bool
        Select scalar class labels from decoded MM/ML calls. Dictionary
        targets accept scalar classes or probability vectors in either mode.
    num_classes : int
        Target-vector width.
    label_coordinates : {"read", "ref"} or None
        Required for dictionary labels: zero-based original 5'-to-3' read
        offsets or reference positions. Omit for MM/ML calls, whose existing
        read-coordinate matching and reference-position output are unchanged.
        Polishing remaps original read offsets.
    pore_type : str or None
        Chemistry selector passed to signal normalization.
    discard_empty_labels : bool, default=False
        Return ``None`` if no chunk contains a target.
    polished_read : PolishedRead or None
        Optional polished sequence. MM/ML read positions removed by polishing
        are discarded and retained positions are remapped.

    Returns
    -------
    dict or None
        Signal, sequence, target, target-mask, and position chunks. ``position``
        follows dictionary ``label_coordinates`` (polished-read offsets when
        polishing). MM/ML output positions remain reference coordinates.
        Only target-containing chunks are retained unless empty-label discard
        is disabled and the read has targets outside all chunks.
    """
    if not use_ML_tag and label_coordinates not in ("read", "ref"):
        raise ValueError("label_coordinates must be 'read' or 'ref'")
    if use_ML_tag and label_coordinates is not None:
        raise ValueError("label_coordinates applies only to dictionary labels")
    read_based = not use_ML_tag and label_coordinates == "read"
    read.signal = normalize_read_signal(read, pore_type)

    if read_based and polished_read is None:
        signal_chunks, base_seq_chunks, _, read_pos_chunks = get_read_chunks_read_based(
            read, size, step,
        )
        query_pos = read_pos_chunks
    else:
        signal_chunks, base_seq_chunks, _, pos_chunks_padded, read_pos_chunks = get_read_chunks_labelled(
            read, size, step, polished_read=polished_read,
        )
        query_pos = read_pos_chunks if (use_ML_tag or read_based) else pos_chunks_padded
    output_positions = read_pos_chunks if read_based else pos_chunks_padded
    
    target_val_np, target_pos_np = target_info
    target_pos_np = np.atleast_1d(target_pos_np)
    if not use_ML_tag and (target_pos_np.ndim != 1 or (
        len(target_pos_np) and (
            not np.issubdtype(target_pos_np.dtype, np.integer)
            or np.any(target_pos_np < 0)
            or np.any(target_pos_np[1:] <= target_pos_np[:-1])
        )
    )):
        raise ValueError("Tagged target positions must be sorted, unique, nonnegative integers")

    if (use_ML_tag or read_based) and polished_read is not None and len(target_pos_np) > 0:
        original_target_pos = target_pos_np.astype(np.int64, copy=False)
        in_range = (
            (original_target_pos >= 0)
            & (original_target_pos < len(polished_read.p_from_read))
        )
        mapped_target_pos = np.full(len(original_target_pos), -1, dtype=np.int64)
        mapped_target_pos[in_range] = polished_read.p_from_read[original_target_pos[in_range]]
        retained = mapped_target_pos >= 0
        target_pos_np = mapped_target_pos[retained]
        target_val_arr = np.asarray(target_val_np)
        shared_vector = not use_ML_tag and target_val_arr.shape == (num_classes,)
        if not shared_vector and target_val_arr.ndim > 0 and len(target_val_arr) == len(retained):
            target_val_np = target_val_arr[retained]
        order = np.argsort(target_pos_np)
        target_pos_np = target_pos_np[order]
        target_val_arr = np.asarray(target_val_np)
        if not shared_vector and target_val_arr.ndim > 0 and len(target_val_arr) == len(order):
            target_val_np = target_val_arr[order]
    
    if len(target_pos_np) == 0:
        return None
        
    device = query_pos.device
    target_pos = torch.from_numpy(target_pos_np).to(device=device, dtype=query_pos.dtype)

    target_val_np = np.asarray(target_val_np)
    if not use_ML_tag and target_val_np.ndim == 2:
        target_prob = torch.from_numpy(target_val_np.astype(np.float32, copy=False)).to(device=device)
        if target_prob.shape[1] != num_classes:
            raise ValueError(
                f"Expected tagged label vectors with {num_classes} classes; got {target_prob.shape[1]}"
            )
    elif not use_ML_tag and target_val_np.ndim == 1 and target_val_np.shape[0] == num_classes:
        target_prob = torch.from_numpy(target_val_np.astype(np.float32, copy=False)).to(device=device)
        target_prob = target_prob.unsqueeze(0).expand(len(target_pos), -1)
    else:
        target_val = torch.from_numpy(np.atleast_1d(target_val_np)).to(device=device, dtype=torch.long)
        if len(target_val) == 1 and len(target_pos) > 1:
            target_val = target_val.expand(len(target_pos))
        target_prob = torch.zeros((len(target_pos), num_classes), dtype=torch.float32, device=device)
        target_prob[torch.arange(len(target_pos), device=device), target_val] = 1.0

    idx = torch.searchsorted(target_pos, query_pos)

    # Clamp insertion points before vectorized equality validation.
    idx = torch.clamp(idx, 0, len(target_pos) - 1)

    valid_mask = (target_pos[idx] == query_pos)

    labels = torch.zeros((*query_pos.shape, num_classes), dtype=torch.float32, device=device)

    # Non-hits carry background-shaped values but remain excluded by labels_idx.
    labels[..., 0] = 1.0

    if valid_mask.any():
        labels[valid_mask] = target_prob[idx[valid_mask]]
    
    labels_idx = valid_mask
    keep = labels_idx.sum(1) > 0
    
    if not keep.any():
        if discard_empty_labels:
            return None
        else:
            keep = torch.ones_like(keep, dtype=torch.bool)
    
    keep_idx = keep.nonzero(as_tuple=True)[0].tolist()
    signals = signal_chunks[keep].transpose(1, 2)
    seqs, aligned_labels, aligned_labels_idx, positions = (
        align_chunk_fields_to_sequences(
            [base_seq_chunks[i] for i in keep_idx],
            labels[keep].float(),
            labels_idx[keep],
            output_positions[keep],
        )
    )
    
    return {
        'signal': signals,
        'seq': seqs,
        'labels': aligned_labels,
        'labels_idx': aligned_labels_idx,
        'position': positions,
    }

class TaggedBamSample:
    """Configuration for a BAM labelled by per-read data or MM/ML calls.

    For dictionary labels, required ``label_coordinates`` selects ``"read"``
    (zero-based original read orientation) or ``"ref"`` (zero-based reference
    positions). Dictionary read mode needs neither FASTA nor index unless
    ``chrom_info`` restricts fetching to
    reference regions. Reference mode requires an indexed FASTA and BAM.
    ``chrom_info`` contains zero-based half-open reference regions.
    With ``ML_tag=False``,
    ``read_label_dict_path`` points to a pickle keyed by read name. With
    ``ML_tag=True``, decoded probabilities strictly above ``ML_threshold``
    receive scalar class ``label``. Optional ``negative_label`` assigns a
    different class to calls strictly below the same threshold; omission or
    ``None`` leaves those calls unlabelled. Equality is always unlabelled.
    Optional ``MM_code`` selects one exact letter or numeric modification code
    before thresholding; reads with no calls for that code are skipped. When
    omitted, decoded codes remain pooled as before. With selection enabled,
    only that code's decoded calls contribute to ``min_read_frac``.
    The fraction of decoded calls retained across both enabled classes must
    reach ``min_read_frac`` (not a fraction of all read bases). MM/ML samples retain their
    mapped/indexed BAM and reference requirements; omit ``label_coordinates``.
    ``label_smoothing`` (training metadata only), ``flag``, class
    count, and polishing options have the same meanings as on
    :class:`LabelledBamSample`.
    """

    def __init__(self, bam_path: str, chrom_info: List[str], ref: Optional[str], num_classes,
                 read_label_dict_path=None, label_smoothing=0, label=1, ML_tag=False, ML_threshold = 0.5, min_read_frac=0, flag=0,
                 polish=False, polish_indel_limit=DEFAULT_POLISH_INDEL_LIMIT,
                 polish_vcf_path=None, *, label_coordinates=None, negative_label=None, MM_code=None):
        polish, polish_indel_limit, polish_vcf_path = validate_polishing_options(
            polish, polish_indel_limit, polish_vcf_path,
        )
        if not ML_tag and label_coordinates not in ("read", "ref"):
            raise ValueError("label_coordinates must be 'read' or 'ref'")
        if ML_tag and label_coordinates is not None:
            raise ValueError("label_coordinates applies only to dictionary labels")
        if MM_code is not None and not ML_tag:
            raise ValueError("MM_code requires ML_tag=True")
        self.MM_code = normalize_mm_code(MM_code) if MM_code is not None else None
        if label_coordinates == "ref" and not ref:
            raise ValueError("Reference-coordinate tagged samples require ref")
        if negative_label is not None:
            if not ML_tag:
                raise ValueError("negative_label requires ML_tag=True")
            if (isinstance(negative_label, (bool, np.bool_))
                or not isinstance(negative_label, (int, np.integer))
                or not 0 <= negative_label < num_classes):
                raise ValueError(f"negative_label must be an integer in [0, {num_classes})")
            if negative_label == label:
                raise ValueError("negative_label must differ from label")
        self.bam_path = bam_path
        self.label_coordinates = label_coordinates
        self.chrom_info = tuple(sorted(chrom_info)) 
        self.ref = ref
        self.read_label_dict_path=read_label_dict_path
        self.flag = flag
        self.label_smoothing = label_smoothing
        self.num_classes=num_classes
        self.label = label
        self.negative_label = negative_label
        self.ML_tag = ML_tag
        self.ML_threshold = ML_threshold
        self.min_read_frac=min_read_frac
        self.polish=polish
        self.polish_indel_limit=polish_indel_limit
        self.polish_vcf_path=polish_vcf_path

    def get_read_label_dict(self):
        """Load the pickle mapping read names to ``(values, positions)`` tuples."""

        with open(self.read_label_dict_path, 'rb') as f:
            read_label_dict = pickle.load(f)

        return read_label_dict
        
class TaggedBamIterableDataset(IterableDataset):
    """Stream labelled chunks from per-read dictionaries or MM/ML BAM tags.

    Constructor and yielded-record semantics match
    :class:`LabelledBamIterableDataset`. Each sample selects either a cached
    pickled target dictionary or MM/ML decoding. Dictionaries require an
    explicit coordinate system. Read dictionaries support unaligned/unindexed
    streams without a FASTA; explicit regions still require an indexed BAM. Reference coordinates
    and MM/ML sources require mapped, indexed BAMs compatible with their FASTA.
    Dictionary ``position`` values follow the selected coordinate system. Read
    streams have an unknown coordinate frontier, as do read-motif streams.
    """

    def __init__(
        self,
        bam_list: List[TaggedBamSample],
        size,
        step,
        move_downsample_factor,
        seq_type=None,
        kmer_table=None,
        reads_per_chunk=1000,
        min_read_length=100,
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        shuffle=True,
        discard_empty_labels=True,
        cycle=True,
        expected_ts_dim=None,
        include_label_smoothing=False,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
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
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.shuffle=shuffle
        self.cycle=cycle
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        self.include_label_smoothing = include_label_smoothing
        
        self.discard_empty_labels = discard_empty_labels
        
    def _yield_buffer(self, data):
        """Compute percentiles, optionally shuffle, and emit buffered chunks."""
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)
            
        signals = signals.reshape(signals.size(0), signals.size(1)//self.move_downsample_factor, signals.size(2)*self.move_downsample_factor)
        percentiles = percentiles.reshape(percentiles.size(0), percentiles.size(1)//self.move_downsample_factor, percentiles.size(2)*self.move_downsample_factor)
        
        idxs = list(range(len(data['seq'])))
        
        if self.shuffle:
            random.shuffle(idxs)
        
        for i in idxs:
            if len(data['seq'][i])>=64:
                item = (
                    data['seq'][i],
                    signals[i].to(dtype=torch.float16),
                    percentiles[i].to(dtype=torch.float16),
                    data['labels'][i],
                    data['labels_idx'][i],
                    data['position'][i],
                    data['frontier_chrom'],
                    data['frontier_start'],
                    data['frontier_chrom_idx'],
                    data['rank'],
                    data['strand'][i],
                    data['read_name'][i],
                    data['ref_name'][i],
                )
                if self.include_label_smoothing:
                    yield item, data['label_smoothing'][i]
                else:
                    yield item

                
    def __iter__(self):
        """Yield this worker's tagged chunks across repeated BAM cycles."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
                
        poly_gaa='GAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAA'
        poly_tgg='TGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGG'
        
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
        
        ref_seq_cache = {}
        read_label_dict_cache = {}
        polish_variant_mask_cache = {}
        data = _new_labelled_buffer()
            
        
        for cycle_counter, bam_sample in iter_file_cycles(my_files, self.cycle):
            
            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info=bam_sample.flag

            use_ML_tag=bam_sample.ML_tag
            read_based = not use_ML_tag and bam_sample.label_coordinates == "read"
            ML_threshold=bam_sample.ML_threshold
            bam_label=bam_sample.label
            min_read_frac=bam_sample.min_read_frac
            if not use_ML_tag:
                if bam_sample.read_label_dict_path in read_label_dict_cache:
                    read_label_dict=read_label_dict_cache[bam_sample.read_label_dict_path]
                else:
                    read_label_dict=bam_sample.get_read_label_dict()
                    read_label_dict_cache[bam_sample.read_label_dict_path]=read_label_dict
            
            chroms=tuple(x[0] for x in chrom_info)
            
            active_chroms = (
                get_mapped_chroms(bam_path) & set(chroms)
                if chrom_info else set()
            )
            if chrom_info and not active_chroms:
                continue
                
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter)}',
                    flush=True,
                )
                
                if not read_based:
                    if bam_sample.ref not in ref_seq_cache:
                        ref_seq_cache[bam_sample.ref] = get_ref_seq_dict(
                            ref=bam_sample.ref, chrom_list=chroms,
                        )
                    ref_seq_dict = ref_seq_cache[bam_sample.ref]

                region_specs = list(enumerate(chrom_info))
                if read_based and not chrom_info:
                    region_specs = [(None, (None, None, None))]
                excluded_flags = (0x900 if read_based and not chrom_info else 0x904) | flag_info
                for contig_idx, (contig, start, end) in region_specs:
                    if contig is not None and contig not in active_chroms:
                        continue
                    print(
                        f'{str(datetime.datetime.now())}: '
                        f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter, contig=contig)}',
                        flush=True,
                    )
                    
                    bam_reads = (
                        bam.fetch(until_eof=True) if contig is None
                        else bam.fetch(contig, start=start, end=end, until_eof=True)
                    )
                    for bam_read in bam_reads:
                        if (bam_read.flag & excluded_flags == 0
                            and passes_min_qscore(bam_read, self.min_qscore)
                            and len(bam_read.seq) >= self.min_read_length
                        ):
                            if self.seq_type=='rna' and (poly_gaa in bam_read.seq or poly_tgg in bam_read.seq): continue
                                
                            if use_ML_tag==False and bam_read.qname not in read_label_dict:
                                continue
                                
                            if read_based:
                                read = Read_Lite(*get_read_info_lite(
                                    bam_read, self.seq_type, self.kmer_table,
                                    expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                ))
                            else:
                                read = Read(*get_read_info(
                                    bam_read, ref_seq_dict, self.seq_type, self.kmer_table,
                                    expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                                ))
                            polished_read = (
                                get_polished_read_for_sample(
                                    bam_read, bam_sample, polish_variant_mask_cache,
                                )
                                if bam_sample.polish
                                else None
                            )
                            
                            strand_val = 0 if bam_read.is_forward else 1
                            read_name_str = read.read_name
                            ref_name_str = bam_read.reference_name or 'unknown'

                            if use_ML_tag:
                                coords, vals = mm_ml_coords(bam_read, mod_code=bam_sample.MM_code)
                                if len(vals) == 0:
                                    continue
                                    
                                positive_calls = vals > ML_threshold
                                negative_calls = (
                                    vals < ML_threshold
                                    if bam_sample.negative_label is not None
                                    else np.zeros_like(positive_calls)
                                )
                                val_filter = positive_calls | negative_calls
                                if np.mean(val_filter)<min_read_frac:
                                    continue
                                # ML selects a hard class; it is not a target weight.
                                target_labels = np.full(sum(val_filter), bam_label, dtype=np.int64)
                                if bam_sample.negative_label is not None:
                                    target_labels[negative_calls[val_filter]] = bam_sample.negative_label
                                target_info = (
                                    target_labels,
                                    coords[val_filter],
                                )
                            else:
                                target_info = read_label_dict[read_name_str]
                            
                            try:
                                processed = prepare_tagged_chunks(
                                    read,
                                    target_info,
                                    self.size,
                                    self.step,
                                    use_ML_tag,
                                    num_classes=bam_sample.num_classes,
                                    label_coordinates=bam_sample.label_coordinates,
                                    pore_type=self.pore_type,
                                    discard_empty_labels=self.discard_empty_labels,
                                    polished_read=polished_read,
                                )
                            except DegenerateQuantileNormalizationError:
                                continue

                            if processed is not None:
                                data['signal'].append(processed['signal'])
                                data['seq'].extend(processed['seq'])
                                data['labels'].extend(processed['labels'])
                                data['labels_idx'].extend(processed['labels_idx'])
                                data['label_smoothing'].extend([bam_sample.label_smoothing] * len(processed['seq']))
                                data['position'].extend(processed['position'])
                                data['strand'].extend([strand_val] * len(processed['position']))
                                data['read_name'].extend([read_name_str] * len(processed['position']))
                                data['ref_name'].extend([ref_name_str] * len(processed['position']))
                                if not read_based:
                                    _update_labelled_buffer_frontier(
                                        data, contig, contig_idx, bam_read.reference_start,
                                    )
                                
                            if len(data['signal']) == self.reads_per_chunk:
                                data['rank']=global_worker_id
                                yield from self._yield_buffer(data)
                                data = _new_labelled_buffer()
        if len(data['signal']) > 0:
            data['rank']=global_worker_id
            yield from self._yield_buffer(data)
