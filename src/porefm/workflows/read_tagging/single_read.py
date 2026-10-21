"""Compare single-read embeddings with a streamed control and emit BAM/Parquet."""

from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
import argparse
import os
import sys
import time
import heapq
from pathlib import Path

from porefm.cli._parser import PoreFMArgumentParser, PoreFMHelpFormatter
from porefm.common.run_outputs import add_output_safety_argument, prepare_run_outputs

class DefaultsHelpFormatter(PoreFMHelpFormatter):
    """Append defaults to optional argument help when not already present."""


class DefaultsArgumentParser(PoreFMArgumentParser):
    """Argument parser that reserves help text for undocumented options."""

    def add_argument(self, *args, **kwargs):
        """Add an argument with temporary blank help when none is supplied."""
        kwargs.setdefault('help', ' ')
        return super().add_argument(*args, **kwargs)


def finalize_default_help(parser):
    """Replace blank help with a default annotation or no help text."""
    for action in parser._actions:
        if action.help is argparse.SUPPRESS or (action.help or '').strip():
            continue
        if (
            action.option_strings
            and action.default is not argparse.SUPPRESS
            and action.default is not None
            and not action.required
        ):
            action.help = '(default: %(default)s)'
        else:
            action.help = None


def build_single_read_arg_parser():
    """Build the CLI parser for single-read versus control comparison."""
    parser = DefaultsArgumentParser(
        formatter_class=DefaultsHelpFormatter,
        description="Comparison of single read deviations from control sample",
    )

    # Core paths and settings
    parser.add_argument('--kmer_dict_path', type=str, required=True)
    parser.add_argument('--seq_type', type=str, choices=['dna', 'rna'], required=True)
    parser.add_argument('--ref', type=str, required=True)

    parser.add_argument('--motif_string', type=str, default='', help="Motif specification supplied as one quoted shell argument, for example 'CG 0;CH 0'.")
    parser.add_argument('--motif_label', type=str, default='', help="Semicolon-delimited class labels corresponding to --motif_string.")
    parser.add_argument('--pos_file', type=str, default='')

    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=100000)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=int, default=10)
    parser.add_argument('--min_control_cov', type=int, default=30, help='Minimum control reads required at a locus/strand before scoring WT reads.')
    parser.add_argument('--scale_factor', type=float, default=2.0, help='Multiplier applied to cosine distances before clipping to [0, 1] for ML tag values.')
    parser.add_argument('--batch_size', type=int, default=256, help="Forward microbatch size and control loader batch size.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained pretraining or classifier checkpoint.')
    parser.add_argument('--ctrl_dir', type=str, default=None, help='Control BAM directory; recursively reads *.bam files.')
    parser.add_argument('--wt_dir', type=str, default=None, help='WT/test BAM directory; recursively reads *.bam files.')
    parser.add_argument('--wga_bam_path', '--ctrl_bam_path', dest='wga_bam_path', type=str, default=None, help='Single control BAM path.')
    parser.add_argument('--wt_bam_path', type=str, default=None, help='Single WT/test BAM path.')
    parser.add_argument('--bam_output', type=str, default=None, help='Path to BAM output; required when --output_format is bam or both.')
    parser.add_argument('--output', type=str, default=None, help='Output directory for parquet files; required when --output_format is parquet or both.')
    parser.add_argument('--output_format', type=str, default='bam', choices=['bam', 'parquet', 'both'], help='Output format.')
    parser.add_argument('--checkpoint_batch_step', type=int, default=100, help='WT/test batches per parquet checkpoint.')
    parser.add_argument('--chrom', type=str, default=None, help='Legacy single chromosome to process.')
    parser.add_argument('--chrom_list', type=str, nargs='+', default=None, help='Chromosomes to process; defaults to all reference contigs.')
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--mod_code', type=str, nargs='+', required=True)
    parser.add_argument('--num_trim', type=int, default=10)
    parser.add_argument('--wt_parallel_loader', default=False, action='store_true', help='Process WT BAM files in parallel and merge by coordinate.')
    parser.add_argument('--ctrl_parallel_loader', default=False, action='store_true', help='Process control BAM files in parallel and merge by coordinate.')
    parser.add_argument('--wt_use_polished_sequence', default=False, action='store_true', help='Use polished WT/test read chunks and write polished WT/test sequences to BAM output.')
    parser.add_argument('--ctrl_use_polished_sequence', default=False, action='store_true', help='Use polished control read chunks when building the control embedding stream.')
    parser.add_argument('--polish_indel_limit', type=int, default=5, help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None, help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')
    parser.add_argument('--wt_num_workers', type=int, default=1, help='WT DataLoader workers when --wt_parallel_loader is not used.')
    parser.add_argument('--control_num_workers', type=int, default=1, help='Control DataLoader workers when --ctrl_parallel_loader is not used.')
    parser.add_argument('--control_buffer_blocks', type=int, default=10, help='Number of genomic blocks to keep in the GPU control circular buffer.')
    parser.add_argument('--bam_threads', type=int, default=4, help='pysam threads for BAM output writing.')
    parser.add_argument('--progress_every', type=int, default=10, help='Print WT batch progress every N batches; 0 disables progress prints.')

    finalize_default_help(parser)
    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_single_read_arg_parser().parse_args()

import seaborn as sns
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import threading
import queue
import array
import torch
import torch.nn as nn
import pysam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from numpy.linalg import norm

import collections

import numpy as np
import numpy_indexed as npi

from porefm.models.loading import load_model as load_checked_model
from porefm.data.bam_headers import add_pg_to_header
from porefm.data.bam import build_lookup_table, get_labelled_arrays, revcomp
from porefm.common.tensor_masks import ignore_edges
from porefm.common.chunking import resolve_downstream_step
from porefm.common.utils import require_single_process_execution
from porefm.data.reader import get_mapped_chroms
from porefm.data.datasets.aligned import AlignedBamIterableDataset, AlignedBamSample
from porefm.data.collation import aligned_collate_fn, collate_modal_aligned_batch
from porefm.workflows.read_tagging.sync_embeddings import process_aligned_batch
from porefm.data.complete_reads import (
    CompleteAlignedReadIterableDataset,
    make_mm_ml_tags,
    stitch_1d_distances_ultra_fast,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE


def load_embedding_model(model_path, device):
    """Load the checked frozen backbone used for single-read scoring."""
    return load_checked_model(
        model_path,
        model_type="backbone",
        add_signal_cross_attn=False,
        device=device,
    )


class SingleGenomicAccumulator:
    """Accumulate strand-specific control means in fixed genomic blocks.

    GPU sum/count tensors are finalized once a monotonically increasing
    coordinate frontier passes their right boundary, then queued on CPU.
    """

    def __init__(self, dim, block_size=100_000, dtype=torch.bfloat16, device='cuda'):
        self.dim = dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        # Track active blocks: maps block_start -> block_data
        self.active_blocks = {}
        
        # Track the minimum coordinate we've seen to safely flush
        self.frontier = 0
        
        # Yields tuple: (block_start, mean_vectors, count_vectors)
        self.ready_queue = collections.deque()

    def _get_or_create_block(self, block_start):
        if block_start not in self.active_blocks:
            # ALLOCATE ON GPU FOR FAST ACCUMULATION
            self.active_blocks[block_start] = {
                'sum': torch.zeros((2, self.block_size, self.dim), dtype=torch.float32, device=self.device),
                'count': torch.zeros((2, self.block_size), dtype=torch.int32, device=self.device)
            }
        return self.active_blocks[block_start]

    def update(self, indices, embeddings, strand, current_min_coord, unique_blocks_cpu):
        """Scatter-add one filtered embedding batch and release safe blocks."""
        self.frontier = current_min_coord

        # Identify blocks
        block_ids = torch.div(indices, self.block_size, rounding_mode='floor')
        
        # Update blocks ON GPU
        for bid in unique_blocks_cpu:
            block_start = int(bid) * self.block_size
            mask = (block_ids == bid)
            
            block_data = self._get_or_create_block(block_start)
            
            for s in (0, 1):
                s_mask = mask & (strand == s)
                
                # --- NO 'IF' STATEMENT ---
                # We blindly slice and add. If s_mask is entirely False, 
                # PyTorch handles the 0-sized tensors natively without blocking the CPU.
                local_indices = indices[s_mask] - block_start
                local_embeds = embeddings[s_mask].float() 
                
                block_data['sum'][s].index_add_(0, local_indices, local_embeds)
                block_data['count'][s].index_add_(0, local_indices, torch.ones_like(local_indices, dtype=torch.int32))

        self._flush_old_blocks(self.frontier)

    def _flush_old_blocks(self, safe_frontier):
        # Cast to list so we can modify the dictionary during iteration
        for block_start in list(self.active_blocks.keys()):
            block_end = block_start + self.block_size
            if block_end <= safe_frontier:
                self._enqueue_block(block_start)
    
    def flush_all_blocks(self):
        """Finalize every active block regardless of the current frontier."""
        for block_start in list(self.active_blocks.keys()):
            self._enqueue_block(block_start)
            
    def _enqueue_block(self, block_start):
        # Pop removes it from active memory
        block_data = self.active_blocks.pop(block_start)
        
        # Calculate Means
        counts = torch.clamp_min(block_data['count'], 1).unsqueeze(-1)
        mean_vectors = (block_data['sum'] / counts).to(self.dtype)
        
        # Push to queue (moving to CPU to free up VRAM)
        self.ready_queue.append((
            block_start, 
            mean_vectors, 
            block_data['count']
        ))

    def pop_ready(self):
        """Yield finalized ``(block_start, means, counts)`` tuples."""
        while self.ready_queue:
            yield self.ready_queue.popleft()
            
def post_process_block(block_start, mean_vectors, 
                       chrom_labelled_arrays, counts, block_size, min_cov=5):
    """Select labelled forward/reverse control positions meeting coverage.

    Returns one ``(positions, counts, mean_embeddings)`` tuple per strand.
    """
        
    count_filter=(counts>=min_cov).numpy()
    
    positions=np.arange(block_start, block_start+block_size)

    fwd_ref_pos=chrom_labelled_arrays[0][0]
    match_idx = np.searchsorted(fwd_ref_pos, positions)
    match_idx = np.clip(match_idx, 0, len(fwd_ref_pos) - 1)
    fwd_pos_idx = positions==fwd_ref_pos[match_idx]
    fwd_pos_idx=fwd_pos_idx&count_filter[0]
    
    fwd_counts = counts[0][fwd_pos_idx]
    fwd_mean_vectors=mean_vectors[0, fwd_pos_idx]
    
    rev_ref_pos=chrom_labelled_arrays[1][0]
    match_idx = np.searchsorted(rev_ref_pos, positions)
    match_idx = np.clip(match_idx, 0, len(rev_ref_pos) - 1)
    rev_pos_idx = positions==rev_ref_pos[match_idx]
    rev_pos_idx=rev_pos_idx&count_filter[1]
    
    rev_counts = counts[1][rev_pos_idx]
    rev_mean_vectors=mean_vectors[1, rev_pos_idx]
    
    final_fwd_pos = positions[fwd_pos_idx]
    final_rev_pos = positions[rev_pos_idx]
    
    return (final_fwd_pos, fwd_counts, fwd_mean_vectors), (final_rev_pos, rev_counts, rev_mean_vectors)      


def _coord_key(value):
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


class SortedSingleChromMerger:
    """Merge finite single-chromosome loaders by their minimum coordinates."""

    def __init__(self, dataloaders, min_start_index):
        self.iters = [iter(dl) for dl in dataloaders]
        self.min_start_index = min_start_index
        self.heap = []

        for loader_idx, loader_iter in enumerate(self.iters):
            try:
                self._push(next(loader_iter), loader_idx)
            except StopIteration:
                pass

    def _push(self, batch, loader_idx):
        min_start = _coord_key(batch[self.min_start_index])
        heapq.heappush(self.heap, (min_start, loader_idx, batch))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.heap:
            raise StopIteration

        _, loader_idx, batch = heapq.heappop(self.heap)
        try:
            self._push(next(self.iters[loader_idx]), loader_idx)
        except StopIteration:
            pass

        return batch


def _loader_kwargs(num_workers, prefetch_factor):
    kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
    return kwargs


def get_sorted_aligned_iter(dataset_kwargs, bam_config_list, batch_size, num_workers, parallel_files):
    """Create a sorted aligned-chunk iterator, optionally one loader per BAM."""
    if parallel_files:
        loaders = [
            DataLoader(
                AlignedBamIterableDataset(bam_list=[bam_config], **dataset_kwargs),
                batch_size=batch_size,
                collate_fn=aligned_collate_fn,
                **_loader_kwargs(1, 8),
            )
            for bam_config in bam_config_list
        ]
        return iter(SortedSingleChromMerger(loaders, min_start_index=-1))

    loader = DataLoader(
        AlignedBamIterableDataset(bam_list=bam_config_list, **dataset_kwargs),
        batch_size=batch_size,
        collate_fn=aligned_collate_fn,
        **_loader_kwargs(num_workers, 4),
    )
    return iter(loader)


def get_sorted_complete_read_iter(dataset_kwargs, bam_config_list, num_workers, parallel_files):
    """Create a sorted complete-read iterator, optionally one loader per BAM."""
    if parallel_files:
        loaders = [
            DataLoader(
                CompleteAlignedReadIterableDataset(bam_list=[bam_config], **dataset_kwargs),
                batch_size=None,
                **_loader_kwargs(1, 4),
            )
            for bam_config in bam_config_list
        ]
        return iter(SortedSingleChromMerger(loaders, min_start_index=-3))

    loader = DataLoader(
        CompleteAlignedReadIterableDataset(bam_list=bam_config_list, **dataset_kwargs),
        batch_size=None,
        **_loader_kwargs(num_workers, 4),
    )
    return iter(loader)


def collect_bam_paths(path, directory, dir_arg, path_arg):
    """Resolve one BAM path or recursively collect BAMs from a directory."""
    if directory:
        bam_paths = natsorted(str(path) for path in Path(directory).rglob('*.bam'))
        if not bam_paths:
            raise ValueError(f"No BAM files found under {dir_arg}={directory}")
        return bam_paths

    if path:
        return [path]

    raise ValueError(f"Provide either {dir_arg} or {path_arg}.")


def get_bam_active_chroms(bam_path):
    """Return contigs with mapped reads, falling back to all BAM references."""
    try:
        return get_mapped_chroms(bam_path)
    except Exception:
        with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
            return set(bam.references)


def index_bam_paths_by_chrom(bam_paths, chrom_names):
    """Map requested contigs to BAMs that report those contigs as active."""
    requested_chroms = set(chrom_names)
    paths_by_chrom = {chrom: [] for chrom in chrom_names}
    for bam_path in bam_paths:
        for chrom in get_bam_active_chroms(bam_path) & requested_chroms:
            paths_by_chrom[chrom].append(bam_path)
    return paths_by_chrom


def make_aligned_samples(
    bam_paths,
    chrom_info,
    ref,
    polish=False,
    polish_indel_limit=5,
    polish_vcf_path=None,
):
    """Create one unlabelled aligned sample per BAM for the given region."""
    return [
        AlignedBamSample(
            bam_path=bam_path,
            chrom_info=chrom_info,
            ref=ref,
            pos_file='',
            motif='',
            motif_label=0,
            flag=0,
            polish=polish,
            polish_indel_limit=polish_indel_limit,
            polish_vcf_path=polish_vcf_path if polish else None,
        )
        for bam_path in bam_paths
    ]


def single_read_bam_writer_worker(write_queue, bam_output, bam_threads, out_header, labelled_arrays, base, mod_code, seq_type):
    """Consume stitched read distances and write MM/ML-annotated BAM records.

    Only reference positions present in ``labelled_arrays`` are encoded. A
    ``None`` queue item terminates the worker after marking the task complete.
    """
    out_header = add_pg_to_header(out_header)
    with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=out_header) as outf:
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break

            chrom, strand_dict, read_wise_metric, read_wise_refpos, pysam_data = item
            chrom_labelled_arrays = labelled_arrays[chrom]

            for i, (rname, r_strand) in enumerate(strand_dict.items()):
                r_metric = read_wise_metric[i].numpy()
                r_pos = read_wise_refpos[i].numpy()
                positions = chrom_labelled_arrays[r_strand][0]

                if len(positions) == 0:
                    r_pos_indices = np.asarray([], dtype=np.int64)
                else:
                    match_idx = np.searchsorted(positions, r_pos)
                    match_idx = np.clip(match_idx, 0, len(positions) - 1)
                    valid_searches_mask = positions[match_idx] == r_pos
                    r_pos_indices = np.where(valid_searches_mask & (~np.isnan(r_metric)))[0]

                read_dict = pysam_data[rname]
                seq = read_dict['seq']
                seq = revcomp(seq) if int(read_dict['flag']) & 16 else seq

                mm_string, ml_array = make_mm_ml_tags(
                    seq,
                    base,
                    r_pos_indices,
                    r_metric[r_pos_indices],
                    seq_type,
                    mod_code=mod_code,
                    strand="+",
                    skipped_flag="?",
                )

                new_read = pysam.AlignedSegment.from_dict(read_dict, out_header)
                new_read.set_tag("MM", mm_string, value_type="Z")
                new_read.set_tag("ML", array.array('B', ml_array))
                new_read.set_tag("MN", len(seq))
                outf.write(new_read)

            write_queue.task_done()


def collect_single_read_parquet_checkpoint_data(
    chrom,
    unique_reads,
    strand_dict,
    read_wise_distance,
    read_wise_refpos,
    labelled_arrays,
):
    """Flatten labelled finite read distances into checkpoint meta/data columns."""
    meta = {
        "readname": [],
        "refname": [],
        "strand": [],
        "length": [],
    }
    data = {
        "read_pos": [],
        "refpos": [],
        "distance": [],
    }
    chrom_labelled_arrays = labelled_arrays[chrom]

    for i, rname in enumerate(unique_reads):
        rname = str(rname)
        r_strand = int(strand_dict[rname])
        r_distance = read_wise_distance[i].numpy()
        r_refpos = read_wise_refpos[i].numpy()
        positions = chrom_labelled_arrays[r_strand][0]

        if len(positions) == 0:
            read_pos = np.asarray([], dtype=np.int64)
            refpos = np.asarray([], dtype=np.int64)
            distances = np.asarray([], dtype=np.float32)
        else:
            match_idx = np.searchsorted(positions, r_refpos)
            match_idx = np.clip(match_idx, 0, len(positions) - 1)
            valid_label_mask = (
                (positions[match_idx] == r_refpos)
                & np.isfinite(r_distance)
                & (r_refpos >= 0)
            )
            read_pos = np.flatnonzero(valid_label_mask).astype(np.int64)
            refpos = r_refpos[valid_label_mask].astype(np.int64)
            distances = r_distance[valid_label_mask].astype(np.float32, copy=False)

        meta["readname"].append(rname)
        meta["refname"].append(chrom)
        meta["strand"].append(r_strand)
        meta["length"].append(len(read_pos))

        data["read_pos"].extend(read_pos.tolist())
        data["refpos"].extend(refpos.tolist())
        data["distance"].extend(distances.tolist())

    return meta, data


def write_single_read_parquet_checkpoint(chunks, output_dir, checkpoint):
    """Merge row chunks and write paired meta/data Parquet checkpoint files."""
    meta = {
        "readname": [],
        "refname": [],
        "strand": [],
        "length": [],
    }
    data = {
        "read_pos": [],
        "refpos": [],
        "distance": [],
    }

    for chunk_meta, chunk_data in chunks:
        for key in meta:
            meta[key].extend(chunk_meta[key])
        for key in data:
            data[key].extend(chunk_data[key])

    if not meta["readname"]:
        return

    os.makedirs(output_dir, exist_ok=True)
    pq.write_table(
        pa.table(meta),
        os.path.join(output_dir, f"checkpoint_{checkpoint}_meta.parquet"),
        compression="snappy",
    )
    pq.write_table(
        pa.table(data),
        os.path.join(output_dir, f"checkpoint_{checkpoint}_data.parquet"),
        compression="snappy",
    )


def get_stream_metrics(model, bam_config_list,
         seq_type, kmer_table,
         device,
         size=2048,
         step=None,
         block_size = 100_000,
         reads_per_chunk=100,
         min_read_length=500,
         min_qscore=5,
         batch_size=128,
         num_trim=16,
         num_workers=1,
         parallel_loader=False,
         announce_step=True):
    """Yield finalized control mean-embedding blocks from aligned BAM streams.

    ``announce_step`` lets the enclosing multi-chromosome workflow suppress a
    duplicate message after it has already resolved and announced the stride.
    """
    
    fwd_data = {'pos': [], 'counts': [], 'mean_vectors':[]}
    rev_data = {'pos': [], 'counts': [], 'mean_vectors':[]}

    if not bam_config_list:
        return
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="single-read control accumulation",
        announce=announce_step,
    )

    dataset_kwargs=dict(size=size, step=step, seq_type=seq_type, kmer_table=kmer_table,
                        reads_per_chunk=reads_per_chunk, min_read_length=min_read_length,
                        min_qscore=min_qscore, pore_type=None, base_seed=0, rank=0, world_size=1,
                        expected_ts_dim=model.config.ts_dim)
    
    r_start, r_end = bam_config_list[0].chrom_info[0][1:]
    ds_iter = get_sorted_aligned_iter(
        dataset_kwargs,
        bam_config_list,
        batch_size,
        num_workers,
        parallel_loader,
    )

    accumulator = SingleGenomicAccumulator(dim=model.config.hidden_dim, block_size=block_size, device=device)

    # Fetch first batches
    batch = next(ds_iter, None)

    t = time.time()
    start=time.time()
    while batch is not None:
        process_aligned_batch(
            batch,
            model,
            accumulator,
            device,
            r_start,
            r_end,
            num_trim,
        )
        batch = next(ds_iter, None)

        # Yield and calculate anything that is ready
        yield from accumulator.pop_ready()
    accumulator.flush_all_blocks()
    yield from accumulator.pop_ready()

    return 

class CircularGPUControlBuffer:
    """Fixed-capacity circular cache of sequential control means and coverage.

    Absolute genomic coordinates map by modulo capacity. Gathered entries are
    valid only while loaded and when coverage reaches ``min_control_cov``.
    """

    def __init__(self, stream_iter, block_size, buffer_blocks, dim, device, dtype=torch.bfloat16, min_control_cov=1):
        self.stream_iter = stream_iter
        self.block_size = block_size
        self.buffer_blocks = buffer_blocks
        self.capacity = block_size * buffer_blocks
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.min_control_cov = min_control_cov
        
        # Allocate VRAM exactly ONCE
        # Shape: [2, capacity, dim]
        self.embs = torch.zeros((2, self.capacity, self.dim), dtype=dtype, device=device)
        self.counts = torch.zeros((2, self.capacity), dtype=torch.int32, device=device)
        
        # Track what absolute genomic coordinates are currently in the buffer
        self.min_loaded = 0
        self.max_loaded = 0
        self.is_initialized = False

    def _clear_block_slot(self, block_start):
        write_start_idx = block_start % self.capacity
        write_end_idx = write_start_idx + self.block_size
        self.embs[:, write_start_idx:write_end_idx, :].zero_()
        self.counts[:, write_start_idx:write_end_idx].zero_()

    def _clear_skipped_slots(self, start_coord, stop_coord):
        if stop_coord <= start_coord:
            return

        if stop_coord - start_coord >= self.capacity:
            self.embs.zero_()
            self.counts.zero_()
            return

        for block_start in range(start_coord, stop_coord, self.block_size):
            self._clear_block_slot(block_start)

    def ensure_coverage(self, min_coord, max_coord):
        """
        Advances the circular buffer by overwriting old blocks in-place.
        ZERO memory re-allocations.
        """
        # If the requested max_coord is completely out of our expected forward stream
        # (e.g., jumping chromosomes), you might need reset logic here. 
        # Assuming contiguous forward streaming for now.
        
        while self.max_loaded <= max_coord:
            try:
                block_start, block_embs, block_counts = next(self.stream_iter)
            except StopIteration:
                break # Stream exhausted
                
            # Initialize the loaded tracking on the very first block
            if not self.is_initialized:
                self.min_loaded = block_start
                self.max_loaded = block_start
                self.is_initialized = True
            elif block_start > self.max_loaded:
                self._clear_skipped_slots(self.max_loaded, block_start)
            
            # Find the exact slot in our pre-allocated tensor
            write_start_idx = block_start % self.capacity
            write_end_idx = write_start_idx + self.block_size
            
            # In-place overwrite (blazing fast, no allocation)
            self.embs[:, write_start_idx:write_end_idx, :] = block_embs.to(self.device, dtype=self.dtype)
            self.counts[:, write_start_idx:write_end_idx] = block_counts.to(self.device, dtype=torch.int32)
            
            # Update our tracker
            self.max_loaded = max(self.max_loaded, block_start + self.block_size)
            
            # The oldest data we can safely access without it having been overwritten
            # is the current max_loaded minus the total capacity of the buffer.
            self.min_loaded = max(0, self.max_loaded - self.capacity)

    def gather(self, ref_positions, strands, valid_mask):
        """
        Gathers embeddings using modulo indexing. 
        """
        B, L = ref_positions.shape
        
        # 1. Identify which requested positions are actually loaded in our circular buffer
        # and aren't padding (valid_mask)
        in_bounds_mask = (ref_positions >= self.min_loaded) & (ref_positions < self.max_loaded)
        
        # 2. Map absolute genomic coordinates to circular buffer indices natively
        # PyTorch % is highly optimized on CUDA
        circ_indices = ref_positions % self.capacity
        
        # Expand strands to match [B, L]
        strand_idx = strands.view(B, 1).expand(B, L)

        ctrl_counts = self.counts[strand_idx, circ_indices]
        coverage_mask = ctrl_counts >= self.min_control_cov
        final_mask = valid_mask & in_bounds_mask & coverage_mask
        
        # 3. Gather directly from the static buffer
        ctrl_embs = self.embs[strand_idx, circ_indices, :]
        
        # 4. Zero out positions that were out of bounds or invalid padding
        ctrl_embs = torch.where(
            final_mask.unsqueeze(-1), 
            ctrl_embs, 
            torch.zeros_like(ctrl_embs) # Or float('nan') if you prefer
        )
        
        return ctrl_embs, final_mask


def call_mod_bams(model, ctrl_bam_paths_by_chrom, wt_bam_paths_by_chrom, wt_header_bam_path, ref, kmer_table,
                  block_size, chrom_info_list, bam_output, seq_type, size, step,
                  base, mod_code, labelled_arrays,
                  output_format='bam', parquet_output=None, checkpoint_batch_step=100,
                  batch_size=32, bam_threads=2, num_trim=16, scale_factor=2,
                  reads_per_chunk=50, min_read_length=100, min_qscore=5,
                  wt_num_workers=1, control_num_workers=1,
                  wt_parallel_loader=False, ctrl_parallel_loader=False,
                  control_buffer_blocks=10, min_control_cov=30, progress_every=10,
                  wt_use_polished_sequence=False, ctrl_use_polished_sequence=False,
                  polish_indel_limit=5, polish_vcf_path=None):
    """Compare complete WT reads to control means and write selected outputs.

    Each chromosome receives a finite control stream cached in a circular GPU
    buffer. WT contextual embeddings are scored by cosine distance at aligned
    coordinates, stitched by read, and asynchronously emitted as BAM MM/ML
    annotations, Parquet checkpoints, or both. Output workers are always joined
    in a ``finally`` block.
    """
    device = torch.device('cuda')
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="single-read anomaly scoring",
    )

    write_queue = None
    writer_thread = None
    if output_format in {"bam", "both"}:
        with pysam.AlignmentFile(wt_header_bam_path, 'rb', check_sq=False) as bam:
            str_header = str(bam.header)
        out_header=pysam.AlignmentHeader.from_text(str_header)

        write_queue = queue.Queue(maxsize=1000) 
        writer_thread = threading.Thread(
            target=single_read_bam_writer_worker,
            args=(write_queue, bam_output, bam_threads, out_header, labelled_arrays, base, mod_code, seq_type),
        )
        writer_thread.start()

    io_executor = None
    futures = []
    parquet_row_chunks = []
    checkpoint = 0
    batch_num = 0
    if output_format in {"parquet", "both"}:
        os.makedirs(parquet_output, exist_ok=True)
        io_executor = ThreadPoolExecutor(max_workers=2)

    counter = 0
    st_time = time.time()
    t = st_time

    try:
        for chrom, r_start, r_end in chrom_info_list:
            chrom_start_time = time.time()
            chrom_info = [(chrom, r_start, r_end)]
            ctrl_bam_paths = ctrl_bam_paths_by_chrom.get(chrom, [])
            wt_bam_paths = wt_bam_paths_by_chrom.get(chrom, [])
            if not wt_bam_paths:
                print(f"Skipping {chrom}: no WT/test BAMs with mapped reads.", flush=True)
                continue
            if not ctrl_bam_paths:
                print(f"Skipping {chrom}: no control BAMs with mapped reads.", flush=True)
                continue
            print(f"Processing {chrom}:{r_start}-{r_end}", flush=True)

            ctrl_bam_config_list = make_aligned_samples(
                ctrl_bam_paths,
                chrom_info,
                ref,
                polish=ctrl_use_polished_sequence,
                polish_indel_limit=polish_indel_limit,
                polish_vcf_path=polish_vcf_path,
            )
            wt_bam_config_list = make_aligned_samples(
                wt_bam_paths,
                chrom_info,
                ref,
                polish=wt_use_polished_sequence,
                polish_indel_limit=polish_indel_limit,
                polish_vcf_path=polish_vcf_path,
            )

            control_stream = get_stream_metrics(
                model,
                ctrl_bam_config_list,
                seq_type,
                kmer_table,
                device,
                size=size,
                step=step,
                block_size=block_size,
                reads_per_chunk=reads_per_chunk,
                min_read_length=min_read_length,
                min_qscore=min_qscore,
                batch_size=batch_size,
                num_trim=num_trim,
                num_workers=control_num_workers,
                parallel_loader=ctrl_parallel_loader,
                announce_step=False,
            )

            control_buffer = CircularGPUControlBuffer(
                stream_iter=iter(control_stream),
                block_size=block_size,
                buffer_blocks=control_buffer_blocks,
                dim=model.config.hidden_dim,
                device=device,
                min_control_cov=min_control_cov,
            )

            wt_dataset_kwargs = dict(
                size=size,
                step=step,
                seq_type=seq_type,
                kmer_table=kmer_table,
                reads_per_chunk=reads_per_chunk,
                min_read_length=min_read_length,
                max_read_length=2e5,
                min_qscore=min_qscore,
                pore_type=None,
                base_seed=0,
                rank=0,
                world_size=1,
                expected_ts_dim=model.config.ts_dim,
            )
            wt_iter = get_sorted_complete_read_iter(
                wt_dataset_kwargs,
                wt_bam_config_list,
                wt_num_workers,
                wt_parallel_loader,
            )

            for batch in wt_iter:
                dna_input, ts_input, percentiles_batch, pad_base_seq, pad_ts, read_names, ref_positions, read_positions, min_start, strands, pysam_data = batch
                read_names = np.asarray(read_names)
                trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
                valid_pos = (ref_positions >= 0) & trimmed_pad_base_seq

                if r_start is not None and r_end is not None:
                    valid_pos = valid_pos & (ref_positions >= r_start) & (ref_positions <= r_end)

                valid_pos = valid_pos.to(device, non_blocking=True)
                strands_gpu = strands.to(device, non_blocking=True)

                if progress_every and counter % progress_every == 0:
                    print(f'counter {counter}: {time.time()-t:.2f}  {chrom}:{_coord_key(min_start)}', flush=True)
                    t = time.time()
                counter += 1

                raw_dist_array = []
                for sub_batch in range(0, dna_input.size(0), batch_size):
                    batch_dna_input = dna_input[sub_batch:sub_batch + batch_size]
                    batch_ts_input = ts_input[sub_batch:sub_batch + batch_size]
                    batch_percentiles_batch = percentiles_batch[sub_batch:sub_batch + batch_size]
                    batch_pad_base_seq = pad_base_seq[sub_batch:sub_batch + batch_size]
                    batch_pad_ts = pad_ts[sub_batch:sub_batch + batch_size]

                    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        out = model(
                            batch_dna_input.to(device, non_blocking=True),
                            batch_ts_input.to(device, non_blocking=True),
                            percentiles_input=batch_percentiles_batch.to(device, non_blocking=True),
                            pad_base_seq=batch_pad_base_seq.to(device, non_blocking=True),
                            pad_ts=batch_pad_ts.to(device, non_blocking=True),
                        )

                    test_embs = out['contextual_base_seq']

                    batch_ref_pos = ref_positions[sub_batch:sub_batch + batch_size].to(device, non_blocking=True)
                    batch_strands = strands_gpu[sub_batch:sub_batch + batch_size]
                    batch_mask = valid_pos[sub_batch:sub_batch + batch_size]

                    if not batch_mask.any():
                        raw_dist_array.append(torch.full(batch_ref_pos.shape, float('nan'), dtype=torch.float32))
                        continue

                    current_min = int(batch_ref_pos[batch_mask].min().item())
                    current_max = int(batch_ref_pos[batch_mask].max().item()) + 1
                    control_buffer.ensure_coverage(current_min, current_max)

                    ctrl_embs, ctrl_valid_mask = control_buffer.gather(batch_ref_pos, batch_strands, batch_mask)

                    cosine_distances = 1 - F.cosine_similarity(test_embs, ctrl_embs, dim=-1)
                    final_valid_mask = ctrl_valid_mask & batch_mask
                    distances = torch.where(
                        final_valid_mask,
                        cosine_distances,
                        torch.tensor(float('nan'), device=device, dtype=cosine_distances.dtype),
                    )
                    raw_dist_array.append(distances.cpu().float())

                raw_base_dist_array = torch.cat(raw_dist_array, dim=0)
                strand_dict = {str(x): y.item() for x, y in zip(read_names, strands.cpu())}

                if output_format in {"bam", "both"}:
                    bam_dist_array = torch.abs(torch.clamp_max(raw_base_dist_array, 1))
                    bam_dist_array = torch.clamp_max(scale_factor * bam_dist_array, 1)
                    _, read_wise_bam_dist, read_wise_refpos = stitch_1d_distances_ultra_fast(
                        bam_dist_array,
                        read_positions,
                        ref_positions,
                        pad_base_seq,
                        read_names,
                    )
                    batch_output = (chrom, strand_dict, read_wise_bam_dist, read_wise_refpos, pysam_data)
                    write_queue.put(batch_output)

                if output_format in {"parquet", "both"}:
                    unique_reads, read_wise_raw_dist, read_wise_raw_refpos = stitch_1d_distances_ultra_fast(
                        raw_base_dist_array,
                        read_positions,
                        ref_positions,
                        pad_base_seq,
                        read_names,
                    )
                    parquet_row_chunks.append(
                        collect_single_read_parquet_checkpoint_data(
                            chrom,
                            unique_reads,
                            strand_dict,
                            read_wise_raw_dist,
                            read_wise_raw_refpos,
                            labelled_arrays,
                        )
                    )

                batch_num += 1
                if output_format in {"parquet", "both"} and batch_num % checkpoint_batch_step == 0:
                    checkpoint += 1
                    futures.append(
                        io_executor.submit(
                            write_single_read_parquet_checkpoint,
                            parquet_row_chunks,
                            parquet_output,
                            checkpoint,
                        )
                    )
                    parquet_row_chunks = []

            print(f"Finished {chrom} in {time.time()-chrom_start_time:.2f}s", flush=True)

        if output_format in {"parquet", "both"} and parquet_row_chunks:
            checkpoint += 1
            futures.append(
                io_executor.submit(
                    write_single_read_parquet_checkpoint,
                    parquet_row_chunks,
                    parquet_output,
                    checkpoint,
                )
            )
            parquet_row_chunks = []

    finally:
        if output_format in {"parquet", "both"}:
            for fut in futures:
                fut.result()
            io_executor.shutdown(wait=True)
        if output_format in {"bam", "both"}:
            write_queue.put(None)
            writer_thread.join()

    print(f"Total Time = {time.time()-st_time:.2f}", flush=True)



def main():
    """Validate CLI inputs, load a frozen backbone, and run read comparison."""
    require_single_process_execution(
        "single_read_tag_vs_control.py",
        "Single-read control comparison accumulates per-locus embeddings before scoring.",
    )
    parser = build_single_read_arg_parser()
    args = parser.parse_args()
    resolve_cli_preprocessing(args)

    if args.output_format in {"bam", "both"} and not args.bam_output:
        raise ValueError("--bam_output is required when --output_format is bam or both.")
    if args.output_format in {"parquet", "both"} and not args.output:
        raise ValueError("--output is required when --output_format is parquet or both.")
    if args.checkpoint_batch_step < 1:
        raise ValueError("--checkpoint_batch_step must be >= 1.")

    prepare_run_outputs(
        args, command="single-read-tag",
        directories={"output": args.output} if args.output_format in {"parquet", "both"} else {},
        files={"bam_output": args.bam_output} if args.output_format in {"bam", "both"} else {},
        output_arguments={"output", "bam_output"},
    )

    model_path = args.model_path
        
    device = torch.device('cuda')
    model = load_embedding_model(model_path, device)
    
    SIZE = model.config.max_ts_len
    B = 8
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _ = model(
            torch.ones(B, SIZE, dtype=torch.long, device=device),
            torch.randn(B, SIZE, model.config.ts_dim, device=device),
            percentiles_input=torch.randn(B, SIZE, model.config.ts_dim, device=device),
            pad_base_seq=torch.ones(B, SIZE, dtype=torch.bool, device=device),
            pad_ts=torch.ones(B, SIZE, dtype=torch.bool, device=device),
        )
    
    model = torch.compile(model, dynamic=True)

    kmer_dict_path = args.kmer_dict_path
    kmer_table = resolve_cli_preprocessing(args).kmer_table
    
    seq_type = args.seq_type

    ref = args.ref

    block_size = args.block_size

    fref=pysam.FastaFile(ref)
    
    if len(args.mod_code) != 1:
        raise ValueError("single_read_tag_vs_control.py writes one MM/ML modification tag; pass exactly one --mod_code.")
    base, mod_code = args.base, args.mod_code[0]

    if args.chrom and args.chrom_list:
        raise ValueError("Use either --chrom or --chrom_list, not both.")

    wt_bam_paths = collect_bam_paths(args.wt_bam_path, args.wt_dir, '--wt_dir', '--wt_bam_path')
    ctrl_bam_paths = collect_bam_paths(args.wga_bam_path, args.ctrl_dir, '--ctrl_dir', '--wga_bam_path/--ctrl_bam_path')

    if args.wt_num_workers < 1:
        raise ValueError("--wt_num_workers must be >= 1.")
    if args.control_num_workers < 1:
        raise ValueError("--control_num_workers must be >= 1.")
    if args.control_buffer_blocks < 1:
        raise ValueError("--control_buffer_blocks must be >= 1.")
    if args.min_control_cov < 1:
        raise ValueError("--min_control_cov must be >= 1.")
    if args.scale_factor < 0:
        raise ValueError("--scale_factor must be >= 0.")

    requested_chrom_names = args.chrom_list or ([args.chrom] if args.chrom else list(fref.references))
    missing_chroms = [chrom for chrom in requested_chrom_names if chrom not in fref.references]
    if missing_chroms:
        raise ValueError(f"Chromosomes not present in --ref: {missing_chroms}")

    wt_bam_paths_by_chrom = index_bam_paths_by_chrom(wt_bam_paths, requested_chrom_names)
    ctrl_bam_paths_by_chrom = index_bam_paths_by_chrom(ctrl_bam_paths, requested_chrom_names)

    chrom_names = [
        chrom for chrom in requested_chrom_names
        if wt_bam_paths_by_chrom.get(chrom) and ctrl_bam_paths_by_chrom.get(chrom)
    ]
    if not chrom_names:
        raise ValueError("No requested reference contigs have both WT/test and control BAM coverage.")

    chrom_info_list = [
        (chrom, 0, fref.get_reference_length(chrom))
        for chrom in chrom_names
    ]

    size, step, batch_size, num_trim = args.size, args.step, args.batch_size, args.num_trim

    labelled_arrays = get_labelled_arrays(
        ref=ref, seq_type=seq_type,
        pos_list=args.pos_file,
        motif=args.motif_string, motif_label=args.motif_label,
        chrom_list=chrom_names)

    call_mod_bams(
        model,
        ctrl_bam_paths_by_chrom,
        wt_bam_paths_by_chrom,
        wt_bam_paths[0],
        ref,
        kmer_table,
        block_size,
        chrom_info_list,
        args.bam_output,
        seq_type,
        size,
        step,
        base,
        mod_code,
        labelled_arrays,
        output_format=args.output_format,
        parquet_output=args.output,
        checkpoint_batch_step=args.checkpoint_batch_step,
        batch_size=batch_size,
        bam_threads=args.bam_threads,
        num_trim=num_trim,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        wt_num_workers=args.wt_num_workers,
        control_num_workers=args.control_num_workers,
        wt_parallel_loader=args.wt_parallel_loader,
        ctrl_parallel_loader=args.ctrl_parallel_loader,
        control_buffer_blocks=args.control_buffer_blocks,
        min_control_cov=args.min_control_cov,
        scale_factor=args.scale_factor,
        progress_every=args.progress_every,
        wt_use_polished_sequence=args.wt_use_polished_sequence,
        ctrl_use_polished_sequence=args.ctrl_use_polished_sequence,
        polish_indel_limit=args.polish_indel_limit,
        polish_vcf_path=args.polish_vcf_path,
    )


if __name__=='__main__':
    main()
