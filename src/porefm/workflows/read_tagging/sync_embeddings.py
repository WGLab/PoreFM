"""Synchronize WT/control BAM streams and write per-position embedding distances."""

from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
import argparse
import datetime
import heapq
import os
import sys
import time
from pathlib import Path

from porefm.cli._parser import PoreFMArgumentParser

def build_parser():
    """Build the CLI parser for synchronized embedding-distance extraction."""
    parser = PoreFMArgumentParser(
        description='Compute synchronized WT/control embedding distances over BAM positions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model_path', type=str, required=True, help='Path to a pretraining checkpoint.')
    parser.add_argument('--kmer_dict_path', type=str, required=True, help='Path to the k-mer expected-signal table.')
    parser.add_argument('--seq_type', type=str, choices=['dna', 'rna'], default='dna')
    parser.add_argument('--ref', type=str, required=True, help='Reference FASTA.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for block parquet files named <chrom>_<block_start>.parquet.')

    parser.add_argument(
        '--wt_bam', '--wt_path',
        dest='wt_inputs',
        nargs='+',
        action='append',
        help='WT/test BAM file(s) or directories. Directories are searched recursively for .bam files.',
    )
    parser.add_argument(
        '--wt_dir',
        dest='wt_inputs',
        nargs='+',
        action='append',
        help='Alias for --wt_bam for compatibility with other pairwise scripts.',
    )
    parser.add_argument(
        '--control_bam', '--ctrl_bam', '--control_path', '--ctrl_path',
        dest='control_inputs',
        nargs='+',
        action='append',
        help='Control BAM file(s) or directories. Directories are searched recursively for .bam files.',
    )
    parser.add_argument(
        '--control_dir', '--ctrl_dir',
        dest='control_inputs',
        nargs='+',
        action='append',
        help='Alias for --control_bam for compatibility with other pairwise scripts.',
    )

    parser.add_argument('--chrom_list', '--chrom', dest='chrom_list', type=str, nargs='+', help='Chromosomes to process. Defaults to all reference contigs.')
    parser.add_argument('--region', type=str, nargs='+', help='Explicit 0-based half-open regions, e.g. chr21:0-46709983.')

    parser.add_argument('--motif_string', type=str, default='', help="Motif specification supplied as one quoted shell argument, for example 'CG 0;CH 0'. If neither --motif_string nor --pos_file is set, all covered positions are stored.")
    parser.add_argument('--motif_label', type=str, default='0', help='Semicolon-delimited class labels corresponding to --motif_string.')
    parser.add_argument('--pos_file', type=str, default='', help='Optional labelled-position file. If set, distances are stored only at positions in the file.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of label classes expected by LabelledBamIterableDataset.')

    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=100000)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_trim', type=int, default=16, help='Trim this many bases from each read/chunk edge before accumulating positions.')
    parser.add_argument('--min_wt_cov', '--min_test_cov', dest='min_wt_cov', type=int, default=5)
    parser.add_argument('--min_control_cov', type=int, default=5)
    parser.add_argument('--pore_type', type=str, default='r10')

    parser.add_argument('--device', type=str, default='auto', help="Use 'auto', 'cuda', 'cuda:0', or 'cpu'.")
    compile_options = parser.add_mutually_exclusive_group()
    compile_options.add_argument('--compile_model', default=True, action='store_true', help='Compile the model with torch.compile before extraction (enabled by default).')
    compile_options.add_argument('--disable_compile', dest='compile_model', action='store_false', default=argparse.SUPPRESS, help='Disable model compilation for diagnostics.')
    parser.add_argument('--num_workers', type=int, default=1, help='Default DataLoader workers for both streams.')
    parser.add_argument('--wt_num_workers', type=int, default=None, help='WT/test DataLoader workers; defaults to --num_workers.')
    parser.add_argument('--control_num_workers', '--ctrl_num_workers', dest='control_num_workers', type=int, default=None, help='Control DataLoader workers; defaults to --num_workers.')
    parser.add_argument('--wt_parallel_loader', default=False, action='store_true', help='Use one labelled DataLoader per WT/test BAM and merge by coordinate.')
    parser.add_argument('--ctrl_parallel_loader', '--control_parallel_loader', dest='ctrl_parallel_loader', default=False, action='store_true', help='Use one labelled DataLoader per control BAM and merge by coordinate.')
    parser.add_argument('--prefetch_factor', type=int, default=8)
    parser.add_argument('--wt_use_polished_sequence', default=False, action='store_true', help='Use reference-guided polished base sequences for the WT/test stream.')
    parser.add_argument('--ctrl_use_polished_sequence', default=False, action='store_true', help='Use reference-guided polished base sequences for the control stream.')
    parser.add_argument('--polish_indel_limit', type=int, default=5, help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None, help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--skip_existing', default=False, action='store_true')

    add_preprocessing_arguments(parser)
    return parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_parser().parse_args()

import seaborn as sns
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from numpy.linalg import norm

import collections

import numpy as np
import numpy_indexed as npi

import pysam

from porefm.common.tensor_masks import ignore_edges
from porefm.common.chunking import resolve_downstream_step
from porefm.data.bam import build_lookup_table
from porefm.data.reader import LabelledBamIterableDataset, LabelledBamSample, labelled_collate_fn
from porefm.data.datasets.aligned import AlignedBamIterableDataset, AlignedBamSample
from porefm.data.collation import aligned_collate_fn, collate_modal_aligned_batch
from porefm.models.loading import load_model as load_checked_model

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

METRIC_NAMES = np.array([
    'mean_squared_distance',
    'cosine_distance',
    'max_pool_mean_squared_distance',
    'max_pool_cosine_distance',
])
COUNT_NAMES = np.array(['control', 'wt'])


def _expand_user_path(path):
    if path is None:
        return None
    return os.path.expanduser(str(path))


def _natsorted(paths):
    try:
        from natsort import natsorted
        return natsorted(paths)
    except ImportError:
        return sorted(paths)


def expand_bam_inputs(inputs):
    """Expand BAM files/directories recursively into a unique natural order."""
    bam_paths = []
    for input_path in inputs:
        path = Path(_expand_user_path(input_path))
        if path.is_dir():
            bam_paths.extend(str(x) for x in path.rglob('*.bam'))
        elif path.is_file():
            bam_paths.append(str(path))
        else:
            raise FileNotFoundError(f'BAM input does not exist: {path}')

    bam_paths = _natsorted(set(bam_paths))
    if not bam_paths:
        raise FileNotFoundError(f'No BAM files found in inputs: {inputs}')
    return bam_paths


def _flatten_arg_lists(values):
    if not values:
        return []
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _as_config_list(bam_config_or_list):
    if isinstance(bam_config_or_list, (list, tuple)):
        return list(bam_config_or_list)
    return [bam_config_or_list]


def _config_region(bam_configs):
    bam_configs = _as_config_list(bam_configs)
    if not bam_configs or not bam_configs[0].chrom_info:
        return None, None
    return bam_configs[0].chrom_info[0][1:]


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concat_or_empty(parts, axis=0, shape=(0,), dtype=np.float32):
    if not parts:
        return np.empty(shape, dtype=dtype)

    arrays = [_to_numpy(x) for x in parts]
    if len(arrays) == 1:
        return arrays[0].astype(dtype, copy=False)
    return np.concatenate(arrays, axis=axis).astype(dtype, copy=False)


def _normalize_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _parse_region(region):
    chrom, interval = region.split(':', 1)
    start, end = interval.replace(',', '').split('-', 1)
    return chrom, int(start), int(end)


def build_chrom_info(ref, chrom_list=None, regions=None):
    """Return explicit or full-contig 0-based half-open processing regions."""
    ref = _expand_user_path(ref)
    if regions:
        return [_parse_region(region) for region in regions]

    with pysam.FastaFile(ref) as fasta:
        if chrom_list is None:
            chrom_list = list(fasta.references)
        return [(chrom, 0, fasta.get_reference_length(chrom)) for chrom in chrom_list]


def make_labelled_bam_configs(
    bam_paths,
    chrom_info,
    ref,
    num_classes,
    pos_file='',
    motif='',
    motif_label=None,
    polish=False,
    polish_indel_limit=5,
    polish_vcf_path=None,
):
    """Create labelled iterable sample configs for every BAM and shared region."""
    return [
        LabelledBamSample(
            bam_path=_expand_user_path(bam_path),
            chrom_info=chrom_info,
            ref=_expand_user_path(ref),
            num_classes=num_classes,
            pos_file=pos_file,
            motif=motif,
            motif_label=motif_label,
            flag=0,
            polish=polish,
            polish_indel_limit=polish_indel_limit,
            polish_vcf_path=polish_vcf_path if polish else None,
        )
        for bam_path in bam_paths
    ]


def load_pretrain_model(model_path, device, compile_model=True):
    """Load a checked frozen backbone, compiling it unless explicitly disabled."""
    model = load_checked_model(
        _expand_user_path(model_path),
        model_type="backbone",
        add_signal_cross_attn=False,
        device=device,
    )

    if compile_model:
        model = torch.compile(model, dynamic=True)
    return model

class SingleGenomicAccumulator:
    """Accumulate strand-specific embedding means in fixed genomic blocks.

    GPU sum/count tensors are finalized behind a monotonically advancing
    coordinate frontier and moved to CPU before queueing.
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
            mean_vectors.cpu().float(), 
            block_data['count'].cpu()
        ))

    def pop_ready(self):
        """Yield finalized ``(block_start, means, counts)`` tuples."""
        while self.ready_queue:
            yield self.ready_queue.popleft()
            
def post_process_block(block_start, mean_vectors, counts, block_size, min_cov=5):
    """Select forward/reverse positions whose single-stream coverage is sufficient."""
        
    count_filter=(counts>=min_cov).numpy()
    
    positions=np.arange(block_start, block_start+block_size)

    fwd_pos_idx = count_filter[0]
    rev_pos_idx = count_filter[1]
    

    fwd_counts = counts[0][fwd_pos_idx]
    fwd_mean_vectors=mean_vectors[0, fwd_pos_idx]
    rev_counts = counts[1][rev_pos_idx]
    rev_mean_vectors=mean_vectors[1, rev_pos_idx]
    
    final_fwd_pos = positions[fwd_pos_idx]
    final_rev_pos = positions[rev_pos_idx]
    
    return (final_fwd_pos, fwd_counts, fwd_mean_vectors), (final_rev_pos, rev_counts, rev_mean_vectors)      

def get_stream_metrics(model, bam_config,
         seq_type, kmer_table,
         device=None,
         size=2048,
         step=None,
         block_size = 100_000,
         reads_per_chunk=100,
         min_read_length=500,
         min_qscore=5,
         batch_size=128,
         num_trim=16,
         select_labelled_positions=True,
         num_workers=1,
         prefetch_factor=8,
         pore_type=None):
    """Extract sorted per-strand mean embeddings and counts from one stream.

    Returns forward and reverse dictionaries containing position, count, and
    mean-vector arrays, sorted by coordinate.
    """
    
    device = _normalize_device(device)
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="single-stream embedding accumulation",
    )

    fwd_data = {'pos': [], 'counts': [], 'mean_vectors':[]}
    rev_data = {'pos': [], 'counts': [], 'mean_vectors':[]}

    bam_configs = _as_config_list(bam_config)

    dataset_kwargs=dict(size=size, step=step, move_downsample_factor=1, seq_type=seq_type, kmer_table=kmer_table,
                        reads_per_chunk=reads_per_chunk, min_read_length=min_read_length,
                        min_qscore=min_qscore, pore_type=pore_type, base_seed=0, rank=0, world_size=1,
                        expected_ts_dim=model.config.ts_dim, cycle=False, shuffle=False,
                        discard_empty_labels=select_labelled_positions)
    
    ds = LabelledBamIterableDataset(bam_list=bam_configs, **dataset_kwargs)
    
    r_start, r_end = _config_region(bam_configs)
    
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=labelled_collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(ds, **loader_kwargs)

    ds_iter = iter(loader)

    accumulator = SingleGenomicAccumulator(dim=model.config.hidden_dim, block_size=block_size, device=device)

    # Fetch first batches
    batch = next(ds_iter, None)

    t = time.time()
    start=time.time()
    while batch is not None:
        process_batch(batch, model, accumulator, device, r_start, r_end, None, num_trim, select_labelled_positions)
        batch = next(ds_iter, None)

        # Yield and calculate anything that is ready
        for block_data in accumulator.pop_ready():
            block_start, mean_vectors, counts = block_data
            print(f"[{time.time()-t:.2f}s] Processing matched block_start: {block_start}")
            fwd_block_data, rev_block_data = post_process_block(block_start, mean_vectors, counts, block_size, min_cov=5)
            fwd_pos, fwd_counts, fwd_mean_vectors = fwd_block_data
            rev_pos, rev_counts, rev_mean_vectors = rev_block_data

            fwd_data['pos'].append(fwd_pos)
            fwd_data['counts'].append(fwd_counts)
            fwd_data['mean_vectors'].append(fwd_mean_vectors)

            rev_data['pos'].append(rev_pos)
            rev_data['counts'].append(rev_counts)
            rev_data['mean_vectors'].append(rev_mean_vectors)

            t = time.time() # Reset timer

    accumulator.flush_all_blocks()
    for block_data in accumulator.pop_ready():
        block_start, mean_vectors, counts = block_data
        print(f"[{time.time()-t:.2f}s] Processing matched block_start: {block_start}")
        fwd_block_data, rev_block_data = post_process_block(block_start, mean_vectors, counts, block_size, min_cov=5)
        fwd_pos, fwd_counts, fwd_mean_vectors = fwd_block_data
        rev_pos, rev_counts, rev_mean_vectors = rev_block_data

        fwd_data['pos'].append(fwd_pos)
        fwd_data['counts'].append(fwd_counts)
        fwd_data['mean_vectors'].append(fwd_mean_vectors)

        rev_data['pos'].append(rev_pos)
        rev_data['counts'].append(rev_counts)
        rev_data['mean_vectors'].append(rev_mean_vectors)

        t = time.time() # Reset timer

    del accumulator    
    print(time.time()-start)
    
    fwd_data['pos'] = _concat_or_empty(fwd_data['pos'], dtype=np.int64)
    fwd_data['counts'] = _concat_or_empty(fwd_data['counts'], dtype=np.int32)
    fwd_data['mean_vectors'] = _concat_or_empty(
        fwd_data['mean_vectors'], axis=0, shape=(0, model.config.hidden_dim), dtype=np.float32
    )

    rev_data['pos'] = _concat_or_empty(rev_data['pos'], dtype=np.int64)
    rev_data['counts'] = _concat_or_empty(rev_data['counts'], dtype=np.int32)
    rev_data['mean_vectors'] = _concat_or_empty(
        rev_data['mean_vectors'], axis=0, shape=(0, model.config.hidden_dim), dtype=np.float32
    )

    sort_idx = np.argsort(fwd_data['pos'])
    fwd_data['pos'] = fwd_data['pos'][sort_idx]
    fwd_data['counts'] = fwd_data['counts'][sort_idx]
    fwd_data['mean_vectors'] = fwd_data['mean_vectors'][sort_idx]
    
    sort_idx = np.argsort(rev_data['pos'])
    rev_data['pos'] = rev_data['pos'][sort_idx]
    rev_data['counts'] = rev_data['counts'][sort_idx]
    rev_data['mean_vectors'] = rev_data['mean_vectors'][sort_idx]

    return fwd_data, rev_data

class DualGenomicAccumulator:
    """Accumulate synchronized WT/control sums and derive blockwise distances.

    The shared frontier is the smaller stream coordinate. Matched blocks are
    finalized only after both streams pass them; outputs retain four distance
    metrics and separate WT/control coverage.
    """

    def __init__(self, dim, block_size=100_000, dtype=torch.bfloat16, device='cuda'):
        self.dim = dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        
        # Track blocks separately for wt and control
        self.active_blocks = {'wt': {}, 'control': {}}
        
        # Track the minimum coordinate we've seen from each stream to safely flush
        self.frontiers = {'wt': 0, 'control': 0}
        
        # Yields tuple: (block_start, wt_embeddings, control_embeddings)
        self.ready_queue = collections.deque()

    def _get_or_create_block(self, stream_name, block_start):
        if block_start not in self.active_blocks[stream_name]:
            # ALLOCATE ON GPU FOR FAST ACCUMULATION
            self.active_blocks[stream_name][block_start] = {
                'sum': torch.zeros((2, self.block_size, self.dim), dtype=torch.float32, device=self.device),
                'count': torch.zeros((2, self.block_size), dtype=torch.int32, device=self.device)
            }
        return self.active_blocks[stream_name][block_start]

    def update(self, stream_name, indices, embeddings, strand, current_min_coord, unique_blocks_cpu):
        """Advance one stream and scatter-add its rows into active blocks."""
        self.frontiers[stream_name] = current_min_coord

        # Identify blocks
        block_ids = torch.div(indices, self.block_size, rounding_mode='floor')
        
        # Update blocks ON GPU
        for bid in unique_blocks_cpu:
            block_start = int(bid) * self.block_size
            mask = (block_ids == bid)
            
            block_data = self._get_or_create_block(stream_name, block_start)
            
            for s in (0, 1):
                s_mask = mask & (strand == s)
                
                # --- NO 'IF' STATEMENT ---
                # We blindly slice and add. If s_mask is entirely False, 
                # PyTorch handles the 0-sized tensors natively without blocking the CPU.
                local_indices = indices[s_mask] - block_start
                local_embeds = embeddings[s_mask].float() 
                
                block_data['sum'][s].index_add_(0, local_indices, local_embeds)
                block_data['count'][s].index_add_(0, local_indices, torch.ones_like(local_indices, dtype=torch.int32))

        safe_frontier = min(self.frontiers['wt'], self.frontiers['control'])
        self._flush_old_blocks(safe_frontier)

    def _flush_old_blocks(self, safe_frontier):
        wt_blocks = set(self.active_blocks['wt'].keys())
        control_blocks = set(self.active_blocks['control'].keys())
        
        # Check ALL active blocks to catch orphans
        all_blocks = wt_blocks.union(control_blocks)
        
        for block_start in sorted(list(all_blocks)):
            block_end = block_start + self.block_size
            if block_end <= safe_frontier:
                if block_start in wt_blocks and block_start in control_blocks:
                    # Block exists in both, process normally
                    self._enqueue_matched_block(block_start)
                else:
                    # ORPHAN BLOCK: We have passed the safe frontier, but this block 
                    # only has data in one stream. Discard it to prevent GPU leaks.
                    self.active_blocks['wt'].pop(block_start, None)
                    self.active_blocks['control'].pop(block_start, None)
    
    def flush_all_blocks(self):
        """Finalize every remaining block containing both stream populations."""
        wt_blocks = set(self.active_blocks['wt'].keys())
        control_blocks = set(self.active_blocks['control'].keys())
        
        all_blocks = wt_blocks.union(control_blocks)
        
        for block_start in sorted(list(all_blocks)):
            if block_start in wt_blocks and block_start in control_blocks:
                self._enqueue_matched_block(block_start)
            else:
                self.active_blocks['wt'].pop(block_start, None)
                self.active_blocks['control'].pop(block_start, None)
            
    def _enqueue_matched_block(self, block_start):
        wt_data = self.active_blocks['wt'].pop(block_start)
        wt_counts = torch.clamp_min(wt_data['count'], 1).unsqueeze(-1)
        wt_mean_vectors = (wt_data['sum'] / wt_counts).to(self.dtype)
        
        control_data = self.active_blocks['control'].pop(block_start)
        control_counts = torch.clamp_min(control_data['count'], 1).unsqueeze(-1)
        control_mean_vectors = (control_data['sum'] / control_counts).to(self.dtype)
        
        kernel_size=5
        diff = torch.mean((wt_mean_vectors-control_mean_vectors)**2, 2)
        max_pool_diff = F.max_pool1d(diff, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        
        cosine_sim = F.cosine_similarity(wt_mean_vectors.float(), control_mean_vectors.float(), dim=-1)
        cosine_dist = torch.clamp(1.0 - cosine_sim, min=0.0, max=2.0)
        max_pool_cosine_dist = F.max_pool1d(cosine_dist, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        
        metrics=(diff.cpu().float(), cosine_dist.cpu().float(), \
                                 max_pool_diff.cpu().float(), max_pool_cosine_dist.cpu().float())
        self.ready_queue.append((block_start, metrics,
                                 control_counts.cpu().squeeze(), 
                                 wt_counts.cpu().squeeze()))
        
#         ({k:v.cpu() for k,v in wt_data.items()}, {k:v.cpu() for k,v in control_data.items()})

    def pop_ready(self):
        """Yield finalized distance blocks in release order."""
        while self.ready_queue:
            yield self.ready_queue.popleft()
            
def post_process_dual_block(block_start, metrics, control_counts, wt_counts,
                            block_size, min_cov_wt=5, min_cov_control=5):
    """Split a distance block by strand after applying both coverage thresholds."""
    
    diff, cosine_dist, max_pool_diff, max_pool_cosine_dist = metrics
    
    count_filter=((control_counts>=min_cov_control)&(wt_counts>=min_cov_wt)).numpy()
    
    positions=np.arange(block_start, block_start+block_size)

    fwd_pos_idx = count_filter[0]
    
    fwd_counts = torch.stack((control_counts[0][fwd_pos_idx], wt_counts[0][fwd_pos_idx]))
    fwd_diff=diff[0, fwd_pos_idx]
    fwd_cosine_dist=cosine_dist[0, fwd_pos_idx]
    mp_fwd_diff=max_pool_diff[0, fwd_pos_idx]
    mp_fwd_cosine_dist=max_pool_cosine_dist[0, fwd_pos_idx]
    
    rev_pos_idx = count_filter[1]
    
    rev_counts = torch.stack((control_counts[1][rev_pos_idx], wt_counts[1][rev_pos_idx]))
    rev_diff=diff[1, rev_pos_idx]
    rev_cosine_dist=cosine_dist[1, rev_pos_idx]
    mp_rev_diff=max_pool_diff[1, rev_pos_idx]
    mp_rev_cosine_dist=max_pool_cosine_dist[1, rev_pos_idx]

    final_fwd_pos = positions[fwd_pos_idx]
    final_rev_pos = positions[rev_pos_idx]
    
    fwd_metrics=torch.vstack([fwd_diff, fwd_cosine_dist, mp_fwd_diff, mp_fwd_cosine_dist])
    rev_metrics=torch.vstack([rev_diff, rev_cosine_dist, mp_rev_diff, mp_rev_cosine_dist])
    
    return (final_fwd_pos, fwd_counts, fwd_metrics), (final_rev_pos, rev_counts, rev_metrics)


def write_distance_block(output_dir, chrom, block_start, fwd_block_data, rev_block_data):
    """Write one non-empty strand-combined distance block as Parquet."""
    fwd_pos, fwd_counts, fwd_metrics = fwd_block_data
    rev_pos, rev_counts, rev_metrics = rev_block_data

    pos = np.concatenate((fwd_pos, rev_pos)).astype(np.int64, copy=False)
    if len(pos) == 0:
        return None

    strand = np.concatenate((np.zeros_like(fwd_pos), np.ones_like(rev_pos))).astype(np.int8, copy=False)
    counts = np.concatenate((_to_numpy(fwd_counts), _to_numpy(rev_counts)), axis=1).astype(np.int32, copy=False)
    metrics = np.concatenate((_to_numpy(fwd_metrics), _to_numpy(rev_metrics)), axis=1).astype(np.float32, copy=False)

    order = np.lexsort((strand, pos))
    pos = pos[order]
    strand = strand[order]
    counts = counts[:, order]
    metrics = metrics[:, order]

    safe_chrom = str(chrom).replace(os.sep, '_')
    os.makedirs(output_dir, exist_ok=True)
    curr_path = os.path.join(output_dir, f'{safe_chrom}_{block_start}.parquet')
    table = pa.Table.from_pydict({
        'chrom': np.asarray([chrom] * len(pos), dtype=object),
        'position': pos,
        'strand': strand,
        'control_count': counts[0],
        'wt_count': counts[1],
        'mean_squared_distance': metrics[0],
        'cosine_distance': metrics[1],
        'max_pool_mean_squared_distance': metrics[2],
        'max_pool_cosine_distance': metrics[3],
    })
    pq.write_table(table, curr_path, compression='snappy')
    return curr_path


class SortedCoordinateMerger:
    """Merge coordinate-sorted labelled DataLoaders with a one-batch heap."""

    def __init__(self, dataloaders, chrom_to_idx):
        self.iters = [iter(dl) for dl in dataloaders]
        self.chrom_to_idx = chrom_to_idx
        self.heap = []

        for loader_idx, iterator in enumerate(self.iters):
            try:
                self._push_to_heap(next(iterator), loader_idx)
            except StopIteration:
                pass

    def _push_to_heap(self, batch, loader_idx):
        min_start = batch[-2]
        min_chrom = batch[-3]
        chrom_idx = self.chrom_to_idx.get(min_chrom, float('inf'))
        heapq.heappush(self.heap, (chrom_idx, min_start, loader_idx, batch))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.heap:
            raise StopIteration

        _, _, loader_idx, batch = heapq.heappop(self.heap)
        try:
            self._push_to_heap(next(self.iters[loader_idx]), loader_idx)
        except StopIteration:
            pass
        return batch


def get_labelled_iter(dataset_kwargs, bam_config_list, batch_size, num_workers, device, chrom_to_idx, parallel_files=False, prefetch_factor=4):
    """Create a labelled iterator, optionally merging one loader per BAM."""
    if parallel_files:
        loader_kwargs = dict(
            batch_size=batch_size,
            collate_fn=labelled_collate_fn,
            num_workers=1,
            pin_memory=device.type == 'cuda',
            prefetch_factor=prefetch_factor,
        )
        loaders = [
            DataLoader(LabelledBamIterableDataset(bam_list=[bam_config], **dataset_kwargs), **loader_kwargs)
            for bam_config in bam_config_list
        ]
        return iter(SortedCoordinateMerger(loaders, chrom_to_idx))

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=labelled_collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(LabelledBamIterableDataset(bam_list=bam_config_list, **dataset_kwargs), **loader_kwargs)
    return iter(loader)


def get_dual_stream_metrics(model, bam_config_A, bam_config_B,
         seq_type, kmer_table,
         device=None,
         size=2048,
         step=None,
         block_size = 100_000,
         reads_per_chunk=100,
         min_read_length=500,
         min_qscore=5,
         batch_size=128,
         num_trim=16,
         min_cov_wt=5,
         min_cov_control=5,
         wt_num_workers=1,
         control_num_workers=1,
         prefetch_factor=8,
         pore_type=None,
         select_labelled_positions=True,
         output_dir=None,
         chrom=None,
         wt_parallel_loader=False,
         ctrl_parallel_loader=False):
    """Synchronize two BAM streams and compute four per-position distances.

    Streams advance in chromosome/coordinate order into a shared accumulator.
    Final blocks are written immediately when ``output_dir`` is supplied;
    otherwise returns ``(positions, counts, metrics, strands)`` arrays, where
    counts have WT/control rows and metrics follow ``METRIC_NAMES``.
    """
    
    device = _normalize_device(device)
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="synchronized embedding comparison",
    )

    fwd_data = {'pos': [], 'counts': [], 'metrics':[]}
    rev_data = {'pos': [], 'counts': [], 'metrics':[]}

    wt_bam_configs = _as_config_list(bam_config_A)
    control_bam_configs = _as_config_list(bam_config_B)

    chrom_order = [info[0] for info in wt_bam_configs[0].chrom_info]
    chrom_to_idx = {c: i for i, c in enumerate(chrom_order)}

    dataset_kwargs=dict(size=size, step=step, move_downsample_factor=1, seq_type=seq_type, kmer_table=kmer_table,
                        reads_per_chunk=reads_per_chunk, min_read_length=min_read_length,
                        min_qscore=min_qscore, pore_type=pore_type, base_seed=0, rank=0, world_size=1,
                        expected_ts_dim=model.config.ts_dim, cycle=False, shuffle=False,
                        discard_empty_labels=select_labelled_positions)
    
    wt_r_start, wt_r_end = _config_region(wt_bam_configs)
    control_r_start, control_r_end = _config_region(control_bam_configs)
    
    starts = [x for x in (wt_r_start, control_r_start) if x is not None]
    r_start = max(starts) if starts else None

    ends = [x for x in (wt_r_end, control_r_end) if x is not None]
    r_end = min(ends) if ends else None
    
    wt_iter = get_labelled_iter(
        dataset_kwargs, wt_bam_configs, batch_size, wt_num_workers, device,
        chrom_to_idx, parallel_files=wt_parallel_loader, prefetch_factor=prefetch_factor
    )
    control_iter = get_labelled_iter(
        dataset_kwargs, control_bam_configs, batch_size, control_num_workers, device,
        chrom_to_idx, parallel_files=ctrl_parallel_loader, prefetch_factor=prefetch_factor
    )


    accumulator = DualGenomicAccumulator(dim=model.config.hidden_dim, block_size=block_size, device=device)

    # Fetch first batches
    wt_batch = next(wt_iter, None)
    control_batch = next(control_iter, None)

    t = time.time()
    start=time.time()
    while wt_batch is not None or control_batch is not None:

        # Decide which stream to advance to keep them synchronized in coordinate space
        # If one is exhausted (None), we advance the other.
        if wt_batch is not None and control_batch is not None:
            wt_min_coord = wt_batch[-2]
            wt_chrom_idx = chrom_to_idx.get(wt_batch[-3], 0)
            control_min_coord = control_batch[-2]
            control_chrom_idx = chrom_to_idx.get(control_batch[-3], 0)

            if (wt_chrom_idx, wt_min_coord) <= (control_chrom_idx, control_min_coord):
                process_batch(wt_batch, model, accumulator, device, r_start, r_end, 'wt', num_trim, select_labelled_positions)
                wt_batch = next(wt_iter, None)
            else:
                process_batch(control_batch, model, accumulator, device, r_start, r_end, 'control', num_trim, select_labelled_positions)
                control_batch = next(control_iter, None)

        elif wt_batch is not None:
            process_batch(wt_batch, model, accumulator, device, r_start, r_end, 'wt', num_trim, select_labelled_positions)
            wt_batch = next(wt_iter, None)

        elif control_batch is not None:
            process_batch(control_batch, model, accumulator, device, r_start, r_end, 'control', num_trim, select_labelled_positions)
            control_batch = next(control_iter, None)

        # Yield and calculate anything that is ready
        for block_data in accumulator.pop_ready():
            block_start, metrics, control_counts, wt_counts = block_data
            print(f"[{time.time()-t:.2f}s] Processing matched block_start: {block_start}")
            fwd_block_data, rev_block_data = post_process_dual_block(
                block_start, metrics, control_counts, wt_counts, block_size,
                min_cov_wt=min_cov_wt, min_cov_control=min_cov_control
            )
            fwd_pos, fwd_counts, fwd_metrics = fwd_block_data
            rev_pos, rev_counts, rev_metrics = rev_block_data

            if output_dir:
                output_path = write_distance_block(output_dir, chrom or chrom_order[0], block_start, fwd_block_data, rev_block_data)
                if output_path is not None:
                    print(f"Wrote {output_path}", flush=True)
            else:
                fwd_data['pos'].append(fwd_pos)
                fwd_data['counts'].append(fwd_counts)
                fwd_data['metrics'].append(fwd_metrics)

                rev_data['pos'].append(rev_pos)
                rev_data['counts'].append(rev_counts)
                rev_data['metrics'].append(rev_metrics)


            t = time.time() # Reset timer

    accumulator.flush_all_blocks()
    for block_data in accumulator.pop_ready():
        block_start, metrics, control_counts, wt_counts = block_data
        print(f"[{time.time()-t:.2f}s] Processing matched block_start: {block_start}")
        fwd_block_data, rev_block_data = post_process_dual_block(
            block_start, metrics, control_counts, wt_counts, block_size,
            min_cov_wt=min_cov_wt, min_cov_control=min_cov_control
        )

        fwd_pos, fwd_counts, fwd_metrics = fwd_block_data
        rev_pos, rev_counts, rev_metrics = rev_block_data

        if output_dir:
            output_path = write_distance_block(output_dir, chrom or chrom_order[0], block_start, fwd_block_data, rev_block_data)
            if output_path is not None:
                print(f"Wrote {output_path}", flush=True)
        else:
            fwd_data['pos'].append(fwd_pos)
            fwd_data['counts'].append(fwd_counts)
            fwd_data['metrics'].append(fwd_metrics)

            rev_data['pos'].append(rev_pos)
            rev_data['counts'].append(rev_counts)
            rev_data['metrics'].append(rev_metrics)

        t = time.time() # Reset timer

    del accumulator
    print(f'Total Time = {time.time()-start:.2f}')

    if output_dir:
        return None
    
    fwd_data['pos'] = _concat_or_empty(fwd_data['pos'], dtype=np.int64)
    fwd_data['counts'] = _concat_or_empty(fwd_data['counts'], axis=1, shape=(2, 0), dtype=np.int32)
    fwd_data['metrics'] = _concat_or_empty(fwd_data['metrics'], axis=1, shape=(4, 0), dtype=np.float32)

    rev_data['pos'] = _concat_or_empty(rev_data['pos'], dtype=np.int64)
    rev_data['counts'] = _concat_or_empty(rev_data['counts'], axis=1, shape=(2, 0), dtype=np.int32)
    rev_data['metrics'] = _concat_or_empty(rev_data['metrics'], axis=1, shape=(4, 0), dtype=np.float32)

    pos = np.concatenate((fwd_data['pos'], rev_data['pos'])).astype(np.int64, copy=False)
    counts = np.concatenate((fwd_data['counts'], rev_data['counts']), axis=1).astype(np.int32, copy=False)
    metrics = np.concatenate((fwd_data['metrics'], rev_data['metrics']), axis=1).astype(np.float32, copy=False)
    strand = np.concatenate((np.zeros_like(fwd_data['pos']), np.ones_like(rev_data['pos']))).astype(np.int8, copy=False)
    
    return pos, counts, metrics, strand

def process_aligned_batch(
    batch,
    model,
    accumulator,
    device,
    r_start=None,
    r_end=None,
    num_trim=16,
):
    """Accumulate embeddings from the canonical 9-field aligned-read batch."""
    (
        dna_input,
        ts_input,
        percentiles_batch,
        pad_base_seq,
        pad_ts,
        _readnames,
        positions,
        strands,
        min_start,
    ) = batch

    trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
    filter_pos = (positions >= 0) & trimmed_pad_base_seq

    if r_start is not None and r_end is not None:
        filter_pos = filter_pos & (positions >= r_start) & (positions <= r_end)

    filter_norm_pos_cpu = positions[filter_pos]
    unique_blocks_cpu = (
        torch.div(
            filter_norm_pos_cpu,
            accumulator.block_size,
            rounding_mode='floor',
        ).unique().tolist()
        if filter_norm_pos_cpu.numel() > 0
        else []
    )

    if filter_norm_pos_cpu.numel() == 0:
        accumulator.update(
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(
                (0, model.config.hidden_dim),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(0, dtype=torch.long, device=device),
            min_start,
            unique_blocks_cpu,
        )
        return

    dna_input = dna_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_batch = percentiles_batch.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    positions = positions.to(device, non_blocking=True)
    filter_pos = filter_pos.to(device, non_blocking=True)
    strands = strands.to(device, non_blocking=True)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
    ):
        out = model(
            dna_input,
            ts_input,
            percentiles_input=percentiles_batch,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
        )

    accumulator.update(
        positions[filter_pos],
        out['contextual_base_seq'][filter_pos],
        strands.unsqueeze(1).expand_as(positions)[filter_pos],
        min_start,
        unique_blocks_cpu,
    )


def process_batch(batch, model, accumulator, device, r_start=None, r_end=None, stream_name=None, num_trim=16, select_labelled_positions=True):
    """Handle the labelled-iterable tuple and select positions via labels_idx when requested."""
    dna_input, ts_input, percentiles_batch, pad_base_seq, pad_ts, labels, labels_idx, positions, strands, readnames, refnames, min_chrom, min_start, rank = batch

    trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
    filter_pos = (positions >= 0) & trimmed_pad_base_seq
    if select_labelled_positions:
        filter_pos = filter_pos & labels_idx.bool()

    if r_start is not None and r_end is not None:
        filter_pos = filter_pos & (positions >= r_start) & (positions <= r_end)

    filter_norm_pos_cpu = positions[filter_pos]
    unique_blocks_cpu = torch.div(
        filter_norm_pos_cpu,
        accumulator.block_size,
        rounding_mode='floor'
    ).unique().tolist() if filter_norm_pos_cpu.numel() > 0 else []

    if filter_norm_pos_cpu.numel() == 0:
        empty_pos = torch.empty(0, dtype=torch.long, device=device)
        empty_ctx = torch.empty((0, model.config.hidden_dim), dtype=torch.float32, device=device)
        empty_strand = torch.empty(0, dtype=torch.long, device=device)
        if stream_name:
            accumulator.update(stream_name, empty_pos, empty_ctx, empty_strand, min_start, unique_blocks_cpu)
        else:
            accumulator.update(empty_pos, empty_ctx, empty_strand, min_start, unique_blocks_cpu)
        return

    dna_input = dna_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_batch = percentiles_batch.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    positions = positions.to(device, non_blocking=True)
    filter_pos = filter_pos.to(device, non_blocking=True)
    strands = strands.to(device, non_blocking=True)

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(dna_input, ts_input,
                    percentiles_input=percentiles_batch,
                    pad_base_seq=pad_base_seq, pad_ts=pad_ts)

    filter_norm_pos = positions[filter_pos]
    filter_ctx = out['contextual_base_seq'][filter_pos]
    filter_strands = strands.unsqueeze(1).expand_as(positions)[filter_pos]

    if stream_name:
        accumulator.update(stream_name, filter_norm_pos, filter_ctx, filter_strands, min_start, unique_blocks_cpu)
    else:
        accumulator.update(filter_norm_pos, filter_ctx, filter_strands, min_start, unique_blocks_cpu)
        
def get_olp(data, control_data):
    """Align two sorted position dictionaries and return squared mean differences."""
    pos, counts, mean_vectors = data['pos'], data['counts'], data['mean_vectors']
    control_pos, control_counts, control_mean_vectors = control_data['pos'], control_data['counts'], control_data['mean_vectors']
    
    idx = np.searchsorted(control_pos, pos)
    clamped_idx = np.minimum(idx, len(control_pos) - 1)
    mask = (pos == control_pos[clamped_idx])

    pos          = pos[mask]
    counts       = counts[mask]
    mean_vectors = mean_vectors[mask]

    matching_control_idx = clamped_idx[mask]

    # Filtering the Control arrays
    control_pos          = control_pos[matching_control_idx]
    control_counts       = control_counts[matching_control_idx]
    control_mean_vectors = control_mean_vectors[matching_control_idx]

    diff=np.mean((mean_vectors-control_mean_vectors)**2, 1)
    
    return pos, diff, counts, control_counts, mean_vectors, control_mean_vectors


def run(args):
    """Validate inputs and process each requested region into Parquet blocks."""
    resolve_cli_preprocessing(args)
    ref = _expand_user_path(args.ref)
    model_path = _expand_user_path(args.model_path)
    kmer_dict_path = _expand_user_path(args.kmer_dict_path) if args.kmer_dict_path else None
    pos_file = _expand_user_path(args.pos_file) if args.pos_file else ''

    wt_inputs = _flatten_arg_lists(args.wt_inputs)
    control_inputs = _flatten_arg_lists(args.control_inputs)
    if not wt_inputs:
        raise ValueError('Provide at least one WT/test BAM input with --wt_bam or --wt_dir.')
    if not control_inputs:
        raise ValueError('Provide at least one control BAM input with --control_bam/--ctrl_bam or --control_dir/--ctrl_dir.')

    wt_bam_paths = expand_bam_inputs(wt_inputs)
    control_bam_paths = expand_bam_inputs(control_inputs)
    chrom_info_list = build_chrom_info(ref, chrom_list=args.chrom_list, regions=args.region)
    output_dir = _expand_user_path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Loading model: {model_path}', flush=True)
    model = load_pretrain_model(model_path, device, compile_model=args.compile_model)
    kmer_table = resolve_cli_preprocessing(args).kmer_table

    wt_num_workers = args.wt_num_workers if args.wt_num_workers is not None else args.num_workers
    control_num_workers = args.control_num_workers if args.control_num_workers is not None else args.num_workers

    if wt_num_workers != 1 or control_num_workers != 1:
        print(
            'Warning: DataLoader workers > 1 are only appropriate when each input source is '
            'coordinate-ordered in a way that preserves frontier correctness.',
            flush=True,
        )

    selection_mode = 'labelled' if (pos_file or args.motif_string) else 'all'
    written = []

    for region in chrom_info_list:
        chrom, start, end = region
        expected_prefix = f'{str(chrom).replace(os.sep, "_")}_'
        existing_outputs = list(Path(output_dir).glob(f'{expected_prefix}*.parquet'))
        if args.skip_existing and existing_outputs:
            print(f'Skipping {chrom}:{start}-{end}; parquet output already exists in {output_dir}', flush=True)
            continue
        if existing_outputs and not args.overwrite:
            raise FileExistsError(
                f'Found existing parquet outputs for {chrom} in {output_dir}. '
                'Use --overwrite to replace matching block files or --skip_existing to skip this region.'
            )

        print(
            f'Processing {chrom}:{start}-{end} with {len(wt_bam_paths)} WT/test BAM(s), '
            f'{len(control_bam_paths)} control BAM(s), selection={selection_mode}',
            flush=True,
        )

        region_chrom_info = [region]
        wt_bam_configs = make_labelled_bam_configs(
            wt_bam_paths,
            region_chrom_info,
            ref,
            args.num_classes,
            pos_file=pos_file,
            motif=args.motif_string,
            motif_label=args.motif_label,
            polish=args.wt_use_polished_sequence,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=args.polish_vcf_path,
        )
        control_bam_configs = make_labelled_bam_configs(
            control_bam_paths,
            region_chrom_info,
            ref,
            args.num_classes,
            pos_file=pos_file,
            motif=args.motif_string,
            motif_label=args.motif_label,
            polish=args.ctrl_use_polished_sequence,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=args.polish_vcf_path,
        )

        get_dual_stream_metrics(
            model,
            wt_bam_configs,
            control_bam_configs,
            args.seq_type,
            kmer_table,
            device,
            size=args.size,
            step=args.step,
            block_size=args.block_size,
            reads_per_chunk=args.reads_per_chunk,
            min_read_length=args.min_read_length,
            min_qscore=args.min_qscore,
            batch_size=args.batch_size,
            num_trim=args.num_trim,
            min_cov_wt=args.min_wt_cov,
            min_cov_control=args.min_control_cov,
            wt_num_workers=wt_num_workers,
            control_num_workers=control_num_workers,
            prefetch_factor=args.prefetch_factor,
            pore_type=args.pore_type,
            select_labelled_positions=selection_mode == 'labelled',
            output_dir=output_dir,
            chrom=chrom,
            wt_parallel_loader=args.wt_parallel_loader,
            ctrl_parallel_loader=args.ctrl_parallel_loader,
        )
        written.append(output_dir)

    return written


def main():
    """Parse CLI arguments and run synchronized embedding extraction."""
    print('\n%s:\nCommand: python %s\n' % (str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
