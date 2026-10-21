"""Extract paired WT/control embeddings for block anomaly scoring or storage."""

from porefm.cli.preprocessing_args import resolve_cli_preprocessing
import sys
from porefm.cli.dual_stream_args import build_pairwise_distance_parser


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    build_pairwise_distance_parser().parse_args()

import seaborn as sns
import pandas as pd
from safetensors.torch import save_file
import torch.distributed as dist
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import heapq
from pathlib import Path

from porefm.workflows.anomaly.detection import process_block_anomaly_detection_vectorized
from porefm.workflows.prototypes.embedding_selection import get_embedding_dimension, select_base_embedding
from natsort import natsorted

#choose one
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from loky import get_reusable_executor

import threading
import queue
import array
import torch
import torch.nn as nn
from numpy.linalg import norm

import collections
import random
import numpy as np
import pysam

from porefm.models.loading import load_model as load_checked_model
from porefm.common.chunking import resolve_downstream_step
from porefm.common.utils import require_single_process_execution
from porefm.common.tensor_masks import ignore_edges
from porefm.data.bam import build_lookup_table
from porefm.data.reader import LabelledBamIterableDataset, LabelledBamSample, labelled_collate_fn
from porefm.data.excluded_regions import get_excluded_regions
from porefm.data.datasets.aligned import AlignedBamIterableDataset, AlignedBamSample
from porefm.data.collation import aligned_collate_fn, collate_modal_aligned_batch
# from sync_embedding_extractor import *
# from completeReadIterable import *

import os
import time
import torch
import numpy as np
import pandas as pd
import collections
import queue
import threading
from torch.utils.data import DataLoader


def build_block_groups(refnames, block_ids_np):
    """
    Build (chromosome, block_id) -> row-index groups in a single pass.

    This replaces the previous set(unique keys) + full-array mask scan per block.
    It preserves row order within each block, so deduplication keeps the same
    first-seen read for a duplicated (position, readname) pair.
    """
    groups = collections.OrderedDict()
    for row_idx, key in enumerate(zip(refnames, block_ids_np)):
        chrom, block_id = key
        key = (chrom, int(block_id))
        groups.setdefault(key, []).append(row_idx)

    return [
        (key, np.asarray(row_indices, dtype=np.int64))
        for key, row_indices in groups.items()
    ]


def _tensor_to_numpy_for_npz(tensor):
    """Return a NumPy payload plus dtype metadata, preserving bfloat16 exactly."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype is torch.bfloat16:
        return tensor.view(torch.uint16).numpy(), 'torch.bfloat16_as_uint16'
    return tensor.numpy(), str(tensor.dtype)


def _npz_stream_payload(prefix, compiled):
    emb_payload, emb_dtype = _tensor_to_numpy_for_npz(compiled['embeddings'])
    return {
        f'{prefix}_position': compiled['positions'].detach().cpu().numpy(),
        f'{prefix}_strand': compiled['strands'].detach().cpu().numpy(),
        f'{prefix}_readname': np.asarray(compiled['readnames'], dtype=str),
        f'{prefix}_embeddings': emb_payload,
        f'{prefix}_embedding_dtype': np.asarray(emb_dtype),
        f'{prefix}_embedding_shape': np.asarray(tuple(compiled['embeddings'].shape), dtype=np.int64),
    }


def write_embeddings_task(b_chrom, block_start, wt_compiled, control_compiled, output_dir, write_mode='fast'):
    """Write already-filtered accumulated embeddings for one matched block.

    fast:       uncompressed torch serialization; preserves tensor dtypes exactly.
    compressed: npz/deflate; stores bfloat16 as raw uint16 bits to avoid float conversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_chrom = str(b_chrom).replace(os.sep, '_')

    if write_mode == 'fast':
        curr_path = os.path.join(output_dir, f'{safe_chrom}_{block_start}.pt')
        torch.save(
            {
                'chrom': b_chrom,
                'block_start': block_start,
                'wt': wt_compiled,
                'control': control_compiled,
            },
            curr_path,
        )
    elif write_mode == 'compressed':
        curr_path = os.path.join(output_dir, f'{safe_chrom}_{block_start}.npz')
        payload = {
            'chrom': np.asarray(str(b_chrom)),
            'block_start': np.asarray(block_start, dtype=np.int64),
        }
        payload.update(_npz_stream_payload('wt', wt_compiled))
        payload.update(_npz_stream_payload('control', control_compiled))
        np.savez_compressed(curr_path, **payload)
    else:
        raise ValueError(f"Unknown embedding write_mode: {write_mode}")

    return b_chrom, block_start


class DualGenomicAccumulator:
    """Accumulate deduplicated WT/control embeddings until blocks are safe.

    Worker-specific coordinate frontiers control release. Only blocks observed
    in both streams are queued, and rows are deduplicated by
    ``(position, readname)`` at compilation.
    """

    def __init__(self, dim, wt_num_workers, control_num_workers, chrom_order, block_size=100_000, dtype=torch.bfloat16):
        self.dim = dim
        self.block_size = block_size
        self.dtype = dtype
        self.wt_num_workers = wt_num_workers
        self.control_num_workers = control_num_workers
        
        # Mapping chromosome names to their global sorted index
        self.chrom_order = chrom_order
        self.chrom_to_idx = {chrom: idx for idx, chrom in enumerate(chrom_order)}
        
        # Active blocks are now keyed by a tuple: (chrom, block_start)
        self.active_blocks = {'wt': {}, 'control': {}}
        
        # Track the frontier for EACH worker as a tuple: (chrom_idx, min_coord)
        effective_control_workers = max(1, control_num_workers)
        effective_wt_workers = max(1, wt_num_workers)
        self.frontiers = {
            'wt': {i: (0, 0) for i in range(effective_wt_workers)},
            'control': {i: (0, 0) for i in range(effective_control_workers)}
        }
        
        self.stream_done = {'wt': False, 'control': False}
        self.ready_queue = collections.deque()

    def set_stream_done(self, stream_name):
        """Mark one stream complete and flush blocks newly behind the frontier."""
        self.stream_done[stream_name] = True
        self._check_and_flush()

    def _get_or_create_block(self, stream_name, chrom, block_start):
        key = (chrom, block_start)
        if key not in self.active_blocks[stream_name]:
            self.active_blocks[stream_name][key] = {
                'positions': [], 'embeddings': [], 'strands': [], 'readnames': []
            }
        return self.active_blocks[stream_name][key]

    def update(self, stream_name, indices, embeddings, strands, readnames, refnames, min_chrom, min_coord, rank, block_groups_cpu):
        """Advance a worker frontier and append pre-grouped embedding rows."""
        # 1. Update this specific worker's 2D frontier
        # If an unknown chromosome appears, default to index 0 safely (or handle as needed)
        chrom_idx = self.chrom_to_idx.get(min_chrom, 0)
        self.frontiers[stream_name][rank] = (chrom_idx, min_coord)
        
        if len(indices) == 0:
            self._check_and_flush()
            return

        # 2. Iterate through pre-built (chrom, block_id) -> row-index groups.
        # This avoids rescanning the entire filtered batch once per block.
        for (b_chrom, bid), row_indices_np in block_groups_cpu:
            if row_indices_np.size == 0:
                continue

            block_start = int(bid) * self.block_size
            row_indices = torch.from_numpy(row_indices_np).to(indices.device)

            local_indices = indices.index_select(0, row_indices).clone()
            local_embeds = embeddings.index_select(0, row_indices).to(self.dtype).clone()
            local_strands = strands.index_select(0, row_indices).clone()
            local_readnames = readnames[row_indices_np]

            block_data = self._get_or_create_block(stream_name, b_chrom, block_start)
            block_data['positions'].append(local_indices)
            block_data['embeddings'].append(local_embeds)
            block_data['strands'].append(local_strands)
            block_data['readnames'].append(local_readnames)

        # 3. Attempt to flush
        self._check_and_flush()

    def _check_and_flush(self):
        # Python natively compares tuples correctly: (0, 100) < (1, 0) is True
        inf_frontier = (float('inf'), float('inf'))
        
        wt_safe = min(self.frontiers['wt'].values()) if not self.stream_done['wt'] else inf_frontier
        control_safe = min(self.frontiers['control'].values()) if not self.stream_done['control'] else inf_frontier

        safe_frontier = min(wt_safe, control_safe)
        
        if safe_frontier != inf_frontier:
            self._flush_old_blocks(safe_frontier)

    def _flush_old_blocks(self, safe_frontier):
        safe_chrom_idx, safe_coord = safe_frontier
        
        wt_blocks = set(self.active_blocks['wt'].keys())
        control_blocks = set(self.active_blocks['control'].keys())
        
        all_blocks = wt_blocks.union(control_blocks)
        
        # Sort blocks by chrom_idx then block_start to maintain chronological order
        sorted_blocks = sorted(list(all_blocks), key=lambda x: (self.chrom_to_idx.get(x[0], 0), x[1]))
            
        for block_key in sorted_blocks:
            b_chrom, block_start = block_key
            b_chrom_idx = self.chrom_to_idx.get(b_chrom, 0)
            block_end = block_start + self.block_size
            
            # Condition to flush: We have fully passed this chromosome, OR we are on it but past the block's end
            if b_chrom_idx < safe_chrom_idx or (b_chrom_idx == safe_chrom_idx and block_end <= safe_coord):
                if block_key in wt_blocks and block_key in control_blocks:
                    self._enqueue_matched_block(block_key)
                else:
                    self.active_blocks['wt'].pop(block_key, None)
                    self.active_blocks['control'].pop(block_key, None)
    
    def flush_all_blocks(self):
        """Release all remaining matched blocks and discard unmatched blocks."""
        wt_blocks = set(self.active_blocks['wt'].keys())
        control_blocks = set(self.active_blocks['control'].keys())
        all_blocks = wt_blocks.union(control_blocks)
        
        sorted_blocks = sorted(list(all_blocks), key=lambda x: (self.chrom_to_idx.get(x[0], 0), x[1]))
        for block_key in sorted_blocks:
            if block_key in wt_blocks and block_key in control_blocks:
                self._enqueue_matched_block(block_key)
            else:
                self.active_blocks['wt'].pop(block_key, None)
                self.active_blocks['control'].pop(block_key, None)
            
    def _compile_block(self, block_data):
        """Concatenates the accumulated lists and deduplicates by (position, readname)."""
        if not block_data['positions']:
            return None
            
        positions = torch.cat(block_data['positions'])
        embeddings = torch.cat(block_data['embeddings'])
        strands = torch.cat(block_data['strands'])
        readnames = np.concatenate(block_data['readnames'])

        pos_np = positions.numpy()
        compound_dtype = np.dtype([('pos', pos_np.dtype), ('readname', 'O')])
        
        compound_arr = np.empty(len(pos_np), dtype=compound_dtype)
        compound_arr['pos'] = pos_np
        compound_arr['readname'] = readnames
        
        _, unique_indices = np.unique(compound_arr, return_index=True)
        unique_indices.sort()
        
        keep_mask_torch = torch.from_numpy(unique_indices)
        
        return {
            'positions': positions[keep_mask_torch],
            'embeddings': embeddings[keep_mask_torch],
            'strands': strands[keep_mask_torch],
            'readnames': readnames[unique_indices]
        }

    def _enqueue_matched_block(self, block_key):
        wt_raw_data = self.active_blocks['wt'].pop(block_key)
        control_raw_data = self.active_blocks['control'].pop(block_key)
        
        wt_compiled = self._compile_block(wt_raw_data)
        control_compiled = self._compile_block(control_raw_data)
        
        # Append the unpacked key along with data
        b_chrom, block_start = block_key
        self.ready_queue.append((b_chrom, block_start, wt_compiled, control_compiled))

    def pop_ready(self):
        """Yield compiled matched blocks in release order until the queue is empty."""
        while self.ready_queue:
            yield self.ready_queue.popleft()


def process_batch(batch, model, accumulator, device, accumulation_queue, key, stream_name=None, num_trim=16):
    """Handles the forward pass and accumulator update for a single batch."""
    dna_input, ts_input, percentiles_batch, pad_base_seq, pad_ts, labels, labels_idx, positions, strands, readnames, refnames, min_chrom, min_start, rank = batch

    trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
            
    filter_pos = trimmed_pad_base_seq & labels_idx
    filter_norm_pos = positions[filter_pos]

    # Avoid constructing full B x L object arrays for readnames/refnames.
    # Boolean selection over a broadcast/tile is equivalent to indexing by the
    # selected read row for each retained base.
    read_rows, _ = filter_pos.nonzero(as_tuple=True)
    strands = strands[read_rows]
    read_rows_np = read_rows.numpy()
    readnames = np.asarray(readnames, dtype=object)[read_rows_np]
    filtered_refnames = np.asarray(refnames, dtype=object)[read_rows_np]
    
    filter_pos = filter_pos.to(device, non_blocking=True)
    
    dna_input = dna_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_batch = percentiles_batch.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(dna_input, ts_input, 
                    percentiles_input=percentiles_batch,
                    pad_base_seq=pad_base_seq, pad_ts=pad_ts)       
        
    filter_ctx = select_base_embedding(out, key)[filter_pos]
    
    accumulation_queue.put({
        'stream_name': stream_name,
        'filter_norm_pos': filter_norm_pos,
        'filter_ctx_gpu': filter_ctx,
        'strands': strands,
        'readnames': readnames,
        'refnames': filtered_refnames, # Needed by CPU worker to build unique combinations
        'min_chrom': min_chrom,        # Added so frontier knows exactly where it is
        'min_start': min_start,
        'rank': rank
    })

def write_and_process_task(b_chrom, block_start, wt_compiled, control_compiled, anomaly_kwargs, output_dir):
    """This runs entirely in a separate process"""
    res = process_block_anomaly_detection_vectorized(
        control_compiled, 
        wt_compiled, 
        **anomaly_kwargs
    )
    if len(res['positions']) > 0:
        table = pa.Table.from_pydict({
            'readname': res['readnames'],
            'position': res['positions'],
            'p_val': res['p_vals'],
            'strand': res['strands'],
            'chrom': [b_chrom] * len(res['positions']),
        })
        curr_path = os.path.join(output_dir, f'{b_chrom}_{block_start}.parquet')
        pq.write_table(table, curr_path)
    return b_chrom, block_start

def setup_distributed():
    """Enforce this workflow's single-process execution requirement."""
    return require_single_process_execution(
        "pairwise_distance_read_tagging-multiworker.py",
        "Pairwise distance tagging accumulates matched per-locus embeddings before scoring.",
    )

class SortedCoordinateMerger:
    """Merge sorted DataLoaders by their batch-level chromosome coordinates."""

    def __init__(self, dataloaders, chrom_to_idx):
        """
        Merges multiple sorted DataLoaders into a single stream, 
        always yielding the batch with the lowest genomic coordinate.
        """
        self.iters = [iter(dl) for dl in dataloaders]
        self.chrom_to_idx = chrom_to_idx
        self.heap = []
        
        # Initialize the heap with the first batch from every dataloader
        for i, it in enumerate(self.iters):
            try:
                batch = next(it)
                self._push_to_heap(batch, i)
            except StopIteration:
                pass # This specific BAM file was empty

    def _push_to_heap(self, batch, loader_idx):
        # Extract the lowest coordinate from this batch to use as the sorting key.
        # Ensure your labelled_collate_fn exposes 'min_chrom' and 'min_start' at the batch level.
        min_start = batch[-2]
        min_chrom = batch[-3]
        
        # Convert chromosome string to a global index
        # If 'unknown', we give it infinity so it gets pushed to the very end of processing
        c_idx = self.chrom_to_idx.get(min_chrom, float('inf'))
        
        # Push to heap. (c_idx, min_start) acts as the sorting priority.
        heapq.heappush(self.heap, (c_idx, min_start, loader_idx, batch))

    def __iter__(self):
        return self

    def __next__(self):
        # If the heap is empty, all BAM files are fully exhausted
        if not self.heap:
            raise StopIteration
        
        # 1. Pop the batch with the absolute lowest coordinate
        c_idx, start, loader_idx, batch = heapq.heappop(self.heap)
        
        # 2. Fetch the next batch from the specific DataLoader that just yielded
        try:
            next_batch = next(self.iters[loader_idx])
            self._push_to_heap(next_batch, loader_idx)
        except StopIteration:
            pass # This specific DataLoader is now exhausted

        return batch
        
        
def get_iters(dataset_kwargs, bam_config_list, batch_size, num_workers, 
              chrom_to_idx=None, parallel_files=False, rank = 0, world_size = 1):
    """Create one labelled-BAM iterator, optionally merging per-file loaders."""
    
    if parallel_files:
        loader_kwargs = dict(batch_size=batch_size, collate_fn=labelled_collate_fn, 
                             num_workers=1, pin_memory=True, prefetch_factor=8)

        loaders = [
            DataLoader(LabelledBamIterableDataset(bam_list=[b], **dataset_kwargs), **loader_kwargs) 
            for b in bam_config_list
        ]
        
        loader_iter = iter(SortedCoordinateMerger(loaders, chrom_to_idx))
        
    else:
        loader_kwargs = dict(batch_size=batch_size, collate_fn=labelled_collate_fn, 
                                 num_workers=num_workers, pin_memory=True, prefetch_factor=4)

        loader = DataLoader(LabelledBamIterableDataset(bam_list = bam_config_list, **dataset_kwargs), **loader_kwargs)
        loader_iter = iter(loader)       
        
    return loader_iter


def effective_num_workers(requested_num_workers, bam_config_list, parallel_files):
    """Validate and cap worker count; parallel-file mode always uses one per file."""
    if requested_num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {requested_num_workers}")
    if not bam_config_list:
        raise ValueError("At least one BAM input is required.")
    if parallel_files:
        return 1
    return min(requested_num_workers, len(bam_config_list))


def get_dual_stream_metrics(model, bam_config_wt_list, bam_config_ctrl_list,
         seq_type, kmer_table, device, key, anomaly_kwargs, 
         wt_num_workers=1, control_num_workers=1, size=2048, step=None, block_size=100_000,
         reads_per_chunk=100, min_read_length=500, min_qscore=5,
         batch_size=128, num_trim=16, return_block_data=False, wt_parallel_loader=False, ctrl_parallel_loader=False,
         anomaly_workers=1, store_embeddings=False, embedding_write_mode='fast', embedding_write_workers=1,
         output_mode=None, anomaly_output=None, embedding_output=None,
         max_pending_block_tasks=None, accumulation_queue_size=128, exclude_bed=None):
    """Extract paired embeddings and asynchronously process complete blocks.

    ``output_mode`` selects p-value Parquet, embedding payloads, or both.
    Coordinate frontiers make blocks safe, while bounded accumulation and future
    queues prevent unbounded tensor retention. The returned legacy block list is
    currently empty because outputs are written by background workers.
    ``exclude_bed`` optionally removes same-strand read-span overlaps from both
    input populations using a ``chrom start end [strand]`` file (both strands
    when column four is omitted).
    """
    
    rank, world_size, local_rank = setup_distributed()
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="legacy pairwise-distance comparison",
        announce=rank == 0,
    )
    if anomaly_workers < 1:
        raise ValueError(f"anomaly_workers must be >= 1, got {anomaly_workers}")
    if embedding_write_workers < 1:
        raise ValueError(f"embedding_write_workers must be >= 1, got {embedding_write_workers}")
    if embedding_write_mode not in {'fast', 'compressed'}:
        raise ValueError("embedding_write_mode must be either 'fast' or 'compressed'")

    if output_mode is None:
        output_mode = 'embeddings' if store_embeddings else 'pvalues'
    if output_mode not in {'pvalues', 'embeddings', 'both'}:
        raise ValueError("output_mode must be one of: 'pvalues', 'embeddings', 'both'")
    run_anomaly_detection = output_mode in {'pvalues', 'both'}
    write_embeddings = output_mode in {'embeddings', 'both'}
    if run_anomaly_detection and not anomaly_output:
        raise ValueError("anomaly_output is required when output_mode is 'pvalues' or 'both'")
    if write_embeddings and not embedding_output:
        raise ValueError("embedding_output is required when output_mode is 'embeddings' or 'both'")
    if run_anomaly_detection:
        os.makedirs(anomaly_output, exist_ok=True)
    if write_embeddings:
        os.makedirs(embedding_output, exist_ok=True)

    if max_pending_block_tasks is None:
        active_workers = 0
        if run_anomaly_detection:
            active_workers += anomaly_workers
        if write_embeddings:
            active_workers += embedding_write_workers
        max_pending_block_tasks = max(2 * active_workers, active_workers + 1)
    if max_pending_block_tasks < 1:
        raise ValueError(f"max_pending_block_tasks must be >= 1, got {max_pending_block_tasks}")
    
    wt_num_workers = effective_num_workers(wt_num_workers, bam_config_wt_list, wt_parallel_loader)
    control_num_workers = effective_num_workers(control_num_workers, bam_config_ctrl_list, ctrl_parallel_loader)
    
    chrom_order = [info[0] for info in bam_config_wt_list[0].chrom_info]
    chrom_to_idx = {c: i for i, c in enumerate(chrom_order)}
    
    dataset_kwargs = dict(size=size, step=step, move_downsample_factor=1, seq_type=seq_type, kmer_table=kmer_table,
                              reads_per_chunk=reads_per_chunk, min_read_length=min_read_length,
                              min_qscore=min_qscore, pore_type=None, base_seed=0, rank=rank, world_size=world_size, cycle=False,
                              shuffle=False, expected_ts_dim=model.config.ts_dim, exclude_bed=exclude_bed)
    
    wt_iter=get_iters(dataset_kwargs, bam_config_wt_list, batch_size, wt_num_workers, 
                      chrom_to_idx=chrom_to_idx, parallel_files=wt_parallel_loader)
    
    control_iter=get_iters(dataset_kwargs, bam_config_ctrl_list, batch_size, control_num_workers, 
                      chrom_to_idx=chrom_to_idx, parallel_files=ctrl_parallel_loader, rank = rank, world_size = world_size)
    
    wt_batch = next(wt_iter, None)
    control_batch = next(control_iter, None)

    # Pass the global chromosome order into the accumulator
    accumulator = DualGenomicAccumulator(dim=get_embedding_dimension(model.config, key), wt_num_workers=wt_num_workers, control_num_workers=control_num_workers, chrom_order=chrom_order, block_size=block_size)

    start = time.time()
    block_data_list = []
    accumulation_queue = queue.Queue(maxsize=accumulation_queue_size) 
    pending_futures = set()
    block_task_errors = []

    def track_future(future, block_desc):
        """Attach block context and retain a submitted background future."""
        future.block_desc = block_desc
        pending_futures.add(future)

    def submit_block_task(b_chrom, block_start, wt_compiled, control_compiled):
        """Submit configured outputs for one block and enforce backpressure."""
        if write_embeddings:
            future = embedding_executor.submit(
                write_embeddings_task,
                b_chrom,
                block_start,
                wt_compiled,
                control_compiled,
                embedding_output,
                embedding_write_mode,
            )
            track_future(future, f"embeddings {b_chrom}:{block_start}")

        if run_anomaly_detection:
            future = anomaly_executor.submit(
                write_and_process_task,
                b_chrom,
                block_start,
                wt_compiled,
                control_compiled,
                anomaly_kwargs,
                anomaly_output
            )
            track_future(future, f"anomaly {b_chrom}:{block_start}")

        # Prevent unbounded queued block payloads from retaining large CPU tensors.
        if len(pending_futures) >= max_pending_block_tasks:
            drain_completed_block_tasks(block=True)

    def drain_completed_block_tasks(block=False):
        """Collect completed background tasks and retain contextualized errors."""
        if not pending_futures:
            return

        if block:
            done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
        else:
            done, _ = wait(pending_futures, timeout=0, return_when=FIRST_COMPLETED)

        for future in done:
            pending_futures.remove(future)
            try:
                future.result()
            except Exception as exc:
                block_desc = getattr(future, 'block_desc', '<unknown block>')
                block_task_errors.append(
                    RuntimeError(f"Block worker failed while processing {block_desc}: {exc}")
                )
    
    def accumulator_worker():
        """Move queued embeddings to CPU, accumulate them, and dispatch safe blocks."""
        t = time.time()
        
        while True:
            job = accumulation_queue.get()
            
            if job is None:
                accumulation_queue.task_done()
                break
                
            if 'stream_done' in job:
                accumulator.set_stream_done(job['stream_done'])
                accumulation_queue.task_done()
                continue
                
            filter_ctx = job['filter_ctx_gpu'].cpu()
            filter_norm_pos = job['filter_norm_pos']
            
            # Build block groups once on CPU; avoids a full scan per unique block.
            blocks_cpu = torch.div(filter_norm_pos, accumulator.block_size, rounding_mode='floor').numpy()
            refnames_cpu = job['refnames']
            block_groups_cpu = build_block_groups(refnames_cpu, blocks_cpu) if len(blocks_cpu) > 0 else []
            
            accumulator.update(
                job['stream_name'] if job['stream_name'] else 'default', 
                filter_norm_pos, 
                filter_ctx, 
                job['strands'], 
                job['readnames'], 
                job['refnames'],
                job['min_chrom'],
                job['min_start'],
                job['rank'],
                block_groups_cpu
            )
            
            for block_data in accumulator.pop_ready():
                b_chrom, block_start, wt_compiled, control_compiled = block_data
                submit_block_task(b_chrom, block_start, wt_compiled, control_compiled)
                drain_completed_block_tasks(block=False)
                
                print(f"[{time.time()-t:.2f}s] Dispatched {b_chrom}:{block_start} for {output_mode}", flush=True)
                t = time.time()
                
            accumulation_queue.task_done()

    anomaly_executor = get_reusable_executor(max_workers=anomaly_workers) if run_anomaly_detection else None
    # Threads avoid pickling/copying large embedding tensors into child processes.
    embedding_executor = ThreadPoolExecutor(max_workers=embedding_write_workers) if write_embeddings else None

    def shutdown_executors(wait=True):
        """Shut down every output executor with the requested wait behavior."""
        if anomaly_executor is not None:
            anomaly_executor.shutdown(wait=wait)
        if embedding_executor is not None:
            embedding_executor.shutdown(wait=wait)
    
    worker_thread = threading.Thread(target=accumulator_worker, daemon=True)
    worker_thread.start()
    
    while wt_batch is not None or control_batch is not None:
        if wt_batch is not None and control_batch is not None:
            wt_min_start = wt_batch[-2]
            wt_min_chrom = wt_batch[-3]
            wt_chrom_idx = chrom_to_idx.get(wt_min_chrom, 0)
            
            control_min_start = control_batch[-2]
            control_min_chrom = control_batch[-3]
            control_chrom_idx = chrom_to_idx.get(control_min_chrom, 0)

            # Compare using 2D location (chromosome index AND coordinate)
            if (wt_chrom_idx, wt_min_start) <= (control_chrom_idx, control_min_start):
                process_batch(wt_batch, model, accumulator, device, accumulation_queue, key, 'wt', num_trim)
                wt_batch = next(wt_iter, None)
                if wt_batch is None:
                    accumulation_queue.put({'stream_done': 'wt'})
            else:
                process_batch(control_batch, model, accumulator, device, accumulation_queue, key, 'control', num_trim)
                control_batch = next(control_iter, None)
                if control_batch is None:
                    accumulation_queue.put({'stream_done': 'control'})

        elif wt_batch is not None:
            process_batch(wt_batch, model, accumulator, device, accumulation_queue, key, 'wt', num_trim)
            wt_batch = next(wt_iter, None)
            if wt_batch is None:
                accumulation_queue.put({'stream_done': 'wt'})

        elif control_batch is not None:
            process_batch(control_batch, model, accumulator, device, accumulation_queue, key, 'control', num_trim)
            control_batch = next(control_iter, None)
            if control_batch is None:
                accumulation_queue.put({'stream_done': 'control'})
            
    accumulation_queue.join() 
    accumulation_queue.put(None) 
    worker_thread.join()
    if block_task_errors:
        shutdown_executors(wait=True)
        raise block_task_errors[0]

    # Flush remaining blocks
    accumulator.flush_all_blocks()
    for block_data in accumulator.pop_ready():
        b_chrom, block_start, wt_compiled, control_compiled = block_data
        submit_block_task(b_chrom, block_start, wt_compiled, control_compiled)
        drain_completed_block_tasks(block=False)
        
        print(f"Dispatched {b_chrom}:{block_start} for {output_mode}", flush=True)
        t = time.time()

    drain_completed_block_tasks(block=True)
    shutdown_executors(wait=True)
    if block_task_errors:
        raise block_task_errors[0]
    
    del accumulator
    print(f'Total Time = {time.time()-start:.2f}')
    
    return block_data_list



def expand_bam_input(input_path, arg_name):
    """Resolve one BAM file or recursively expand a BAM directory."""
    path = Path(input_path)
    if path.is_dir():
        bam_paths = natsorted(str(x) for x in path.rglob('*.bam'))
    elif path.is_file():
        if path.suffix.lower() != '.bam':
            raise ValueError(f"{arg_name} must point to a .bam file or a directory containing .bam files: {input_path}")
        bam_paths = [str(path)]
    else:
        raise FileNotFoundError(f"{arg_name} does not exist: {input_path}")

    if not bam_paths:
        raise FileNotFoundError(f"No .bam files found for {arg_name}: {input_path}")
    return bam_paths


def load_embedding_model(model_path, device):
    """Load the checked frozen backbone used for distance scoring."""
    return load_checked_model(
        model_path,
        model_type="backbone",
        add_signal_cross_attn=False,
        device=device,
    )



def main():
    """Load the pretrained model and run paired-distance block processing."""
    
    print('\n%s:\nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    parser = build_pairwise_distance_parser()
    args = parser.parse_args()
    resolve_cli_preprocessing(args)
    get_excluded_regions(args.exclude_bed)

    model_path = args.model_path

    device = torch.device('cuda')
    model = load_embedding_model(model_path, device)
    
    max_base_seq_len = model.config.max_base_seq_len
    max_ts_len = model.config.max_ts_len
    ts_dim = model.config.ts_dim
    B = 8
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
        ss = model(torch.ones(B, max_base_seq_len).long().to(device),
                   torch.randn(B, max_ts_len, ts_dim).to(device),
                   percentiles_input=torch.randn(B, max_ts_len, ts_dim).to(device),
                   pad_base_seq=torch.ones(B, max_base_seq_len, dtype=torch.bool).to(device),
                   pad_ts=torch.ones(B, max_ts_len, dtype=torch.bool).to(device))
    
    model = torch.compile(model, dynamic=True)
    
    kmer_dict_path = args.kmer_dict_path
    kmer_table = resolve_cli_preprocessing(args).kmer_table
    
    seq_type = args.seq_type

    ref = args.ref
    
    output_mode = args.output_mode
    if output_mode is None:
        output_mode = 'embeddings' if args.store_embeddings else 'pvalues'
    anomaly_output = args.anomaly_output
    embedding_output = args.embedding_output
    
    fref = pysam.FastaFile(ref)
    chrom_list=args.chrom_list or fref.references
    chrom_list = [(chrom, 0, fref.get_reference_length(chrom)-1) for chrom in chrom_list]

    ctrl_bam_path_list = expand_bam_input(args.ctrl_input, '--ctrl_dir/--ctrl_bam_path')
    
    ctrl_bam_config_list = []
    
    motif_string = args.motif_string
    motif_label = args.motif_label
    
    pos_file = args.pos_file
    
    for ctrl_bam_path in ctrl_bam_path_list:
        ctrl_bam_config_list.append(LabelledBamSample(
                bam_path=ctrl_bam_path,
                chrom_info=chrom_list,
                ref=ref,
                num_classes=2,
                pos_file=pos_file,
                motif=motif_string,
                motif_label=motif_label,
                flag=0,
                polish=args.ctrl_use_polished_sequence,
                polish_indel_limit=args.polish_indel_limit,
                polish_vcf_path=(
                    args.polish_vcf_path if args.ctrl_use_polished_sequence else None
                ),
            ))
    
    wt_bam_path_list = expand_bam_input(args.wt_input, '--wt_dir/--wt_bam_path')
    
    wt_bam_config_list = []
    
    for wt_bam_path in wt_bam_path_list:
        wt_bam_config_list.append(LabelledBamSample(
                bam_path=wt_bam_path,
                chrom_info=chrom_list,
                ref=ref,
                num_classes=2,
                pos_file=pos_file,
                motif=motif_string,
                motif_label=motif_label,
                flag=0,
                polish=args.wt_use_polished_sequence,
                polish_indel_limit=args.polish_indel_limit,
                polish_vcf_path=(
                    args.polish_vcf_path if args.wt_use_polished_sequence else None
                ),
            ))

    anomaly_kwargs = dict(min_control_cov=args.min_control_cov, 
                          min_test_cov=args.min_test_cov, 
                          max_fit_cov=args.max_fit_cov, 
                          n_comp=args.n_comp, 
                          batch_size=args.anomaly_batch_size, 
                          score_batch_size=args.score_batch_size,
                          max_score_cov=args.max_score_cov,
                          pvalue_method=args.pvalue_method,
                          calibration_fraction=args.calibration_fraction,
                          device='cuda')
    
    
    res_list = get_dual_stream_metrics(model, wt_bam_config_list, ctrl_bam_config_list,
             seq_type, kmer_table,
             device,
             args.embedding_key,
             anomaly_kwargs,
             wt_num_workers=args.wt_num_workers,
             control_num_workers=args.control_num_workers,
             size=args.size,
             step=args.step,
             block_size=args.block_size,
             reads_per_chunk=args.reads_per_chunk,
             min_read_length=args.min_read_length,
             min_qscore=args.min_qscore,
             batch_size=args.batch_size,
             num_trim=args.num_trim,
             wt_parallel_loader=args.wt_parallel_loader,
             ctrl_parallel_loader=args.ctrl_parallel_loader,
             anomaly_workers=args.anomaly_workers,
             store_embeddings=args.store_embeddings,
             exclude_bed=args.exclude_bed,
             embedding_write_mode=args.embedding_write_mode,
             embedding_write_workers=args.embedding_write_workers,
             output_mode=output_mode,
             anomaly_output=anomaly_output,
             embedding_output=embedding_output,
             max_pending_block_tasks=args.max_pending_block_tasks,
             accumulation_queue_size=args.accumulation_queue_size,)

if __name__=='__main__':
    main()
