"""Legacy streaming control-embedding comparison and BAM tagging utilities."""

from porefm.preprocessing import resolve_preprocessing
import sys
import time
import seaborn as sns
import pandas as pd
from safetensors.torch import save_file

import threading
import queue
import array
import torch
import torch.nn as nn
from numpy.linalg import norm

import collections

import numpy as np
import numpy_indexed as npi
import pysam
from torch.nn import functional as F
from torch.utils.data import DataLoader

from porefm.common.tensor_masks import ignore_edges
from porefm.common.chunking import resolve_downstream_step
from porefm.data.bam import build_lookup_table
from porefm.data.datasets.aligned import AlignedBamIterableDataset, AlignedBamSample
from porefm.data.collation import aligned_collate_fn, collate_modal_aligned_batch
from porefm.workflows.read_tagging.sync_embeddings import process_batch
from porefm.data.complete_reads import bam_writer_worker, stitch_1d_distances_ultra_fast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

class ControlGenomeAccumulator:
    """Accumulate strand-specific mean embeddings in fixed genomic blocks.

    GPU sums/counts are released to a CPU queue once the monotonically advancing
    coordinate frontier passes a block. The class assumes one chromosome and
    0-based non-negative coordinates.
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
        """Scatter-add one filtered batch and flush blocks behind the frontier."""
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
                
                # Empty indexed updates remain on device and avoid a synchronizing
                # Python branch for strands absent from the batch.
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
    """Select labelled forward/reverse positions meeting minimum coverage.

    Returns one ``(positions, counts, mean_embeddings)`` tuple per strand for
    the 0-based interval beginning at ``block_start``.
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

def get_stream_metrics(model, bam_config,
         seq_type, kmer_table,
         device,
         size=2048,
         step=None,
         block_size = 100_000,
         reads_per_chunk=100,
         min_read_length=500,
         min_qscore=5,
         batch_size=128,
         num_trim=16):
    """Yield finalized control-embedding blocks for one aligned BAM sample.

    A finite coordinate-ordered dataset is forwarded through ``model`` and
    accumulated into strand-specific genomic means.
    """
    step = resolve_downstream_step(
        size,
        seq_type,
        step,
        context="legacy control embedding accumulation",
    )
    
    fwd_data = {'pos': [], 'counts': [], 'mean_vectors':[]}
    rev_data = {'pos': [], 'counts': [], 'mean_vectors':[]}

    dataset_kwargs=dict(size=size, step=step, seq_type=seq_type, kmer_table=kmer_table,
                        reads_per_chunk=reads_per_chunk, min_read_length=min_read_length,
                        min_qscore=min_qscore, pore_type=None, base_seed=0, rank=0, world_size=1)
    
    ds = AlignedBamIterableDataset(bam_list=[bam_config], **dataset_kwargs)
    
    r_start, r_end = bam_config.chrom_info[0][1:]
    
    loader_kwargs = dict(batch_size=batch_size, collate_fn=aligned_collate_fn, 
                         num_workers=1, pin_memory=True, prefetch_factor=8)

    loader = DataLoader(ds, **loader_kwargs)

    ds_iter = iter(loader)

    accumulator = ControlGenomeAccumulator(dim=model.config.hidden_dim, block_size=block_size, device=device)

    # Fetch first batches
    batch = next(ds_iter, None)

    t = time.time()
    start=time.time()
    while batch is not None:
        process_batch(batch, model, accumulator, device, r_start, r_end, None, num_trim)
        batch = next(ds_iter, None)

        # Yield and calculate anything that is ready
        yield from accumulator.pop_ready()
    accumulator.flush_all_blocks()
    yield from accumulator.pop_ready()

    return 

class CircularGPUControlBuffer:
    """Fixed-capacity circular cache of sequential control embedding blocks.

    Absolute genomic coordinates map to slots by modulo ``capacity``. Advancing
    the stream overwrites the oldest blocks, so callers must request positions
    inside ``[min_loaded, max_loaded)``.
    """

    def __init__(self, stream_iter, block_size, buffer_blocks, dim, device, dtype=torch.bfloat16):
        self.stream_iter = stream_iter
        self.block_size = block_size
        self.buffer_blocks = buffer_blocks
        self.capacity = block_size * buffer_blocks
        self.dim = dim
        self.device = device
        self.dtype = dtype
        
        # Allocate VRAM exactly ONCE
        # Shape: [2, capacity, dim]
        self.embs = torch.zeros((2, self.capacity, self.dim), dtype=dtype, device=device)
        
        # Track what absolute genomic coordinates are currently in the buffer
        self.min_loaded = 0
        self.max_loaded = 0
        self.is_initialized = False

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
                block_start, block_embs, _ = next(self.stream_iter)
            except StopIteration:
                break # Stream exhausted
                
            # Initialize the loaded tracking on the very first block
            if not self.is_initialized:
                self.min_loaded = block_start
                self.max_loaded = block_start
                self.is_initialized = True
            
            # Find the exact slot in our pre-allocated tensor
            write_start_idx = block_start % self.capacity
            write_end_idx = write_start_idx + self.block_size
            
            # In-place overwrite (blazing fast, no allocation)
            self.embs[:, write_start_idx:write_end_idx, :] = block_embs.to(self.device, dtype=self.dtype)
            
            # Update our tracker
            self.max_loaded = block_start + self.block_size
            
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
        final_mask = valid_mask & in_bounds_mask
        
        # 2. Map absolute genomic coordinates to circular buffer indices natively
        # PyTorch % is highly optimized on CUDA
        circ_indices = ref_positions % self.capacity
        
        # Expand strands to match [B, L]
        strand_idx = strands.view(B, 1).expand(B, L)
        
        # 3. Gather directly from the static buffer
        ctrl_embs = self.embs[strand_idx, circ_indices, :]
        
        # 4. Zero out positions that were out of bounds or invalid padding
        ctrl_embs = torch.where(
            final_mask.unsqueeze(-1), 
            ctrl_embs, 
            torch.zeros_like(ctrl_embs) # Or float('nan') if you prefer
        )
        
        return ctrl_embs, final_mask
    
def call_mod_bam(model, wga_bam_path, wt_bam_path, kmer_dict_path,
                block_size, chrom, r_start, r_end, 
                bam_output, seq_type, size, step, base,
                mod_code, chrom_labelled_arrays,
                batch_size = 32, bam_threads=2, num_trim=16, scale_factor=2):
    """Run the legacy control-distance workflow and write modification BAM tags.

    Control embeddings are streamed into a circular GPU buffer and compared to
    complete WT read embeddings before a background BAM writer emits MM/ML
    annotations. This entry point depends on legacy module-level runtime state
    for the reference path and device; callers should prefer newer packaged
    read-tagging workflows when possible.
    """
    
    preprocessing = resolve_preprocessing(preprocessing=getattr(model, "preprocessing", None),
                                          seq_type=seq_type, kmer_dict_path=kmer_dict_path)
    seq_type, kmer_table = preprocessing.seq_type, preprocessing.kmer_table
    
    wga_bam_config = AlignedBamSample(
        bam_path=wga_bam_path,
        chrom_info=[(chrom, r_start, r_end)],
        ref=ref,
        pos_file='',
        motif='',
        motif_label=0,
        flag=0,
    )

    control_stream = get_stream_metrics(model, wga_bam_config,
             seq_type, kmer_table,
             device,
             size=size,
             step=step,
             block_size = block_size,
             reads_per_chunk=100,
             min_read_length=500,
             min_qscore=10,
             batch_size=128,
             num_trim=32)

    control_stream_iter = iter(control_stream)

    control_buffer = CircularGPUControlBuffer(
        stream_iter=control_stream_iter, 
        block_size=block_size, 
        buffer_blocks=10, 
        dim=model.config.hidden_dim, 
        device=device
    )

    t=time.time()
    st_time=t
    wt_bam_config=AlignedBamSample(bam_path=wt_bam_path,
                                 chrom_info=[(chrom, r_start, r_end)],
                                 ref=ref,
                                 pos_file='',
                                 motif='',
                                 motif_label=0,
                                 flag=0)

    str_header=str(pysam.AlignmentFile(wt_bam_config.bam_path).header)
    out_header=pysam.AlignmentHeader.from_text(str_header)

    ds = CompeletAlignedReadIterableDataset(bam_list=[wt_bam_config],
            size=size, step=step, seq_type=seq_type, kmer_table=kmer_table,
            reads_per_chunk=50, min_read_length=100, max_read_length=2e5, min_qscore=0,
            pore_type=None, base_seed=0, rank=0, world_size=1)


    write_queue = queue.Queue(maxsize=1000) 
    writer_thread = threading.Thread(
        target=bam_writer_worker, 
        args=(write_queue, bam_output, bam_threads, out_header, chrom_labelled_arrays, base, mod_code, seq_type)
    )
    writer_thread.start()

    counter=0
    
    data_dict = {}
    dataloader = DataLoader(
        ds,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True
        )

    for batch in dataloader:
        dna_input, ts_input, percentiles_batch, pad_base_seq, pad_ts, read_names, ref_positions, read_positions, min_start, strands, pysam_data = batch
        read_names=np.array(read_names)
        ref_positions=ref_positions
        trimmed_pad_base_seq = ignore_edges(pad_base_seq, num_trim)
        valid_pos=((ref_positions>=0)&trimmed_pad_base_seq)

        if r_start is not None and r_end is not None:
            valid_pos=valid_pos&(ref_positions>=r_start)&(ref_positions<=r_end)

        valid_pos=valid_pos.to(device)
        strands=strands.to(device)

        print(f'counter {counter}: {time.time()-t:.2f}  Coordinate: {min_start}')
        t=time.time()
        counter+=1

        # NEW: Store distances instead of embeddings
        dist_array = [] 
        
        
        for sub_batch in range(0, dna_input.size(0), batch_size):
            
            batch_dna_input = dna_input[sub_batch : sub_batch + batch_size]
            batch_ts_input = ts_input[sub_batch : sub_batch + batch_size]
            batch_percentiles_batch = percentiles_batch[sub_batch : sub_batch + batch_size]
            batch_pad_base_seq = pad_base_seq[sub_batch : sub_batch + batch_size]
            batch_pad_ts = pad_ts[sub_batch : sub_batch + batch_size]
            
#             if batch_dna_input.shape[0]<batch_size:
#                 continue
#             print('read_wise', batch_dna_input.shape, batch_ts_input.shape)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
                out = model(batch_dna_input.to(device), batch_ts_input.to(device), 
                            percentiles_input=batch_percentiles_batch.to(device),
                            pad_base_seq=batch_pad_base_seq.to(device), pad_ts=batch_pad_ts.to(device))

            test_embs = out['contextual_base_seq'] 
            
            # Slice coordinates and masks for the current sub-batch and move to GPU
            batch_ref_pos = ref_positions[sub_batch : sub_batch + batch_size].to(device)
            batch_strands = strands[sub_batch : sub_batch + batch_size]
            batch_mask = valid_pos[sub_batch : sub_batch + batch_size]

            # Skip if the whole sub-batch is padding
            if not batch_mask.any():
                dummy_dist = torch.full(batch_ref_pos.shape, float('nan'), dtype=torch.float32)
                dist_array.append(dummy_dist)
                continue

            # Get the span of coordinates for this specific sub-batch
            current_min = int(batch_ref_pos[batch_mask].min().item())
            current_max = int(batch_ref_pos[batch_mask].max().item()) + 1

            # Advance the control buffer to ensure it covers our test reads
            control_buffer.ensure_coverage(current_min, current_max)

            # Gather the matched control embeddings
            ctrl_embs, ctrl_valid_mask = control_buffer.gather(batch_ref_pos, batch_strands, batch_mask)

            # Calculate L2 Distance on the GPU
    #         l2_distances = torch.mean((test_embs - ctrl_embs)**2, dim=-1)
            cosine_distances = 1 - F.cosine_similarity(test_embs, ctrl_embs, dim=-1)
            cosine_distances = torch.clamp_max(cosine_distances, 1)

            # Mask out padding and out-of-bounds positions with NaNs
            distances = cosine_distances
            final_valid_mask = ctrl_valid_mask & batch_mask
            distances = torch.where(
                final_valid_mask, 
                distances, 
                torch.tensor(float('nan'), device=device, dtype=distances.dtype)
            )

            # Move only the 1D distances to CPU RAM
            dist_array.append(distances.cpu().float())
#         continue
        # Concatenate the 1D array
        base_dist_array = torch.abs(torch.cat(dist_array))
        base_dist_array=torch.clamp_max(scale_factor*base_dist_array, 1)

        # Call the updated stitching function
        unique_reads, read_wise_dist, read_wise_refpos = stitch_1d_distances_ultra_fast(
            base_dist_array, read_positions, ref_positions, pad_base_seq, read_names
        )
        
        for x,y,z in zip(read_names, read_wise_dist, read_wise_refpos):
            data_dict[str(x)]=[y.cpu().float(), z.cpu()]
        strand_dict={str(x):y.item() for x,y in zip(read_names, strands.cpu())}

        write_queue.put((strand_dict, read_wise_dist, read_wise_refpos, pysam_data))

    # 5. Cleanup after the loop finishes
    write_queue.put(None) # Send sentinel to tell the thread to stop
    writer_thread.join()  # Wait for the remaining queue to write to disk
    
    print(time.time()-st_time)
    
    return data_dict
