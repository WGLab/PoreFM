"""Iterable datasets for aligned nanopore reads."""

from porefm.preprocessing import resolve_preprocessing
from porefm.data.chunk_boundaries import validate_chunking_mode
import datetime
import random
from typing import List, Optional

import numpy as np
import pysam
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from porefm.common.utils import DegenerateQuantileNormalizationError
from porefm.data.bam import Read, get_read_info, passes_min_qscore, pre_process
from porefm.data.chunks import get_read_chunks_aligned
from porefm.data.polishing import polish_read_with_maps
from porefm.data.polishing_options import (
    DEFAULT_POLISH_INDEL_LIMIT,
    get_polishing_variant_mask,
    validate_polishing_options,
)
from porefm.data.reader import (
    format_iterable_bam_log,
    get_percentiles,
    normalize_read_signal,
)


class AlignedBamSample:
    """Describe one aligned BAM source and its reference preprocessing.

    Parameters
    ----------
    bam_path : str
        Path to a coordinate-queryable BAM file.
    chrom_info : list
        Regions as ``(contig, start, end)`` triples using zero-based,
        half-open pysam coordinates. Entries are sorted and stored as a tuple.
    ref : str
        Indexed reference FASTA path.
    pos_file : str or None
        Optional tab-separated labelled-position file.
    motif : str or None
        Optional reference motif specification used by preprocessing.
    motif_label : int or None
        Label assigned to motif positions.
    flag : int, default=0
        Additional SAM flag bits that cause a record to be excluded.
    polish : bool, default=False
        Enable reference-guided sequence polishing.
    polish_indel_limit : int, default=5
        Largest unmasked indel corrected during polishing.
    polish_vcf_path : str or None
        VCF/BCF of variants that polishing must preserve. Requires
        ``polish=True``.
    """

    def __init__(
        self,
        bam_path: str,
        chrom_info: List,
        ref: str,
        pos_file: str,
        motif: str,
        motif_label: Optional[int],
        flag=0,
        polish=False,
        polish_indel_limit=DEFAULT_POLISH_INDEL_LIMIT,
        polish_vcf_path=None,
    ):
        polish, polish_indel_limit, polish_vcf_path = validate_polishing_options(
            polish, polish_indel_limit, polish_vcf_path,
        )
        self.bam_path = bam_path
        self.chrom_info = tuple(sorted(chrom_info))
        self.ref = ref
        self.pos_file = pos_file
        self.motif = motif
        self.motif_label = motif_label
        self.flag=flag
        self.polish=polish
        self.polish_indel_limit=polish_indel_limit
        self.polish_vcf_path=polish_vcf_path

class AlignedBamIterableDataset(IterableDataset):
    """Stream normalized, fixed-signal windows from aligned nanopore BAMs.

    BAM files are deterministically sharded across the product of distributed
    ranks and DataLoader workers. Records are filtered for secondary/
    supplementary flags, optional flag bits, read length, and quality before
    being converted to :class:`~porefm.data.bam.Read` objects. Each yielded
    window includes variable-length base tokens and zero-based reference
    positions.

    Parameters
    ----------
    bam_list : sequence of AlignedBamSample
        Input samples. Every worker receives a disjoint strided subset.
    size : int
        Signal time steps per model window.
    step : int
        Signal time-step offset between consecutive windows.
    seq_type : str, optional
        Sequence orientation/type; inferred from the profile.
    kmer_table : numpy.ndarray, optional
        Dense expected-current lookup; inferred from the profile.
    preprocessing : PreprocessingSpec or mapping, optional
        Shared contract, e.g. ``model.preprocessing``. Explicit sequence,
        pore, and table values must agree with it.
    chemistry : str, optional
        Standalone selection: ``dna_r10.4.1_e8.2_400bps``, ``rna004``, or
        ``rna002``; no model or external table is needed.
    reads_per_chunk : int, default=1000
        Number of read-level chunk tensors buffered before percentile
        calculation and yielding.
    min_read_length : int, default=100
        Minimum query-sequence length in bases.
    min_qscore : float, default=10
        Minimum ``qs`` BAM tag when that optional tag is present.
    pore_type : str or None
        Optional chemistry assertion; inferred from the resolved profile.
    base_seed : int, default=1234
        Base for deterministic per-worker Python, NumPy, and Torch seeds.
    rank, world_size : int
        Distributed rank and process count used for file sharding.
    expected_ts_dim : int or None
        Expected move-table stride. A mismatch raises ``ValueError`` while
        decoding the read.

    Yields
    ------
    tuple
        Sequence tokens, ``float16`` signal window, ``float16`` percentile
        features, read name, reference positions, buffer minimum position, and
        strand index (``0`` forward, ``1`` reverse). Sequences shorter than 64
        tokens are omitted.
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
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
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
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank=rank
        self.world_size=world_size
        self.kmer_table = kmer_table
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim


    def _yield_buffer(self, data):
        """Compute shared percentile features and emit valid buffered windows."""
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)

        if data['position']:
            pos_cat = torch.cat(data['position'])
            valid_pos = pos_cat[pos_cat > 0]
            min_start = valid_pos.min().item() if valid_pos.numel() > 0 else 1e12
        else:
            min_start = 1e12

        for i in range(len(data['seq'])):
            if len(data['seq'][i]) >= 64:
                yield (
                    data['seq'][i],
                    signals[i].to(dtype=torch.float16),
                    percentiles[i].to(dtype=torch.float16),
                    data['read_name'][i],
                    data['position'][i],
                    min_start,
                    data['strand'][i]
                )


    def __iter__(self):
        """Yield this worker's deterministic shard of aligned read windows."""
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

        data = {'signal': [], 'seq': [],'read_name': [], 'position': [], 'strand': []}

        for bam_sample in my_files:
            bam_path, chrom_info = bam_sample.bam_path, bam_sample.chrom_info
            flag_info=bam_sample.flag
            polish=bam_sample.polish
            variant_mask = get_polishing_variant_mask(
                bam_sample, polish_variant_mask_cache,
            ) if polish else None

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

                            polished_read = polish_read_with_maps(
                                bam_read,
                                variant_mask=variant_mask,
                                indel_lim=bam_sample.polish_indel_limit,
                            ) if polish else None

                            signal_chunks, base_seq_chunks, _, pos_chunks = get_read_chunks_aligned(read, polished_read, self.size, self.step)

                            if start!=None and end!=None:

                                if isinstance(pos_chunks, list) and len(pos_chunks) > 0:
                                    pos_tensor = pad_sequence(pos_chunks, batch_first=True, padding_value=-1)
                                else:
                                    pos_tensor = pos_chunks

                                keep = torch.any((pos_tensor >= start) & (pos_tensor <= end), dim=1)

                            else:
                                keep = torch.ones(len(data['seq']), dtype=bool)

                            if keep.any():
                                data['signal'].append(signal_chunks[keep])

                                keep_idx = keep.nonzero(as_tuple=True)[0].tolist()

                                data['seq'].extend([base_seq_chunks[i] for i in keep_idx])
                                data['read_name'].extend([read_name_str] * len(keep_idx))
                                data['position'].extend([pos_chunks[i] for i in keep_idx])
                                data['strand'].extend([strand_val] * len(keep_idx))

                            if len(data['signal']) == self.reads_per_chunk:
                                yield from self._yield_buffer(data)
                                data = {'signal': [], 'seq': [], 'read_name': [], 'position': [], 'strand': []}
        if len(data['signal']) > 0:
            yield from self._yield_buffer(data)
