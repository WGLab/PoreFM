"""Extract bounded oligo neighborhood means for paired footprint fitting.

Uses the production complete-read signal path and freezes the backbone. Only
exact, contiguous, Q-filtered reference k-mers contribute, and each reference
position contributes at most once per read. All events remain in the modified
sample: simultaneous-event deconvolution occurs later in the paired fitter.
"""

from __future__ import annotations

from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
import math
from pathlib import Path

from porefm.cli.chunking_args import add_chunking_arguments
from porefm.cli._parser import PoreFMArgumentParser


def accumulate_read(read, reference, statistics, *, k, radius, min_base_quality, max_sites):
    """Add one stitched read to site sums; return its number of accepted sites.

    ``reference`` is the forward FASTA string, ``statistics`` maps
    (contig, genomic position, strand) to [oriented context, sum[D], count].
    The exact-match window uses read orientation; both strands retain 0-based
    genomic site coordinates. An alignment gap invalidates the exact k-mer.
    """
    import numpy as np
    import torch
    from porefm.data.reference_match import build_exact_match_kmer_arrays_from_read_arrays
    ref = reference.upper().replace('U', 'T')
    strand = read['strand']
    complement = str.maketrans('ACGTN', 'TGCAN')
    positions = read['ref_positions'].numpy()
    sequence = read['sequence']
    exact = np.array([0 <= p < len(ref) and sequence[i] == (ref[p].translate(complement) if strand else ref[p])
                      for i, p in enumerate(positions)])
    quality = read['base_error_probabilities'].numpy() <= 10 ** (-min_base_quality / 10)
    tokens = np.array(['ACGT'.find(s) for s in sequence], dtype=np.int64)
    _, valid = build_exact_match_kmer_arrays_from_read_arrays(
        tokens, positions, exact, k=k, quality_ok=quality,
        eligible_center_mask=read['valid'].numpy())
    half = k // 2 + radius
    seen, accepted = set(), 0
    for index in np.flatnonzero(valid):
        pos = int(positions[index])
        if pos in seen or pos < half or pos + half >= len(ref):
            continue
        context = ref[pos - half:pos + half + 1]
        if strand:
            context = context.translate(complement)[::-1]
        if set(context) - set('ACGT'):
            continue
        value = read['embeddings'][index].double()
        if not torch.isfinite(value).all():
            continue
        key = (read['chrom'], pos, strand)
        if key not in statistics:
            if len(statistics) >= max_sites:
                raise ValueError('max_sites exceeded; this extractor is for bounded oligo statistics, not IVT-wide reservoirs')
            statistics[key] = [context, torch.zeros_like(value), 0]
        row = statistics[key]
        if row[0] != context:
            raise ValueError('reference context changed within extraction')
        row[1] += value
        row[2] += 1
        seen.add(pos)
        accepted += 1
    return accepted


def statistics_payload(statistics, *, embedding_key, k, radius, metadata):
    """Serialize site sums as sorted means/counts under the single v1 contract."""
    import torch
    keys = sorted(statistics)
    if not keys:
        raise ValueError('no eligible neighborhood sites were observed')
    means = torch.stack([statistics[key][1] / statistics[key][2] for key in keys]).float()
    return dict(artifact_type='effect_neighborhood_statistics', format_version=1,
                embedding_key=embedding_key, embedding_dim=means.shape[1], kmer_size=k, radius=radius,
                chroms=[key[0] for key in keys], positions=torch.tensor([key[1] for key in keys]),
                strands=torch.tensor([key[2] for key in keys]), contexts=[statistics[key][0] for key in keys],
                means=means, counts=torch.tensor([statistics[key][2] for key in keys]), metadata=metadata)


def parse_args(argv=None):
    """Parse an oligo-only extraction job with explicit production QC settings."""
    from .embedding_selection import EMBEDDING_KEY_CHOICES
    from porefm.common.run_outputs import add_output_safety_argument
    parser = PoreFMArgumentParser(description=__doc__)
    add_output_safety_argument(parser)
    add_chunking_arguments(parser)
    for name in ('bam_path', 'model_path', 'kmer_dict_path', 'ref', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--chrom', nargs='+', default=None, help='Default: all contigs in the bounded oligo reference.')
    parser.add_argument('--embedding_key', choices=EMBEDDING_KEY_CHOICES, default='contextual_base_seq')
    parser.add_argument('--seq_type', choices=['rna', 'dna'], default='rna')
    parser.add_argument('--pore_type', default='r10')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--disable_compile', action='store_true')
    parser.add_argument('--use_polished_sequence', action='store_true')
    parser.add_argument('--polish_vcf_path', default=None)
    for name, default in [('kmer_size', 5), ('radius', 4), ('size', 2048), ('step', None), ('num_trim', 8),
                          ('reads_per_chunk', 16), ('batch_size', 32), ('num_workers', 1), ('prefetch_factor', 2),
                          ('min_read_length', 80), ('flag', 0), ('seed', 0), ('log_every', 10),
                          ('max_batches', None), ('max_sites', 100000), ('polish_indel_limit', 5), ('threads', 4)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('max_read_length', 5e5), ('max_alignment_length', 1e6),
                          ('min_qscore', 10), ('min_base_quality', 10)]:
        parser.add_argument('--' + name, type=float, default=default)
    add_preprocessing_arguments(parser)
    args = parser.parse_args(argv)
    from .effects import validate_k
    validate_k(args.kmer_size)
    if not 1 <= args.radius <= 12:
        parser.error('--radius must be in [1, 12]')
    if args.num_trim < 0 or args.num_workers < 0 or not math.isfinite(args.min_base_quality) or args.min_base_quality < 0:
        parser.error('trim, workers, and base quality must be nonnegative')
    if any(getattr(args, key) < 1 for key in ('max_sites', 'threads', 'log_every', 'batch_size', 'reads_per_chunk', 'prefetch_factor')):
        parser.error('counts and thread options must be positive')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max_batches must be positive')
    return args


def main(argv=None):
    """Extract paired-fit sufficient statistics and provenance, without labels."""
    args = parse_args(argv)
    resolve_cli_preprocessing(args)
    from porefm.common.run_outputs import prepare_run_outputs
    prepare_run_outputs(args, command="prototype extract-effect-neighborhoods", files={"output": args.output})
    import torch
    import pysam
    from .effects import checkpoint_digest
    from .effects_pipeline import iter_read_embeddings
    torch.set_num_threads(args.threads)
    target = Path(args.output)
    with pysam.FastaFile(args.ref) as fasta:
        if args.chrom is None:
            args.chrom = list(fasta.references)
        if sum(fasta.get_reference_length(chrom) for chrom in args.chrom) > 10000000:
            raise ValueError('selected reference exceeds 10 Mb; use canonical k-mer reservoirs for IVT data')
        references = {chrom: fasta.fetch(chrom).upper() for chrom in args.chrom}
    metadata = dict(checkpoint_sha256=checkpoint_digest(args.model_path),
                    model_path=str(Path(args.model_path).resolve()), arguments=vars(args),
                    coordinate_system='0-based genomic; contexts and offsets read-oriented',
                    deduplication='one reference position per read; max chunk edge margin',
                    exact_match_k=args.kmer_size)
    stats, count, accepted = {}, 0, 0
    iterator = iter_read_embeddings(args, args.embedding_key, None, quality_features=True)
    try:
        for read in iterator:
            accepted += accumulate_read(read, references[read['chrom']], stats, k=args.kmer_size,
                                         radius=args.radius, min_base_quality=args.min_base_quality,
                                         max_sites=args.max_sites)
            count += 1
    finally:
        iterator.close()
    metadata.update(reads=count, accepted_read_sites=accepted)
    metadata['qc'] = {key: getattr(args, key) for key in (
        'size', 'step', 'seq_type', 'num_trim', 'min_qscore', 'min_base_quality', 'use_polished_sequence')}
    payload = statistics_payload(stats, embedding_key=args.embedding_key, k=args.kmer_size,
                                 radius=args.radius, metadata=metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    print(f'Wrote {target}: reads={count} sites={len(stats)} observations={accepted}', flush=True)


if __name__ == '__main__':
    main()
