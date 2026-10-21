"""Read-aligned extraction and opt-in conditional effect classification.

All aligned chunks are loaded regardless of output labels. Overlap embeddings
are selected by their distance from the chunk edge before any joint scoring.
This module does not change the backbone or the existing prototype scorers.
"""

from __future__ import annotations

from porefm.preprocessing import validate_artifact_preprocessing
from porefm.cli.preprocessing_args import resolve_cli_preprocessing
from porefm.cli.chunking_args import resolve_chunking_mode
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from . import utils as query
from .embedding_selection import select_base_embedding, get_embedding_dimension


def stitch_embeddings(batch, embeddings, num_trim, qualities=None):
    """Yield whole-read arrays from the production complete-read batch contract.

    ``batch`` has the 13 complete-read fields; ``embeddings`` is (chunks,L,D).
    Embedding values remain on ``embeddings.device``; coordinate/selection
    metadata stays on CPU. Optional base error probabilities are (chunks,L). Overlapping tokens are
    deduplicated by greatest edge margin, with stable first-chunk tie breaking.
    Read positions are oriented indices and reference positions remain 0-based
    genomic coordinates on both strands. Missing read indices have N/-1/False.
    Validity describes observed read tokens, independent of reference mapping:
    inserted and soft-clipped bases may supply neighbor evidence. Alignment
    restrictions are applied separately by output selection or by the exact
    read/reference-match filter in neighborhood-statistics extraction.
    """
    seq, _, _, mask, _, _, names, chroms, refs, reads, _, strands, _ = batch
    if embeddings.shape[:2] != seq.shape or num_trim < 0:
        raise ValueError('invalid embedding shape or trim')
    groups = {}
    for chunk in range(len(seq)):
        key = (str(names[chunk]), str(chroms[chunk]), int(strands[chunk]))
        groups.setdefault(key, []).append(chunk)
    for (name, chrom, strand), chunks in groups.items():
        length = max((int(reads[c][mask[c].bool()].max()) + 1 for c in chunks if mask[c].any()), default=0)
        if length == 0:
            continue
        values = torch.zeros(length, embeddings.shape[-1], device=embeddings.device)
        tokens = torch.zeros(length, dtype=torch.long)
        reference = torch.full((length,), -1, dtype=torch.long)
        best = torch.full((length,), -1, dtype=torch.long)
        errors = torch.ones(length)
        for c in chunks:
            indices = torch.where(mask[c].bool() & (reads[c] >= 0))[0]
            if not len(indices):
                continue
            target = reads[c, indices].long()
            margin = torch.minimum(torch.arange(len(indices)), torch.arange(len(indices) - 1, -1, -1))
            old = best[target] >= 0
            if ((tokens[target[old]] != seq[c, indices[old]])
                    | (reference[target[old]] != refs[c, indices[old]])).any():
                raise ValueError('overlapping chunks disagree on sequence/reference coordinates')
            use = margin > best[target]
            target, indices, margin = target[use], indices[use], margin[use]
            best[target] = margin
            tokens[target] = seq[c, indices].long()
            reference[target] = refs[c, indices].long()
            values[target.to(embeddings.device)] = embeddings[c, indices.to(embeddings.device)].float()
            if qualities is not None:
                errors[target] = qualities[c, indices].float()
        valid = (best >= num_trim) & (tokens >= 1) & (tokens <= 4)
        yield dict(readname=name, chrom=chrom, strand=strand,
                   sequence=''.join('ACGT'[x - 1] if 1 <= x <= 4 else 'N' for x in tokens.tolist()),
                   embeddings=values, valid=valid, ref_positions=reference,
                   read_positions=torch.arange(length), base_error_probabilities=errors)


def embed_batch(batch, model, device, embedding_key, embedding_dim, batch_size, num_trim,
                qualities=None, *, embedding_device='cpu'):
    """Embed and stitch on the requested storage device using unchanged forwards.

    ``embedding_device`` defaults to CPU for neighborhood-statistics consumers.
    GPU effect scoring supplies CUDA so no high-dimensional embeddings are
    copied to CPU. Every forward output is released after copying its selected
    embedding; suspended generators retain only the required stitching buffer.
    Stitching retains usable unaligned tokens as evidence in both input modes;
    reference coordinates determine output selection, not embedding validity.
    """
    if batch_size < 1:
        raise ValueError('embedding batch_size must be positive')
    device = torch.device(device)
    seq, signal, pct, mask, signal_mask = batch[:5]
    embeddings = torch.empty(*seq.shape, embedding_dim, device=embedding_device)
    for start in range(0, len(seq), batch_size):
        sl = slice(start, start + batch_size)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                   enabled=device.type == 'cuda'):
            out = model(seq[sl].to(device), signal[sl].to(device), percentiles_input=pct[sl].to(device),
                        pad_base_seq=mask[sl].to(device), pad_ts=signal_mask[sl].to(device), embeddings_only=True)
            h = select_base_embedding(out, embedding_key)
        if h.shape != (*seq[sl].shape, embedding_dim):
            raise ValueError('query embedding shape differs from effect artifact')
        embeddings[sl] = h.detach().to(device=embedding_device, dtype=torch.float32)
        del h, out
    yield from stitch_embeddings(batch, embeddings, num_trim, qualities)


def iter_read_embeddings(args, embedding_key, embedding_dim, *, quality_features=False, embedding_device='cpu'):
    """Stream complete read embeddings with label-independent chunk retention.

    Optional quality mode explicitly consumes the production 16-field scale
    feature contract. A stopped/closed iterator shuts down its loader workers.
    No extraction inputs or checkpoints are modified.
    ``embedding_device`` controls stitched tensor storage without changing the
    backbone execution device or the CPU metadata/complete-read contract.
    An optional ``args.read_based`` selects motifs in read space; ``args.chrom``
    can restrict indexed input and ``args.ref`` can retain genomic coordinates.
    Without a contig selection, read-based input streams all BAM records.
    Neighborhood-extraction callers omit that option and retain aligned mode.
    BAM output retains source SAM dictionaries in ``pysam_record`` without
    changing the complete-read batch contract or embedding orientation.
    Optional ``args.exclude_bed`` removes whole reads by BAM reference-span
    overlap before signal decoding, including in read-based mode without FASTA.
    Optional ``args.pod5`` uses one BAM and a saved index prepared before loader
    startup; POD5 files are partitioned among workers, with unordered output.
    """
    resolve_cli_preprocessing(args)
    from porefm.common.chunking import resolve_downstream_step
    pod5_options = query.prepare_cli_pod5_input(args)
    query.lazy_import_runtime_dependencies()
    query.require_single_process_environment('effect prototypes', 'Joint read inference runs in one process.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, config = query.load_pretrain_model(args.model_path, device)
    if embedding_dim is None:
        embedding_dim = get_embedding_dimension(config, embedding_key)
    model.eval()
    if not args.disable_compile:
        model = torch.compile(model, dynamic=True)
    step = resolve_downstream_step(args.size, args.seq_type, args.step, context='effect prototypes')
    args.step = step
    read_based = getattr(args, 'read_based', False)
    retain_bam = getattr(args, 'output_format', 'parquet') in ('bam', 'both')
    from .effects_bam import BAM_TAGS_TO_DROP
    chrom_info = (
        query.get_read_based_chrom_info(args.ref, args.chrom)
        if read_based else query.get_chrom_info(args.ref, args.chrom)
    )
    samples = [query.LabelledBamSample(
        bam_path=p, chrom_info=chrom_info, ref=args.ref, num_classes=2,
        pos_file=None, motif='A 0;C 0;G 0;T 0' if read_based else None,
        read_based=read_based, flag=args.flag, polish=args.use_polished_sequence,
        **pod5_options,
        polish_indel_limit=args.polish_indel_limit,
        polish_vcf_path=args.polish_vcf_path if args.use_polished_sequence else None,
    ) for p in query.expand_bam_paths(args.bam_path)]
    dataset = query.CompleteLabelledReadIterableDataset(
        bam_list=samples, size=args.size, step=step, move_downsample_factor=1,
        seq_type=args.seq_type, chunking_mode=resolve_chunking_mode(args), kmer_table=resolve_cli_preprocessing(args).kmer_table,
        reads_per_chunk=args.reads_per_chunk, min_read_length=args.min_read_length,
        max_read_length=args.max_read_length, max_alignment_length=args.max_alignment_length,
        min_qscore=args.min_qscore, pore_type=args.pore_type, base_seed=args.seed,
        rank=0, world_size=1, discard_empty_labels=False, return_pysam=retain_bam,
        tags_to_drop=BAM_TAGS_TO_DROP,
        return_scale_features=quality_features, label_all_aligned_bases=not read_based,
        exclude_bed=getattr(args, 'exclude_bed', None),
        expected_ts_dim=config.ts_dim, allow_pod5=True)
    num_workers = query.inference_worker_count(dataset, args.num_workers)
    kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=device.type == 'cuda')
    if num_workers:
        kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    loader = query.DataLoader(dataset, **kwargs)
    iterator = iter(loader)
    start, reads = time.time(), 0
    try:
        for index, batch in enumerate(iterator, 1):
            if quality_features:
                (*core, qualities, _global_features, _read_lengths) = batch
                if len(core) != 13:
                    raise ValueError('expected complete-read scale-feature batch contract')
            else:
                core, qualities = batch, None
                if len(core) != 13:
                    raise ValueError('expected complete-read batch contract')
            for read in embed_batch(core, model, device, embedding_key, embedding_dim,
                                    args.batch_size, args.num_trim, qualities, embedding_device=embedding_device):
                if retain_bam:
                    read['pysam_record'] = core[12][read['readname']]
                reads += 1
                yield read
            if index % args.log_every == 0:
                print(f'[{time.time() - start:.1f}s] batches={index} reads={reads}', flush=True)
            if args.max_batches is not None and index >= args.max_batches:
                break
    finally:
        query.shutdown_dataloader_iterator(iterator)


def select_output(read, selectors, *, read_based=False, read_motifs=None):
    """Select output sites independently of inferred modification states.

    ``selectors`` holds reference-position labels in aligned mode. Read-based
    mode instead uses parsed read motifs, or all observed canonical tokens
    when ``read_motifs`` is None. Selection does not restrict neighbor evidence.
    """
    positions = read['ref_positions']
    if read_based:
        if read_motifs is None:
            return torch.tensor([base in 'ACGT' for base in read['sequence']], dtype=torch.bool)
        from porefm.data.bam import get_read_motif_targets
        targets, _ = get_read_motif_targets(read['sequence'], read_motifs, num_classes=1)
        selected = torch.zeros(len(positions), dtype=torch.bool)
        selected[torch.as_tensor(targets).long()] = True
        return selected
    if selectors is None:
        return positions >= 0
    entry = selectors.get(read['chrom'], {}).get(read['strand'])
    if entry is None:
        return torch.zeros(len(positions), dtype=torch.bool)
    return (positions >= 0) & torch.isin(positions, torch.as_tensor(entry[0]).long())


def iter_scoring_groups(reads, *, max_reads, max_bases):
    """Group complete reads while bounding retained embedding bases.

    A read longer than ``max_bases`` is processed alone with tiled scoring;
    it is never split into independent inference problems. Ordering is stable.
    Consumers may clear a yielded list after releasing its GPU embeddings.
    """
    if min(max_reads, max_bases) < 1:
        raise ValueError('effect scoring read/base limits must be positive')
    pending, bases = [], 0
    for read in reads:
        length = len(read['sequence'])
        if pending and (len(pending) >= max_reads or bases+length > max_bases):
            yield pending
            pending, bases = [], 0
        pending.append(read)
        bases += length
        if len(pending) >= max_reads or bases >= max_bases:
            yield pending
            pending, bases = [], 0
    if pending:
        yield pending


def effect_rows(read, result, model, selected):
    """Build reference/debug records; production output uses typed columns.

    Unscorable selected positions are retained with class index -1 and null
    class/scores, not relabelled canonical. Off-family class columns are null.
    Read-based calls retain oriented zero-based ``read_pos``; missing genomic
    coordinates are written as ``position=-1`` (unmapped: chrom='*', strand=-1).
    CPU tensors are converted in batches before record construction, avoiding
    per-scalar tensor indexing/isfinite dispatch on the decoder thread.
    """
    labels, k = model['class_labels'], model['kmer_size']
    positions = torch.where(selected)[0]
    indices = positions.tolist()
    values = {key: result[key][positions].tolist() for key in (
        'predicted_class_index', 'isolated_class_index', 'scores',
        'neighbor_count', 'canonical_energy')}
    references = read['ref_positions'][positions].tolist()
    fields = [f'score_{label}' for label in labels]
    rows = []
    for index, pos in enumerate(indices):
        idx = values['predicted_class_index'][index]
        isolated = values['isolated_class_index'][index]
        scores = values['scores'][index]
        row = dict(readname=read['readname'], chrom=read['chrom'], strand=read['strand'],
                   read_pos=pos, position=references[index],
                   kmer=read['sequence'][max(0, pos - k // 2):pos + k // 2 + 1],
                   predicted_class_index=idx, predicted_class=labels[idx] if idx >= 0 else None,
                   score=scores[idx] if idx >= 0 else None,
                   score_semantics='conditional_uncalibrated',
                   isolated_class=labels[isolated] if isolated >= 0 else None,
                   neighbor_count=values['neighbor_count'][index],
                   canonical_energy=values['canonical_energy'][index] if idx >= 0 else None,
                   inference_converged=result['converged'], call_status='called' if idx >= 0 else 'unavailable_context')
        for j, score_key in enumerate(fields):
            row[score_key] = scores[j] if math.isfinite(scores[j]) else None
        rows.append(row)
    return rows


def write_effect_rows(rows, output_dir, part, labels, *, force_overwrite=False):
    """Write typed parquet rows, allowing replacement after an explicit force.

    ``force_overwrite`` is supplied by the run's output preflight, which has
    already warned about occupied destinations before any result writes.
    Empty batches produce no file; all-null/no-call columns retain their types.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not rows:
        return None
    integer = {'strand', 'read_pos', 'position', 'predicted_class_index', 'neighbor_count'}
    floating = {'score', 'canonical_energy'}
    floating.update(f'{field}_{label}' for field in ('score',) for label in labels)
    schema = pa.schema([(key, pa.int64() if key in integer else pa.float32() if key in floating
                         else pa.bool_() if key == 'inference_converged' else pa.string()) for key in rows[0]
                         if key != 'probability' and not key.startswith('probability_')])
    path = Path(output_dir) / f'part_{part:06d}.parquet'
    if path.exists() and not force_overwrite:
        raise FileExistsError(f'refusing to overwrite {path}')
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression='snappy')
    return path


def run(args):
    """Apply a fitted artifact through the public prototype-classification CLI."""
    resolve_cli_preprocessing(args)
    from porefm.data.excluded_regions import get_excluded_regions
    if args.exclude_bed:
        args.exclude_bed_sha256 = get_excluded_regions(args.exclude_bed).sha256
    from porefm.common.run_outputs import prepare_run_outputs
    prepare_run_outputs(args, command="prototype classify residual-effects",
                        directories={"output_dir": args.output_dir},
                        output_arguments={"classifier_output"})
    query.prepare_cli_pod5_input(args)
    from .effects import (EffectInferenceCache, load_effect_model, checkpoint_digest,
                          score_effect_read, score_effect_packet)
    from .effects_decode import decoder_backend, warm_decoder
    from .effects_output import EffectWriter, effect_columns, LAYOUT
    decoder = decoder_backend(args.effect_decoder)
    print(f'Effect CPU decoder: {decoder} (requested {args.effect_decoder}); initializing', flush=True)
    warm_decoder(decoder)
    from porefm.data.bam import get_labelled_arrays
    model = load_effect_model(args.effect_model)
    validate_artifact_preprocessing(model, args.preprocessing)
    if checkpoint_digest(args.model_path) != model['metadata']['checkpoint_sha256']:
        raise ValueError('query backbone checkpoint does not match fitted effect artifact')
    if args.embedding_key is not None and args.embedding_key != model['embedding_key']:
        raise ValueError('query embedding_key does not match fitted effect artifact')
    write_parquet = args.output_format in ('parquet', 'both')
    write_bam = args.output_format in ('bam', 'both')
    if write_bam:
        from .effects_bam import EffectBamWriter, bam_families, effect_bam_header
        families = bam_families(model)
        bam_header = effect_bam_header(query.expand_bam_paths(args.bam_path))
    torch.set_num_threads(args.effect_threads)
    backbone_device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    effect_device = backbone_device if args.effect_device == 'auto' else torch.device(args.effect_device)
    if effect_device.type not in ('cpu', 'cuda'):
        raise ValueError('effect scoring supports CPU or CUDA')
    if effect_device.type == 'cuda':
        from .effects_device import DeviceEffectScorer
        scorer = DeviceEffectScorer(model, effect_device, cache_bytes=args.effect_cache_mb * 1024 ** 2,
                                    block_size=args.effect_block_size)
    else:
        cache = EffectInferenceCache(model, max_bytes=args.effect_cache_mb * 1024 ** 2)
    output = Path(args.output_dir)
    force_overwrite = bool(getattr(args, 'force_overwrite', False))
    read_based = args.read_based
    selectors, read_motifs = None, None
    if read_based and not query.is_all_base_motif(args.motif):
        from porefm.data.complete_reads import parse_read_based_motifs
        read_motifs = parse_read_based_motifs(args.motif)
    elif not read_based and (args.pos_file or (args.motif and not query.is_all_base_motif(args.motif))):
        selectors = get_labelled_arrays(args.ref, args.seq_type, args.pos_file, args.motif,
                                        args.motif_label, args.chrom, num_classes=2)
    metadata = dict(scorer='residual-effects', effect_model=str(Path(args.effect_model).resolve()),
                    effect_model_sha256=checkpoint_digest(args.effect_model), arguments=vars(args),
                    effect_format_version=model['format_version'], fit_settings=model['metadata'].get('fit_settings'),
                    class_labels=model['class_labels'], calibration='none', effect_device=str(effect_device),
                    decoder_workers=1, decoder_backend=decoder, score_queue=(effect_device.type == 'cuda'),
                    output_layout=LAYOUT if write_parquet else 'bam_mm_ml',
                    observation_validity='observed_read_tokens_independent_of_reference_alignment',
                    score_semantics='conditional scores given inferred neighbors; uncalibrated')
    if write_bam:
        metadata['bam'] = dict(path='output.bam', complete=False,
                              probability_transform='family_softmax',
                              probability_temperature=args.effect_probability_temperature,
                              probability_semantics='conditional_uncalibrated',
                              sam_codes={model['class_labels'][index]: f'{base}+{code}'
                                         for base, indices, codes in families
                                         for index, code in zip(indices[1:], codes)},
                              skipped_sites='unknown', sort_order='unsorted')
    (output / 'effect_run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    def append_result(read, result):
        """Convert local family scores to columns on the decoder thread."""
        selected = select_output(read, selectors, read_based=read_based, read_motifs=read_motifs)
        if write_parquet:
            table, meta = effect_columns(read, result, model, selected)
            writer.append(table, meta)
        if write_bam:
            bam_writer.append(read, result, selected)

    def decode_packet(payload):
        read, packet = payload
        result = score_effect_packet(packet, model['class_labels'], beam_width=args.effect_beam_width,
                                      max_sweeps=args.effect_max_sweeps, decoder=decoder)
        append_result(read, result)

    if effect_device.type == 'cuda':
        iterator = iter_read_embeddings(args, model['embedding_key'], model['embedding_dim'], embedding_device=effect_device)
    else:
        iterator = iter_read_embeddings(args, model['embedding_key'], model['embedding_dim'])
    try:
        with ExitStack() as stack:
            if write_bam:
                bam_writer = stack.enter_context(EffectBamWriter(
                    output, bam_header, families, seq_type=args.seq_type,
                    temperature=args.effect_probability_temperature, threads=args.bam_threads,
                    force_overwrite=force_overwrite))
            if write_parquet:
                writer = stack.enter_context(EffectWriter(
                    output, model, rows_per_part=args.effect_write_rows,
                    max_bytes=args.effect_write_mb*1024**2, workers=args.effect_writer_threads,
                    force_overwrite=force_overwrite))
            if effect_device.type == 'cuda':
                from .effects_queue import BoundedScoreQueue
                with BoundedScoreQueue(decode_packet, max_items=args.effect_queue_size,
                                       max_bytes=args.effect_queue_mb * 1024 ** 2) as queue:
                    for group in iter_scoring_groups(iterator, max_reads=args.effect_batch_reads,
                                                    max_bases=args.effect_batch_bases):
                        queue.check()
                        packets = scorer.prepare_many(group, neighbors=args.effect_neighbors == 'on',
                                                      penalty=args.effect_penalty)
                        # Release grouped GPU embeddings before CPU backpressure.
                        metadata_keys = ('readname', 'chrom', 'strand', 'sequence', 'ref_positions')
                        if write_bam:
                            metadata_keys += ('pysam_record',)
                        metadata_reads = [{key: read[key] for key in metadata_keys}
                                          for read in group]
                        group.clear()
                        for read, packet in zip(metadata_reads, packets):
                            nbytes = packet.nbytes + read['ref_positions'].numel() * read['ref_positions'].element_size()
                            nbytes += sum(len(read[key].encode('utf-8')) for key in ('readname', 'chrom', 'sequence'))
                            if write_bam:
                                nbytes += len(json.dumps(read['pysam_record']).encode('utf-8'))
                            queue.submit((read, packet), nbytes)
                print(dict(effect_queue_peak_items=queue.peak_items, effect_queue_peak_bytes=queue.peak_bytes), flush=True)
            else:
                for read in iterator:
                    result = score_effect_read(read['sequence'], read['embeddings'], read['valid'], model,
                                               neighbors=args.effect_neighbors == 'on', penalty=args.effect_penalty,
                                               beam_width=args.effect_beam_width, max_sweeps=args.effect_max_sweeps,
                                               cache=cache, decoder=decoder)
                    append_result(read, result)
    finally:
        iterator.close()
    if write_parquet:
        print(dict(total_rows=writer.total_rows, no_calls=writer.no_calls, output_parts=writer.part,
                   output_layout=LAYOUT, writer_peak_pending_bytes=writer.peak_pending_bytes,
                   writer_peak_buffer_bytes=writer.peak_buffer_bytes,
                   writer_peak_pending_parts=writer.peak_pending_parts, calibration='none'), flush=True)
    if write_bam:
        metadata['bam'].update(complete=True, total_reads=bam_writer.total_reads,
                               tagged_reads=bam_writer.tagged_reads)
        (output / 'effect_run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(dict(output_bam=str(bam_writer.path), bam_reads=bam_writer.total_reads,
                   bam_tagged_reads=bam_writer.tagged_reads, calibration='none'), flush=True)
