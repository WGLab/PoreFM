"""Arrow reconstruction and manifest snapshots for compact prototype calls."""

from itertools import chain
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._columns import without_probabilities
from ._batching import split_reads
from ._filters import filter_table

LAYOUT = 'effect_family_meta_data_v1'

def _expanded_schema(metadata, columns):
    """Resolve projected logical fields, retaining types for empty selections."""
    types = dict(readname=pa.string(), chrom=pa.string(), strand=pa.int64(),
                 read_pos=pa.int32(), position=pa.int64(), kmer=pa.string(),
                 predicted_class_index=pa.int32(), predicted_class=pa.string(),
                 score=pa.float32(), score_semantics=pa.string(),
                 isolated_class=pa.string(), neighbor_count=pa.int32(),
                 canonical_energy=pa.float32(), inference_converged=pa.bool_(), call_status=pa.string())
    types.update({f'{field}_{label}': pa.float32()
                  for label in metadata['class_labels'] for field in ('score',)})
    default = list(types)
    types.update(family_index=pa.int8(), isolated_class_index=pa.int32())
    columns = default if columns is None else without_probabilities(columns)
    unknown = set(columns) - types.keys()
    if unknown:
        raise ValueError(f'unknown effect call columns: {sorted(unknown)}')
    return pa.schema([(name, types[name]) for name in columns])


def _expanded_table(batch, meta, metadata, row_indices, schema, read_ends, sequences):
    """Reconstruct retained rows using their original absolute part offsets.

    ``row_indices`` must refer to rows before filtering so metadata/read
    boundaries remain correct when entire reads or batches are rejected.
    Read ends and optional sequence strings are cached once per part, avoiding
    repeated full-metadata conversions when yielding individual reads.
    """
    data = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
    read_indices = np.searchsorted(read_ends, row_indices, side='right')
    take = pa.array(read_indices)
    labels = metadata['class_labels']
    predicted = data['predicted_class_index'].to_numpy()
    isolated = data['isolated_class_index'].to_numpy()
    numeric = ('read_pos', 'position', 'family_index', 'predicted_class_index',
               'isolated_class_index', 'neighbor_count', 'canonical_energy')
    columns = schema.names
    result = {}
    for name in columns:
        if name in numeric:
            result[name] = data[name]
        elif name in ('readname', 'chrom', 'strand', 'inference_converged'):
            result[name] = pc.take(meta[name], take)
        elif name in ('predicted_class', 'isolated_class'):
            ids = predicted if name == 'predicted_class' else isolated
            result[name] = pc.take(pa.array(labels), pa.array(ids, mask=ids < 0))
        elif name == 'score_semantics':
            result[name] = pa.repeat(metadata['score_semantics'], len(data))
        elif name == 'call_status':
            result[name] = pc.if_else(pa.array(predicted >= 0), 'called', 'unavailable_context')
        elif name == 'kmer':
            k = metadata['kmer_size']
            result[name] = pa.array([sequences[r][max(0, int(p)-k//2):int(p)+k//2+1]
                                     for r, p in zip(read_indices, data['read_pos'].to_numpy())])
        elif name != 'score' and not any(
                name == f'{field}_{label}' for label in labels for field in ('score',)):
            raise ValueError(f'unknown effect call column: {name}')
    family = data['family_index'].to_numpy()
    for field, stored in (('score', 'scores'),):
        requested = [name for name in columns if name == field or name.startswith(field+'_') and
                     name != 'score_semantics']
        if not requested:
            continue
        values = np.full((len(data), len(labels)), np.nan, dtype=np.float32)
        lists = data[stored].combine_chunks()
        offsets, flat = lists.offsets.to_numpy(), lists.values.to_numpy(zero_copy_only=False)
        for base, classes in enumerate(metadata['family_class_indices']):
            rows = np.flatnonzero((family == base) & (predicted >= 0))
            if len(rows) and classes:
                values[np.ix_(rows, classes)] = flat[offsets[rows, None]+np.arange(len(classes))]
        for name in requested:
            if name == field:
                selected = values[np.arange(len(data)), np.maximum(predicted, 0)]
                result[name] = pa.array(selected, mask=predicted < 0)
            else:
                selected = values[:, labels.index(name[len(field)+1:])]
                result[name] = pa.array(selected, mask=~np.isfinite(selected))
    return pa.table({name: result[name] for name in columns}, schema=schema)


def _incomplete_parts(root, manifest, manifest_mtime_ns):
    """Snapshot published pairs belonging to the still-incomplete run.

    The writer creates the incomplete manifest before publishing any part and
    renames data before metadata. Require both final filenames, timestamps at
    least as new as that manifest, and identical embedded run-layout metadata.
    This excludes temporary files and old pairs left by an overwrite. Footer
    errors and incompatible pairs are surfaced rather than silently skipped.
    """
    parts = []
    for data_path in sorted(root.glob('part_[0-9][0-9][0-9][0-9][0-9][0-9]_data.parquet')):
        meta_path = data_path.with_name(data_path.name.replace('_data.parquet', '_meta.parquet'))
        if not meta_path.exists():
            continue
        if min(data_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns) < manifest_mtime_ns:
            continue
        data, meta = pq.ParquetFile(data_path), pq.ParquetFile(meta_path)
        data_info = json.loads((data.schema_arrow.metadata or {}).get(b'porefm_effects', b'{}'))
        meta_info = json.loads((meta.schema_arrow.metadata or {}).get(b'porefm_effects', b'{}'))
        if (data_info.get('layout') != LAYOUT or data_info != meta_info
                or any(manifest.get(key) != value for key, value in data_info.items())):
            raise ValueError(f'effect part metadata differs from manifest: {data_path}')
        parts.append(dict(data=data_path.name, meta=meta_path.name,
                          rows=data.metadata.num_rows, reads=meta.metadata.num_rows))
    return parts


def _effect_snapshot(root, allow_incomplete):
    """Resolve a manifest and a fixed part list for Arrow or SQL consumers."""
    manifest_path = root / 'effect_parts.json'
    # Atomic finalization may replace the pathname while this snapshot opens.
    with manifest_path.open() as handle:
        manifest = json.load(handle)
        manifest_mtime_ns = os.fstat(handle.fileno()).st_mtime_ns
    if manifest.get('layout') != LAYOUT:
        raise ValueError(f'unsupported effect output layout: {manifest_path}')
    if not manifest.get('complete'):
        if not allow_incomplete:
            raise ValueError(f'incomplete effect output: {manifest_path}; wait for inference to finish '
                             'or pass allow_incomplete=True to inspect already written pairs')
        parts = _incomplete_parts(root, manifest, manifest_mtime_ns)
    else:
        parts = manifest['parts']
    for part in parts:
        if any(Path(part[key]).name != part[key] for key in ('data', 'meta')):
            raise ValueError('effect part paths must be basenames')
    return manifest, parts


def _iter_effect_rows(output_dir, *, columns, batch_size, _snapshot,
                      whole_reads=False, filters=()):
    """Expand projected physical batches from an already validated snapshot.

    Establish read boundaries using original metadata lengths before filtering.
    Class/status filters run before expensive score/k-mer expansion; other
    predicates are handled by the common reader. Legacy flat files go through
    the common physical reader. Probability payloads are never scanned.
    """
    root = Path(output_dir)
    manifest, parts = _snapshot
    schema = _expanded_schema(manifest, columns)
    columns = schema.names
    fast_filters = [item for item in filters if item[0] in ('predicted_class', 'call_status')]
    fast_schema = _expanded_schema(manifest, list(dict.fromkeys(item[0] for item in fast_filters)))
    for part in parts:
        meta_columns = sorted({'length'} | (set(columns) &
            {'readname', 'chrom', 'strand', 'inference_converged'}) |
            ({'sequence'} if 'kmer' in columns else set()))
        meta = pq.read_table(root / part['meta'], columns=meta_columns)
        data = pq.ParquetFile(root / part['data'])
        lengths = meta['length']
        if (not pa.types.is_integer(lengths.type) or lengths.null_count
                or any(n < 0 for n in lengths.to_pylist())
                or sum(lengths.to_pylist()) != data.metadata.num_rows
                or data.metadata.num_rows != part['rows'] or len(meta) != part['reads']):
            raise ValueError('effect metadata lengths do not match data rows')
        read_ends = np.cumsum(lengths.to_numpy())
        sequences = meta['sequence'].to_pylist() if 'kmer' in columns else None
        needed = {'predicted_class_index', 'isolated_class_index', 'family_index'}
        needed.update(set(columns) & set(data.schema_arrow.names))
        if 'kmer' in columns:
            needed.add('read_pos')
        if any(name == 'score' or name.startswith('score_') and name != 'score_semantics'
               for name in columns):
            needed.add('scores')
        tables = (pa.Table.from_batches([batch]) for batch in
                  data.iter_batches(batch_size=batch_size, columns=sorted(needed)))
        if whole_reads:
            tables = split_reads(tables, lengths.to_pylist())
        row_start = 0
        for table in tables:
            row_indices = np.arange(row_start, row_start + len(table))
            row_start += len(table)
            if fast_filters:
                fast = _expanded_table(table, meta, manifest, row_indices, fast_schema, read_ends, sequences)
                fast = fast.append_column('__offset', pa.array(np.arange(len(table))))
                keep = filter_table(fast, fast_filters)['__offset']
                if not len(keep):
                    continue
                table = table.take(keep)
                row_indices = row_indices[keep.to_numpy()]
            if len(table):
                yield _expanded_table(table, meta, manifest, row_indices, schema, read_ends, sequences)


def iter_effect_tables(output_dir, *, columns=None, batch_size=None, reads_per_batch=None,
                       allow_incomplete=False, call_status=None, predicted_classes=None,
                       _snapshot=None):
    """Yield complete-read Arrow batches, or row batches with explicit batch_size.

    Defaults to 32 reads; reads_per_batch counts records retaining matching rows.
    Historical probabilities are ignored. Published-part snapshots, projection
    and filter semantics follow OutputDataset.iter_tables. Pair metadata always
    refers to original unfiltered offsets. Memory scales with selected read size.
    """
    from .dataset import open_output
    if call_status not in (None, 'called', 'unavailable_context'):
        raise ValueError("call_status must be 'called', 'unavailable_context', or None")
    dataset = open_output(output_dir, format='prototypes', allow_incomplete=allow_incomplete)
    if _snapshot is not None:
        from .dataset import Part
        dataset._manifest, dataset._records = _snapshot
        dataset.parts = [Part(Path(output_dir)/p['data'], Path(output_dir)/p['meta']) for p in dataset._records]
    filters = {}
    if call_status is not None:
        filters['call_status'] = call_status
    if predicted_classes is not None:
        labels = [predicted_classes] if isinstance(predicted_classes, str) else list(predicted_classes)
        if any(not isinstance(label, str) for label in labels):
            raise TypeError('predicted_classes must contain class-label strings')
        if dataset._effect and set(labels) - set(dataset.class_labels):
            raise ValueError(f'unknown predicted classes: {set(labels) - set(dataset.class_labels)}')
        filters['predicted_class'] = labels
    yield from dataset.iter_tables(columns=columns, filters=filters, batch_size=batch_size,
                                   reads_per_batch=reads_per_batch)


def read_effect_table(output_dir, **options):
    """Load all selected calls as a chunked table; accepts iter_effect_tables options.

    Arrow concatenation shares buffers. Convert to pandas once if needed; all
    selected data must fit in RAM. Use the iterator for larger runs.
    """
    return pa.concat_tables(list(iter_effect_tables(output_dir, **options)))


def open_effect_reader(output_dir, **options):
    """Open a single-pass Arrow reader with the requested table batch boundaries.

    Accepts iter_effect_tables options. Combining chunks copies only the current
    yielded table so each RecordBatch keeps all selected rows of its reads.
    For SQL use register_effect_calls, avoiding Python worker callbacks.
    """
    tables = iter_effect_tables(output_dir, **options)
    first = next(tables)
    batches = (batch for table in chain((first,), tables)
               for batch in table.combine_chunks().to_batches())
    return pa.RecordBatchReader.from_batches(first.schema, batches)
