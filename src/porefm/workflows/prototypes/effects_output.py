"""Compact, columnar effect calls with bounded asynchronous Parquet writing.

Each committed part pairs numeric ``*_data.parquet`` rows with one metadata
row per read in ``*_meta.parquet``. Metadata lengths partition the data in read
order. Scores contain only the receiving base's class family;
the family class order is stored once in Arrow schema metadata. No-call lists
are empty and predicted class indices remain -1. A completion manifest lists
exactly the committed parts, so readers never ingest stale or partial files.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


LAYOUT = 'effect_family_meta_data_v1'
BASES = 'ACGT'


def output_metadata(model):
    """Describe local score ordering and the lossless metadata/data pairing."""
    return dict(layout=LAYOUT, class_labels=list(model['class_labels']),
                kmer_size=model['kmer_size'], family_bases=list(BASES),
                family_class_indices=[([model['families'][base]['canonical_class_index']] +
                                       [m['class_index'] for m in model['families'][base]['modifications']])
                                      if base in model['families'] else [] for base in BASES],
                pairing='data rows are contiguous in metadata row order; length is the output-row count',
                score_semantics='conditional_uncalibrated',
                no_call='predicted_class_index=-1; empty score lists',
                coordinates='read_pos is oriented zero-based; position is zero-based genomic or -1')


def effect_columns(read, result, model, selected):
    """Build one read's typed Arrow data and small metadata record.

    ``selected[L]`` is a CPU boolean tensor; result tensors follow
    ``score_effect_packet``. Only selected positions are serialized, in read
    order. Local lists contain canonical plus modifications for that base and
    never off-family null placeholders. Metadata keeps the full oriented read
    sequence so reference-free calls and exact k-mer reconstruction are retained.
    Empty selections return a typed empty table and a zero-length metadata row.
    """
    positions = np.flatnonzero(selected.numpy()).astype(np.int32)
    predicted = result['predicted_class_index'].numpy()[positions].astype(np.int32)
    called = predicted >= 0
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.frombuffer(b'ACGT', dtype=np.uint8)] = np.arange(4)
    sequence = read['sequence'].upper().replace('U', 'T')
    families = lookup[np.frombuffer(sequence.encode('ascii', errors='replace'), dtype=np.uint8)[positions]]
    metadata = output_metadata(model)
    counts = np.zeros(len(positions), dtype=np.int32)
    for base, classes in enumerate(metadata['family_class_indices']):
        counts[(families == base) & called] = len(classes)
    offsets = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(counts, dtype=np.int32)])
    local_scores = np.empty(int(offsets[-1]), dtype=np.float32)
    scores = result['scores'].numpy()
    for base, classes in enumerate(metadata['family_class_indices']):
        rows = np.flatnonzero((families == base) & called)
        if len(rows) and classes:
            target = offsets[rows, None] + np.arange(len(classes))
            source = np.ix_(positions[rows], classes)
            local_scores[target] = scores[source]

    def lists(values):
        return pa.ListArray.from_arrays(offsets, pa.array(values, mask=~np.isfinite(values)))

    energy = result['canonical_energy'].numpy()[positions].astype(np.float32)
    table = pa.table(dict(
        read_pos=pa.array(positions),
        position=pa.array(read['ref_positions'].numpy()[positions], type=pa.int64()),
        family_index=pa.array(families),
        predicted_class_index=pa.array(predicted),
        isolated_class_index=pa.array(result['isolated_class_index'].numpy()[positions], type=pa.int32()),
        neighbor_count=pa.array(result['neighbor_count'].numpy()[positions], type=pa.int32()),
        canonical_energy=pa.array(energy, mask=~called),
        scores=lists(local_scores),
    )).replace_schema_metadata({b'porefm_effects': json.dumps(metadata).encode()})
    meta = dict(readname=read['readname'], chrom=read['chrom'], strand=int(read['strand']),
                sequence=read['sequence'], inference_converged=bool(result['converged']),
                length=len(positions))
    return table, meta


class EffectWriter:
    """Accumulate whole reads and write large parts on a bounded thread pool.

    Parameters
    ----------
    output_dir : path-like
        Prepared run directory. No file is replaced unless ``force_overwrite``.
    model : dict
        Immutable effect artifact providing class-family and k-mer metadata.
    rows_per_part : int
        Target output rows per part (default four million). Flushes at a read
        boundary; one read may exceed the row target.
    max_bytes : int
        Byte budget for the current buffer and, independently, all outstanding
        writer tasks, including in-flight tasks. Arrow buffers and UTF-8 metadata
        are accounted; allocator, compression, and filesystem buffers are extra.
        One read larger than this budget is rejected with an actionable error.
    workers : int
        Concurrent part writers. They receive immutable Arrow buffers, not
        per-base Python dictionaries. Decoder calls to ``append`` stay serial.

    Failures propagate on append, flush, or close. The final completion manifest
    is published only after all paired files succeed. On errors, pending tasks
    are cancelled, running tasks are joined, and no complete manifest is emitted.
    """

    def __init__(self, output_dir, model, *, rows_per_part=4_000_000,
                 max_bytes=512*1024**2, workers=2, force_overwrite=False):
        if min(rows_per_part, max_bytes, workers) < 1:
            raise ValueError('effect output row/byte/worker limits must be positive')
        self.output = Path(output_dir)
        self.metadata = output_metadata(model)
        self.rows_per_part, self.max_bytes = rows_per_part, max_bytes
        self.workers, self.force = workers, force_overwrite
        self.tables, self.reads, self.pending = [], [], deque()
        self.buffer_bytes = self.buffer_rows = self.pending_bytes = 0
        self.total_rows = self.no_calls = self.total_reads = 0
        self.peak_pending_bytes = self.peak_buffer_bytes = self.peak_pending_parts = 0
        self.parts = []
        self.part = 0
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='effect-write')
        self.closed = False

    def __enter__(self):
        self._manifest(complete=False)
        return self

    def _manifest(self, *, complete):
        target = self.output / 'effect_parts.json'
        if not complete and target.exists() and not self.force:
            raise FileExistsError(f'refusing to overwrite {target}')
        payload = dict(**self.metadata, complete=complete, parts=self.parts,
                       total_rows=self.total_rows, no_calls=self.no_calls,
                       total_reads=self.total_reads)
        temp = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
        temp.write_text(json.dumps(payload, indent=2)+'\n')
        temp.replace(target)

    def _finish_oldest(self):
        future, size = self.pending.popleft()
        self.pending_bytes -= size
        self.parts.append(future.result())

    def check(self):
        """Surface asynchronous failures and release completed FIFO buffers."""
        while self.pending and self.pending[0][0].done():
            self._finish_oldest()

    def append(self, table, meta):
        """Append one read's selected columns without materializing base records."""
        self.check()
        if not len(table):
            return
        size = table.nbytes + sum(len(str(meta[key]).encode()) for key in ('readname', 'chrom', 'sequence')) + 64
        if size > self.max_bytes:
            raise ValueError(f'one read requires {size} output bytes; increase --effect_write_mb')
        if self.tables and self.buffer_bytes + size > self.max_bytes:
            self.flush()
        self.tables.append(table)
        self.reads.append(meta)
        self.buffer_bytes += size
        self.buffer_rows += len(table)
        self.total_rows += len(table)
        self.no_calls += int(pc.sum(pc.cast(pc.less(table['predicted_class_index'], 0), pa.int64())).as_py())
        self.total_reads += 1
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.buffer_bytes)
        if self.buffer_rows >= self.rows_per_part:
            self.flush()

    def flush(self):
        """Submit immutable column buffers, waiting only at the configured bounds."""
        self.check()
        if not self.tables:
            return
        while self.pending and (len(self.pending) >= self.workers or
                                self.pending_bytes+self.buffer_bytes > self.max_bytes):
            self._finish_oldest()
        table = pa.concat_tables(self.tables)
        meta = pa.table({key: [read[key] for read in self.reads] for key in self.reads[0]})
        meta = meta.replace_schema_metadata(table.schema.metadata)
        size = self.buffer_bytes
        future = self.executor.submit(self._write_part, table, meta, self.part)
        self.pending.append((future, size))
        self.pending_bytes += size
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        self.peak_pending_parts = max(self.peak_pending_parts, len(self.pending))
        self.part += 1
        self.tables, self.reads = [], []
        self.buffer_bytes = self.buffer_rows = 0

    def _write_part(self, data, meta, part):
        """Publish paired files; the completion manifest controls visibility."""
        paths = [self.output / f'part_{part:06d}_{kind}.parquet' for kind in ('data', 'meta')]
        for path in paths:
            if path.exists() and not self.force:
                raise FileExistsError(f'refusing to overwrite {path}')
        for index, (table, path) in enumerate(zip((data, meta), paths)):
            temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
            # Family-local lists interleave zeros and nonzero floats. Byte-stream
            # splitting improves their compression without changing any bits;
            # dictionary encoding is useful only for low-cardinality integers.
            encoding = (dict(use_dictionary=['family_index', 'predicted_class_index',
                                             'isolated_class_index', 'neighbor_count'],
                             use_byte_stream_split=['canonical_energy', 'scores.list.element']) if index == 0 else {})
            pq.write_table(table, temp, compression='zstd', compression_level=1,
                           row_group_size=262144, **encoding)
            temp.replace(path)
        return dict(data=paths[0].name, meta=paths[1].name, rows=len(data), reads=len(meta))

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.flush()
                while self.pending:
                    self._finish_oldest()
                self._manifest(complete=True)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()
            self.tables, self.reads = [], []
            self.pending_bytes = self.buffer_bytes = self.buffer_rows = 0
            self.closed = True


# Public reader imports retained for existing callers; implementation is centralized.
from porefm.outputs._effects import iter_effect_tables, read_effect_table, open_effect_reader
