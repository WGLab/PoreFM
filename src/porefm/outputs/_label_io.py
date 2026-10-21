"""Disk-backed intermediates shared by read-label selection helpers."""

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._filters import literal
from .summaries import _connection


def report(enabled, message):
    """Print completed stages without affecting selection or ordering."""
    if enabled:
        print(message, flush=True)


@contextmanager
def workspace(calls, temp_directory, output_path, memory_limit, threads, progress):
    """Own a unique intermediate/spill directory and close SQL before cleanup.

    ``temp_directory`` names its parent, not a shared DuckDB spill directory.
    None uses the system temporary directory. Outputs must be new files outside
    the input-part directories so subsequent discovery cannot ingest labels.
    """
    if not isinstance(progress, bool):
        raise ValueError('progress must be a boolean')
    target = None if output_path is None else Path(output_path).resolve()
    if target is not None:
        if target.exists():
            raise FileExistsError(f'refusing to overwrite {target}')
        input_dirs = {p.resolve().parent for part in calls.parts
                      for p in (part.data, part.meta) if p is not None}
        if target.parent in input_dirs:
            raise ValueError('Label output must be outside input-part directories')
    if temp_directory is not None:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='porefm-labels-', dir=temp_directory) as directory:
        root = Path(directory)
        report(progress, f'Label work directory: {root}')
        with _connection(memory_limit, root / 'spill', int(threads)) as con:
            yield con, root, target


def parquet_source(paths):
    """Build a SQL scan of an explicit, fixed list of intermediate files."""
    return 'read_parquet([' + ', '.join(literal(str(p)) for p in paths) + '], hive_partitioning=false)'


def copy_parquet(con, query, path):
    """Write a bounded-stage query directly to Parquet, including empty results."""
    con.execute(f'COPY ({query}) TO {literal(str(path))} '
                '(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)')


def empty_labels(con, calls):
    """Register a typed empty result when a snapshot has no input parts."""
    schema = pa.schema([calls.schema.field('readname'), calls.schema.field('read_pos'),
                        pa.field('label', pa.int64())])
    con.register('empty_labels', schema.empty_table())
    return 'SELECT * FROM empty_labels'


def finish(con, query, target, progress):
    """Globally deduplicate/sort, returning pandas or atomically publishing Parquet."""
    query = ('SELECT DISTINCT readname, read_pos, label FROM (' + query + ') '
             'ORDER BY readname, read_pos, label')
    report(progress, 'Finalizing labels: global deduplication and sorting')
    if target is None:
        result = con.execute(query).df()
        report(progress, f'Completed: {len(result):,} labels returned to pandas')
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    try:
        copy_parquet(con, query, temporary)
        # A hard-link publication cannot overwrite a file created after the
        # initial existence check. The temporary file is on the same filesystem.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    report(progress, f'Completed: labels written to {target}')
    return target


def sample_indices(path, population, count, random_state, *, chunk_size=262144):
    """Write exactly RandomState.choice(..., replace=False)'s selected indices.

    Pandas uses NumPy's legacy permutation-based uniform sampler for integer
    seeds. Shuffle the same arange in a disk-backed memmap, then write the first
    ``count`` indices in bounded chunks. Disk space is O(population), but no
    population-sized Python array or negative-label DataFrame is allocated.
    The OS manages mapped-page residency; this is not a hard process-RAM cap.
    Call only for 0 < count < population; retaining none/all needs no sampling.
    """
    permutation_path = path.with_suffix('.permutation')
    permutation = np.memmap(permutation_path, mode='w+', dtype=np.intp, shape=(population,))
    try:
        for start in range(0, population, chunk_size):
            stop = min(start + chunk_size, population)
            permutation[start:stop] = np.arange(start, stop, dtype=np.intp)
        rng = np.random if random_state is None else np.random.RandomState(int(random_state))
        rng.shuffle(permutation)
        schema = pa.schema([('sample_index', pa.int64())])
        with pq.ParquetWriter(path, schema, compression='zstd') as writer:
            for start in range(0, count, chunk_size):
                values = permutation[start:min(start + chunk_size, count)]
                writer.write_table(pa.table({'sample_index': pa.array(values, type=pa.int64())}))
    finally:
        # Close the mapping before its owning temporary directory is cleaned up.
        del permutation
