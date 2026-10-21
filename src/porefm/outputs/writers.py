"""Shared contiguous metadata/data serialization for row-oriented workflows."""

from pathlib import Path
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _compact_metadata(table, metadata_columns):
    """Extract contiguous identities without combining source string chunks.

    Scan at most 65,536 rows at a time, carrying the last identity across
    Arrow chunk and scan boundaries. Select run starts within each record
    batch before assembling the compact table. This keeps temporary decoded
    strings bounded and avoids the 32-bit offset overflow that a table-level
    ``take`` can cause when repeated strings total more than 2 GiB.

    Preserve metadata field types and annotations; append an int64 ``length``
    column counting source rows per identity run. The nonempty input table
    stays in producer order and is not modified.
    """
    identities = table.select(metadata_columns)
    selected_batches = []
    run_starts = []
    previous = None
    offset = 0
    for batch in identities.to_batches(max_chunksize=65536):
        if not batch.num_rows:
            continue
        boundary = np.zeros(batch.num_rows, dtype=bool)
        boundary[0] = previous is None
        last_values = []
        for index, column in enumerate(batch.columns):
            values = column.to_numpy(zero_copy_only=False)
            boundary[1:] |= values[1:] != values[:-1]
            if previous is not None:
                boundary[0] |= values[0] != previous[index]
            last_values.append(values[-1])
        starts = np.flatnonzero(boundary)
        if starts.size:
            # RecordBatch columns are single arrays: take never concatenates
            # the original per-base string column before selecting identities.
            selected_batches.append(batch.take(pa.array(starts)))
            run_starts.append(starts + offset)
        previous = last_values
        offset += batch.num_rows
    starts = np.concatenate(run_starts)
    meta = pa.Table.from_batches(selected_batches, schema=identities.schema)
    return meta.append_column(
        'length', pa.array(np.diff(np.r_[starts, len(table)]), type=pa.int64()),
    )


def write_split_part(table, output_dir, stem, *, format, metadata_columns=('readname', 'chrom', 'strand'),
                     compression='zstd', compression_level=None, use_dictionary=True, use_byte_stream_split=False):
    """Factor repeated identities into metadata and atomically publish a pair.

    ``table`` contains one row per scored read position, in producer order.
    Consecutive equal identities form metadata rows; ``length`` counts scored
    rows, not read bases. Recurring identities need not be adjacent: a later run
    gets another metadata row. Data order, dtypes, field metadata and score
    values are unchanged. No input table is mutated or reordered.

    Return the data Path. Publish data first, metadata last; readers only accept
    complete pairs. Each filename is atomically replaced, but the pair is not a
    transactional overwrite. The workflow must prepare a fresh output directory
    and must not replace a run being read. Compression mappings are projected
    onto each file's columns. Large parts bound overhead from per-file metadata.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if not len(table):
        raise ValueError('Cannot write an empty split part')
    if Path(stem).name != stem:
        raise ValueError('stem must be a basename')
    metadata_columns = list(metadata_columns)
    if 'length' in table.column_names or not metadata_columns:
        raise ValueError('length is reserved and metadata_columns must be nonempty')
    meta = _compact_metadata(table, metadata_columns)
    data = table.drop(metadata_columns)
    schema_metadata = {**(table.schema.metadata or {}), b'porefm_output_format': str(format).encode(),
                       b'layout': b'inference_meta_data_v1',
                       b'pairing': b'data rows are contiguous in meta row order'}
    paths = []
    for kind, payload in [('data', data), ('meta', meta)]:
        path = root / f'{stem}_{kind}.parquet'
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        options = dict(compression=compression, compression_level=compression_level,
                       use_dictionary=use_dictionary, use_byte_stream_split=use_byte_stream_split)
        for key, value in options.items():
            if isinstance(value, dict):
                options[key] = {name: item for name, item in value.items() if name in payload.column_names}
            elif isinstance(value, list):
                options[key] = [name for name in value if name in payload.column_names]
        pq.write_table(payload.replace_schema_metadata(schema_metadata), temporary, **options)
        temporary.replace(path)
        paths.append(path)
    return paths[0]
