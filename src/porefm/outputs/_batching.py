"""Read-boundary slicing shared by compact and historical flat readers."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def batching(format, batch_size, reads_per_batch):
    """Resolve mutually exclusive row/read batch controls."""
    for name, value in [('batch_size', batch_size), ('reads_per_batch', reads_per_batch)]:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if batch_size is not None and reads_per_batch is not None:
        raise ValueError('Specify only one of batch_size and reads_per_batch')
    if format == 'pairwise':
        if reads_per_batch is not None:
            raise ValueError('Pairwise files can split reads across genomic blocks; use batch_size')
        return batch_size or 65536, None
    return (batch_size, None) if batch_size is not None else (65536, reads_per_batch or 32)


def split_reads(tables, lengths=None):
    """Yield whole stored read records from one part's unfiltered table stream.

    Split parts use original metadata lengths, including zero-row records.
    Flat parts use contiguous readname/chrom/strand identities. Only an
    unfinished read is carried between physical batches; no global grouping or
    deduplication is performed. The producer must keep each record in one part.
    """
    pending = []
    if lengths is not None:
        remaining_lengths = iter(int(n) for n in lengths if n > 0)
        remaining = next(remaining_lengths, None)
        for table in tables:
            offset = 0
            while offset < len(table):
                if remaining is None:
                    raise ValueError('Data exceeds metadata lengths')
                size = min(remaining, len(table) - offset)
                pending.append(table.slice(offset, size))
                offset += size
                remaining -= size
                if remaining == 0:
                    yield pa.concat_tables(pending)
                    pending = []
                    remaining = next(remaining_lengths, None)
        if remaining is not None:
            raise ValueError('Data shorter than metadata lengths')
    else:
        last = None
        for table in tables:
            if not len(table):
                continue
            keys = [name for name in ('readname', 'chrom', 'strand') if name in table.column_names]
            if 'readname' not in keys:
                raise ValueError('Complete-read batching requires readname; use batch_size for row scans')
            boundary = np.zeros(len(table), dtype=bool)
            for name in keys:
                values = table[name]
                if pa.types.is_null(values.type):
                    continue
                same = pc.fill_null(pc.equal(values.slice(1), values.slice(0, len(table)-1)), False)
                both_null = pc.and_(pc.is_null(values.slice(1)), pc.is_null(values.slice(0, len(table)-1)))
                boundary[1:] |= ~pc.or_(same, both_null).to_numpy()
            first = tuple(table[name][0].as_py() for name in keys)
            if pending and first != last:
                yield pa.concat_tables(pending)
                pending = []
            starts = np.r_[0, np.flatnonzero(boundary), len(table)]
            for start, end in zip(starts[:-1], starts[1:]):
                if start and pending:
                    yield pa.concat_tables(pending)
                    pending = []
                pending.append(table.slice(int(start), int(end-start)))
            last = tuple(table[name][-1].as_py() for name in keys)
        if pending:
            yield pa.concat_tables(pending)


def pack_reads(tables, reads_per_batch):
    """Combine nonempty filtered read tables without splitting any record."""
    pending = []
    for table in tables:
        if len(table):
            pending.append(table)
            if len(pending) == reads_per_batch:
                yield pa.concat_tables(pending)
                pending = []
    if pending:
        yield pa.concat_tables(pending)
