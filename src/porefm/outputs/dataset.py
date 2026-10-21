"""Format discovery, bounded Arrow reads and reusable native DuckDB views."""

from dataclasses import dataclass
from copy import copy
from glob import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._filters import filter_table, identifier, literal, predicates, sql_filter
from ._batching import batching, pack_reads, split_reads
from ._columns import without_probabilities


ALIASES = {'refname': 'chrom', 'refpos': 'position', 'read_position': 'read_pos',
           'p_val': 'pvalue', 'p_value': 'pvalue'}
FORMATS = ('canonical_null', 'pairwise', 'prototypes', 'supervised')


@dataclass(frozen=True)
class Part:
    """One flat file or contiguous metadata/data pair."""
    data: Path
    meta: Path | None = None


def _discover(path, allow_incomplete):
    """Inspect one run, never recursively mix read scores and site statistics."""
    root = Path(path) if isinstance(path, (str, Path)) else None
    if root is not None and root.is_dir():
        if (root / 'read_scores').is_dir():
            root = root / 'read_scores'
        paths = sorted(root.glob('*.parquet'))
    else:
        inputs = [path] if root is not None else list(path)
        paths = sorted({Path(p) for value in inputs for p in glob(str(value))})
    if not paths:
        raise FileNotFoundError(f'No Parquet files found: {path}')
    split = any(p.name.endswith(('_meta.parquet', '_data.parquet')) for p in paths)
    if not split:
        return [Part(p.resolve()) for p in paths]
    if any(not p.name.endswith(('_meta.parquet', '_data.parquet')) for p in paths):
        raise ValueError('Mixed flat and split files; select one output run/layout explicitly')
    stems = sorted({str(p).removesuffix('_meta.parquet').removesuffix('_data.parquet') for p in paths})
    parts = []
    for stem in stems:
        data, meta = Path(stem + '_data.parquet'), Path(stem + '_meta.parquet')
        if not data.is_file() or not meta.is_file():
            if allow_incomplete:
                continue
            raise FileNotFoundError(f'Incomplete split pair: {stem}; use allow_incomplete=True for a snapshot')
        parts.append(Part(data.resolve(), meta.resolve()))
    if not parts:
        raise FileNotFoundError('No complete metadata/data pairs yet')
    return parts


def _lengths(part):
    values = pq.read_table(part.meta, columns=['length'])['length']
    if values.null_count or not pa.types.is_integer(values.type):
        raise ValueError(f'Metadata lengths must be nonnegative integers: {part.meta}')
    lengths = values.to_numpy()
    if (lengths < 0).any() or sum(map(int, lengths)) != pq.read_metadata(part.data).num_rows:
        raise ValueError(f'Metadata lengths do not match data rows: {part.data}')
    return lengths


def _detect(names):
    if 'predicted_class' in names or 'predicted_class_index' in names:
        return 'prototypes'
    if any(name.startswith('prob_') for name in names):
        return 'supervised'
    if 'distance' in names and 'kmer' in names:
        return 'canonical_null'
    if set(names) & {'pvalue', 'mean_squared_distance', 'cosine_distance'}:
        return 'pairwise'
    raise ValueError('Cannot detect output format; pass format= explicitly')


class OutputDataset:
    """A snapshot of one PoreFM output run with a common logical column schema.

    Construct using :func:`open_output`. Common names are ``readname``, ``chrom``,
    ``position`` (zero-based reference), ``read_pos`` (producer-oriented read),
    ``strand`` (0 forward, 1 reverse), and ``pvalue`` where available. Other score
    columns retain their producer names/precision. Negative positions remain in
    read-level results but are always excluded by ``summarize_loci``.

    Read methods accept ``columns`` and ``filters``. Filters are an AND mapping
    of column to scalar (equality) or list (membership), or a list of
    ``(column, operator, value)`` triples. Operators: ==, !=, <, <=, >, >=, in,
    not in. Filters may use columns absent from the projection. No deduplication
    is applied; input files must belong to one run. Open again to see new parts.
    """

    def __init__(self, path, *, format='auto', allow_incomplete=False):
        if format != 'auto' and format not in FORMATS:
            raise ValueError(f'format must be auto or one of {FORMATS}')
        self.path = path
        self.allow_incomplete = allow_incomplete
        root = Path(path) if isinstance(path, (str, Path)) else None
        effect_part = None
        if (root is not None and root.is_file() and root.name.endswith(('_data.parquet', '_meta.parquet'))
                and (root.parent / 'effect_parts.json').is_file()):
            effect_part = root.name
            root = root.parent
            self.path = root
        self._effect = root is not None and (root / 'effect_parts.json').is_file()
        self._derived = set()
        self._prob_columns = []
        if self._effect:
            from ._effects import _effect_snapshot, _expanded_schema
            self.format = 'prototypes'
            self._manifest, records = _effect_snapshot(root, allow_incomplete)
            if effect_part is not None:
                records = [p for p in records if effect_part in (p['data'], p['meta'])]
                if not records:
                    raise ValueError(f'Part is not in the published manifest snapshot: {effect_part}')
            self._records = records
            self.parts = [Part(root / p['data'], root / p['meta']) for p in records]
            self.schema = _expanded_schema(self._manifest, None)
            self.class_labels = tuple(self._manifest['class_labels'])
        else:
            self.parts = _discover(path, allow_incomplete)
            if any(b'porefm_effects' in (pq.read_schema(p.data).metadata or {}) for p in self.parts):
                raise ValueError('Family-local prototype parts require their effect_parts.json; '
                                 'open the run directory or a single published part with its manifest')
            self._origins, schema = self._part_schema(self.parts[0])
            self.format = _detect(schema.names) if format == 'auto' else format
            if self.format == 'prototypes':
                kept = without_probabilities(schema.names)
                schema = pa.schema([schema.field(n) for n in kept])
                self._origins = {n: self._origins[n] for n in kept}
            for part in self.parts:
                origins, other = self._part_schema(part)
                if self.format == 'prototypes':
                    kept = without_probabilities(other.names)
                    other = pa.schema([other.field(n) for n in kept])
                    origins = {n: origins[n] for n in kept}
                if origins != self._origins or not schema.equals(other, check_metadata=True):
                    raise ValueError(f'Inconsistent schemas within output run: {part.data}')
                if part.meta:
                    _lengths(part)
            if self.format == 'supervised':
                prefix = 'prob_'
                self._prob_columns = [name for name in schema.names if name.startswith(prefix)]
                if not self._prob_columns:
                    raise ValueError('No probability columns found')
                self.class_labels = tuple(name[len(prefix):] for name in self._prob_columns)
                for name, dtype in [('predicted_class', pa.string()), ('probability', pa.float64()),
                                    ('call_status', pa.string())]:
                    if name not in schema.names:
                        schema = schema.append(pa.field(name, dtype))
                        self._derived.add(name)
            else:
                self.class_labels = tuple(n[len('score_'):] for n in schema.names
                                          if n.startswith('score_') and n != 'score_semantics')
                if self.format == 'prototypes' and 'call_status' not in schema.names:
                    schema = schema.append(pa.field('call_status', pa.string()))
                    self._derived.add('call_status')
            self.schema = schema
        if format != 'auto' and format != self.format:
            raise ValueError(f'Expected {format}, found {self.format}')

    @staticmethod
    def _part_schema(part):
        origins, fields = {}, []
        for kind, path in [('d', part.data), ('m', part.meta)]:
            if path is None:
                continue
            for field in pq.read_schema(path):
                if kind == 'm' and field.name == 'length':
                    continue
                name = ALIASES.get(field.name, field.name)
                if name in origins:
                    raise ValueError(f'Overlapping logical column {name}: {part.data}')
                origins[name] = (kind, field.name)
                dtype = field.type.value_type if pa.types.is_dictionary(field.type) else field.type
                fields.append(pa.field(name, dtype, metadata=field.metadata))
        return origins, pa.schema(fields)

    def _selection(self, columns, filters):
        columns = list(self.schema.names if columns is None else columns)
        if self.format == 'prototypes':
            columns = without_probabilities(columns)
        if not columns or len(set(columns)) != len(columns) or set(columns) - set(self.schema.names):
            raise ValueError(f'columns must be a nonempty unique subset of {self.schema.names}')
        items = predicates(filters, self.schema.names)
        needed = set(columns) | {name for name, _, _ in items}
        return columns, items, [name for name in self.schema.names if name in needed]

    def _subset(self, parts):
        """Keep a subset of this fixed snapshot without rediscovering live files."""
        result = copy(self)
        result.parts = list(parts)
        if self._effect:
            names = {part.data.name for part in result.parts}
            result._records = [r for r in self._records if r['data'] in names]
        return result

    def _prune_metadata(self, items):
        """Exclude whole split parts using metadata-only AND predicates.

        Read only predicate columns plus lengths. Never remove metadata rows
        from a surviving part: its original boundaries must remain intact for
        ASOF reconstruction. No scored data columns are scanned in this pass.
        """
        if self._effect:
            origins = {name: name for name in ('readname', 'chrom', 'strand', 'inference_converged')}
        else:
            origins = {name: raw for name, (kind, raw) in self._origins.items() if kind == 'm'}
        selected = [item for item in items if item[0] in origins]
        if not selected:
            return self
        names = list(dict.fromkeys(item[0] for item in selected))
        parts = []
        for part in self.parts:
            meta = pq.read_table(part.meta, columns=['length'] + [origins[n] for n in names])
            table = pa.table({n: pc.cast(meta[origins[n]], self.schema.field(n).type) for n in names})
            table = table.append_column('__length', meta['length'])
            matching = filter_table(table, selected)
            if len(matching) and pc.any(pc.greater(matching['__length'], 0)).as_py():
                parts.append(part)
        return self._subset(parts)

    def _derive(self, table, needed):
        if not (self._derived & set(needed)):
            return table
        if self.format == 'supervised':
            values = np.column_stack([table[name].to_numpy() for name in self._prob_columns])
            valid = np.all(np.isfinite(values) & (values >= 0) & (values <= 1), axis=1)
            chosen = np.argmax(np.where(np.isfinite(values), values, -np.inf), axis=1)
            derived = {
                'predicted_class': pc.take(pa.array(self.class_labels), pa.array(chosen, mask=~valid)),
                'probability': pa.array(values[np.arange(len(values)), chosen], mask=~valid, type=pa.float64()),
                'call_status': pc.if_else(pa.array(valid), 'called', 'unavailable_context'),
            }
        else:
            valid = pc.is_valid(table['predicted_class'])
            derived = {'call_status': pc.if_else(valid, 'called', 'unavailable_context')}
        for name in self._derived & set(needed):
            table = table.append_column(name, derived[name])
        return table

    def iter_tables(self, *, columns=None, filters=None, batch_size=None,
                    reads_per_batch=None, decode_context=True):
        """Yield filtered Arrow tables, preserving complete stored reads by default.

        Canonical-null, prototype and supervised output default to 32 nonempty
        reads per batch. ``reads_per_batch`` sets this count; row counts and RAM
        vary with read length. Split metadata defines records; historical flat
        files use contiguous readname/chrom/strand groups within each part.
        Duplicate names/alignments are not globally combined. Filters retain all
        matching rows of each read together, not all original biological bases.
        Explicit ``batch_size`` instead selects ordinary row batching and is
        mutually exclusive with ``reads_per_batch``. Pairwise defaults to 65536
        rows and supports only row batching because reads span genomic blocks.
        Metadata filters first skip wholly unmatched split parts. Surviving
        mixed parts retain row filtering; general Parquet predicate pushdown
        is not promised by the Arrow iterator.
        Identity reconstruction always uses original physical offsets. Packed
        canonical-null context is decoded to text when selected unless
        ``decode_context=False``. Empty selections yield one typed empty table.
        Iteration preserves file/physical row order. Close/discard the iterator
        to stop early; no background Python callback workers are started.
        """
        batch_size, reads_per_batch = batching(self.format, batch_size, reads_per_batch)
        columns, items, needed = self._selection(columns, filters)
        selected = self._prune_metadata(items)
        yield from selected._iter_selected_tables(columns=columns, items=items, needed=needed,
            batch_size=batch_size, reads_per_batch=reads_per_batch, decode_context=decode_context)

    def _iter_selected_tables(self, *, columns, items, needed, batch_size, reads_per_batch, decode_context):
        """Read already-pruned parts, retaining original offsets in mixed parts."""
        if reads_per_batch and not self._effect and any(p.meta is None for p in self.parts):
            needed = list(dict.fromkeys(needed + [n for n in ('readname', 'chrom', 'strand')
                                                  if n in self.schema.names]))
        schema = pa.schema([self.schema.field(name) for name in columns])
        emitted = False
        if self._effect:
            from ._effects import _iter_effect_rows
            # Use the fixed manifest snapshot, including for an active run.
            def effect_tables():
                for record in self._records:
                    yield from _iter_effect_rows(self.path, columns=needed, batch_size=batch_size,
                                                 _snapshot=(self._manifest, [record]),
                                                 whole_reads=bool(reads_per_batch), filters=items)
            tables = effect_tables()
        else:
            tables = self._iter_physical(needed, batch_size, whole_reads=bool(reads_per_batch))
        # Text context predicates need decoded values even when context is not
        # projected. Otherwise defer context decoding until rows are selected.
        decode_for_filter = decode_context and any(name == 'sequence_context' for name, _, _ in items)
        tables = (filter_table(self._decode_context(table, decode_for_filter), items).select(columns)
                  for table in tables)
        if reads_per_batch:
            tables = pack_reads(tables, reads_per_batch)
        for table in tables:
            table = self._decode_context(table, decode_context)
            schema = table.schema
            if len(table):
                emitted = True
                yield table
        if not emitted:
            yield self._decode_context(schema.empty_table(), decode_context)

    @staticmethod
    def _decode_context(table, decode):
        if not decode or 'sequence_context' not in table.column_names:
            return table
        field = table.schema.field('sequence_context')
        if not pa.types.is_binary(field.type):
            return table
        metadata = field.metadata or {}
        if metadata.get(b'encoding') != b'dna2bit-nmask-v1' or b'length' not in metadata:
            raise ValueError('Unknown packed sequence_context encoding/length')
        from porefm.workflows.canonical_null.sequence_context import decode_sequence_context
        values = decode_sequence_context(table['sequence_context'].to_pylist(), int(metadata[b'length']))
        metadata = {**metadata, b'encoding': b'utf8'}
        return table.set_column(table.schema.get_field_index('sequence_context'),
                                pa.field('sequence_context', pa.string(), metadata=metadata), pa.array(values, type=pa.string()))

    def _iter_physical(self, needed, batch_size, whole_reads=False):
        physical = set(needed) - self._derived
        if self._derived & set(needed):
            physical.update(self._prob_columns or ['predicted_class'])
        for part in self.parts:
            stream = self._iter_part(part, needed, physical, batch_size)
            yield from split_reads(stream, _lengths(part) if part.meta else None) if whole_reads else stream

    def _iter_part(self, part, needed, physical, batch_size):
        """Reconstruct one part in physical row order before any filtering."""
        data_names = [raw for name, (kind, raw) in self._origins.items() if name in physical and kind == 'd']
        meta_names = [raw for name, (kind, raw) in self._origins.items() if name in physical and kind == 'm']
        meta = pq.read_table(part.meta, columns=['length'] + meta_names) if part.meta else None
        ends = np.cumsum(meta['length'].to_numpy()) if meta is not None else None
        offset = 0
        for batch in pq.ParquetFile(part.data).iter_batches(batch_size=batch_size, columns=data_names):
            data = pa.Table.from_batches([batch])
            indices = pa.array(np.searchsorted(ends, np.arange(offset, offset + len(batch)), side='right')) if meta is not None else None
            offset += len(batch)
            arrays = {}
            for name, (kind, raw) in self._origins.items():
                if name in physical:
                    array = data[raw] if kind == 'd' else pc.take(meta[raw], indices)
                    arrays[name] = pc.cast(array, self.schema.field(name).type)
            fields = [self.schema.field(name) for name in arrays]
            table = pa.table(arrays, schema=pa.schema(fields))
            yield self._derive(table, needed).select(needed)

    def read_table(self, **options):
        """Load selected rows into one chunked Arrow table; all rows must fit RAM.

        Accepts ``iter_tables`` options. Arrow concatenation shares buffers;
        convert to pandas once, optionally with ``types_mapper=pd.ArrowDtype``.
        """
        return pa.concat_tables(list(self.iter_tables(**options)))

    def register(self, con, *, name='calls', columns=None, filters=None):
        """Register a reusable native DuckDB view on a caller-owned connection.

        SQL projects/filters Parquet directly; no Python generator is registered.
        Metadata predicates prune wholly unmatched split files before the query
        is built. Mixed parts retain their complete metadata boundaries.
        Packed contexts remain binary in SQL. Split rows are mapped by physical
        file row number, so filters cannot shift read boundaries. Query order is
        unspecified. DuckDB controls threads, memory limits and disk spilling.
        """
        columns, items, needed = self._selection(columns, filters)
        selected = self._prune_metadata(items)
        if not selected.parts:
            # Arrow registration replaces Arrow views, but not a SQL-created
            # view from an earlier nonempty part on this same connection.
            con.execute(f'DROP VIEW IF EXISTS temp.main.{identifier(name)}')
            con.register(name, pa.schema([self.schema.field(n) for n in columns]).empty_table())
            return name
        if selected._effect:
            from .prototypes import register_effect_calls
            internal = name + '__source'
            register_effect_calls(con, self.path, name=internal, columns=needed,
                                  allow_incomplete=self.allow_incomplete,
                                  _snapshot=(selected._manifest, selected._records))
            query = f'SELECT * FROM {identifier(internal)}'
        else:
            query = selected._native_query()
        select = ', '.join(map(identifier, columns))
        con.execute(f'CREATE OR REPLACE TEMP VIEW {identifier(name)} AS '
                    f'SELECT {select} FROM ({query}) WHERE {sql_filter(items, self.schema)}')
        return name

    def _native_query(self):
        files = '[' + ', '.join(literal(str(part.data)) for part in self.parts) + ']'
        base = f'read_parquet({files}, file_row_number=true, filename=true) d'
        prefix = ''
        if self.parts[0].meta:
            files = '[' + ', '.join(literal(str(part.meta)) for part in self.parts) + ']'
            prefix = f'''WITH boundaries AS (
                SELECT *, regexp_replace(filename, '_meta[.]parquet$', '_data.parquet') AS data_filename,
                    sum(length) OVER (PARTITION BY filename ORDER BY file_row_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - length AS start_row
                FROM read_parquet({files}, file_row_number=true, filename=true)
            ) '''
            base += ' ASOF JOIN (SELECT * FROM boundaries WHERE length > 0) m ON d.filename=m.data_filename AND d.file_row_number >= m.start_row'
        select = ', '.join(f'{kind}.{identifier(raw)} AS {identifier(name)}'
                           for name, (kind, raw) in self._origins.items())
        query = prefix + f'SELECT {select} FROM {base}'
        if not self._derived:
            return query
        if self.format == 'supervised':
            cols = list(map(identifier, self._prob_columns))
            valid = ' AND '.join(f'({col} BETWEEN 0 AND 1 AND isfinite({col}))' for col in cols)
            values = '[' + ', '.join(cols) + ']'
            labels = '[' + ', '.join(map(literal, self.class_labels)) + ']'
            maximum = f'list_max({values})'
            expressions = {'predicted_class': f'CASE WHEN {valid} THEN {labels}[list_position({values}, {maximum})] END',
                           'probability': f'CASE WHEN {valid} THEN CAST({maximum} AS DOUBLE) END',
                           'call_status': f"CASE WHEN {valid} THEN 'called' ELSE 'unavailable_context' END"}
        else:
            expressions = {'call_status': "CASE WHEN predicted_class IS NOT NULL THEN 'called' ELSE 'unavailable_context' END"}
        return f'SELECT *, ' + ', '.join(f'{expressions[n]} AS {identifier(n)}' for n in sorted(self._derived)) + f' FROM ({query})'

    def summarize_loci(self, **options):
        """Aggregate with format-specific semantics; see ``summarize_loci``."""
        from .summaries import summarize_loci
        return summarize_loci(self, **options)

    def iter_site_tables(self, *, kind='anomaly_site_metrics', **options):
        """Read existing pairwise site metrics for this snapshot's read parts.

        ``kind`` is anomaly_site_metrics or locus_metrics. Only matching stems
        are selected, without scanning read rows. Columns/filters/batch_size
        follow ``iter_tables``; returned names use the shared chrom convention.
        Missing companions raise. These are producer statistics, so read filters
        passed to other methods do not recompute their coverage/covariances.
        """
        if self.format != 'pairwise' or kind not in ('anomaly_site_metrics', 'locus_metrics'):
            raise ValueError('Site companions are supported for pairwise anomaly_site_metrics/locus_metrics')
        paths = []
        for part in self.parts:
            if part.meta is None or part.data.parent.name != 'read_scores':
                raise ValueError('Site companions require a comparison run with read_scores/')
            path = part.data.parent.parent / kind / (part.data.name.removesuffix('_data.parquet') + '.parquet')
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        yield from open_output(paths, format='pairwise').iter_tables(**options)

    def read_site_table(self, **options):
        """Load existing pairwise site metrics; accepts iter_site_tables options."""
        return pa.concat_tables(list(self.iter_site_tables(**options)))


def open_output(path, *, format='auto', allow_incomplete=False):
    """Open a single run directory, Parquet file/glob, or list of part files.

    Format is auto-detected or one of canonical_null, pairwise, prototypes,
    supervised. Comparison run roots select ``read_scores/`` only; siblings
    containing locus metrics are never mixed into read rows. Split file input
    resolves its companion. Schemas and metadata lengths are checked up front.

    ``allow_incomplete`` permits snapshots of published pairs; no claim of run
    completion is made for layouts without a manifest. Never restart/overwrite
    a directory while reading it. Legacy non-atomic writers can expose an
    unfinished footer: retry opening after that pair finishes, rather than
    silently ignoring a corrupt file. Prototype manifests retain their stricter
    completed-run/stale-file checks.

    Supervised calls use stored ``prob_*`` columns for derived argmax calls;
    probability and logit columns are retained. Ties pick the first stored
    class. Any missing, nonfinite or out-of-range class probability produces
    an unavailable call.
    """
    return OutputDataset(path, format=format, allow_incomplete=allow_incomplete)
