"""Native DuckDB access and genomic-site summaries for prototype calls."""

from pathlib import Path
from glob import glob
import uuid

import pyarrow.parquet as pq

from ._columns import without_probabilities
from ._effects import _effect_snapshot, _expanded_schema


CLASSES = ('A', '6mA', 'C', '5mC', '5hmC')


def _literal(value):
    """Quote a SQL string literal, including paths and arbitrary class labels."""
    return "'" + str(value).replace("'", "''") + "'"


def _identifier(value):
    """Quote a SQL identifier independently of string-literal escaping."""
    return '"' + str(value).replace('"', '""') + '"'


def _sql_list(values):
    return '[' + ', '.join(_literal(value) for value in values) + ']'


def register_effect_calls(con, path, *, name='calls', columns=None, call_status=None,
                          predicted_classes=None, allow_incomplete=False, file_pattern='part*.parquet', _snapshot=None):
    """Create a reusable DuckDB view of read-level calls without pandas expansion.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Caller-owned connection; its memory limit, spill directory and threads
        govern queries. DuckDB is imported only by the aggregation helper.
    path : path-like or sequence of path-like
        Split-output directory, historical flat Parquet directory/file/glob,
        or list of historical flat files. Split input requires its manifest.
    name : str
        Temporary view name, replaced if already present on this connection.
    columns : sequence of str or None
        Logical call columns; None exposes the standard reconstructed fields.
    call_status : {'called', 'unavailable_context'} or None
        Optional callability filter. Independent of projected output columns.
    predicted_classes : str or sequence of str or None
        Keep any listed prediction, combined with call_status using AND.
        An empty sequence keeps no rows. Split labels are validated.
    allow_incomplete : bool
        Snapshot finished pairs while inference is running. Re-register to
        include later parts; do not overwrite/restart the source during queries.
    file_pattern : str
        Glob for historical flat directories; ignored for manifest-based input.

    Returns
    -------
    str
        Registered view name. It can be queried repeatedly on ``con``.

    Notes
    -----
    Native Parquet ``file_row_number`` preserves original positions after SQL
    filtering. An ASOF join maps each data row to the preceding metadata read
    boundary within its part, avoiding a many-to-many interval join. Metadata
    lengths are validated before registration. Genomic and read coordinates,
    class indices, nulls and family-local score ordering match the Arrow reader.
    """
    if call_status not in (None, 'called', 'unavailable_context'):
        raise ValueError("call_status must be 'called', 'unavailable_context', or None")
    if predicted_classes is not None:
        predicted_classes = ([predicted_classes] if isinstance(predicted_classes, str)
                             else list(predicted_classes))
        if any(not isinstance(label, str) for label in predicted_classes):
            raise TypeError('predicted_classes must contain class-label strings')
    root = None if isinstance(path, (list, tuple)) else Path(path)
    split = root is not None and root.is_dir() and (root / 'effect_parts.json').exists()
    conditions = []
    if not split:
        if root is not None and root.is_dir() and any(root.glob('part_*_data.parquet')):
            raise FileNotFoundError(f'split output requires effect_parts.json: {root}')
        files = ([str(p) for p in path] if root is None else
                 [str(root / file_pattern) if root.is_dir() else str(root)])
        if not files:
            raise ValueError('No parquet files supplied.')
        if call_status is not None:
            conditions.append(f'call_status = {_literal(call_status)}')
        if predicted_classes is not None:
            conditions.append(f'predicted_class IN ({", ".join(map(_literal, predicted_classes))})'
                              if predicted_classes else 'false')
        available = con.execute(f'DESCRIBE SELECT * FROM read_parquet({_sql_list(files)})').fetchall()
        selected = without_probabilities([row[0] for row in available] if columns is None else columns)
        if not selected:
            raise ValueError('Select at least one non-probability column')
        selection = ', '.join(map(_identifier, selected))
        query = f'SELECT {selection} FROM read_parquet({_sql_list(files)})'
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
    else:
        root = root.resolve()
        manifest, parts = _snapshot if _snapshot is not None else _effect_snapshot(root, allow_incomplete)
        labels = manifest['class_labels']
        schema = _expanded_schema(manifest, columns)
        if predicted_classes is not None:
            unknown = set(predicted_classes) - set(labels)
            if unknown:
                raise ValueError(f'unknown predicted classes: {sorted(unknown)}; available: {labels}')
            ids = [labels.index(label) for label in predicted_classes]
            conditions.append(f'predicted_class_index IN ({", ".join(map(str, ids))})' if ids else 'false')
        if call_status is not None:
            conditions.append('predicted_class_index >= 0' if call_status == 'called' else 'predicted_class_index < 0')
        if not parts:
            # This is a zero-row Arrow table, not a Python callback stream.
            # Replace a preceding SQL-backed temporary view as well; register()
            # alone only replaces views it previously registered itself.
            con.execute(f'DROP VIEW IF EXISTS temp.main.{_identifier(name)}')
            con.register(name, schema.empty_table())
            return name
        for part in parts:
            lengths = pq.read_table(root / part['meta'], columns=['length'])['length'].to_numpy()
            rows = pq.read_metadata(root / part['data']).num_rows
            if (len(lengths) != part['reads'] or (lengths <= 0).any()
                    or int(lengths.sum()) != rows or rows != part['rows']):
                raise ValueError('effect metadata lengths do not match data rows')
        data_paths = _sql_list(str(root / part['data']) for part in parts)
        meta_paths = _sql_list(str(root / part['meta']) for part in parts)
        expressions = {key: f'd.{key}' for key in (
            'read_pos', 'position', 'family_index', 'predicted_class_index',
            'isolated_class_index', 'neighbor_count', 'canonical_energy')}
        expressions.update({key: f'm.{key}' for key in ('readname', 'chrom', 'strand', 'inference_converged')})
        label_list = _sql_list(labels)
        for field, index in [('predicted_class', 'predicted_class_index'), ('isolated_class', 'isolated_class_index')]:
            expressions[field] = f'CASE WHEN d.{index} >= 0 THEN {label_list}[d.{index} + 1] END'
        half = int(manifest['kmer_size']) // 2
        start = f'greatest(0, d.read_pos - {half})'
        expressions['kmer'] = f'substring(m.sequence, {start} + 1, d.read_pos + {half} + 1 - {start})'
        expressions['call_status'] = "CASE WHEN d.predicted_class_index >= 0 THEN 'called' ELSE 'unavailable_context' END"
        expressions['score_semantics'] = _literal(manifest['score_semantics'])
        families = manifest['family_class_indices']
        for field, stored in [('score', 'scores')]:
            branches = [f'WHEN {base} THEN list_position({classes}, d.predicted_class_index)'
                        for base, classes in enumerate(families) if classes]
            local = 'CASE d.family_index ' + ' '.join(branches) + ' END'
            expressions[field] = f'CASE WHEN d.predicted_class_index >= 0 THEN d.{stored}[{local}] END'
            for index, label in enumerate(labels):
                branches = [f'WHEN {base} THEN d.{stored}[{classes.index(index) + 1}]'
                            for base, classes in enumerate(families) if index in classes]
                expressions[f'{field}_{label}'] = ('CASE d.family_index ' + ' '.join(branches) + ' END'
                                                   if branches else 'CAST(NULL AS FLOAT)')
        select = ', '.join(f'{expressions[field]} AS {_identifier(field)}' for field in schema.names)
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        query = f'''
            WITH d AS (
                SELECT * FROM read_parquet({data_paths}, filename=true, file_row_number=true){where}
            ), m AS (
                SELECT *,
                    regexp_replace(filename, '_meta[.]parquet$', '_data.parquet') AS data_filename,
                    sum(length) OVER (PARTITION BY filename ORDER BY file_row_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - length AS start_row
                FROM read_parquet({meta_paths}, filename=true, file_row_number=true)
            )
            SELECT {select} FROM d ASOF JOIN m
                ON d.filename = m.data_filename AND d.file_row_number >= m.start_row
        '''
    con.execute(f'CREATE OR REPLACE TEMP VIEW {_identifier(name)} AS {query}')
    return name


def make_meth_df_from_predictions(path, classes=CLASSES, group_by_kmer=False, output_path=None,
                                   file_pattern='part*.parquet', memory_limit=None,
                                   temp_directory=None, threads=None, *, call_status=None,
                                   predicted_classes=None, allow_incomplete=False, overwrite=False):
    """Collapse read-level calls to genomic sites using native DuckDB scans.

    ``path``, ``file_pattern`` and reader filters follow
    :func:`register_effect_calls`. ``classes`` chooses frequency columns only;
    it does not filter reads or change the denominator. Defaults preserve the
    original DNA utility's A/6mA/C/5mC/5hmC columns. Frequencies are hard-call
    counts divided by all non-null predictions retained by the input filters,
    not mean probabilities or calibrated modification fractions.

    Group by zero-based ``(chrom, position, strand)``; negative positions
    (including -1/-2), missing chromosomes and strands outside 0/1 are always
    excluded. ``group_by_kmer=True`` adds the read k-mer to the key. Otherwise
    report its mode among retained rows, breaking ties lexicographically.
    ``total_count`` includes retained no-calls; ``called_count`` excludes them.
    A zero called count gives null frequencies. Filtering to modifications
    alone changes the denominator: include canonical classes when estimating
    modification fractions among canonical-plus-modified calls.

    ``memory_limit`` (e.g. '16GB'), ``temp_directory`` and positive ``threads``
    configure DuckDB aggregation/spilling. These bounds do not limit pandas
    output memory or every allocation in the process. Return a pandas DataFrame
    when ``output_path`` is None, otherwise write Zstd Parquet directly and
    return its Path. Existing output requires ``overwrite=True``; publication
    is atomic. Input files and manifests, including symlink aliases, are rejected
    as destinations even with overwrite=True. Row order is not
    guaranteed. An incomplete input produces a snapshot, not final coverage.
    """
    import duckdb

    classes = [classes] if isinstance(classes, str) else list(classes)
    if not classes or any(not isinstance(label, str) for label in classes) or len(set(classes)) != len(classes):
        raise ValueError('classes must be a nonempty sequence of unique class-label strings')
    if threads is not None and (int(threads) != threads or threads < 1):
        raise ValueError('threads must be a positive integer')
    target = None if output_path is None else Path(output_path)
    snapshot = None
    if target is not None:
        root = None if isinstance(path, (list, tuple)) else Path(path)
        if root is not None and root.is_dir() and (root / 'effect_parts.json').exists():
            snapshot = _effect_snapshot(root, allow_incomplete)
            inputs = [root / 'effect_parts.json'] + [root / part[key]
                      for part in snapshot[1] for key in ('data', 'meta')]
        else:
            patterns = (path if root is None else
                        [root / file_pattern if root.is_dir() else root])
            # Freeze the flat-file list so registration scans exactly the files
            # checked here, including lists/globs and custom directory patterns.
            path = [name for pattern in patterns for name in glob(str(pattern), recursive=True)]
            inputs = [Path(name) for name in path]
        if target.resolve() in {source.resolve() for source in inputs}:
            raise ValueError('Summary output must not overwrite an input part or manifest')
    if target is not None and target.exists() and not overwrite:
        raise FileExistsError(f'refusing to overwrite {target}; pass overwrite=True')
    with duckdb.connect(database=':memory:') as con:
        con.execute('SET preserve_insertion_order = false')
        if threads is not None:
            con.execute(f'SET threads = {int(threads)}')
        if memory_limit is not None:
            con.execute(f'SET memory_limit = {_literal(memory_limit)}')
        if temp_directory is not None:
            spill = Path(temp_directory)
            spill.mkdir(parents=True, exist_ok=True)
            con.execute(f'SET temp_directory = {_literal(spill)}')
        register_effect_calls(con, path, columns=['chrom', 'position', 'strand', 'kmer', 'predicted_class'],
                              call_status=call_status, predicted_classes=predicted_classes,
                              allow_incomplete=allow_incomplete, file_pattern=file_pattern,
                              _snapshot=snapshot)
        group = 'chrom, position, strand' + (', kmer' if group_by_kmer else '')
        mode = '' if group_by_kmer else ', mode(kmer ORDER BY kmer) AS kmer'
        counts = ', '.join(f'count(*) FILTER (WHERE predicted_class = {_literal(label)})::UBIGINT AS n_class_{i}'
                           for i, label in enumerate(classes))
        frequencies = ', '.join(f'n_class_{i}::DOUBLE / nullif(called_count, 0) AS {_identifier("freq_" + label)}'
                                for i, label in enumerate(classes))
        where = "chrom IS NOT NULL AND chrom != '*' AND position >= 0 AND strand IN (0, 1)"
        if group_by_kmer:
            where += ' AND kmer IS NOT NULL'
        query = f'''
            WITH collapsed AS (
                SELECT {group}{mode}, count(*)::UBIGINT AS total_count,
                    count(predicted_class)::UBIGINT AS called_count, {counts}
                FROM calls WHERE {where} GROUP BY {group}
            )
            SELECT chrom, position, strand, kmer, total_count, called_count, {frequencies}
            FROM collapsed
        '''
        if target is None:
            return con.execute(query).df()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
        con.execute(f'COPY ({query}) TO {_literal(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        temp.replace(target)
        return target
