"""Select read-position labels using exact, group-specific score quantiles."""

import math
from numbers import Integral, Real

import pyarrow as pa

from ._filters import identifier, literal
from ._label_io import copy_parquet, empty_labels, finish, parquet_source, report, workspace


def _strings(values, name, *, allow_empty=False):
    """Normalize a single string or iterable without treating strings as characters."""
    values = [values] if isinstance(values, str) else list(values)
    if (not values and not allow_empty) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f'{name} must contain nonempty strings')
    return list(dict.fromkeys(values))


def select_quantile_labels(calls, *, score_column, positive_classes, negative_classes,
                           positive_quantile=0.75, negative_quantile=0.05, chroms=None,
                           positive_label=1, negative_label=0, group_column='kmer',
                           call_status='called', threads=8, memory_limit='16GB',
                           temp_directory=None, quantile_partitions=32,
                           output_path=None, progress=False):
    """Return positive upper-tail and negative lower-tail read-position labels.

    Parameters
    ----------
    calls : OutputDataset
        An existing ``open_output`` snapshot exposing ``readname``, ``read_pos``,
        ``predicted_class``, the score/group columns, and any filter columns.
        Native DuckDB registration is used; the snapshot is not reopened.
    score_column : str
        Numeric column used for both cohorts, for example ``score_5hmC`` or
        ``prob_6mA``. Values are promoted to double before computing quantiles.
    positive_classes, negative_classes : str or iterable of str
        Nonempty, disjoint sets of accepted ``predicted_class`` values. Classes
        within each cohort are pooled before quantile calculation. Class names
        need not follow any particular modification naming convention.
    positive_quantile, negative_quantile : float, default 0.75, 0.05
        Quantiles in [0, 1], not percentages. Within each cohort and group,
        retain positive scores strictly above their quantile and negative
        scores strictly below theirs. Ties are excluded, so retained fractions
        may be smaller than 1 - positive_quantile and negative_quantile.
    chroms : str or iterable of str, optional
        Filter chromosomes BEFORE calculating thresholds. None includes all
        chromosomes; an empty iterable selects no rows.
    positive_label, negative_label : int, default 1, 0
        Distinct, nonnegative int64 output class IDs. For a C/5mC/5hmC label
        dictionary, use positive_label=2 when selecting 5hmC examples.
    group_column : str, default 'kmer'
        Column defining independent quantile groups within each cohort.
        Rows with null group values are excluded.
    call_status : str or None, default 'called'
        Filter status before calculating thresholds. None disables this filter
        and does not require a ``call_status`` column.
    threads : int, default 8
        Number of DuckDB worker threads; no Python worker processes are started.
    memory_limit : str or None, default '16GB'
        DuckDB memory setting. Exact quantiles and the final pandas result still
        require memory; this is not a hard limit on total process RAM.
    temp_directory : path-like, optional
        Parent for a unique work directory containing score partitions, labels,
        and DuckDB spill files. None uses the system temporary directory, not
        the working directory's .tmp. Use disk-backed local scratch with enough
        space; the directory is cleaned after completion or failure.
    quantile_partitions : int, default 32
        Hash partitions of (group, cohort) processed separately for exact
        quantiles. More partitions reduce aggregate working memory at the cost
        of more temporary files. A single large group still needs its exact
        aggregate state; this is not a hard process-memory limit.
    output_path : path-like, optional
        Write sorted, deduplicated Parquet instead of materializing pandas.
        Must be a new file outside input-part directories. Publication is atomic.
    progress : bool, default False
        Print the work directory and progress through both passes and quantiles.

    Returns
    -------
    pandas.DataFrame or pathlib.Path
        Columns ``readname``, ``read_pos``, ``label``, sorted by those columns
        with duplicate triples removed. Coordinates retain the reader's
        oriented, zero-based read positions. Conflicting labels at the same
        read/position are retained, matching triple-based deduplication. With
        output_path, return the absolute path instead of a DataFrame.

    Notes
    -----
    Nonfinite and null scores are excluded. Thresholds use exact interpolated
    quantiles over all eligible rows, including duplicates; deduplication occurs
    only after selection. The first source pass reconstructs one part at a time
    and writes only group/cohort/score to disk. Each score partition then pools
    ALL parts for exact quantiles. The second pass applies those thresholds one
    part at a time and writes only selected label triples. Global deduplication
    and sorting can spill to disk. The final DataFrame must still fit RAM unless
    output_path is supplied. Input files must remain unchanged during both
    passes. This function neither builds nor saves tag dicts.
    """
    positive_classes = _strings(positive_classes, 'positive_classes')
    negative_classes = _strings(negative_classes, 'negative_classes')
    if set(positive_classes) & set(negative_classes):
        raise ValueError('positive_classes and negative_classes must be disjoint')
    for name, value in [('positive_quantile', positive_quantile),
                        ('negative_quantile', negative_quantile)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite number in [0, 1]')
    for name, value in [('positive_label', positive_label), ('negative_label', negative_label)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**63 - 1:
            raise ValueError(f'{name} must be a nonnegative int64')
    if positive_label == negative_label:
        raise ValueError('positive_label and negative_label must differ')
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError('threads must be a positive integer')
    if (isinstance(quantile_partitions, bool) or not isinstance(quantile_partitions, Integral)
            or quantile_partitions < 1):
        raise ValueError('quantile_partitions must be a positive integer')
    if call_status is not None and (not isinstance(call_status, str) or not call_status):
        raise ValueError('call_status must be a nonempty string or None')
    for name, value in [('score_column', score_column), ('group_column', group_column)]:
        if not isinstance(value, str) or value not in calls.schema.names:
            raise ValueError(f'Unknown {name}: {value!r}')
    dtype = calls.schema.field(score_column).type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)):
        raise ValueError(f'score_column must be numeric: {score_column!r}')

    filters = [('predicted_class', 'in', positive_classes + negative_classes)]
    if call_status is not None:
        filters.append(('call_status', '==', call_status))
    if chroms is not None:
        filters.append(('chrom', 'in', _strings(chroms, 'chroms', allow_empty=True)))
    score, group = identifier(score_column), identifier(group_column)
    positives = ', '.join(map(literal, positive_classes))
    score_columns = list(dict.fromkeys(['predicted_class', group_column, score_column]))
    label_columns = list(dict.fromkeys(['readname', 'read_pos'] + score_columns))
    candidate_query = f'''
            SELECT {group} AS quantile_group,
                CAST({score} AS DOUBLE) AS value,
                predicted_class IN ({positives}) AS is_positive
            FROM calls
            WHERE {group} IS NOT NULL AND isfinite(CAST({score} AS DOUBLE))
    '''
    with workspace(calls, temp_directory, output_path, memory_limit, threads, progress) as (con, root, target):
        buckets = {}
        for index, part in enumerate(calls.parts):
            calls._subset([part]).register(con, name='calls', columns=score_columns, filters=filters)
            directory = root / f'scores_{index:06d}'
            con.execute(f'''
                COPY (SELECT *, hash(quantile_group, is_positive) % {int(quantile_partitions)} AS bucket
                      FROM ({candidate_query})) TO {literal(str(directory))}
                (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(bucket), ROW_GROUP_SIZE 65536)
            ''')
            for path in sorted(directory.glob('bucket=*/*.parquet')):
                bucket = int(path.parent.name.split('=', 1)[1])
                buckets.setdefault(bucket, []).append(path)
            report(progress, f'Quantile pass 1: {index + 1}/{len(calls.parts)} parts')

        schema = pa.schema([pa.field('quantile_group', calls.schema.field(group_column).type),
                            pa.field('is_positive', pa.bool_()), pa.field('cutoff', pa.float64())])
        con.register('threshold_schema', schema.empty_table())
        con.execute('CREATE TEMP TABLE thresholds AS SELECT * FROM threshold_schema')
        # Hashing keeps every group's full population together. Exact aggregate
        # states are released between partitions rather than coexisting globally.
        for index, (_, paths) in enumerate(sorted(buckets.items())):
            con.execute(f'''
            INSERT INTO thresholds
            SELECT quantile_group, is_positive,
                CASE WHEN is_positive THEN
                    quantile_cont(value, {literal(positive_quantile)}) FILTER (WHERE is_positive)
                ELSE
                    quantile_cont(value, {literal(negative_quantile)}) FILTER (WHERE NOT is_positive)
                END AS cutoff
            FROM {parquet_source(paths)}
            GROUP BY quantile_group, is_positive
            ''')
            report(progress, f'Exact quantiles: {index + 1}/{len(buckets)} partitions')

        selected_paths = []
        for index, part in enumerate(calls.parts):
            calls._subset([part]).register(con, name='calls', columns=label_columns, filters=filters)
            labels = f'''
            SELECT c.readname, c.read_pos,
                CAST(CASE WHEN c.is_positive THEN {literal(positive_label)}
                     ELSE {literal(negative_label)} END AS BIGINT) AS label
            FROM (SELECT readname, read_pos, {group} AS quantile_group,
                         CAST({score} AS DOUBLE) AS value,
                         predicted_class IN ({positives}) AS is_positive
                  FROM calls WHERE {group} IS NOT NULL AND isfinite(CAST({score} AS DOUBLE))) AS c
            JOIN thresholds AS t USING (quantile_group, is_positive)
            WHERE (c.is_positive AND c.value > t.cutoff)
               OR (NOT c.is_positive AND c.value < t.cutoff)
            '''
            path = root / f'labels_{index:06d}.parquet'
            copy_parquet(con, labels, path)
            selected_paths.append(path)
            report(progress, f'Quantile pass 2: {index + 1}/{len(calls.parts)} parts')
        query = ('SELECT * FROM ' + parquet_source(selected_paths) if selected_paths
                 else empty_labels(con, calls))
        return finish(con, query, target, progress)
