"""Select fixed-threshold labels with optional negative cohort sampling."""

import math
from numbers import Integral, Real

import pyarrow as pa

from ._filters import identifier, literal
from ._label_io import (copy_parquet, empty_labels, finish, parquet_source, report,
                       sample_indices, workspace)
from .quantiles import _strings


def select_threshold_labels(calls, *, score_column, positive_classes, negative_classes,
                            positive_threshold, negative_threshold, negative_ratio=1.0,
                            random_state=42, chroms=None, positive_label=1, negative_label=0,
                            call_status='called', threads=8, memory_limit='16GB',
                            temp_directory=None, output_path=None, progress=False):
    """Select fixed score tails, sample negatives, then deduplicate label triples.

    Parameters
    ----------
    calls : OutputDataset
        Existing ``open_output`` snapshot exposing ``readname``, ``read_pos``,
        ``predicted_class``, the score column, and any requested filter columns.
        The native DuckDB reader is used without reopening the snapshot.
    score_column : str
        Numeric score used for both cohorts. Comparisons promote it to double.
        Null and nonfinite scores are excluded. No k-mer column is required.
    positive_classes, negative_classes : str or iterable of str
        Nonempty, disjoint sets of accepted ``predicted_class`` values. All
        classes in each cohort share its threshold and output label.
    positive_threshold, negative_threshold : float
        Finite global cutoffs. Retain positive scores strictly greater than
        positive_threshold and negative scores strictly less than
        negative_threshold, excluding ties.
    negative_ratio : float or None, default 1.0
        Sample floor(negative_ratio * number_of_positive_rows) negative rows,
        without replacement. Counts are taken AFTER filtering and BEFORE
        deduplication, matching ``neg_df.sample(n=len(pos_df))`` at ratio 1.
        None keeps all eligible negatives; 0 keeps none. With a numeric ratio,
        zero positives gives zero sampled negatives. Insufficient negatives
        raise ValueError rather than silently changing the requested ratio.
    random_state : int or None, default 42
        Pandas-compatible sampling seed in [0, 2**32 - 1]; None is unseeded.
        Eligible rows are sorted by readname/read_pos/label before sampling to
        make an integer seed reproducible across DuckDB thread counts. The
        sampled identities match the previous helper's sorted pandas sample,
        including sampling duplicate rows before deduplication. They can differ
        from sampling an unsorted Arrow iterator with the same seed.
    chroms : str or iterable of str, optional
        Chromosome filter applied before score selection and sampling. None
        includes all chromosomes; an empty iterable selects no rows.
    positive_label, negative_label : int, default 1, 0
        Distinct, nonnegative int64 class IDs in the returned labels.
    call_status : str or None, default 'called'
        Required status, or None to disable the filter and column requirement.
    threads : int, default 8
        Positive number of DuckDB worker threads.
    memory_limit : str or None, default '16GB'
        DuckDB memory setting, not a limit on pandas or total process RAM.
    temp_directory : path-like, optional
        Parent for a unique work directory containing eligible-label Parquets,
        a disk-backed sampling permutation, and DuckDB spill files. None uses
        the system temporary directory rather than the working directory's
        .tmp. Use disk-backed local scratch with enough free space. Cleanup
        occurs after completion or failure.
    output_path : path-like, optional
        Write sorted, deduplicated Parquet instead of materializing pandas.
        Must be a new file outside input-part directories; publication is atomic.
    progress : bool, default False
        Print the work directory, processed parts, counts, and sampling stages.

    Returns
    -------
    pandas.DataFrame or pathlib.Path
        Sorted ``readname``, ``read_pos``, ``label`` columns with duplicate
        triples removed. Read coordinates retain their oriented, zero-based
        meaning. Deduplication can reduce the final negative-to-positive ratio.
        Conflicting labels at the same read/position remain separate rows.
        With output_path, return the absolute path instead of a DataFrame.

    Notes
    -----
    Each source part is filtered independently without loading sequences or
    k-mers. Only qualifying label triples are written to temporary Parquet.
    Sampling reproduces NumPy's legacy permutation using a disk-backed integer
    array (8 bytes per eligible negative on a 64-bit system); mapped-page
    residency is managed by the OS, not the DuckDB memory limit. Global sorting
    and deduplication can spill to disk. Only final labels enter pandas, which
    must still fit RAM unless output_path is supplied. No tag dict is saved.
    """
    positive_classes = _strings(positive_classes, 'positive_classes')
    negative_classes = _strings(negative_classes, 'negative_classes')
    if set(positive_classes) & set(negative_classes):
        raise ValueError('positive_classes and negative_classes must be disjoint')
    for name, value in [('positive_threshold', positive_threshold),
                        ('negative_threshold', negative_threshold)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if negative_ratio is not None and (isinstance(negative_ratio, bool)
            or not isinstance(negative_ratio, Real) or not math.isfinite(negative_ratio)
            or negative_ratio < 0):
        raise ValueError('negative_ratio must be a finite nonnegative number or None')
    if random_state is not None and (isinstance(random_state, bool)
            or not isinstance(random_state, Integral) or not 0 <= random_state < 2**32):
        raise ValueError('random_state must be an integer in [0, 2**32 - 1] or None')
    for name, value in [('positive_label', positive_label), ('negative_label', negative_label)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**63 - 1:
            raise ValueError(f'{name} must be a nonnegative int64')
    if positive_label == negative_label:
        raise ValueError('positive_label and negative_label must differ')
    if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
        raise ValueError('threads must be a positive integer')
    if call_status is not None and (not isinstance(call_status, str) or not call_status):
        raise ValueError('call_status must be a nonempty string or None')
    if not isinstance(score_column, str) or score_column not in calls.schema.names:
        raise ValueError(f'Unknown score_column: {score_column!r}')
    dtype = calls.schema.field(score_column).type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)):
        raise ValueError(f'score_column must be numeric: {score_column!r}')

    filters = [('predicted_class', 'in', positive_classes + negative_classes)]
    if call_status is not None:
        filters.append(('call_status', '==', call_status))
    if chroms is not None:
        filters.append(('chrom', 'in', _strings(chroms, 'chroms', allow_empty=True)))
    columns = list(dict.fromkeys(['readname', 'read_pos', 'predicted_class', score_column]))
    score = f'CAST({identifier(score_column)} AS DOUBLE)'
    positive = f'predicted_class IN ({", ".join(map(literal, positive_classes))})'
    with workspace(calls, temp_directory, output_path, memory_limit, threads, progress) as (con, root, target):
        paths = []
        for index, part in enumerate(calls.parts):
            calls._subset([part]).register(con, name='calls', columns=columns, filters=filters)
            # Keep duplicates: the requested ratio is based on pre-dedup counts.
            query = f'''
            SELECT readname, read_pos,
                CAST(CASE WHEN {positive} THEN {literal(positive_label)}
                     ELSE {literal(negative_label)} END AS BIGINT) AS label
            FROM calls
            WHERE isfinite({score})
              AND (({positive} AND {score} > {literal(positive_threshold)})
                   OR (NOT ({positive}) AND {score} < {literal(negative_threshold)}))
            '''
            path = root / f'eligible_{index:06d}.parquet'
            copy_parquet(con, query, path)
            paths.append(path)
            report(progress, f'Threshold filtering: {index + 1}/{len(calls.parts)} parts')
        source = ('SELECT * FROM ' + parquet_source(paths) if paths else empty_labels(con, calls))
        con.execute(f'CREATE TEMP VIEW eligible AS {source}')
        n_positive, n_negative = con.execute(f'''
            SELECT count(*) FILTER (WHERE label = {literal(positive_label)}),
                   count(*) FILTER (WHERE label = {literal(negative_label)}) FROM eligible
        ''').fetchone()
        report(progress, f'Eligible rows: {n_positive:,} positive, {n_negative:,} negative')
        count = n_negative
        if negative_ratio is not None:
            requested = negative_ratio * n_positive
            if not math.isfinite(requested):
                raise ValueError('negative_ratio requests too many negative rows')
            count = math.floor(requested)
            if count > n_negative:
                raise ValueError(f'Requested {count} negative rows for {n_positive} positive rows '
                                 f'(negative_ratio={negative_ratio}), but only {n_negative} qualify. '
                                 'Reduce negative_ratio or use None to keep all eligible negatives.')

        selected = 'SELECT * FROM eligible'
        if count == 0:
            selected += f' WHERE label = {literal(positive_label)}'
        elif count < n_negative:
            report(progress, f'Sampling exactly {count:,}/{n_negative:,} negatives on disk')
            indices = root / 'sample_indices.parquet'
            sample_indices(indices, n_negative, count, random_state)
            # Ranking matches the previous helper's SQL order before pandas
            # sampling. Identical tied triples are interchangeable after dedup.
            ranked = root / 'ranked_negatives.parquet'
            copy_parquet(con, f'''
                SELECT *, row_number() OVER (ORDER BY readname, read_pos) - 1 AS sample_index
                FROM eligible WHERE label = {literal(negative_label)}
            ''', ranked)
            selected = f'''
                SELECT readname, read_pos, label FROM eligible WHERE label = {literal(positive_label)}
                UNION ALL
                SELECT n.readname, n.read_pos, n.label FROM {parquet_source([ranked])} n
                SEMI JOIN {parquet_source([indices])} s USING (sample_index)
            '''
        return finish(con, selected, target, progress)
