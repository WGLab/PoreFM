"""Native SQL locus reductions; only collapsed results cross into Python."""

from pathlib import Path
from contextlib import ExitStack
from numbers import Real
import re
from tempfile import TemporaryDirectory
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._filters import identifier, literal, predicates


def _connection(memory_limit, temp_directory, threads):
    import duckdb

    config = {'preserve_insertion_order': False}
    if memory_limit is not None:
        config['memory_limit'] = memory_limit
    if temp_directory is not None:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        config['temp_directory'] = str(temp_directory)
    if threads is not None:
        if not isinstance(threads, int) or threads < 1:
            raise ValueError('threads must be a positive integer')
        config['threads'] = threads
    return duckdb.connect(config=config)


def _metrics(dataset, metric_columns, metric_stats):
    """Discover numeric metrics only when an aggregation was requested."""
    if not metric_stats:
        return []
    metrics = list(metric_columns) if metric_columns is not None else [
        n for n in dataset.schema.names if n in ('distance', 's_cos', 'z_cos', 's_whiten', 'z_whiten',
            'mean_squared_distance', 'cosine_distance', 'squared_mahalanobis_distance')
        or re.fullmatch(r'pc_[1-9][0-9]*', n)]
    for name in metrics:
        if name not in dataset.schema.names:
            raise ValueError(f'Unknown metric column: {name}')
        dtype = dataset.schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f'Metric must be numeric: {name}')
    if len(set(metrics)) != len(metrics):
        raise ValueError('metric_columns must be unique')
    return metrics


def _pvalue_query(dataset, cutoff, minimum, thresholds, metrics, metric_stats, combine):
    reductions = []
    functions = dict(count='count', mean='avg', std='stddev_samp', median='median')
    for name in metrics:
        col = identifier(name)
        clean = f'CASE WHEN isfinite({col}) THEN CAST({col} AS DOUBLE) END'
        for stat in metric_stats:
            reductions.append(f'{functions[stat]}({clean}) AS {identifier(name + "_" + stat)}')
    if 'pvalue' in dataset.schema.names:
        valid = 'isfinite(pvalue) AND pvalue BETWEEN 0 AND 1'
        reductions += [f'count(*) FILTER (WHERE {valid})::BIGINT AS total_count',
                       f'count(*) FILTER (WHERE {valid} AND CAST(pvalue AS DOUBLE) <= {cutoff!r})::BIGINT AS modified_count']
        if combine:
            reductions.append(f'sum(CASE WHEN {valid} THEN ln(greatest(CAST(pvalue AS DOUBLE), {minimum!r})) END) AS log_p_sum')
        for i, threshold in enumerate(thresholds):
            reductions.append(f'count(*) FILTER (WHERE {valid} AND CAST(pvalue AS DOUBLE) < {threshold!r})::BIGINT AS threshold_{i}')
    else:
        reductions.append('count(*)::BIGINT AS total_count')
    return 'SELECT chrom, position, strand, count(*)::BIGINT AS n_observations, ' + ', '.join(reductions) + " FROM calls WHERE chrom IS NOT NULL AND chrom != '*' AND position >= 0 AND strand IN (0, 1) GROUP BY chrom, position, strand"


def _finish_pvalues(table, minimum, thresholds, confidence):
    """Apply requested Fisher/Wilson transforms to site batches only."""
    if 'modified_count' not in table.column_names:
        return table
    counts = table['total_count'].to_numpy().astype(float)
    modified = table['modified_count'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = modified / counts
    table = table.append_column('fraction_modified', pa.array(fraction, from_pandas=True))
    drop = ['modified_count']
    if 'log_p_sum' in table.column_names:
        from scipy.stats import chi2
        with np.errstate(divide='ignore', invalid='ignore'):
            combined = -chi2.logsf(-2 * table['log_p_sum'].to_numpy(), 2 * counts) / np.log(10)
        table = table.append_column('combined_neglog10_pvalue', pa.array(np.clip(combined, 0, -np.log10(minimum)), from_pandas=True))
        drop.append('log_p_sum')
    table = table.append_column('n_invalid_p', pa.array(table['n_observations'].to_numpy() - counts.astype(np.int64)))
    if confidence is not None:
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)

        def interval(result, prefix, fractions):
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = 1 + z*z/counts
                center = (fractions + z*z/(2*counts)) / denominator
                radius = z*np.sqrt(fractions*(1-fractions)/counts + z*z/(4*counts**2)) / denominator
            for suffix, values in [('ci_low', np.maximum(0, center-radius)), ('ci_high', np.minimum(1, center+radius))]:
                result = result.append_column(prefix + '_' + suffix, pa.array(values, from_pandas=True))
            return result

        table = interval(table, 'fraction_modified', fraction)
    for i, threshold in enumerate(thresholds):
        count = table[f'threshold_{i}'].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction = count / counts
        prefix = 'p_lt_' + format(threshold, '.12g').replace('.', 'p').replace('-', 'm').replace('+', '')
        table = table.append_column(prefix + '_count', pa.array(count))
        table = table.append_column(prefix + '_fraction', pa.array(fraction, from_pandas=True))
        if confidence is not None:
            table = interval(table, prefix, fraction)
        drop.append(f'threshold_{i}')
    return table.drop(drop)


def _class_families(dataset, classes, class_families, available_classes):
    """Resolve full prediction families independently of selected output classes.

    Effect manifests are authoritative. Legacy files need an explicit mapping
    for modifications; canonical labels map to themselves (T and U share a
    family). Never infer a modification's parent from its spelling.
    """
    mapping = {}
    if dataset._effect:
        for base, indices in zip(dataset._manifest['family_bases'],
                                 dataset._manifest['family_class_indices']):
            for index in indices:
                mapping[dataset.class_labels[index]] = base.replace('U', 'T')
    if class_families is not None:
        for label, base in class_families.items():
            if not isinstance(label, str) or base not in ('A', 'C', 'G', 'T', 'U'):
                raise ValueError('class_families must map class labels to A/C/G/T/U')
            base = base.replace('U', 'T')
            if label in mapping and mapping[label] != base:
                raise ValueError(f'class_families conflicts with manifest for {label}')
            mapping[label] = base
    for base in 'ACGTU':
        if base in mapping and mapping[base] != base.replace('U', 'T'):
            raise ValueError(f'Invalid canonical family for {base}')
        mapping[base] = base.replace('U', 'T')
    missing = (set(classes) | set(available_classes)) - mapping.keys()
    if missing:
        raise ValueError(f'No canonical family for {sorted(missing)}; supply class_families')
    return [tuple(label for label, base in mapping.items() if base == mapping[c])
            for c in classes]


def _class_score_columns(dataset, classes):
    """Prefer the winning score; legacy outputs may store only class columns."""
    columns = ['score' if 'score' in dataset.schema.names else 'score_' + label
               for label in classes]
    for name in dict.fromkeys(columns):
        if name not in dataset.schema.names:
            raise ValueError(f'mean_scores requires score or the class score column {name!r}')
        dtype = dataset.schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f'Score must be numeric: {name}')
    return columns


def _binary_decision(dataset, threshold, positive_class):
    """Validate a summary-only binary rule and return its column and SQL.

    The existing derived call supplies validity across both probabilities.
    The explicit positive label is independent of stored/report class order.
    """
    if threshold is None:
        if positive_class is not None:
            raise ValueError('binary_positive_class requires binary_threshold')
        return None, None
    if dataset.format != 'supervised' or len(dataset.class_labels) != 2:
        raise ValueError('binary_threshold requires supervised output with exactly two classes')
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real) or not 0 <= threshold <= 1:
        raise ValueError('binary_threshold must be a finite number in [0, 1]')
    if not isinstance(positive_class, str) or positive_class not in dataset.class_labels:
        raise ValueError('binary_positive_class must name one of the two stored classes')
    index = dataset.class_labels.index(positive_class)
    column = dataset._prob_columns[index]
    negative_class = dataset.class_labels[1 - index]
    decision = (
        f'CASE WHEN predicted_class IS NULL THEN NULL '
        f'WHEN CAST({identifier(column)} AS DOUBLE) >= {float(threshold)!r} '
        f'THEN {literal(positive_class)} ELSE {literal(negative_class)} END'
    )
    return column, decision


def _classification_query(dataset, classes, group_by_kmer, include_kmer, mean_probabilities,
                          families=(), score_columns=(), binary_decision=None):
    groups = ['chrom', 'position', 'strand']
    has_kmer = 'kmer' in dataset.schema.names
    if group_by_kmer and not has_kmer:
        raise ValueError('This output has no kmer column')
    if group_by_kmer:
        groups.append('kmer')
    select = list(groups)
    if include_kmer and has_kmer and not group_by_kmer:
        select.append('mode(kmer ORDER BY kmer) AS kmer')
    select += ['count(*)::BIGINT AS total_count', 'count(predicted_class)::BIGINT AS called_count']
    for i, label in enumerate(classes):
        count = f'count(*) FILTER (WHERE predicted_class={literal(label)})'
        select.append(f'{count}::BIGINT AS class_count_{i}')
        if score_columns:
            col = identifier(score_columns[i])
            clean = f'CASE WHEN predicted_class={literal(label)} AND isfinite({col}) THEN CAST({col} AS DOUBLE) END'
            # Merge sums and valid counts, never averages of per-part averages.
            select += [f'sum({clean}) AS score_sum_{i}',
                       f'count({clean})::BIGINT AS score_count_{i}']
    for i, family in enumerate(families):
        labels = ', '.join(map(literal, family))
        select.append(f'count(*) FILTER (WHERE predicted_class IN ({labels}))::BIGINT AS family_count_{i}')
    if dataset.format == 'supervised' and mean_probabilities:
        for label, column in zip(dataset.class_labels, dataset._prob_columns):
            select.append(f"avg(CASE WHEN call_status='called' THEN CAST({identifier(column)} AS DOUBLE) END) AS {identifier('mean_probability_' + label)}")
    where = "chrom IS NOT NULL AND chrom != '*' AND position >= 0 AND strand IN (0, 1)"
    if group_by_kmer:
        where += ' AND kmer IS NOT NULL'
    # Filters already selected rows in calls. Reassign only the summary class,
    # preserving invalid calls and the denominator on both aggregation paths.
    source = ('calls' if binary_decision is None else
              f'(SELECT * REPLACE ({binary_decision} AS predicted_class) FROM calls)')
    return f'SELECT {", ".join(select)} FROM {source} WHERE {where} GROUP BY {", ".join(groups)}'


def _finish_classification(query, classes, families, include_class_counts, mean_scores=False):
    """Form frequencies and means after additive site statistics are merged."""
    hidden = [f'class_count_{i}' for i in range(len(classes))]
    unique_families = list(dict.fromkeys(families))
    hidden += [f'family_count_{i}' for i in range(len(unique_families))]
    if mean_scores:
        hidden += [f'score_{stat}_{i}' for i in range(len(classes)) for stat in ('sum', 'count')]
    fields = ['* EXCLUDE (' + ', '.join(hidden) + ')' if hidden else '*']
    for i, label in enumerate(classes):
        count = f'class_count_{i}'
        fields.append(f'{count}::DOUBLE / nullif(called_count, 0) AS {identifier("freq_" + label)}')
        if families:
            denom = f'family_count_{unique_families.index(families[i])}'
            fields.append(f'{count}::DOUBLE / nullif({denom}, 0) AS {identifier("freq_" + label + "_rel")}')
        if include_class_counts:
            fields.append(f'{count} AS {identifier("count_" + label)}')
        if mean_scores:
            fields.append(f'score_sum_{i} / nullif(score_count_{i}, 0) AS {identifier("mean_score_" + label)}')
    return f'SELECT {", ".join(fields)} FROM ({query})'


def _partition_counts(con, dataset, needed, filters, query, directory):
    """Reduce one input part at a time, then merge only additive site state.

    This bounds ASOF sorting to one file. Temporary Parquets contain site
    counts, score sums and log-p sums, never raw read rows; all are removed on
    context exit. Fractions, score means and Fisher transforms are computed
    only after the final merge.
    Optional nonadditive statistics retain the general SQL path.
    """
    paths = []
    for i, part in enumerate(dataset.parts):
        subset = dataset._subset([part])
        subset.register(con, name='calls', columns=needed, filters=filters)
        path = Path(directory) / f'{i:06d}.parquet'
        con.execute(f'COPY ({query}) TO {literal(str(path))} (FORMAT PARQUET, COMPRESSION ZSTD)')
        paths.append(path)
    groups = ['chrom', 'position', 'strand']
    fields = list(groups)
    for field in pq.read_schema(paths[0]):
        if field.name in groups:
            continue
        name = identifier(field.name)
        cast = '::BIGINT' if pa.types.is_integer(field.type) else ''
        fields.append(f'sum({name}){cast} AS {name}')
    files = '[' + ', '.join(literal(str(path)) for path in paths) + ']'
    return f'SELECT {", ".join(fields)} FROM read_parquet({files}) GROUP BY {", ".join(groups)}'


def summarize_loci(source, *, filters=None, classes=None, group_by_kmer=False,
                   relative_frequencies=False, include_class_counts=False, class_families=None,
                   binary_threshold=None, binary_positive_class=None,
                   modified_pvalue_cutoff=0.05, min_pvalue=1e-300,
                   pvalue_thresholds=(), confidence=None,
                   metric_columns=None, metric_stats=(), combine_pvalues=None,
                   include_kmer=False, mean_probabilities=False, mean_scores=False,
                   output_path=None, overwrite=False,
                   memory_limit='8GB', temp_directory=None, threads=None):
    """Collapse one run to reference sites without loading its read table.

    Parameters
    ----------
    source : OutputDataset or path-like
        An opened dataset or a path accepted by ``open_output``. Keys are
        zero-based chrom/position/strand. All negative positions, missing/'*'
        chromosomes and strands outside 0/1 are excluded, before grouping.
    filters : mapping or predicate triples, optional
        Read-level filters, as in ``OutputDataset.iter_tables``. Filtering changes
        denominators. Class frequencies require canonical AND modified rows;
        selecting only modifications cannot estimate modification fractions.
    classes : sequence of str, optional
        Classes reported in frequencies, optional counts and score means,
        defaulting to stored class labels.
        Does not filter rows. Frequencies divide hard-call counts by called_count;
        total_count also includes retained no-calls.
    relative_frequencies : bool
        Prototype-only: add freq_LABEL_rel, dividing each hard-call count by
        called predictions in its canonical-plus-modified family. Default off.
        A zero family count gives null, not zero. Families are taken from the
        effect manifest, or class_families for older outputs. Selecting classes
        does not remove sibling classes from the denominator; filters do.
    include_class_counts : bool
        Add integer count_LABEL columns for selected classification classes,
        default off. Counts and both frequencies use the same retained calls.
    binary_threshold : float or None
        Supervised output with exactly two stored classes only. Assign each
        valid call to binary_positive_class when its selected probability is
        >= this threshold in [0, 1], otherwise to the other class. None keeps
        argmax. Invalid calls stay null; no rows are removed and called_count
        is unchanged. Uses the stored class probabilities.
        Applies only to summary counts/frequencies, after existing filters;
        filters on predicted_class/probability still use the original argmax.
        Stored probabilities, mean probabilities and read-level calls are unchanged.
    binary_positive_class : str or None
        Required with binary_threshold: exact stored label whose probability
        is thresholded. Independent of classes (reported columns) and stored
        class order. Supplying it without a threshold is an error.
    class_families : mapping of str to str, optional
        Prototype class label to canonical A/C/G/T/U. For legacy files, include
        every modification that should contribute to family denominators,
        including unselected classes. Canonical labels are included automatically;
        T/U share a family. Manifest assignments cannot be overridden. Requires
        relative_frequencies=True. No reference FASTA or k-mer inference is used.
    group_by_kmer : bool
        Add kmer to classification grouping. Unavailable for p-value summaries.
    modified_pvalue_cutoff : float
        Inclusive cutoff for fraction_modified. This is an anomaly-call fraction,
        not a calibrated biological modification fraction. Only finite p-values
        in [0, 1] count toward total_count; n_observations includes invalid values.
    min_pvalue : float
        Positive floor for Fisher's sum of log p-values and the combined -log10
        p-value cap. Fisher's usual independence assumption is not validated here.
    pvalue_thresholds : sequence of float
        Additional STRICT p < threshold counts and fractions; default none.
        This preserves the cutoff distinction in the original pairwise utility.
    confidence : float or None
        Enable Wilson intervals for primary and extra fractions; default off.
        Excludes assay/calibration uncertainty.
    metric_columns : sequence of str, optional
        Null/pairwise metrics; discover available distances/PCs only when stats
        are requested. Nonfinite observations are excluded independently.
    metric_stats : sequence of {count, mean, std, median}
        Numeric statistics; default none. std uses sample standard deviation.
        Exact median may require substantial memory.
    combine_pvalues : bool or None
        Fisher combination defaults on for calibrated canonical-null, off for
        pairwise. Explicit booleans override; absent p-values produce no Fisher field.
    include_kmer : bool
        Include representative kmer mode in classification summaries, default off.
    mean_probabilities : bool
        Include supervised class mean probabilities, default off.
    mean_scores : bool
        Prototype-only: add mean_score_LABEL for each selected class, default
        off. Average finite scores only over rows predicted as that class;
        null/NaN/infinite scores are excluded without changing class counts.
        No finite matching scores gives null/NaN. Uses the winning score column
        (or score_LABEL for legacy files lacking it), preserving stored score
        semantics. Reads score payloads only when enabled or explicitly filtered.
        Per-part score sums and valid counts are merged before division.
    output_path : path-like, optional
        Write only site batches to Zstd Parquet and return Path, avoiding a full
        pandas site table. Otherwise return a pandas DataFrame of collapsed sites.
        Existing files require overwrite=True. Output must not replace an input.
    memory_limit, temp_directory, threads : optional
        DuckDB configuration. Aggregates and split joins can spill to disk; these
        settings are not a hard cap on every process allocation or pandas results.

    Notes
    -----
    Counts are observations, not deduplicated read IDs. Do not mix independent
    runs. Site order is unspecified. An incomplete dataset is a snapshot only.
    No raw read table is retained in Python; only aggregated site batches are
    transformed there. No calibrated p-values means counts plus explicitly requested metrics.
    Metadata filters skip unrelated split parts before scanning scored columns.
    Default counts/fractions, score means and p-value summaries aggregate each part first,
    then merge temporary site-only Parquets. This bounds metadata joins to one
    part. Temporary files use temp_directory or the system temporary directory
    and are cleaned up on success or failure. Numeric metrics, k-mer summaries,
    and mean probabilities use the general SQL aggregation path over selected
    parts. Returned pandas site tables still need to fit in memory.
    """
    from .dataset import OutputDataset, open_output

    dataset = source if isinstance(source, OutputDataset) else open_output(source)
    binary_column, binary_decision = _binary_decision(dataset, binary_threshold, binary_positive_class)
    if (relative_frequencies or class_families is not None) and dataset.format != 'prototypes':
        raise ValueError('relative_frequencies/class_families apply only to prototypes')
    if class_families is not None and not relative_frequencies:
        raise ValueError('class_families requires relative_frequencies=True')
    if include_class_counts and dataset.format not in ('prototypes', 'supervised'):
        raise ValueError('include_class_counts applies only to classification output')
    if mean_scores and dataset.format != 'prototypes':
        raise ValueError('mean_scores applies only to prototypes')
    cutoff, minimum = float(modified_pvalue_cutoff), float(min_pvalue)
    thresholds = tuple(map(float, pvalue_thresholds))
    if not 0 <= cutoff <= 1 or not 0 < minimum <= 1 or (confidence is not None and not 0 < confidence < 1):
        raise ValueError('Invalid p-value floor/cutoff or confidence')
    if (any(not 0 < value <= 1 for value in thresholds)
            or len({format(value, '.12g') for value in thresholds}) != len(thresholds)):
        raise ValueError('pvalue_thresholds must lie in (0, 1] with unique output prefixes')
    metric_stats = (metric_stats,) if isinstance(metric_stats, str) else tuple(metric_stats)
    if len(set(metric_stats)) != len(metric_stats) or set(metric_stats) - {'count', 'mean', 'std', 'median'}:
        raise ValueError('metric_stats must contain unique count/mean/std/median names')
    if combine_pvalues is not None and type(combine_pvalues) is not bool:
        raise ValueError('combine_pvalues must be True, False, or None')
    combine = dataset.format == 'canonical_null' if combine_pvalues is None else combine_pvalues
    metrics = _metrics(dataset, metric_columns, metric_stats)
    missing = {'chrom', 'position', 'strand'} - set(dataset.schema.names)
    if missing:
        raise ValueError(f'Missing genomic keys: {missing}')
    target = None if output_path is None else Path(output_path)
    if target is not None:
        inputs = [p for part in dataset.parts for p in (part.data, part.meta) if p]
        if dataset._effect:
            inputs.append(Path(dataset.path) / 'effect_parts.json')
        if any(target.resolve() == p.resolve() for p in inputs):
            raise ValueError('Summary output must not overwrite an input part or manifest')
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
    # Prune once before opening scored columns. Surviving mixed parts retain
    # complete metadata boundaries; row predicates are still applied normally.
    dataset = dataset._prune_metadata(predicates(filters, dataset.schema.names))
    with _connection(memory_limit, temp_directory, threads) as con, ExitStack() as cleanup:
        needed = ['chrom', 'position', 'strand']
        if dataset.format in ('prototypes', 'supervised'):
            needed.append('predicted_class')
            if binary_column is not None:
                needed.append(binary_column)
            if (include_kmer or group_by_kmer) and 'kmer' in dataset.schema.names:
                needed.append('kmer')
            if mean_probabilities and dataset.format == 'supervised':
                needed.extend(['call_status'] + dataset._prob_columns)
        else:
            needed.extend(metrics)
            if 'pvalue' in dataset.schema.names:
                needed.append('pvalue')
        needed = list(dict.fromkeys(needed))
        partitioned = (len(dataset.parts) > 1 and not metric_stats and not group_by_kmer
                       and not include_kmer and not mean_probabilities)
        if dataset.format in ('prototypes', 'supervised'):
            available_classes = dataset.class_labels
            if not available_classes and (classes is None or relative_frequencies):
                dataset.register(con, name='calls', columns=needed, filters=filters)
                available_classes = tuple(row[0] for row in con.execute(
                    'SELECT DISTINCT predicted_class FROM calls WHERE predicted_class IS NOT NULL').fetchall())
            if classes is None:
                classes = available_classes
            classes = [classes] if isinstance(classes, str) else list(classes)
            if len(set(classes)) != len(classes) or any(not isinstance(c, str) for c in classes):
                raise ValueError('classes must contain unique string labels')
            families = (_class_families(dataset, classes, class_families, available_classes)
                        if relative_frequencies else [])
            score_columns = _class_score_columns(dataset, classes) if mean_scores else []
            needed = list(dict.fromkeys(needed + score_columns))
            query = _classification_query(dataset, classes, group_by_kmer, include_kmer, mean_probabilities,
                                          families=list(dict.fromkeys(families)), score_columns=score_columns,
                                          binary_decision=binary_decision)
            transform = lambda table: table
        else:
            if classes is not None or group_by_kmer:
                raise ValueError('classes/group_by_kmer apply only to classification output')
            query = _pvalue_query(dataset, cutoff, minimum, thresholds, metrics, metric_stats, combine)
            transform = lambda table: _finish_pvalues(table, minimum, thresholds, confidence)
        if partitioned:
            directory = cleanup.enter_context(TemporaryDirectory(prefix='porefm-loci-', dir=temp_directory))
            query = _partition_counts(con, dataset, needed, filters, query, directory)
        else:
            dataset.register(con, name='calls', columns=needed, filters=filters)
        if dataset.format in ('prototypes', 'supervised'):
            query = _finish_classification(query, classes, families, include_class_counts, mean_scores)
        reader = con.execute(query).to_arrow_reader(batch_size=65536)
        if target is None:
            tables = [transform(pa.Table.from_batches([batch])) for batch in reader]
            result = pa.concat_tables(tables) if tables else transform(reader.schema.empty_table())
            return result.to_pandas()
        temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
        # Even empty results retain the complete transformed output schema.
        schema = transform(reader.schema.empty_table()).schema
        with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
            for batch in reader:
                writer.write_table(transform(pa.Table.from_batches([batch])))
        temporary.replace(target)
        return target
