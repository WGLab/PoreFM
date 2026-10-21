"""Summarize pairwise read scores and diagnose unusual low-p-value reads.

Only combine partitions from ONE comparison run: PCs are fitted separately for
each locus and cannot be pooled across loci or independently fitted runs.
No model execution is needed. Dependencies: numpy, pandas, scipy, scikit-learn,
PyArrow for loading, and optionally Polars for numeric aggregation.

The low-p-value fraction is an observed anomaly-call fraction, not a calibrated
modification proportion. Cluster distances/ranks are descriptive diagnostics,
not new p-values or modification probabilities. Cross-fitting excludes a scored
read from its cluster fit, but does not undo selection by the original p-value
or the original joint test/control PCA fit.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import os
import pysam

LOCUS = ["refname", "position", "strand"]
DISTANCES = ["mean_squared_distance", "cosine_distance", "squared_mahalanobis_distance"]
READ_COLUMN_RENAMES = {
    "mean_squared_distance": "full_embedding_mse_to_control",
    "cosine_distance": "full_embedding_cosine_distance_to_control",
    "squared_mahalanobis_distance": "pc_mahalanobis_sq_to_control_reported",
    "p_value": "pc_control_p_value",
}


class _ReferenceSiteFilter:
    """Select complete sites and track original metadata offsets across batches.

    Site keys/coverage occupy O(sites); saved identity/base codes occupy
    O(retained rows). Temporary key lookup and identity expansion are bounded
    by the decoded batch size. Nonmatching rows are validated but not retained.
    """

    def __init__(self, sites, identities, lengths, fasta_path, reference_bases):
        bases = {base.upper() for base in reference_bases}
        if not bases or not bases <= set("ACGTN"):
            raise ValueError("reference_bases must be a nonempty selection of A, C, G, T, N")
        keys = sites.rename(columns={"chrom": "refname"})
        if keys.columns.duplicated().any():
            raise ValueError("Site table must have chrom or refname, not both")
        required = LOCUS + ["scored_test_coverage"]
        if missing := set(required) - set(keys):
            raise ValueError(f"Missing site columns for reference filtering: {sorted(missing)}")
        if keys[LOCUS].isna().any().any() or keys.duplicated(LOCUS).any():
            raise ValueError("Site table must have unique non-null locus keys")
        coverage = keys["scored_test_coverage"]
        if coverage.isna().any() or not pd.api.types.is_integer_dtype(coverage) or (coverage < 0).any():
            raise ValueError("scored_test_coverage must contain nonnegative integers")
        self.expected = coverage.to_numpy(dtype=np.int64)
        if sum(int(count) for count in self.expected) != sum(int(length) for length in lengths):
            raise ValueError("Total scored_test_coverage does not match metadata lengths")
        # Annotate only sites; all reads at a site share this reference base.
        annotated = add_reference_base(sites, fasta_path, ref_col="chrom" if "chrom" in sites else "refname")
        self.bases = pd.Categorical(annotated["ref_base"])
        self.selected = annotated["ref_base"].str.upper().isin(bases).to_numpy()
        self.sites = annotated.loc[self.selected].reset_index(drop=True)
        self.index = pd.MultiIndex.from_frame(keys[LOCUS])
        self.observed = np.zeros(len(sites), dtype=np.int64)
        self.identities = identities
        self.ends = np.cumsum(lengths, dtype=np.int64)
        self.metadata_rows = []
        self.base_codes = []

    def select(self, batch, offset):
        """Return a keep mask using physical row offsets, never filtered offsets."""
        positions = batch.column(batch.schema.get_field_index("position"))
        if positions.null_count:
            raise ValueError("Read positions must not be null")
        metadata_rows = np.searchsorted(
            self.ends, np.arange(offset, offset + batch.num_rows, dtype=np.int64), side="right",
        )
        keys = pd.MultiIndex.from_arrays([
            self.identities["refname"].array.take(metadata_rows),
            positions.to_numpy(zero_copy_only=False),
            self.identities["strand"].array.take(metadata_rows),
        ], names=LOCUS)
        site_rows = self.index.get_indexer(keys)
        if (site_rows < 0).any():
            raise ValueError(f"Read locus absent from site metrics: {keys[np.flatnonzero(site_rows < 0)[0]]}")
        self.observed += np.bincount(site_rows, minlength=len(self.index))
        keep = self.selected[site_rows]
        self.metadata_rows.append(metadata_rows[keep])
        self.base_codes.append(self.bases.codes[site_rows[keep]])
        return keep

    def validate_counts(self):
        """Reject partial/mismatched tables, including excluded sites in scanned files."""
        if not np.array_equal(self.observed, self.expected):
            bad = np.flatnonzero(self.observed != self.expected)[0]
            raise ValueError(f"Read count does not match scored_test_coverage at {self.index[bad]}")


def _read_compact_scores(path, *, float_dtype, pvalue_dtype, batch_size, site_filter=None):
    """Decode and cast Parquet batches before materializing a pandas table.

    Integer narrowing is checked (including fractional values). Finite floats
    outside the target range raise instead of silently becoming infinity.
    Float32 rounding/subnormal underflow is allowed when explicitly selected.
    Unknown columns retain their stored Arrow types.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if float_dtype not in {None, "float32", "float64"} or pvalue_dtype not in {None, "float32", "float64"}:
        raise ValueError("float_dtype and pvalue_dtype must be None, 'float32', or 'float64'")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow.remove_metadata()
    fields = []
    for field in schema:
        dtype = field.type
        if field.name == "read_position":
            dtype = pa.int32()
        elif field.name == "position":
            dtype = pa.int64()
        elif field.name == "p_value" and pvalue_dtype is not None:
            dtype = pa.from_numpy_dtype(pvalue_dtype)
        elif float_dtype is not None and (field.name in DISTANCES or re.fullmatch(r"pc_[1-9][0-9]*", field.name)):
            dtype = pa.from_numpy_dtype(float_dtype)
        fields.append(pa.field(field.name, dtype, nullable=field.nullable))
    target = pa.schema(fields)
    if site_filter is not None and "position" not in schema.names:
        raise ValueError("Missing read column: position")
    batches = []
    # No payload decoding when the site's base selection is empty. Footer and
    # metadata totals have still been checked; per-site read counts cannot be
    # audited without scanning the skipped file.
    decode = site_filter is None or site_filter.selected.any()
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size) if decode else ():
        if site_filter is not None:
            keep = site_filter.select(batch, offset)
            offset += batch.num_rows
            batch = batch.filter(pa.array(keep))
            if not batch.num_rows:
                continue
        arrays = []
        for array, field in zip(batch.columns, target):
            if pa.types.is_floating(field.type) and field.type != array.type:
                limit = np.finfo(field.type.to_pandas_dtype()).max
                outside = pc.and_(pc.is_finite(array), pc.greater(pc.abs(array), float(limit)))
                if pc.any(outside).as_py():
                    raise ValueError(f"{field.name} contains finite values outside {field.type} range")
            try:
                arrays.append(pc.cast(array, field.type, safe=True))
            except pa.ArrowInvalid as exc:
                raise ValueError(f"Cannot safely cast {field.name} to {field.type}: {exc}") from exc
        batches.append(pa.RecordBatch.from_arrays(arrays, schema=target))
    if site_filter is not None and decode:
        site_filter.validate_counts()
    # Arrow chunks hold already compact buffers; split_blocks avoids pandas
    # consolidating all same-dtype columns into another full-width allocation.
    table = pa.Table.from_batches(batches, schema=target)
    return table.to_pandas(split_blocks=True)


def _compact_identities(meta, lengths, readname_dtype):
    """Encode identities before expansion; estimate read-name storage per partition."""
    if readname_dtype not in {"auto", "category", "string"}:
        raise ValueError("readname_dtype must be 'auto', 'category', or 'string'")
    identities = meta[["readname", "refname", "strand"]].copy()
    for col in ("refname", "strand"):
        identities[col] = identities[col].astype("category")
    names = identities["readname"].astype("string[pyarrow]")
    if readname_dtype == "string":
        identities["readname"] = names
        return identities
    categorical = names.astype("category")
    nrows = sum(int(length) for length in lengths)
    category_bytes = categorical.cat.categories.memory_usage(deep=True) + nrows * categorical.cat.codes.dtype.itemsize
    # Arrow strings use UTF-8 bytes plus 64-bit offsets in pandas. Estimate
    # the expanded storage without ever expanding the strings to Python objects.
    name_bytes = names.str.encode("utf-8").str.len().fillna(0).to_numpy(dtype=np.int64)
    string_bytes = sum((int(size) + 8) * int(length) for size, length in zip(name_bytes, lengths))
    identities["readname"] = categorical if readname_dtype == "category" or category_bytes < string_bytes else names
    return identities


def load_pairwise_partition(
    meta_path, *, float_dtype=None, pvalue_dtype=None,
    readname_dtype="auto", batch_size=262_144,
    fasta_path=None, reference_bases=None,
):
    """Return (read_scores, site_metrics) for one current comparison partition.

    ``meta_path`` is ``RUN/read_scores/CHROM_BLOCK_meta.parquet``. The matching
    data file and ``RUN/anomaly_site_metrics/CHROM_BLOCK.parquet`` must exist.
    Metadata lengths count contiguous scored rows, not physical read lengths.
    Coordinates and strand codes are preserved (zero-based; 0 forward, 1 reverse).

    Floating columns retain their stored precision by default. Optional
    ``float_dtype='float32'`` narrows distances/PCs; ``pvalue_dtype='float32'``
    also permits probability rounding and underflow. These are lossy options
    and cannot guarantee identical statistics to the original stored values.
    ``read_position`` uses checked int32; genomic ``position`` stays int64.
    ``refname``/``strand`` become categories. ``readname_dtype='auto'`` chooses
    category or Arrow-backed string from estimated expanded storage; 'category'
    and 'string' force either representation. Categories retain original values.
    Site metrics retain producer dtypes, including control covariance precision.
    ``batch_size`` bounds decoding/casting temporaries, not the returned table:
    one complete retained partition must still fit in memory.

    With ``reference_bases={'A'}`` and ``fasta_path``, annotate sites first and
    retain only matching read rows from each decoded batch. Matching ignores
    letter case; returned ``ref_base`` preserves FASTA case and the existing
    strand-aware complement convention (reverse-strand genomic T selects A).
    Both tables receive ``ref_base``. Row order is preserved, indexes reset.
    All scanned read keys/counts must match unique site keys and
    ``scored_test_coverage``; mismatches raise. No matching sites skips payload
    decoding (only metadata/footer totals are checked). Original read identities
    are recovered before filtering. Site covariance/coverage is never refitted.
    The FASTA index is created if absent, as in ``add_reference_base``.
    """
    meta_path = Path(meta_path)
    suffix = "_meta.parquet"
    if not meta_path.name.endswith(suffix):
        raise ValueError(f"Expected a filename ending in {suffix}")
    stem = meta_path.name.removesuffix(suffix)
    if (fasta_path is None) != (reference_bases is None):
        raise ValueError("Supply fasta_path and reference_bases together")
    sites = pd.read_parquet(meta_path.parent.parent / "anomaly_site_metrics" / (stem + ".parquet"))
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    data_path = meta_path.with_name(stem + "_data.parquet")
    lengths = meta["length"]
    if lengths.isna().any() or not pd.api.types.is_integer_dtype(lengths) or (lengths < 0).any():
        raise ValueError("Metadata lengths must be nonnegative integers")
    lengths = lengths.to_numpy(dtype=np.int64)
    import pyarrow.parquet as pq

    if sum(int(length) for length in lengths) != pq.ParquetFile(data_path).metadata.num_rows:
        raise ValueError("Metadata lengths do not match the data row count")
    identities = _compact_identities(meta, lengths, readname_dtype)
    if identities.isna().any().any():
        raise ValueError("Metadata identities must not be null")
    if set(identities.columns) & set(pq.read_schema(data_path).names):
        raise ValueError("Metadata and score columns overlap")
    site_filter = None if reference_bases is None else _ReferenceSiteFilter(
        sites, identities, lengths, fasta_path, reference_bases,
    )
    if site_filter is not None and "ref_base" in pq.read_schema(data_path).names:
        raise ValueError("Read scores already contain ref_base; expected the current unannotated format")
    data = _read_compact_scores(data_path, float_dtype=float_dtype, pvalue_dtype=pvalue_dtype,
                                batch_size=batch_size, site_filter=site_filter)
    if site_filter is None:
        row_index = np.repeat(np.arange(len(meta)), lengths)
    else:
        row_index = np.concatenate(site_filter.metadata_rows) if site_filter.metadata_rows else np.empty(0, dtype=np.int64)
        site_filter.metadata_rows.clear()
    columns = {col: identities[col].array.take(row_index) for col in identities}
    del row_index
    columns.update({col: data[col].array for col in data})
    reads = pd.DataFrame(columns, copy=False)
    if site_filter is not None:
        codes = np.concatenate(site_filter.base_codes) if site_filter.base_codes else np.empty(0, dtype=np.int8)
        reads["ref_base"] = pd.Categorical.from_codes(codes, categories=site_filter.bases.categories)
        sites = site_filter.sites
    return reads, sites


def _pairwise_meta_files(run_dir):
    """Find current-format partitions, failing on missing companion files upfront."""
    run_dir = Path(run_dir)
    meta_files = sorted((run_dir / "read_scores").glob("*_meta.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.parquet files found in {run_dir / 'read_scores'}")
    for meta_file in meta_files:
        stem = meta_file.name.removesuffix("_meta.parquet")
        for path in (meta_file.with_name(stem + "_data.parquet"), run_dir / "anomaly_site_metrics" / (stem + ".parquet")):
            if not path.is_file():
                raise FileNotFoundError(path)
    return meta_files


def iter_pairwise_directory(run_dir, **load_options):
    """Yield ``(stem, reads, sites)`` in filename order, one partition at a time.

    ``load_options`` are keyword options for ``load_pairwise_partition``.
    Consume and release each result before advancing; collecting this iterator
    into a list defeats the memory bound. No rows or loci are split into batches.
    """
    for meta_path in _pairwise_meta_files(run_dir):
        reads, sites = load_pairwise_partition(meta_path, **load_options)
        yield meta_path.name.removesuffix("_meta.parquet"), reads, sites
        del reads, sites


def _concat_compact(tables):
    """Concatenate tables while retaining categorical columns across dictionaries."""
    from pandas.api.types import union_categoricals

    for col in set.intersection(*(set(table.columns) for table in tables)):
        if all(isinstance(table[col].dtype, pd.CategoricalDtype) for table in tables):
            categories = union_categoricals([
                pd.Categorical([], categories=table[col].cat.categories) for table in tables
            ]).categories
            for table in tables:
                table[col] = table[col].cat.set_categories(categories)
        elif any(isinstance(table[col].dtype, pd.CategoricalDtype) for table in tables):
            # Auto-selected read-name representations can differ by partition.
            for table in tables:
                table[col] = table[col].astype("string[pyarrow]")
    return pd.concat(tables, ignore_index=True)


def load_pairwise_directory(run_dir, **load_options):
    """Load all read scores and site metrics from one comparison run.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Directory containing ``read_scores/`` and ``anomaly_site_metrics/``.
        Loads each ``read_scores/*_meta.parquet`` in filename order with its
        matching data and site-metrics files; subdirectories are not searched.

    Returns
    -------
    reads_df, site_df : pandas.DataFrame
        Concatenated read scores and site statistics with fresh integer indexes.
        Original columns, coordinates, and strand codes are preserved. All
        partitions are loaded into memory; concatenation also needs temporary
        storage. Dtypes follow ``load_pairwise_partition`` and its options can
        be passed as keyword arguments. Categorical dictionaries are unified
        across partitions, so concatenation does not silently produce objects.
        For larger runs use ``iter_pairwise_locus_summaries`` instead.

    Raises
    ------
    FileNotFoundError
        If no metadata files are found or a matching partition file is missing.
        Invalid partitions raise the errors from ``load_pairwise_partition``;
        no partition is silently skipped.
    """
    read_tables = []
    site_tables = []
    for _, reads, sites in iter_pairwise_directory(run_dir, **load_options):
        read_tables.append(reads)
        site_tables.append(sites)

    return (
        _concat_compact(read_tables),
        _concat_compact(site_tables),
    )


def pvalue_prefix(threshold):
    """Return an explicit PC-control p-value prefix, e.g. ``pc_control_p_lt_0p05``."""
    return "pc_control_p_lt_" + format(float(threshold), ".12g").replace(".", "p").replace("-", "m").replace("+", "")


def _wilson(count, total, confidence):
    """Vectorized two-sided Wilson intervals; undefined denominators yield NaN."""
    total = total.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = count / total
        z = norm.ppf((1 + confidence) / 2)
        denominator = 1 + z * z / total
        center = (fraction + z * z / (2 * total)) / denominator
        radius = z * np.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2)) / denominator
    return fraction, np.maximum(0, center - radius), np.minimum(1, center + radius)


def _fit_gaussian(x, estimator, ridge, seed):
    """Estimate a location and positive-definite scatter matrix in fitting units.

    Ledoit-Wolf is fast but not robust to contamination. MCD estimates the
    dominant unimodal cloud and is more expensive. Rank-deficient input is
    rejected rather than made to look confidently compact by an arbitrary ridge.
    """
    from sklearn.covariance import LedoitWolf, MinCovDet

    if np.linalg.matrix_rank(x - x.mean(axis=0)) < x.shape[1]:
        raise ValueError("rank-deficient PC cloud")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        if estimator == "mcd":
            model = MinCovDet(random_state=seed, store_precision=False).fit(x)
        else:
            model = LedoitWolf(store_precision=False).fit(x)
    cov = np.atleast_2d(model.covariance_).astype(float)
    if not np.isfinite(cov).all() or np.linalg.matrix_rank(cov) < x.shape[1]:
        raise ValueError("rank-deficient fitted covariance")
    cov += np.eye(x.shape[1]) * ridge * max(float(np.trace(cov) / x.shape[1]), 1e-12)
    return model.location_, cov, bool(caught)


def _mahalanobis_squared(x, mean, cov):
    """Compute squared distances with a Cholesky solve instead of an inverse."""
    standardized = np.linalg.solve(np.linalg.cholesky(cov), (x - mean).T)
    return np.einsum("ij,ij->j", standardized, standardized)


def _score_pc_gaussian(x, mean, cov, control_cov=None):
    """Score one Gaussian in original, retained PC coordinates.

    All arrays use the SAME site's PC basis. The log density includes both the
    covariance determinant and D*log(2*pi). The optional control covariance
    supplies a common metric for comparing distances to different centers.
    """
    centered = x - mean
    own_d2 = _mahalanobis_squared(x, mean, cov)
    result = {
        "l2_sq": np.einsum("ij,ij->i", centered, centered),
        "mahalanobis_sq": own_d2,
        "logpdf": -0.5 * (own_d2 + np.linalg.slogdet(cov)[1] + x.shape[1] * np.log(2 * np.pi)),
    }
    if control_cov is not None:
        result["control_cov_mahalanobis_sq"] = _mahalanobis_squared(x, mean, control_cov)
    return result


def _numeric_summaries(columns, codes, *, engine, valid_p=None, quantiles=False):
    """Aggregate one numeric column at a time using float64 working precision.

    Avoid an N-by-metrics float64 copy. Input arrays may be float32; only the
    current column is promoted. Dense locus codes are reused without hashing
    them again for each pandas column; all quantiles share one sort per column.
    Both engines exclude nonfinite values, use sample std (ddof=1), and
    linear-interpolated exact quantiles. Output column
    ordering follows the historical basic-then-quantiles schema.
    """
    ngroups = int(codes.max()) + 1 if len(codes) else 0
    if engine == "polars":
        import polars as pl

        group_codes = pl.Series("_group", codes)
    else:
        # Codes already enumerate every observed locus. Keeping all categories
        # allows pandas to reuse these codes without factorizing N rows again.
        group_codes = pd.Categorical.from_codes(codes, categories=pd.RangeIndex(ngroups), ordered=True)
    basic, tails = {}, {}
    for name, values in columns.items():
        values = pd.Series(values, copy=False).to_numpy(dtype=np.float64, na_value=np.nan)
        keep = np.isfinite(values)
        if name == "p_value" and valid_p is not None:
            keep &= valid_p
        output_name = READ_COLUMN_RENAMES.get(name, name)
        if not keep.any():
            # Skipped diagnostic fits leave entire columns NaN. Their exact
            # summaries are known without groupby or sorting all those NaNs.
            for stat in ("count", "mean", "std", "median"):
                basic[f"{output_name}_{stat}"] = np.zeros(ngroups, dtype=np.int64) if stat == "count" else np.full(ngroups, np.nan)
            if quantiles:
                for stat in ("q10", "q90", "q95"):
                    tails[f"{output_name}_{stat}"] = np.full(ngroups, np.nan)
            continue
        # No copy is needed for the common case of fully finite metric columns.
        if not keep.all():
            values = np.where(keep, values, np.nan)
        if engine == "polars":
            frame = pl.DataFrame([group_codes, pl.Series("_value", values, nan_to_null=True)])
            value = pl.col("_value")
            expressions = [value.count().cast(pl.Int64).alias("count"), value.mean().alias("mean"),
                           value.std(ddof=1).alias("std"), value.median().alias("median")]
            if quantiles:
                expressions += [value.quantile(q, interpolation="linear").alias(f"q{round(q * 100)}")
                                for q in (0.1, 0.9, 0.95)]
            result = frame.group_by("_group").agg(expressions).sort("_group").drop("_group").to_pandas()
        else:
            groups = pd.Series(values).groupby(group_codes, observed=False, sort=True)
            result = groups.agg(["count", "mean", "std", "median"]).reset_index(drop=True)
            if quantiles:
                quantile_values = groups.quantile([0.1, 0.9, 0.95]).to_numpy().reshape(ngroups, 3)
                for i, q in enumerate((0.1, 0.9, 0.95)):
                    result[f"q{round(q * 100)}"] = quantile_values[:, i]
        for stat in result:
            target = tails if stat.startswith("q") else basic
            target[f"{output_name}_{stat}"] = result[stat].to_numpy()
    return pd.DataFrame({**basic, **tails})


def summarize_pairwise_loci(
    reads,
    site_metrics=None,
    *,
    pvalue_thresholds=(0.001, 0.01, 0.05),
    cluster_pvalue=0.05,
    min_cluster_reads=20,
    min_cluster_fraction=0.0,
    covariance="ledoit_wolf",
    crossfit_folds=3,
    confidence=0.95,
    ridge=1e-6,
    seed=0,
    compute_pc=True,
    selected_prior=None,
    engine="pandas",
    return_annotated=True,
):
    """Return per-locus statistics and annotated reads from ONE comparison run.

    Parameters
    ----------
    reads : pandas.DataFrame
        Flattened scores with readname, refname, position, strand, p_value, and
        optional distance/pc_1...pc_D columns. Rows must be unique per
        (refname, position, strand, readname); duplicates raise an error.
        Nonfinite/out-of-range p-values are excluded from frequency denominators.
        Nonfinite metric values are excluded independently from their summaries.
    site_metrics : pandas.DataFrame, optional
        Matching anomaly_site_metrics table, keyed by chrom/position/strand.
        Control PCA covariance is a flattened row-major D*D vector. When given,
        PCs are whitened using this control covariance and centered on its mean.
        Missing site rows disable that locus's PC fit; covariance errors raise.
        Coverage fields are joined for auditing input completeness.
    pvalue_thresholds : iterable of float
        Strict p < threshold cuts in (0, 1]. Each produces *_count, *_fraction,
        *_ci_low, *_ci_high, with prefixes from ``pvalue_prefix``. Wilson bounds
        assume independent Bernoulli observations; they do not include assay or
        p-value calibration error. Fractions refer only to valid scored rows.
    cluster_pvalue : float
        Strict p-value cutoff defining the candidate cloud, independently of
        the reported thresholds. Default 0.05. Only these reads get crossfit
        scores; allfit scores also evaluate nonselected reads at fitted loci.
    min_cluster_reads : int
        Minimum candidate reads with finite PCs. Also requires >= 2*D+4 reads
        and enough training rows per cross-fit fold. Sparse clouds remain NaN.
    min_cluster_fraction : float
        Skip PC fitting below this observed p < cluster_pvalue fraction; useful
        for cheaply restricting expensive MCD fits to shortlisted loci.
    covariance : {'ledoit_wolf', 'mcd'}
        Fast shrinkage covariance, or robust minimum covariance determinant.
        MCD targets one dominant cloud; neither option proves unimodality.
    crossfit_folds : int
        Balanced folds, >= 2. Each *_crossfit score uses a fit excluding the
        candidate's fold. Folds are deterministic under input row permutation
        because candidates are first ordered by read name. No in-sample fallback.
    confidence : float
        Coverage of two-sided Wilson intervals, in (0, 1).
    ridge : float
        Positive relative diagonal regularization after covariance estimation.
    seed : int
        Nonnegative seed for folds and MCD. Does not change the original scores.
    compute_pc : bool
        False performs only vectorized summaries/threshold counts and site joins.
    selected_prior : float, optional
        Explicit prior probability in (0, 1) for a two-Gaussian interpretation.
        Adds log(prior/(1-prior)) to the log density ratio to report model log
        odds. No prior is inferred from the observed low-p-value fraction.
        These model odds are not calibrated biological modification odds.
    engine : {'polars', 'pandas'}
        Engine for numeric grouped statistics. Pandas is the default to retain
        historical numeric reductions. Polars implements the same statistics,
        but floating-point reduction order can differ in the last bits. Inputs
        and outputs remain pandas tables. Both aggregate one column at a time
        in float64 rather than copying all metrics into a float64 matrix.
    return_annotated : bool
        If False, the second return value is None. With ``compute_pc=False``,
        this also avoids all per-read diagnostic output arrays. With PC fitting,
        diagnostic arrays are still needed for exact derived median summaries.

    Returns
    -------
    loci : pandas.DataFrame
        One row per refname/position/strand. Includes observation/valid-p counts;
        count, mean, sample std (ddof=1), median, q10, q90, q95 for available
        distance/PC columns and valid p_value; and threshold frequencies/bounds.
        ``pc_selected_fit_status`` explains skipped/failed fits. ``ok_with_warning``
        retains numerical diagnostics but should be inspected before label use.
        ``pc_selected_center_allfit`` and flattened row-major
        ``pc_selected_covariance_allfit`` ALWAYS use the original pc_1...pc_D
        coordinates, as do the corresponding trace/logdet/condition columns.
        ``pc_selected_to_control_variance_ratio`` = trace(C_control^-1 C_selected)/D;
        ``pc_selected_center_control_mahalanobis_sq_allfit`` measures center
        separation using control covariance. Derived per-read scores receive
        count/mean/std/median summary suffixes, retaining their space/fit labels.
    annotated_reads : pandas.DataFrame or None
        Original index and row order, with metric columns renamed according to
        READ_COLUMN_RENAMES. New PC scores include squared Euclidean distances,
        own-covariance squared Mahalanobis distances, distances to the selected
        center under the CONTROL covariance (a shared metric), and Gaussian log
        densities in original PC coordinates. The stored control distance is
        retained as *_reported; its independently reconstructed value has no
        suffix. *_allfit uses the saved selected center/covariance and covers
        all finite-PC reads. *_crossfit uses a different held-out fit for each
        candidate fold and is NaN for nonselected reads. Log density ratios are
        log f_selected - log f_control in natural logs; positive favors selected.
        No selected prior is assumed unless supplied explicitly.
        ``pc_selected_mahalanobis_percentile_crossfit`` uses (rank-0.5)/N among
        selected reads at THAT locus; larger is more peripheral. Unscored values
        are NaN. Full-embedding distances to a selected center cannot be derived
        from these tables because the full embedding vectors are not present.

    Notes
    -----
    Basic summaries use one locus factorization, numeric aggregations and
    vectorized threshold counts. Only eligible PC clouds use Python loops and
    small D*D covariance fits. Float32 input storage is retained; covariance
    fits and derived diagnostics use float64. PC matrices are promoted only for
    the current locus. Numeric columns, site parameters and categorical read
    labels are cached once; candidate names/PCs are gathered only for eligible
    fits. Control scores and covariance validation still run at sparse loci.
    For runs exceeding RAM, use
    ``iter_pairwise_locus_summaries`` and write each result before advancing.
    Each complete partition must fit in memory, including exact quantiles and
    PC diagnostics; this is not an arbitrary-row streaming reduction.
    Negative reference positions are excluded before grouping and annotation.
    Raw PC spread is comparable only within its own locus.
    A very low control p-value AND high selected-cloud percentile identifies
    a read unusual relative to both references; it does not identify its cause.
    """
    if engine not in {"polars", "pandas"}:
        raise ValueError("engine must be 'polars' or 'pandas'")
    required = LOCUS + ["readname", "p_value"]
    if missing := set(required) - set(reads.columns):
        raise ValueError(f"Missing read columns: {sorted(missing)}")
    # Unmapped/insertion/soft-clip sentinels are not genomic loci.
    reads = reads.loc[reads["position"].ge(0)].copy(deep=False)
    if reads[LOCUS + ["readname"]].isna().any().any():
        raise ValueError("Locus keys and read names must not be null")
    if reads.duplicated(LOCUS + ["readname"]).any():
        raise ValueError("Duplicate read/locus rows; do not combine overlapping partitions or runs")
    thresholds = tuple(float(value) for value in pvalue_thresholds)
    if any(not 0 < value <= 1 for value in thresholds + (cluster_pvalue,)):
        raise ValueError("P-value thresholds must lie in (0, 1]")
    if len({pvalue_prefix(value) for value in thresholds}) != len(thresholds):
        raise ValueError("P-value thresholds must have unique column prefixes")
    if covariance not in {"ledoit_wolf", "mcd"}:
        raise ValueError("covariance must be 'ledoit_wolf' or 'mcd'")
    for name, value, minimum in (("min_cluster_reads", min_cluster_reads, 2), ("crossfit_folds", crossfit_folds, 2), ("seed", seed, 0)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0 < confidence < 1 or not 0 <= min_cluster_fraction <= 1:
        raise ValueError("Invalid confidence or min_cluster_fraction")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if selected_prior is not None and not 0 < selected_prior < 1:
        raise ValueError("selected_prior must lie in (0, 1)")
    if set(READ_COLUMN_RENAMES.values()) & set(reads.columns):
        raise ValueError("Supply the original comparison columns, not previously annotated output")

    # Group integer/category codes directly, avoiding one Python tuple per read
    # from factorizing a full-length MultiIndex. Build a MultiIndex only for loci.
    grouping = reads.groupby(LOCUS, sort=True, observed=True)
    codes = grouping.ngroup().to_numpy(dtype=np.int64)
    unique = grouping.size().index
    loci = unique.to_frame(index=False).reset_index(drop=True)
    del grouping
    ngroups = len(loci)
    total = np.bincount(codes, minlength=ngroups)
    p = reads["p_value"].to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    count_valid = np.bincount(codes[valid], minlength=ngroups)
    loci["n_observations"] = total
    loci["n_valid_p"] = count_valid
    loci["n_invalid_p"] = total - count_valid
    for threshold in thresholds:
        count = np.bincount(codes[valid & (p < threshold)], minlength=ngroups)
        fraction, low, high = _wilson(count, count_valid, confidence)
        prefix = pvalue_prefix(threshold)
        for suffix, value in (("count", count), ("fraction", fraction), ("ci_low", low), ("ci_high", high)):
            loci[f"{prefix}_{suffix}"] = value

    pcs = sorted((col for col in reads if re.fullmatch(r"pc_[1-9][0-9]*", col)), key=lambda col: int(col[3:]))
    if pcs and pcs != [f"pc_{i}" for i in range(1, len(pcs) + 1)]:
        raise ValueError("PC columns must be contiguous from pc_1 through pc_D")
    columns = [col for col in DISTANCES if col in reads] + ["p_value"] + pcs
    basic = _numeric_summaries({col: reads[col] for col in columns}, codes,
                               engine=engine, valid_p=valid, quantiles=True)
    loci = pd.concat([loci, basic], axis=1)

    matched = None
    if site_metrics is not None:
        sites = site_metrics.rename(columns={"chrom": "refname"})
        if sites.columns.duplicated().any():
            raise ValueError("Site table must have chrom or refname, not both")
        if missing := set(LOCUS + ["control_pca_mean", "control_pca_covariance"]) - set(sites):
            raise ValueError(f"Missing site columns: {sorted(missing)}")
        if sites[LOCUS].isna().any().any() or sites.duplicated(LOCUS).any():
            raise ValueError("Site table must have unique non-null locus keys")
        matched = sites.set_index(LOCUS).reindex(unique)
        loci["site_metrics_present"] = unique.isin(pd.MultiIndex.from_frame(sites[LOCUS]))
        for col in sites:
            if col.endswith("coverage"):
                loci[col] = matched[col].to_numpy()
        if "ref_base" in matched:
            loci["ref_base"] = matched["ref_base"].to_numpy()
        if "scored_test_coverage" in matched:
            loci["scored_rows_match_site_table"] = total == matched["scored_test_coverage"].to_numpy()
        if "test_coverage" in matched:
            denominator = matched["test_coverage"].to_numpy(dtype=float)
            loci["scored_fraction_of_test"] = np.divide(total, denominator, out=np.full(ngroups, np.nan), where=denominator > 0)

    selected = valid & (p < cluster_pvalue)
    nselected = np.bincount(codes[selected], minlength=ngroups)
    loci["n_selected"] = nselected
    loci["selected_fraction"] = np.divide(nselected, count_valid, out=np.full(ngroups, np.nan), where=count_valid > 0)
    pc_valid = np.full(len(reads), bool(pcs))
    pc_columns = []
    for col in pcs:
        series = reads[col]
        values = (series.to_numpy(copy=False)
                  if isinstance(series.dtype, np.dtype) and pd.api.types.is_numeric_dtype(series.dtype)
                  else series.to_numpy(dtype=float, na_value=np.nan))
        pc_valid &= np.isfinite(values)
        pc_columns.append(values)
    nselected_pc = np.bincount(codes[selected & pc_valid], minlength=ngroups)
    loci["n_selected_valid_pc"] = nselected_pc
    fit_status = np.full(ngroups, "not_requested" if not compute_pc else "no_pc_columns", dtype=object)
    loci["pc_selected_fit_status"] = fit_status
    loci["pc_selected_estimation_space"] = "control_whitened_pc" if site_metrics is not None else "raw_pc"
    fit_parameters = {}
    for col in ("pc_selected_center_allfit", "pc_selected_covariance_allfit"):
        fit_parameters[col] = np.full(ngroups, None, dtype=object)
        loci[col] = fit_parameters[col]
    for col in ("pc_selected_cov_trace_allfit", "pc_selected_cov_logdet_allfit",
                "pc_selected_cov_condition_allfit", "pc_selected_to_control_variance_ratio",
                "pc_selected_center_control_mahalanobis_sq_allfit"):
        fit_parameters[col] = np.full(ngroups, np.nan)
        loci[col] = fit_parameters[col]

    # A broadcast NaN is read-only and occupies only one scalar when no fits
    # will run. Materialize outputs only if annotations were requested.
    fit_pc = compute_pc and bool(pcs)
    def empty_scores():
        return np.full(len(reads), np.nan) if fit_pc or return_annotated else np.broadcast_to(np.nan, (len(reads),))

    per_read = {name: empty_scores() for name in (
        "pc_l2_sq_to_control", "pc_mahalanobis_sq_to_control", "pc_logpdf_control",
        "pc_selected_mahalanobis_percentile_crossfit",
    )}
    for fit_kind in ("allfit", "crossfit"):
        for name in ("pc_l2_sq_to_selected", "pc_mahalanobis_sq_to_selected",
                     "pc_control_cov_mahalanobis_sq_to_selected", "pc_logpdf_selected",
                     "pc_log_density_ratio_selected_vs_control"):
            per_read[f"{name}_{fit_kind}"] = empty_scores()
        if selected_prior is not None:
            per_read[f"pc_log_odds_selected_vs_control_{fit_kind}"] = empty_scores()

    if compute_pc and pcs:
        d = len(pcs)
        order = np.argsort(codes, kind="stable")
        boundaries = np.r_[0, np.cumsum(total)]
        required_n = max(min_cluster_reads, 2 * d + 4)
        enough = (nselected_pc >= required_n) & (nselected_pc - np.ceil(nselected_pc / crossfit_folds) >= d + 2)
        selected_fractions = loci["selected_fraction"].to_numpy()
        read_names = reads["readname"].array
        if enough.any() and isinstance(reads["readname"].dtype, pd.CategoricalDtype):
            # Converting a categorical slice to strings repeatedly converts its
            # entire category dictionary, even for a handful of selected reads.
            name_labels = reads["readname"].cat.categories.astype(str).to_numpy()
            name_codes = reads["readname"].cat.codes.to_numpy()
        else:
            name_labels = name_codes = None
        if matched is not None:
            site_present = loci["site_metrics_present"].to_numpy()
            control_means = matched["control_pca_mean"].to_numpy()
            control_covariances = matched["control_pca_covariance"].to_numpy()
        for gid in range(ngroups):
            group_idx = order[boundaries[gid]:boundaries[gid + 1]]
            all_idx = group_idx[pc_valid[group_idx]]
            control_mean = np.zeros(d)
            control_cov = None
            control_chol = np.eye(d)
            if matched is not None:
                if not site_present[gid]:
                    fit_status[gid] = "missing_control_statistics"
                    continue
                control_mean = np.asarray(control_means[gid], dtype=float)
                control_cov = np.asarray(control_covariances[gid], dtype=float)
                if (control_mean.shape != (d,) or control_cov.size != d * d
                        or not np.isfinite(control_mean).all() or not np.isfinite(control_cov).all()):
                    raise ValueError(f"Invalid control PCA dimensions/values at {unique[gid]}")
                control_cov = control_cov.reshape(d, d)
                if not np.allclose(control_cov, control_cov.T, rtol=1e-5, atol=1e-8):
                    raise ValueError(f"Non-symmetric control covariance at {unique[gid]}")
                try:
                    control_chol = np.linalg.cholesky(control_cov)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f"Non-positive-definite control covariance at {unique[gid]}") from exc
                all_pc = np.column_stack([column[all_idx] for column in pc_columns]).astype(float, copy=False)
                control_scores = _score_pc_gaussian(all_pc, control_mean, control_cov)
                for key, column in (("l2_sq", "pc_l2_sq_to_control"),
                                    ("mahalanobis_sq", "pc_mahalanobis_sq_to_control"),
                                    ("logpdf", "pc_logpdf_control")):
                    per_read[column][all_idx] = control_scores[key]

            if not enough[gid]:
                fit_status[gid] = "too_few_selected"
                continue
            if selected_fractions[gid] < min_cluster_fraction:
                fit_status[gid] = "below_min_fraction"
                continue
            # Prepare candidate identities/PCs only for eligible fits. Read order
            # within each locus and the float64 arithmetic remain unchanged.
            idx = all_idx[selected[all_idx]]
            names = name_labels[name_codes[idx]] if name_labels is not None else np.asarray(read_names.take(idx), dtype=str)
            idx = idx[np.argsort(names, kind="stable")]
            n = len(idx)
            selected_pc = np.column_stack([column[idx] for column in pc_columns]).astype(float, copy=False)
            if matched is None:
                all_pc = np.column_stack([column[all_idx] for column in pc_columns]).astype(float, copy=False)
            # Fit in control-whitened coordinates for stable shrinkage, then
            # map EVERY scored Gaussian back into the original PC coordinates.
            # This ensures comparable log densities (including the Jacobian)
            # and stores parameters that reproduce the allfit distances exactly.
            x = np.linalg.solve(control_chol, (selected_pc - control_mean).T).T
            try:
                center, scatter, warned = _fit_gaussian(x, covariance, ridge, seed)
                raw_center = control_mean + control_chol @ center
                raw_scatter = control_chol @ scatter @ control_chol.T
                allfit = _score_pc_gaussian(all_pc, raw_center, raw_scatter, control_cov)
                folds = np.empty(n, dtype=int)
                folds[np.random.default_rng(seed).permutation(n)] = np.arange(n) % min(crossfit_folds, n)
                crossfit = {name: np.empty(n) for name in allfit}
                for fold in range(min(crossfit_folds, n)):
                    heldout = folds == fold
                    mu, sigma, fold_warned = _fit_gaussian(x[~heldout], covariance, ridge, seed)
                    warned |= fold_warned
                    fold_scores = _score_pc_gaussian(
                        selected_pc[heldout], control_mean + control_chol @ mu,
                        control_chol @ sigma @ control_chol.T, control_cov,
                    )
                    for name, values in fold_scores.items():
                        crossfit[name][heldout] = values
                if any(not np.isfinite(value).all() for result in (allfit, crossfit) for value in result.values()):
                    raise ValueError("nonfinite Gaussian scores")
            except (ValueError, np.linalg.LinAlgError) as exc:
                fit_status[gid] = f"fit_failed: {exc}"
                continue

            for kind, row_idx, result in (("allfit", all_idx, allfit), ("crossfit", idx, crossfit)):
                for key, column in (("l2_sq", "pc_l2_sq_to_selected"),
                                    ("mahalanobis_sq", "pc_mahalanobis_sq_to_selected"),
                                    ("control_cov_mahalanobis_sq", "pc_control_cov_mahalanobis_sq_to_selected"),
                                    ("logpdf", "pc_logpdf_selected")):
                    if key in result:
                        per_read[f"{column}_{kind}"][row_idx] = result[key]
                if control_cov is not None:
                    ratio = result["logpdf"] - per_read["pc_logpdf_control"][row_idx]
                    per_read[f"pc_log_density_ratio_selected_vs_control_{kind}"][row_idx] = ratio
                    if selected_prior is not None:
                        per_read[f"pc_log_odds_selected_vs_control_{kind}"][row_idx] = (
                            ratio + np.log(selected_prior) - np.log1p(-selected_prior)
                        )
            per_read["pc_selected_mahalanobis_percentile_crossfit"][idx] = (
                rankdata(crossfit["mahalanobis_sq"], method="average") - 0.5
            ) / n
            eigenvalues = np.linalg.eigvalsh(raw_scatter)
            fit_status[gid] = "ok_with_warning" if warned else "ok"
            fit_parameters["pc_selected_center_allfit"][gid] = raw_center.tolist()
            fit_parameters["pc_selected_covariance_allfit"][gid] = raw_scatter.ravel().tolist()
            fit_parameters["pc_selected_cov_trace_allfit"][gid] = np.trace(raw_scatter)
            fit_parameters["pc_selected_cov_logdet_allfit"][gid] = np.linalg.slogdet(raw_scatter)[1]
            fit_parameters["pc_selected_cov_condition_allfit"][gid] = eigenvalues[-1] / eigenvalues[0]
            if control_cov is not None:
                fit_parameters["pc_selected_to_control_variance_ratio"][gid] = np.trace(scatter) / d
                fit_parameters["pc_selected_center_control_mahalanobis_sq_allfit"][gid] = center @ center

    loci["pc_selected_fit_status"] = fit_status
    for col, values in fit_parameters.items():
        loci[col] = values

    annotated = None
    if return_annotated:
        # Construct column views once instead of copying every diagnostic array
        # during repeated DataFrame assignments. Preserve input index and attrs.
        annotated_columns = {READ_COLUMN_RENAMES.get(col, col): reads[col].array for col in reads}
        annotated_columns.update(selected_by_pc_control_pvalue=selected, **per_read)
        annotated = pd.DataFrame(annotated_columns, index=reads.index, copy=False)
        annotated.attrs = reads.attrs.copy()
    # Summaries of derived distances retain the exact per-read column name.
    # The count distinguishes all-read allfit summaries from selected-only
    # crossfit summaries; zero eligible observations produce count=0 and NaN.
    if fit_pc:
        derived = _numeric_summaries(per_read, codes, engine=engine)
    else:
        derived = pd.DataFrame({f"{col}_{stat}": np.zeros(ngroups, dtype=np.int64) if stat == "count" else np.full(ngroups, np.nan)
                                for col in per_read for stat in ("count", "mean", "std", "median")})
    loci = pd.concat([loci, derived.reset_index(drop=True)], axis=1)
    settings = dict(pvalue_thresholds=thresholds, cluster_pvalue=cluster_pvalue,
                    covariance=covariance, crossfit_folds=crossfit_folds, confidence=confidence,
                    min_cluster_reads=min_cluster_reads, min_cluster_fraction=min_cluster_fraction,
                    ridge=ridge, seed=seed, selected_prior=selected_prior, output_schema_version=2)
    loci.attrs["pairwise_locus_settings"] = settings
    if annotated is not None:
        annotated.attrs["pairwise_locus_settings"] = settings
    return loci, annotated


def iter_pairwise_locus_summaries(run_dir, *, load_options=None, **summary_options):
    """Yield ``(stem, loci, annotated_or_none)`` for each complete partition.

    Load and summarize one partition at a time. ``load_options`` is a dict of
    ``load_pairwise_partition`` options; other keywords go to
    ``summarize_pairwise_loci``. ``return_annotated`` defaults to False here.
    Write/consume each result before advancing to avoid accumulating read tables.

    Loci must not occur in multiple partitions: exact medians, PCA fits, and
    crossfit scores cannot be combined from partial-locus results. Seen locus
    keys are retained (O(number of loci)); an overlap raises before yielding
    that partition. Earlier yielded results must be discarded if a later
    partition fails validation. Only combine partitions of ONE comparison run.
    Memory scales with the largest partition plus the seen-key set, not all
    read rows. Extremely large individual partitions still need sufficient RAM.
    """
    summary_options.setdefault("return_annotated", False)
    seen = set()
    for stem, reads, sites in iter_pairwise_directory(run_dir, **(load_options or {})):
        keys = set(reads.loc[reads['position'].ge(0), LOCUS].drop_duplicates().itertuples(index=False, name=None))
        if seen.intersection(keys):
            raise ValueError(f"Loci overlap across partitions at {stem}; do not split or pool locus fits")
        seen.update(keys)
        loci, annotated = summarize_pairwise_loci(reads, sites, **summary_options)
        del reads, sites, keys
        yield stem, loci, annotated
        del loci, annotated



def add_reference_base(
    df,
    fasta_path,
    ref_col=None,
    pos_col="position",
    strand_col="strand",
    out_col="ref_base",
):
    """
    Add the strand-aware reference base to each row of a DataFrame.

    Expected columns
    ----------------
    refname   : reference/contig/chromosome name matching the FASTA
    position  : 0-based reference position
    strand    : 0 = forward, 1 = reverse

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    fasta_path : str or pathlib.Path
        Path to FASTA file.

    ref_col : str, optional
        Column containing FASTA reference names. Defaults to chrom for the
        shared output API, or refname for the existing pairwise API.

    pos_col : str, default "position"
        Column containing 0-based reference positions.

    strand_col : str, default "strand"
        Column containing strand:
            0 = forward
            1 = reverse

    out_col : str, default "ref_base"
        Name of the output column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with `out_col` added.

    Notes
    -----
    - One interval spanning the requested positions is fetched per reference.
    - Position is assumed to be 0-based.
    - Reverse-strand positions are NOT coordinate-reversed.
      The base at that reference coordinate is simply complemented.
    """

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    if ref_col is None:
        ref_col = 'chrom' if 'chrom' in df.columns else 'refname'
    required = {ref_col, pos_col, strand_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    # Empty dataframe
    if result.empty:
        result[out_col] = pd.Series(dtype="string")
        return result

    # ------------------------------------------------------------------
    # Validate positions
    # ------------------------------------------------------------------
    if result[pos_col].isna().any():
        raise ValueError(f"{pos_col!r} contains missing values.")

    positions_all = result[pos_col].to_numpy()

    if not np.issubdtype(positions_all.dtype, np.integer):
        # Allow things like 10.0 only if they are exact integers
        numeric = pd.to_numeric(result[pos_col], errors="raise")

        if not np.all(numeric == np.floor(numeric)):
            raise ValueError(
                f"{pos_col!r} must contain integer 0-based positions."
            )

        result[pos_col] = numeric.astype(np.int64)

    if (result[pos_col] < 0).any():
        raise ValueError(
            f"{pos_col!r} contains negative positions."
        )

    # ------------------------------------------------------------------
    # Validate strand
    # ------------------------------------------------------------------
    if result[strand_col].isna().any():
        raise ValueError(f"{strand_col!r} contains missing values.")

    invalid_strand = ~result[strand_col].isin([0, 1])

    if invalid_strand.any():
        bad = result.loc[invalid_strand, strand_col].unique()
        raise ValueError(
            f"{strand_col!r} must contain only 0 (forward) or "
            f"1 (reverse). Found: {bad[:10]}"
        )

    # ------------------------------------------------------------------
    # Create FASTA index if necessary
    # ------------------------------------------------------------------
    fasta_path = os.fspath(fasta_path)
    fai_path = fasta_path + ".fai"

    if not os.path.exists(fai_path):
        pysam.faidx(fasta_path)

    # ------------------------------------------------------------------
    # Byte-level complement lookup table
    #
    # This avoids Python loops over individual bases.
    # ------------------------------------------------------------------
    complement = np.arange(256, dtype=np.uint8)

    complement[ord("A")] = ord("T")
    complement[ord("C")] = ord("G")
    complement[ord("G")] = ord("C")
    complement[ord("T")] = ord("A")
    complement[ord("N")] = ord("N")

    complement[ord("a")] = ord("t")
    complement[ord("c")] = ord("g")
    complement[ord("g")] = ord("c")
    complement[ord("t")] = ord("a")
    complement[ord("n")] = ord("n")

    # Preallocate output
    output = np.empty(len(result), dtype="U1")

    # ------------------------------------------------------------------
    # Open FASTA ONCE
    # ------------------------------------------------------------------
    with pysam.FastaFile(fasta_path) as fasta:

        fasta_lengths = dict(
            zip(fasta.references, fasta.lengths)
        )

        # groupby.indices returns integer row positions
        groups = result.groupby(
            ref_col,
            sort=False,
            observed=True,
        ).indices

        # --------------------------------------------------------------
        # Process one reference at a time
        # --------------------------------------------------------------
        for refname, row_idx in groups.items():

            if pd.isna(refname):
                raise ValueError(
                    f"{ref_col!r} contains missing values."
                )

            if refname not in fasta_lengths:
                raise ValueError(
                    f"Reference {refname!r} not found in FASTA."
                )

            ref_length = fasta_lengths[refname]

            positions = result.iloc[row_idx][
                pos_col
            ].to_numpy(dtype=np.int64, copy=False)

            # ----------------------------------------------------------
            # Bounds checking
            # ----------------------------------------------------------
            invalid = positions >= ref_length

            if invalid.any():
                bad_positions = positions[invalid][:10]

                raise ValueError(
                    f"Position(s) outside reference {refname!r}. "
                    f"Reference length = {ref_length}; "
                    f"invalid positions include "
                    f"{bad_positions.tolist()}"
                )

            # ----------------------------------------------------------
            # Fetch only the interval needed by this partition/site table.
            # ----------------------------------------------------------
            start = int(positions.min())
            seq = fasta.fetch(refname, start, int(positions.max()) + 1).encode("ascii")

            # Zero-copy byte representation of FASTA sequence
            seq_array = np.frombuffer(seq, dtype=np.uint8)

            # ----------------------------------------------------------
            # Vectorized lookup of all requested bases
            # ----------------------------------------------------------
            bases = seq_array[positions - start].copy()

            # ----------------------------------------------------------
            # Reverse strand:
            # 0 = forward
            # 1 = reverse
            # ----------------------------------------------------------
            strands = result.iloc[row_idx][
                strand_col
            ].to_numpy(dtype=np.uint8, copy=False)

            reverse_mask = strands == 1

            # Complement only reverse-strand rows
            bases[reverse_mask] = complement[bases[reverse_mask]]

            # Convert ASCII byte values -> one-character strings
            output[row_idx] = bases.view("S1").astype("U1")

    # Add output column
    result[out_col] = output

    return result
