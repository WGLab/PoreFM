"""Fit per-k-mer prototype classifiers and score base-aligned embeddings.

The workflow routes each query to the classifier family matching its k-mer's
canonical center base.  It supports centroid, diagonal-Gaussian, and locally
shrunk LDA scorers, with shared extraction and output handling.
"""

from porefm.preprocessing import validate_artifact_preprocessing
from porefm.cli.preprocessing_args import add_preprocessing_arguments, resolve_cli_preprocessing
from porefm.cli.chunking_args import add_chunking_arguments, resolve_chunking_mode
import datetime
import math
import os
import sys
import time

from porefm.cli._parser import PoreFMArgumentParser, PoreFMHelpFormatter
from porefm.cli.model_args import add_model_arguments, resolve_model_arguments
from porefm.cli.dual_stream_args import add_exclude_bed_argument
import porefm.workflows.prototypes.utils as query_pipeline
from porefm.common.chunking import resolve_downstream_step
from porefm.common.model_bundle import apply_bundle_arguments, use_model_bundle
from porefm.workflows.prototypes.embedding_selection import EMBEDDING_KEY_CHOICES, select_base_embedding


np = None
pa = None
pq = None
torch = None
DataLoader = None


SCORER_TYPES = (
    'nearest-prototype',
    'centroid-l2',
    'centroid-cosine',
    'centroid-rbf',
    'diagonal-gaussian',
    'per-kmer-lda',
)


def lazy_import_runtime_dependencies():
    """Initialize shared tensor, parquet, and data-loader dependencies lazily."""
    global np, pa, pq, torch, DataLoader

    if torch is not None:
        return

    query_pipeline.lazy_import_runtime_dependencies()
    np = query_pipeline.np
    pa = query_pipeline.pa
    pq = query_pipeline.pq
    torch = query_pipeline.torch
    DataLoader = query_pipeline.DataLoader


class DefaultsFormatter(PoreFMHelpFormatter):
    """Argument formatter that shows defaults for otherwise blank help text."""


def validate_class_inputs(reservoir_paths, class_labels):
    """Validate the one-to-one paths and unique labels for at least two classes."""
    if len(reservoir_paths) < 2:
        raise ValueError("Prototype classification requires at least two classes")
    if len(reservoir_paths) != len(class_labels):
        raise ValueError(
            "--prototype_embeddings and --class_labels must contain the same "
            f"number of values, got {len(reservoir_paths)} and {len(class_labels)}"
        )
    if any(not label.strip() for label in class_labels):
        raise ValueError("--class_labels values must be non-empty")
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("--class_labels values must be unique")


def normalize_class_priors(class_priors, num_classes):
    """Validate and normalize class priors, defaulting to a uniform vector."""
    if class_priors is None:
        return [1.0 / num_classes] * num_classes
    if len(class_priors) != num_classes:
        raise ValueError(
            f"--class_priors requires {num_classes} values, got {len(class_priors)}"
        )
    if any((not math.isfinite(value)) or value <= 0 for value in class_priors):
        raise ValueError("--class_priors values must be finite and strictly positive")
    total = sum(class_priors)
    return [value / total for value in class_priors]


def validate_nearest_options(metric, class_priors, num_classes):
    """Require an unweighted cosine/L2 distance with uniform class priors."""
    if metric not in ('cosine', 'l2'):
        raise ValueError("nearest-prototype metric must be cosine or l2")
    priors = normalize_class_priors(class_priors, num_classes)
    if any(value != priors[0] for value in priors):
        raise ValueError("nearest-prototype requires uniform --class_priors")


def validate_prototype_payload(
    payload,
    path,
    class_label,
    *,
    require_reservoir=False,
):
    """Validate one prototype payload and return normalized summary tensors.

    Every payload must contain aligned per-k-mer IDs, exact means, and source
    observation counts.  Reservoir-based scorers additionally require sampled
    embeddings and consistent prefix offsets.  Passing ``require_reservoir``
    rejects mean-only prototype files.
    """
    required_keys = {
        'kmer_ids',
        'counts',
        'mean_embeddings',
    }
    reservoir_keys = {
        'reservoir_embeddings',
        'reservoir_offsets',
        'reservoir_counts',
    }
    if require_reservoir:
        required_keys.update(reservoir_keys)
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(
            f"Prototype {class_label!r} in {path} is missing keys: {', '.join(missing)}"
        )
    present_reservoir_keys = reservoir_keys.intersection(payload)
    if present_reservoir_keys and present_reservoir_keys != reservoir_keys:
        missing_reservoir = sorted(reservoir_keys.difference(payload))
        raise ValueError(
            f"Prototype {class_label!r} in {path} has an incomplete reservoir; "
            f"missing keys: {', '.join(missing_reservoir)}"
        )

    kmer_ids = payload['kmer_ids'].long().reshape(-1)
    counts = payload['counts'].long().reshape(-1)
    means = payload['mean_embeddings'].float()
    metadata = payload.get('metadata', {})
    kmer_size = int(metadata.get('kmer_size', 9))

    if means.ndim != 2:
        raise ValueError(
            f"Prototype {class_label!r} mean_embeddings must be two-dimensional, "
            f"got shape {tuple(means.shape)}"
        )
    if len(kmer_ids) != len(counts) or len(kmer_ids) != len(means):
        raise ValueError(
            f"Prototype {class_label!r} has inconsistent kmer_ids, counts, and "
            "mean_embeddings lengths"
        )
    if len(torch.unique(kmer_ids)) != len(kmer_ids):
        raise ValueError(f"Prototype {class_label!r} contains duplicate k-mer IDs")
    if (counts <= 0).any():
        raise ValueError(f"Prototype {class_label!r} contains a non-positive count")
    if not torch.isfinite(means).all():
        raise ValueError(f"Prototype {class_label!r} contains non-finite means")

    num_kmers = 4 ** kmer_size
    if ((kmer_ids < 0) | (kmer_ids >= num_kmers)).any():
        raise ValueError(
            f"Prototype {class_label!r} contains a k-mer ID outside [0, {num_kmers}) "
            f"for kmer_size={kmer_size}"
        )

    recorded_dim = metadata.get('embedding_dim')
    if recorded_dim is not None and int(recorded_dim) != means.shape[1]:
        raise ValueError(
            f"Prototype {class_label!r} records embedding_dim={recorded_dim}, "
            f"but its embeddings have dimension {means.shape[1]}"
        )

    reservoir_counts = None
    if present_reservoir_keys:
        samples = payload['reservoir_embeddings']
        offsets = payload['reservoir_offsets'].long().reshape(-1)
        reservoir_counts = payload['reservoir_counts'].long().reshape(-1)
        if samples.ndim != 2:
            raise ValueError(
                f"Prototype {class_label!r} reservoir_embeddings must be "
                f"two-dimensional, got shape {tuple(samples.shape)}"
            )
        if len(reservoir_counts) != len(kmer_ids) or len(offsets) != len(kmer_ids) + 1:
            raise ValueError(
                f"Prototype {class_label!r} has inconsistent reservoir offsets/counts"
            )
        if samples.shape[1] != means.shape[1]:
            raise ValueError(
                f"Prototype {class_label!r} sample dimension {samples.shape[1]} "
                f"does not match centroid dimension {means.shape[1]}"
            )
        if len(offsets) == 0 or offsets[0].item() != 0:
            raise ValueError(f"Prototype {class_label!r} offsets must start at zero")
        if (offsets[1:] < offsets[:-1]).any():
            raise ValueError(
                f"Prototype {class_label!r} offsets must be non-decreasing"
            )
        if offsets[-1].item() != len(samples):
            raise ValueError(
                f"Prototype {class_label!r} final offset {offsets[-1].item()} "
                f"does not match {len(samples)} sampled embeddings"
            )
        if not torch.equal(offsets[1:] - offsets[:-1], reservoir_counts):
            raise ValueError(
                f"Prototype {class_label!r} reservoir counts do not match offsets"
            )
        if (reservoir_counts < 0).any() or (reservoir_counts > counts).any():
            raise ValueError(
                f"Prototype {class_label!r} has an invalid reservoir count"
            )
        if not torch.isfinite(samples).all():
            raise ValueError(
                f"Prototype {class_label!r} contains non-finite reservoir embeddings"
            )

    return {
        'kmer_ids': kmer_ids,
        'counts': counts,
        'mean_embeddings': means,
        'reservoir_counts': reservoir_counts,
        'metadata': metadata,
        'kmer_size': kmer_size,
        'embedding_dim': means.shape[1],
    }


def validate_reservoir_payload(payload, path, class_label):
    """Validate a prototype payload that must include reservoir samples."""
    return validate_prototype_payload(
        payload,
        path,
        class_label,
        require_reservoir=True,
    )


def load_reservoir_centroids(
    reservoir_paths,
    class_labels,
    min_reservoir_count=2,
    requested_embedding_key=None,
    *,
    require_reservoir=True,
    min_prototype_count=1,
):
    """Load exact class centroids on their eligible shared k-mer intersection.

    Returns a mapping containing ``(classes, common_kmers, embedding_dim)``
    centroids, source/sample counts, shared k-mer IDs, embedding metadata, and
    per-class summaries. Reservoir-free centroid scorers set
    ``require_reservoir=False`` and filter by exact source count instead.
    """
    validate_class_inputs(reservoir_paths, class_labels)
    if min_reservoir_count < 1:
        raise ValueError(
            f"min_reservoir_count must be at least one, got {min_reservoir_count}"
        )
    if min_prototype_count < 1:
        raise ValueError(
            f"min_prototype_count must be at least one, got {min_prototype_count}"
        )

    summaries = []
    expected_kmer_size = None
    expected_dim = None
    for path, class_label in zip(reservoir_paths, class_labels):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        summary = validate_prototype_payload(
            payload,
            path,
            class_label,
            require_reservoir=require_reservoir,
        )
        del payload

        if expected_kmer_size is None:
            expected_kmer_size = summary['kmer_size']
            expected_dim = summary['embedding_dim']
        elif summary['kmer_size'] != expected_kmer_size:
            raise ValueError(
                f"Prototype {class_label!r} uses kmer_size={summary['kmer_size']}, "
                f"expected {expected_kmer_size}"
            )
        elif summary['embedding_dim'] != expected_dim:
            raise ValueError(
                f"Prototype {class_label!r} uses embedding dimension "
                f"{summary['embedding_dim']}, expected {expected_dim}"
            )
        summaries.append(summary)

    metadata_records = [
        {'name': label, 'metadata': summary['metadata']}
        for label, summary in zip(class_labels, summaries)
    ]
    embedding_key = query_pipeline.resolve_prototype_embedding_key(
        metadata_records,
        requested_key=requested_embedding_key,
    )

    num_kmers = 4 ** expected_kmer_size
    available_masks = []
    for summary in summaries:
        available = torch.zeros(num_kmers, dtype=torch.bool)
        eligible = summary['counts'] >= min_prototype_count
        if require_reservoir:
            eligible &= summary['reservoir_counts'] >= min_reservoir_count
        available[summary['kmer_ids'][eligible]] = True
        available_masks.append(available)

    common_available = torch.stack(available_masks).all(dim=0)
    common_kmer_ids = common_available.nonzero(as_tuple=True)[0]
    if len(common_kmer_ids) == 0:
        raise ValueError(
            "Prototype files do not share an eligible k-mer in every class"
        )

    centroids = torch.empty(
        (len(class_labels), len(common_kmer_ids), expected_dim),
        dtype=torch.float32,
    )
    source_counts = torch.empty(
        (len(class_labels), len(common_kmer_ids)),
        dtype=torch.long,
    )
    sampled_counts = torch.zeros_like(source_counts)
    for class_index, summary in enumerate(summaries):
        source_index = torch.full((num_kmers,), -1, dtype=torch.long)
        source_index[summary['kmer_ids']] = torch.arange(len(summary['kmer_ids']))
        selected = source_index[common_kmer_ids]
        centroids[class_index] = summary['mean_embeddings'][selected]
        source_counts[class_index] = summary['counts'][selected]
        if summary['reservoir_counts'] is not None:
            sampled_counts[class_index] = summary['reservoir_counts'][selected]

    class_summaries = []
    for path, class_label, summary, available in zip(
        reservoir_paths,
        class_labels,
        summaries,
        available_masks,
    ):
        class_summaries.append({
            'class_label': class_label,
            'path': str(path),
            'stored_kmer_count': len(summary['kmer_ids']),
            'eligible_kmer_count': int(available.sum().item()),
            'common_kmer_count': len(common_kmer_ids),
            'metadata': summary['metadata'],
        })

    return {
        'class_labels': list(class_labels),
        'common_kmer_ids': common_kmer_ids,
        'centroids': centroids,
        'source_counts': source_counts,
        'sampled_counts': sampled_counts,
        'kmer_size': expected_kmer_size,
        'embedding_dim': expected_dim,
        'embedding_key': embedding_key,
        'class_summaries': class_summaries,
    }


class PooledCovarianceAccumulator:
    """Accumulate covariance and Ledoit-Wolf moments without storing residuals."""

    def __init__(self, embedding_dim, device='cpu'):
        self.embedding_dim = int(embedding_dim)
        self.device = torch.device(device)
        self.sum_outer = torch.zeros(
            (self.embedding_dim, self.embedding_dim),
            dtype=torch.float64,
            device=self.device,
        )
        self.sum_norm_four = torch.zeros((), dtype=torch.float64, device=self.device)
        self.num_samples = 0

    def update(self, residuals):
        """Add an ``(n_samples, embedding_dim)`` residual batch to the moments."""
        if residuals.numel() == 0:
            return
        if residuals.ndim != 2 or residuals.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected residuals with shape (N, {self.embedding_dim}), got "
                f"{tuple(residuals.shape)}"
            )
        residuals = residuals.to(self.device, dtype=torch.float32, non_blocking=True)
        self.sum_outer.add_((residuals.T @ residuals).to(torch.float64))
        squared_norm = residuals.square().sum(dim=1)
        self.sum_norm_four.add_(squared_norm.square().sum(dtype=torch.float64))
        self.num_samples += len(residuals)

    def finalize(self, regularization='ledoit-wolf', ridge=1e-4):
        """Estimate a regularized pooled covariance without retaining samples.

        ``regularization`` selects Ledoit-Wolf shrinkage toward scaled identity
        or a fixed diagonal ``ridge``. The returned tuple contains regularized
        and empirical covariance matrices, the applied scalar, and sample count.
        """
        if self.num_samples == 0:
            raise ValueError("Cannot estimate covariance without residual samples")
        empirical = self.sum_outer / self.num_samples
        identity = torch.eye(
            self.embedding_dim,
            dtype=torch.float64,
            device=self.device,
        )

        if regularization == 'ledoit-wolf':
            mean_variance = torch.trace(empirical) / self.embedding_dim
            delta = (empirical - mean_variance * identity).square().sum()
            empirical_norm = empirical.square().sum()
            beta = (
                self.sum_norm_four / self.num_samples - empirical_norm
            ) / self.num_samples
            beta = torch.clamp(beta, min=0.0)
            if delta.item() == 0.0:
                shrinkage = torch.ones((), dtype=torch.float64, device=self.device)
            else:
                shrinkage = torch.clamp(beta / delta, min=0.0, max=1.0)
            covariance = (
                (1.0 - shrinkage) * empirical
                + shrinkage * mean_variance * identity
            )
            regularization_value = float(shrinkage.item())
        elif regularization == 'ridge':
            if not math.isfinite(ridge) or ridge < 0:
                raise ValueError(f"ridge must be finite and non-negative, got {ridge}")
            covariance = empirical + float(ridge) * identity
            regularization_value = float(ridge)
        else:
            raise ValueError(f"Unknown covariance regularization: {regularization}")

        return (
            covariance.cpu(),
            empirical.cpu(),
            regularization_value,
            self.num_samples,
        )


class PerKmerCovarianceAccumulator:
    """Accumulate within-class covariance moments separately for each k-mer."""

    def __init__(self, num_kmers, embedding_dim, device='cpu'):
        self.num_kmers = int(num_kmers)
        self.embedding_dim = int(embedding_dim)
        self.device = torch.device(device)
        self.sum_outer = torch.zeros(
            (self.num_kmers, self.embedding_dim, self.embedding_dim),
            dtype=torch.float64,
            device=self.device,
        )
        self.sum_norm_four = torch.zeros(
            self.num_kmers,
            dtype=torch.float64,
            device=self.device,
        )
        self.sample_counts = torch.zeros(
            self.num_kmers,
            dtype=torch.long,
            device=self.device,
        )

    def update(self, kmer_index, residuals):
        """Add residual samples to one dense common-k-mer index."""
        if residuals.numel() == 0:
            return
        if not 0 <= int(kmer_index) < self.num_kmers:
            raise ValueError(f"kmer_index out of range: {kmer_index}")
        if residuals.ndim != 2 or residuals.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected residuals with shape (N, {self.embedding_dim}), got "
                f"{tuple(residuals.shape)}"
            )

        residuals = residuals.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        self.sum_outer[kmer_index].add_(
            (residuals.T @ residuals).to(torch.float64)
        )
        squared_norm = residuals.square().sum(dim=1)
        self.sum_norm_four[kmer_index].add_(
            squared_norm.square().sum(dtype=torch.float64)
        )
        self.sample_counts[kmer_index] += len(residuals)

    def finalize(
        self,
        global_regularization='ledoit-wolf',
        ridge=1e-4,
        local_shrinkage=None,
    ):
        """Estimate global and per-k-mer within-class covariance matrices.

        Local empirical covariances are shrunk toward the regularized global
        covariance. Shrinkage is moment-estimated per k-mer unless an explicit
        value in ``[0, 1]`` is supplied.
        """
        if (self.sample_counts == 0).any():
            missing = (self.sample_counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                "Cannot estimate local covariance without residual samples for "
                f"k-mer indices: {missing[:10]}"
            )

        global_accumulator = PooledCovarianceAccumulator(
            self.embedding_dim,
            device=self.device,
        )
        global_accumulator.sum_outer.copy_(self.sum_outer.sum(dim=0))
        global_accumulator.sum_norm_four.copy_(self.sum_norm_four.sum())
        global_accumulator.num_samples = int(self.sample_counts.sum().item())
        (
            global_covariance_cpu,
            global_empirical_cpu,
            global_regularization_value,
            global_sample_count,
        ) = global_accumulator.finalize(
            regularization=global_regularization,
            ridge=ridge,
        )
        global_covariance = global_covariance_cpu.to(self.device)
        global_empirical = global_empirical_cpu.to(self.device)

        sample_counts_float = self.sample_counts.to(torch.float64)
        local_empirical = self.sum_outer / sample_counts_float[:, None, None]
        if local_shrinkage is None:
            local_empirical_norm = local_empirical.square().sum(dim=(-2, -1))
            beta = (
                self.sum_norm_four / sample_counts_float
                - local_empirical_norm
            ) / sample_counts_float
            beta = torch.clamp(beta, min=0.0)
            delta = (
                local_empirical - global_covariance.unsqueeze(0)
            ).square().sum(dim=(-2, -1))
            shrinkage_values = torch.where(
                delta > 0,
                torch.clamp(beta / delta, min=0.0, max=1.0),
                torch.ones_like(delta),
            )
        else:
            if not math.isfinite(local_shrinkage) or not 0 <= local_shrinkage <= 1:
                raise ValueError(
                    "local_shrinkage must be between zero and one, got "
                    f"{local_shrinkage}"
                )
            shrinkage_values = torch.full(
                (self.num_kmers,),
                float(local_shrinkage),
                dtype=torch.float64,
                device=self.device,
            )

        local_covariances = (
            (1.0 - shrinkage_values[:, None, None]) * local_empirical
            + shrinkage_values[:, None, None] * global_covariance.unsqueeze(0)
        )
        return {
            'global_covariance': global_covariance,
            'global_empirical_covariance': global_empirical,
            'global_regularization_value': global_regularization_value,
            'global_sample_count': global_sample_count,
            'local_empirical_covariances': local_empirical,
            'local_covariances': local_covariances,
            'local_shrinkage_values': shrinkage_values,
            'local_sample_counts': self.sample_counts,
        }


def resolve_fit_device(requested_device):
    """Resolve ``auto`` for covariance fitting and reject unavailable CUDA."""
    if requested_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA fit device requested but unavailable: {requested_device}")
    return str(device)


def accumulate_reservoir_residuals(
    accumulator,
    reservoir_paths,
    class_labels,
    common_kmer_ids,
    centroids,
    kmer_size,
    covariance_batch_size,
):
    """Stream class-centered reservoir residuals into a per-k-mer accumulator.

    Only k-mers in ``common_kmer_ids`` contribute. Large reservoir segments are
    split into ``covariance_batch_size`` rows to bound device memory.
    """
    num_kmers = 4 ** kmer_size
    common_lookup = torch.full((num_kmers,), -1, dtype=torch.long)
    common_lookup[common_kmer_ids] = torch.arange(len(common_kmer_ids))

    for class_index, (path, class_label) in enumerate(
        zip(reservoir_paths, class_labels)
    ):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        summary = validate_reservoir_payload(payload, path, class_label)
        samples = payload['reservoir_embeddings'].float()
        selected_sample_count = 0
        for source_index, kmer_id in enumerate(summary['kmer_ids'].tolist()):
            common_index = int(common_lookup[kmer_id].item())
            if common_index < 0:
                continue
            sample_start = int(payload['reservoir_offsets'][source_index].item())
            sample_end = int(payload['reservoir_offsets'][source_index + 1].item())
            for start in range(sample_start, sample_end, covariance_batch_size):
                end = min(start + covariance_batch_size, sample_end)
                residuals = samples[start:end] - centroids[class_index, common_index]
                accumulator.update(common_index, residuals)
            selected_sample_count += sample_end - sample_start

        del payload, samples
        print(
            f"Local residuals from class {class_label}: "
            f"{selected_sample_count} samples",
            flush=True,
        )


def _fit_single_center_base_classifier(
    reservoir_paths,
    class_labels,
    regularization='ledoit-wolf',
    ridge=1e-4,
    class_priors=None,
    min_prototype_count=1,
    min_reservoir_count=2,
    fit_device='auto',
    covariance_batch_size=16384,
    local_shrinkage=None,
    requested_embedding_key=None,
):
    lazy_import_runtime_dependencies()
    if covariance_batch_size < 1:
        raise ValueError(
            f"covariance_batch_size must be positive, got {covariance_batch_size}"
        )

    reservoir_info = load_reservoir_centroids(
        reservoir_paths,
        class_labels,
        min_reservoir_count=min_reservoir_count,
        requested_embedding_key=requested_embedding_key,
        min_prototype_count=min_prototype_count,
    )
    priors = normalize_class_priors(class_priors, len(class_labels))
    resolved_fit_device = resolve_fit_device(fit_device)
    covariance_accumulator = PerKmerCovarianceAccumulator(
        len(reservoir_info['common_kmer_ids']),
        reservoir_info['embedding_dim'],
        device=resolved_fit_device,
    )
    accumulate_reservoir_residuals(
        covariance_accumulator,
        reservoir_paths,
        class_labels,
        reservoir_info['common_kmer_ids'],
        reservoir_info['centroids'],
        reservoir_info['kmer_size'],
        covariance_batch_size,
    )
    covariance_result = covariance_accumulator.finalize(
        global_regularization=regularization,
        ridge=ridge,
        local_shrinkage=local_shrinkage,
    )
    local_covariances = covariance_result['local_covariances']
    cholesky, cholesky_info = torch.linalg.cholesky_ex(
        local_covariances,
        check_errors=False,
    )
    if (cholesky_info != 0).any():
        failed_indices = (cholesky_info != 0).nonzero(as_tuple=True)[0]
        failed_kmers = reservoir_info['common_kmer_ids'][
            failed_indices.cpu()
        ].tolist()
        raise RuntimeError(
            "One or more locally shrunk covariance matrices are not positive "
            f"definite (first failing k-mer IDs: {failed_kmers[:10]}). Increase "
            "--local_shrinkage, use --regularization ledoit-wolf, or increase "
            "--ridge."
        )
    precision = torch.cholesky_inverse(cholesky)

    centroids64 = reservoir_info['centroids'].to(
        resolved_fit_device,
        dtype=torch.float64,
    )
    coefficients = torch.einsum('kmd,mde->kme', centroids64, precision)
    log_priors = torch.log(
        torch.tensor(priors, dtype=torch.float64, device=resolved_fit_device)
    )
    intercepts = (
        -0.5 * (centroids64 * coefficients).sum(dim=-1)
        + log_priors[:, None]
    )

    num_kmers = 4 ** reservoir_info['kmer_size']
    kmer_lookup = torch.full((num_kmers,), -1, dtype=torch.long)
    kmer_lookup[reservoir_info['common_kmer_ids']] = torch.arange(
        len(reservoir_info['common_kmer_ids'])
    )

    return {
        'format_version': 2,
        'classifier_type': 'per_kmer_shrunk_lda',
        'class_labels': list(class_labels),
        'class_priors': torch.tensor(priors, dtype=torch.float64),
        'kmer_size': reservoir_info['kmer_size'],
        'embedding_key': reservoir_info['embedding_key'],
        'embedding_dim': reservoir_info['embedding_dim'],
        'kmer_ids': reservoir_info['common_kmer_ids'],
        'kmer_lookup': kmer_lookup,
        'centroids': reservoir_info['centroids'],
        'source_counts': reservoir_info['source_counts'],
        'reservoir_counts': reservoir_info['sampled_counts'],
        'empirical_pooled_covariance': covariance_result[
            'global_empirical_covariance'
        ].cpu().to(torch.float32),
        'pooled_covariance': covariance_result[
            'global_covariance'
        ].cpu().to(torch.float32),
        'local_covariances': local_covariances.cpu().to(torch.float32),
        'precision': precision.cpu().to(torch.float32),
        'coefficients': coefficients.cpu().to(torch.float32),
        'intercepts': intercepts.cpu().to(torch.float32),
        'local_shrinkage_values': covariance_result[
            'local_shrinkage_values'
        ].cpu().to(torch.float32),
        'local_residual_sample_counts': covariance_result[
            'local_sample_counts'
        ].cpu(),
        'metadata': {
            'global_regularization': regularization,
            'global_regularization_value': covariance_result[
                'global_regularization_value'
            ],
            'local_shrinkage': (
                'auto' if local_shrinkage is None else float(local_shrinkage)
            ),
            'ridge': float(ridge),
            'min_reservoir_count': int(min_reservoir_count),
            'min_prototype_count': int(min_prototype_count),
            'residual_sample_count': covariance_result['global_sample_count'],
            'fit_device': resolved_fit_device,
            'covariance_batch_size': int(covariance_batch_size),
            'raw_embeddings_no_additional_normalization': True,
            'class_summaries': reservoir_info['class_summaries'],
        },
    }


def kmer_center_base_ids(kmer_ids, kmer_size):
    """Extract zero-based A/C/G/T center-base IDs from encoded odd k-mers."""
    if kmer_size < 1 or kmer_size % 2 == 0:
        raise ValueError(
            f"Center-base classification requires a positive odd k-mer size, got {kmer_size}"
        )
    center_index = kmer_size // 2
    shift = 2 * (kmer_size - 1 - center_index)
    return (kmer_ids.long() >> shift) & 3


def discover_center_base_class_groups(
    reservoir_paths,
    class_labels,
    min_reservoir_count=2,
    requested_embedding_key=None,
    *,
    require_reservoir=True,
    min_prototype_count=1,
):
    """Group prototype classes by the encoded center base of eligible k-mers."""
    validate_class_inputs(reservoir_paths, class_labels)
    if min_reservoir_count < 1:
        raise ValueError(
            f"min_reservoir_count must be at least one, got {min_reservoir_count}"
        )
    if min_prototype_count < 1:
        raise ValueError(
            f"min_prototype_count must be at least one, got {min_prototype_count}"
        )

    groups = {}
    metadata_records = []
    expected_kmer_size = None
    expected_dim = None
    bases = 'ACGT'
    for class_index, (path, class_label) in enumerate(
        zip(reservoir_paths, class_labels)
    ):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        summary = validate_prototype_payload(
            payload,
            path,
            class_label,
            require_reservoir=require_reservoir,
        )
        del payload

        if expected_kmer_size is None:
            expected_kmer_size = summary['kmer_size']
            expected_dim = summary['embedding_dim']
        elif summary['kmer_size'] != expected_kmer_size:
            raise ValueError(
                f"Prototype {class_label!r} uses kmer_size={summary['kmer_size']}, "
                f"expected {expected_kmer_size}"
            )
        elif summary['embedding_dim'] != expected_dim:
            raise ValueError(
                f"Prototype {class_label!r} uses embedding dimension "
                f"{summary['embedding_dim']}, expected {expected_dim}"
            )

        eligible = summary['counts'] >= min_prototype_count
        if require_reservoir:
            eligible &= summary['reservoir_counts'] >= min_reservoir_count
        eligible_kmer_ids = summary['kmer_ids'][eligible]
        if len(eligible_kmer_ids) == 0:
            threshold_description = (
                f"{min_reservoir_count} sampled embeddings"
                if require_reservoir
                else f"{min_prototype_count} source observations"
            )
            raise ValueError(
                f"Prototype {class_label!r} has no k-mer with at least "
                f"{threshold_description}"
            )
        center_ids = torch.unique(
            kmer_center_base_ids(eligible_kmer_ids, summary['kmer_size'])
        ).tolist()
        if len(center_ids) != 1:
            center_bases = ', '.join(bases[index] for index in center_ids)
            raise ValueError(
                f"Prototype {class_label!r} has eligible k-mers spanning center "
                f"bases {center_bases}; each class prototype must represent one "
                "canonical center base"
            )

        center_base_id = int(center_ids[0])
        groups.setdefault(center_base_id, []).append(class_index)
        metadata_records.append({
            'name': class_label,
            'metadata': summary['metadata'],
        })

    for center_base_id, class_indices in groups.items():
        if len(class_indices) < 2:
            only_label = class_labels[class_indices[0]]
            raise ValueError(
                f"Center-base family {bases[center_base_id]} contains only class "
                f"{only_label!r}; at least two classes are required per center base"
            )

    embedding_key = query_pipeline.resolve_prototype_embedding_key(
        metadata_records,
        requested_key=requested_embedding_key,
    )
    preprocessing = validate_artifact_preprocessing(metadata_records)
    return {
        'groups': [
            {
                'center_base_id': center_base_id,
                'center_base': bases[center_base_id],
                'class_indices': class_indices,
            }
            for center_base_id, class_indices in sorted(groups.items())
        ],
        'kmer_size': expected_kmer_size,
        'embedding_dim': expected_dim,
        'embedding_key': embedding_key,
        'preprocessing': preprocessing.to_dict() if preprocessing is not None else None,
    }


def fit_lda_classifier(
    reservoir_paths,
    class_labels,
    regularization='ledoit-wolf',
    ridge=1e-4,
    class_priors=None,
    min_prototype_count=1,
    min_reservoir_count=2,
    fit_device='auto',
    covariance_batch_size=16384,
    local_shrinkage=None,
    requested_embedding_key=None,
):
    """Fit independent per-k-mer LDAs for each represented center base."""
    lazy_import_runtime_dependencies()
    grouping = discover_center_base_class_groups(
        reservoir_paths,
        class_labels,
        min_reservoir_count=min_reservoir_count,
        requested_embedding_key=requested_embedding_key,
        min_prototype_count=min_prototype_count,
    )
    normalized_priors = normalize_class_priors(class_priors, len(class_labels))
    base_classifiers = []
    family_lookup = torch.full(
        (4 ** grouping['kmer_size'],),
        -1,
        dtype=torch.long,
    )

    for family_index, group in enumerate(grouping['groups']):
        class_indices = group['class_indices']
        family_labels = [class_labels[index] for index in class_indices]
        family_paths = [reservoir_paths[index] for index in class_indices]
        family_priors = [normalized_priors[index] for index in class_indices]
        print(
            f"Fitting center-base family {group['center_base']}: "
            f"{', '.join(family_labels)}",
            flush=True,
        )
        family = _fit_single_center_base_classifier(
            family_paths,
            family_labels,
            regularization=regularization,
            ridge=ridge,
            class_priors=family_priors,
            min_prototype_count=min_prototype_count,
            min_reservoir_count=min_reservoir_count,
            fit_device=fit_device,
            covariance_batch_size=covariance_batch_size,
            local_shrinkage=local_shrinkage,
            requested_embedding_key=requested_embedding_key,
        )
        if family['embedding_key'] != grouping['embedding_key']:
            raise ValueError(
                f"Center-base family {group['center_base']} resolved embedding "
                f"{family['embedding_key']!r}, expected {grouping['embedding_key']!r}"
            )
        overlap = family_lookup[family['kmer_ids']] >= 0
        if overlap.any():
            overlapping_ids = family['kmer_ids'][overlap].tolist()
            raise ValueError(
                "Center-base classifier families overlap at k-mer IDs "
                f"{overlapping_ids[:10]}"
            )
        family_lookup[family['kmer_ids']] = family_index
        family['center_base_id'] = group['center_base_id']
        family['center_base'] = group['center_base']
        family['global_class_indices'] = torch.tensor(
            class_indices,
            dtype=torch.long,
        )
        base_classifiers.append(family)

    all_kmer_ids = (family_lookup >= 0).nonzero(as_tuple=True)[0]
    return {
        'format_version': 3,
        'classifier_type': 'center_base_partitioned_per_kmer_shrunk_lda',
        'class_labels': list(class_labels),
        'class_priors': torch.tensor(normalized_priors, dtype=torch.float64),
        'kmer_size': grouping['kmer_size'],
        'embedding_key': grouping['embedding_key'],
        'embedding_dim': grouping['embedding_dim'],
        'kmer_ids': all_kmer_ids,
        'family_lookup': family_lookup,
        'base_classifiers': base_classifiers,
        'metadata': {
            'global_regularization': regularization,
            'preprocessing': grouping['preprocessing'],
            'local_shrinkage': (
                'auto' if local_shrinkage is None else float(local_shrinkage)
            ),
            'ridge': float(ridge),
            'min_reservoir_count': int(min_reservoir_count),
            'min_prototype_count': int(min_prototype_count),
            'fit_device': resolve_fit_device(fit_device),
            'covariance_batch_size': int(covariance_batch_size),
            'raw_embeddings_no_additional_normalization': True,
            'family_count': len(base_classifiers),
            'residual_sample_count': sum(
                family['metadata']['residual_sample_count']
                for family in base_classifiers
            ),
            'family_summaries': [
                {
                    'center_base': family['center_base'],
                    'class_labels': family['class_labels'],
                    'kmer_count': len(family['kmer_ids']),
                    'global_regularization_value': family['metadata'][
                        'global_regularization_value'
                    ],
                }
                for family in base_classifiers
            ],
        },
    }


def fit_diagonal_variances(
    reservoir_paths,
    class_labels,
    common_kmer_ids,
    centroids,
    kmer_size,
    *,
    covariance_batch_size=16384,
    shrinkage=0.1,
    variance_floor=1e-6,
):
    """Estimate class-specific diagonal variances with a shared feature target.

    Reservoir residual squares are accumulated around the exact full-dataset
    means stored in the prototype payload.  Every class/k-mer diagonal is then
    shrunk toward a feature-wise target pooled across all applicable classes
    and k-mers.  The returned variances have shape
    ``(n_classes, n_kmers, embedding_dim)``.
    """
    if covariance_batch_size < 1:
        raise ValueError(
            f"covariance_batch_size must be positive, got {covariance_batch_size}"
        )
    if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError(f"diagonal shrinkage must be in [0, 1], got {shrinkage}")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError(
            f"variance_floor must be finite and positive, got {variance_floor}"
        )

    num_classes, num_common, embedding_dim = centroids.shape
    num_kmers = 4 ** kmer_size
    common_lookup = torch.full((num_kmers,), -1, dtype=torch.long)
    common_lookup[common_kmer_ids] = torch.arange(num_common)
    sum_squared = torch.zeros(
        (num_classes, num_common, embedding_dim),
        dtype=torch.float64,
    )
    sample_counts = torch.zeros(
        (num_classes, num_common),
        dtype=torch.long,
    )

    for class_index, (path, class_label) in enumerate(
        zip(reservoir_paths, class_labels)
    ):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        summary = validate_reservoir_payload(payload, path, class_label)
        samples = payload['reservoir_embeddings'].float()
        for source_index, kmer_id in enumerate(summary['kmer_ids'].tolist()):
            common_index = int(common_lookup[kmer_id].item())
            if common_index < 0:
                continue
            sample_start = int(payload['reservoir_offsets'][source_index].item())
            sample_end = int(payload['reservoir_offsets'][source_index + 1].item())
            centroid = centroids[class_index, common_index]
            for start in range(sample_start, sample_end, covariance_batch_size):
                end = min(start + covariance_batch_size, sample_end)
                residuals = samples[start:end] - centroid
                sum_squared[class_index, common_index].add_(
                    residuals.to(torch.float64).square().sum(dim=0)
                )
            sample_counts[class_index, common_index] = sample_end - sample_start
        del payload, samples

    if (sample_counts == 0).any():
        raise ValueError(
            "Cannot estimate diagonal variance for a class/k-mer without "
            "reservoir samples"
        )
    empirical = sum_squared / sample_counts.to(torch.float64).unsqueeze(-1)
    global_variance = sum_squared.sum(dim=(0, 1)) / sample_counts.sum()
    variances = (
        (1.0 - float(shrinkage)) * empirical
        + float(shrinkage) * global_variance.reshape(1, 1, -1)
    ).clamp_min(float(variance_floor))
    return {
        'diagonal_variances': variances.to(torch.float32),
        'diagonal_empirical_variances': empirical.to(torch.float32),
        'diagonal_global_variance': global_variance.to(torch.float32),
        'diagonal_sample_counts': sample_counts,
    }


def _fit_single_simple_classifier(
    reservoir_paths,
    class_labels,
    scorer_type,
    *,
    class_priors=None,
    min_prototype_count=1,
    min_reservoir_count=2,
    covariance_batch_size=16384,
    score_temperature=1.0,
    rbf_bandwidth=1.0,
    diagonal_shrinkage=0.1,
    variance_floor=1e-6,
    requested_embedding_key=None,
    metric='cosine',
):
    """Fit one center-base family for a non-LDA scorer."""
    require_reservoir = scorer_type == 'diagonal-gaussian'
    reservoir_info = load_reservoir_centroids(
        reservoir_paths,
        class_labels,
        min_reservoir_count=min_reservoir_count,
        requested_embedding_key=requested_embedding_key,
        require_reservoir=require_reservoir,
        min_prototype_count=min_prototype_count,
    )
    priors = normalize_class_priors(class_priors, len(class_labels))
    if not math.isfinite(score_temperature) or score_temperature <= 0:
        raise ValueError(
            f"score_temperature must be finite and positive, got {score_temperature}"
        )
    if not math.isfinite(rbf_bandwidth) or rbf_bandwidth <= 0:
        raise ValueError(
            f"rbf_bandwidth must be finite and positive, got {rbf_bandwidth}"
        )

    num_kmers = 4 ** reservoir_info['kmer_size']
    kmer_lookup = torch.full((num_kmers,), -1, dtype=torch.long)
    kmer_lookup[reservoir_info['common_kmer_ids']] = torch.arange(
        len(reservoir_info['common_kmer_ids'])
    )
    family = {
        'format_version': 1,
        'scorer_type': scorer_type,
        'class_labels': list(class_labels),
        'class_priors': torch.tensor(priors, dtype=torch.float64),
        'kmer_size': reservoir_info['kmer_size'],
        'embedding_key': reservoir_info['embedding_key'],
        'embedding_dim': reservoir_info['embedding_dim'],
        'kmer_ids': reservoir_info['common_kmer_ids'],
        'kmer_lookup': kmer_lookup,
        'centroids': reservoir_info['centroids'],
        'source_counts': reservoir_info['source_counts'],
        'reservoir_counts': reservoir_info['sampled_counts'],
        'score_temperature': float(score_temperature),
        'rbf_bandwidth': float(rbf_bandwidth),
        'metadata': {
            'min_prototype_count': int(min_prototype_count),
            'min_reservoir_count': (
                int(min_reservoir_count) if require_reservoir else None
            ),
            'embedding_transform': (
                'rowwise_l2_normalization'
                if scorer_type == 'centroid-cosine'
                else 'none'
            ),
            'class_summaries': reservoir_info['class_summaries'],
        },
    }
    if scorer_type == 'nearest-prototype':
        family['metric'] = metric
    if scorer_type == 'diagonal-gaussian':
        family.update(
            fit_diagonal_variances(
                reservoir_paths,
                class_labels,
                reservoir_info['common_kmer_ids'],
                reservoir_info['centroids'],
                reservoir_info['kmer_size'],
                covariance_batch_size=covariance_batch_size,
                shrinkage=diagonal_shrinkage,
                variance_floor=variance_floor,
            )
        )
        family['metadata'].update({
            'diagonal_shrinkage': float(diagonal_shrinkage),
            'variance_floor': float(variance_floor),
            'covariance_batch_size': int(covariance_batch_size),
        })
    return family


def fit_prototype_classifier(
    prototype_paths,
    class_labels,
    *,
    scorer_type='per-kmer-lda',
    class_priors=None,
    min_prototype_count=1,
    min_reservoir_count=2,
    regularization='ledoit-wolf',
    ridge=1e-4,
    fit_device='auto',
    covariance_batch_size=16384,
    local_shrinkage=None,
    score_temperature=1.0,
    rbf_bandwidth=1.0,
    diagonal_shrinkage=0.1,
    variance_floor=1e-6,
    requested_embedding_key=None,
    metric='cosine',
):
    """Fit a center-base-partitioned prototype classifier.

    Nearest-prototype and centroid scorers consume exact means and counts and therefore accept
    mean-only prototype files. Diagonal Gaussian and LDA scorers require the
    aligned reservoir fields produced by prototype extraction. For nearest-prototype,
    ``metric`` selects cosine or Euclidean distance; only uniform priors are
    accepted and score temperature does not affect distances.
    """
    lazy_import_runtime_dependencies()
    if scorer_type not in SCORER_TYPES:
        raise ValueError(
            f"scorer_type must be one of {SCORER_TYPES}, got {scorer_type!r}"
        )
    if scorer_type == 'nearest-prototype':
        validate_class_inputs(prototype_paths, class_labels)
        validate_nearest_options(metric, class_priors, len(class_labels))
    if scorer_type == 'per-kmer-lda':
        classifier = fit_lda_classifier(
            prototype_paths,
            class_labels,
            regularization=regularization,
            ridge=ridge,
            class_priors=class_priors,
            min_prototype_count=min_prototype_count,
            min_reservoir_count=min_reservoir_count,
            fit_device=fit_device,
            covariance_batch_size=covariance_batch_size,
            local_shrinkage=local_shrinkage,
            requested_embedding_key=requested_embedding_key,
        )
        classifier['format_version'] = 4
        classifier['classifier_type'] = 'center_base_partitioned_prototype_classifier'
        classifier['scorer_type'] = scorer_type
        classifier['metadata']['legacy_classifier_type'] = (
            'center_base_partitioned_per_kmer_shrunk_lda'
        )
        for family in classifier['base_classifiers']:
            family['scorer_type'] = scorer_type
        return classifier

    grouping = discover_center_base_class_groups(
        prototype_paths,
        class_labels,
        min_reservoir_count=min_reservoir_count,
        requested_embedding_key=requested_embedding_key,
        require_reservoir=(scorer_type == 'diagonal-gaussian'),
        min_prototype_count=min_prototype_count,
    )
    normalized_priors = normalize_class_priors(class_priors, len(class_labels))
    family_lookup = torch.full((4 ** grouping['kmer_size'],), -1, dtype=torch.long)
    base_classifiers = []
    for family_index, group in enumerate(grouping['groups']):
        class_indices = group['class_indices']
        family_labels = [class_labels[index] for index in class_indices]
        family_paths = [prototype_paths[index] for index in class_indices]
        family_priors = [normalized_priors[index] for index in class_indices]
        family = _fit_single_simple_classifier(
            family_paths,
            family_labels,
            scorer_type,
            class_priors=family_priors,
            metric=metric,
            min_prototype_count=min_prototype_count,
            min_reservoir_count=min_reservoir_count,
            covariance_batch_size=covariance_batch_size,
            score_temperature=score_temperature,
            rbf_bandwidth=rbf_bandwidth,
            diagonal_shrinkage=diagonal_shrinkage,
            variance_floor=variance_floor,
            requested_embedding_key=requested_embedding_key,
        )
        overlap = family_lookup[family['kmer_ids']] >= 0
        if overlap.any():
            raise ValueError(
                "Center-base classifier families overlap at k-mer IDs "
                f"{family['kmer_ids'][overlap].tolist()[:10]}"
            )
        family_lookup[family['kmer_ids']] = family_index
        family['center_base_id'] = group['center_base_id']
        family['center_base'] = group['center_base']
        family['global_class_indices'] = torch.tensor(class_indices, dtype=torch.long)
        base_classifiers.append(family)

    classifier = {
        'format_version': 4,
        'classifier_type': 'center_base_partitioned_prototype_classifier',
        'scorer_type': scorer_type,
        'class_labels': list(class_labels),
        'class_priors': torch.tensor(normalized_priors, dtype=torch.float64),
        'kmer_size': grouping['kmer_size'],
        'embedding_key': grouping['embedding_key'],
        'embedding_dim': grouping['embedding_dim'],
        'kmer_ids': (family_lookup >= 0).nonzero(as_tuple=True)[0],
        'family_lookup': family_lookup,
        'base_classifiers': base_classifiers,
        'metadata': {
            'scorer_type': scorer_type,
            'preprocessing': grouping['preprocessing'],
            'min_prototype_count': int(min_prototype_count),
            'min_reservoir_count': (
                int(min_reservoir_count)
                if scorer_type == 'diagonal-gaussian'
                else None
            ),
            'family_count': len(base_classifiers),
            'embedding_transform': (
                'rowwise_l2_normalization'
                if scorer_type == 'centroid-cosine'
                else 'none'
            ),
        },
    }

    if scorer_type == 'nearest-prototype':
        classifier['metric'] = metric
    return classifier


def prepare_classifier_for_prediction(classifier, device):
    """Move only scorer parameters required by batched prediction to ``device``."""
    scorer_type = classifier.get('scorer_type', 'per-kmer-lda')
    prepared_families = []
    for family in classifier['base_classifiers']:
        family_scorer = family.get('scorer_type', scorer_type)
        prepared = {
            'scorer_type': family_scorer,
            'center_base': family['center_base'],
            'global_class_indices': family['global_class_indices'].to(device),
            'kmer_lookup': family['kmer_lookup'].to(device),
        }
        if family_scorer == 'per-kmer-lda':
            prepared.update({
                'coefficients': family['coefficients'].to(device),
                'intercepts': family['intercepts'].to(device),
            })
        else:
            prepared.update({
                'centroids': family['centroids'].to(device),
                'class_priors': family['class_priors'].to(device, dtype=torch.float32),
                'score_temperature': float(family['score_temperature']),
                'rbf_bandwidth': float(family['rbf_bandwidth']),
            })
            if family_scorer == 'nearest-prototype':
                prepared['metric'] = family['metric']
            if family_scorer == 'diagonal-gaussian':
                prepared['diagonal_variances'] = family[
                    'diagonal_variances'
                ].to(device)
        prepared_families.append(prepared)
    return {
        'num_classes': len(classifier['class_labels']),
        'scorer_type': scorer_type,
        'base_classifiers': prepared_families,
    }


def score_family_embeddings(embeddings, classifier_indices, family):
    """Return class scores for routed embeddings from one center-base family.

    Parameters
    ----------
    embeddings : torch.Tensor
        Query tensor with shape ``(n_queries, embedding_dim)``.
    classifier_indices : torch.Tensor
        Dense family-local k-mer row for every query, shape ``(n_queries,)``.
    family : dict
        Device-prepared family state from
        :func:`prepare_classifier_for_prediction`.

    Returns
    -------
    torch.Tensor
        Scores with shape ``(n_queries, n_applicable_classes)``. Softmax over
        the class dimension would give uncalibrated conditional weights; these
        are not generated or saved. Nearest-prototype returns negative raw
        distances internally, without priors or temperature scaling.
    """
    scorer_type = family.get('scorer_type', 'per-kmer-lda')
    if scorer_type == 'per-kmer-lda':
        coefficients = family['coefficients'][
            :, classifier_indices, :
        ].permute(1, 0, 2)
        intercepts = family['intercepts'][:, classifier_indices].T
        return (embeddings.unsqueeze(1) * coefficients).sum(dim=-1) + intercepts

    centroids = family['centroids'][:, classifier_indices, :].permute(1, 0, 2)
    if scorer_type == 'nearest-prototype':
        # Negated distances let the shared max/argmax path retain class-order ties.
        # The nearest-prototype writer restores raw distances for public output.
        return -query_pipeline.compute_distance(
            embeddings.unsqueeze(1), centroids, None, family['metric'],
        )
    log_priors = torch.log(family['class_priors']).unsqueeze(0)
    expanded_embeddings = embeddings.unsqueeze(1)
    if scorer_type == 'centroid-l2':
        distances = torch.linalg.vector_norm(
            expanded_embeddings - centroids,
            ord=2,
            dim=-1,
        )
        return -distances / family['score_temperature'] + log_priors
    if scorer_type == 'centroid-cosine':
        similarities = torch.nn.functional.cosine_similarity(
            expanded_embeddings,
            centroids,
            dim=-1,
        )
        return similarities / family['score_temperature'] + log_priors
    if scorer_type == 'centroid-rbf':
        squared_distances = (expanded_embeddings - centroids).square().sum(dim=-1)
        denominator = 2.0 * family['rbf_bandwidth'] ** 2
        return -squared_distances / denominator + log_priors
    if scorer_type == 'diagonal-gaussian':
        variances = family['diagonal_variances'][
            :, classifier_indices, :
        ].permute(1, 0, 2)
        centered = expanded_embeddings - centroids
        return (
            -0.5
            * (centered.square() / variances + torch.log(variances)).sum(dim=-1)
            + log_priors
        )
    raise ValueError(f"Unsupported prepared scorer type: {scorer_type!r}")


def compute_prototype_classifier_batch(
    batch,
    model,
    classifier_state,
    kmer_size,
    embedding_dim,
    device,
    embedding_key,
    num_trim,
    batch_size,
):
    """Apply a center-base-partitioned prototype classifier to one chunk batch.

    Each eligible query embedding is routed by k-mer to its center-base family;
    family-local discriminant outputs are scattered into global class columns.
    Invalid, unlabelled, padded, and edge-trimmed locations remain ``NaN`` with
    predicted class index ``-1``.

    Returns
    -------
    dict
        CPU per-class scores shaped
        ``(n_chunks, length, n_classes)``, winning score/index
        arrays, k-mer IDs, positions and masks, and read metadata.
    """
    (
        base_seq_input,
        ts_input,
        percentiles_input,
        pad_base_seq,
        pad_ts,
        labels_idx,
        readnames,
        refnames,
        ref_positions,
        read_positions,
        _min_start,
        strands,
        _pysam_data,
    ) = batch

    device = torch.device(device)
    num_classes = classifier_state['num_classes']
    scores = torch.full(
        (*base_seq_input.shape, num_classes),
        float('nan'),
        dtype=torch.float32,
    )
    max_score = torch.full(base_seq_input.shape, float('nan'), dtype=torch.float32)
    predicted_class_index = torch.full(base_seq_input.shape, -1, dtype=torch.long)
    kmer_ids_cpu = torch.full(base_seq_input.shape, -1, dtype=torch.long)

    for sub_batch in range(0, base_seq_input.size(0), batch_size):
        sl = slice(sub_batch, sub_batch + batch_size)
        batch_base_seq_input = base_seq_input[sl].to(device, non_blocking=True)
        batch_ts_input = ts_input[sl].to(device, non_blocking=True)
        batch_percentiles_input = percentiles_input[sl].to(device, non_blocking=True)
        batch_pad_base_seq = pad_base_seq[sl].to(device, non_blocking=True)
        batch_pad_ts = pad_ts[sl].to(device, non_blocking=True)
        batch_labels_idx = labels_idx[sl].to(device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == 'cuda',
        ):
            out = model(
                batch_base_seq_input,
                batch_ts_input,
                percentiles_input=batch_percentiles_input,
                pad_base_seq=batch_pad_base_seq,
                pad_ts=batch_pad_ts,
                embeddings_only=True,
            )
            embeddings = select_base_embedding(out, embedding_key).float()

        if embeddings.shape[-1] != embedding_dim:
            raise ValueError(
                f"Query embedding {embedding_key!r} has dimension "
                f"{embeddings.shape[-1]}, but the prototype classifier expects "
                f"dimension {embedding_dim}"
            )

        kmer_ids, kmer_valid = query_pipeline.compute_chunk_kmer_ids(
            batch_base_seq_input,
            kmer_size,
        )
        valid_kmer_pos = kmer_valid & (kmer_ids >= 0)
        labelled_kmer_pos = (
            query_pipeline.ignore_edges(batch_pad_base_seq.bool(), num_trim)
            & batch_labels_idx.bool()
            & valid_kmer_pos
        )
        batch_scores = torch.full(
            (*batch_base_seq_input.shape, num_classes),
            float('nan'),
            dtype=torch.float32,
            device=device,
        )
        batch_max_score = torch.full(
            batch_base_seq_input.shape,
            float('nan'),
            dtype=torch.float32,
            device=device,
        )
        batch_predicted_index = torch.full(
            batch_base_seq_input.shape,
            -1,
            dtype=torch.long,
            device=device,
        )
        for family in classifier_state['base_classifiers']:
            classifier_indices = family['kmer_lookup'][kmer_ids.clamp_min(0)]
            classifier_available = labelled_kmer_pos & (classifier_indices >= 0)
            if not classifier_available.any():
                continue

            selected_classifier_indices = classifier_indices[classifier_available]
            selected_scores = score_family_embeddings(
                embeddings[classifier_available],
                selected_classifier_indices,
                family,
            )
            selected_max_score, selected_local_index = selected_scores.max(dim=1)
            selected_global_index = family['global_class_indices'][
                selected_local_index
            ]
            selected_batch, selected_position = classifier_available.nonzero(
                as_tuple=True
            )
            batch_scores[
                selected_batch[:, None],
                selected_position[:, None],
                family['global_class_indices'][None, :],
            ] = selected_scores
            batch_max_score[classifier_available] = selected_max_score
            batch_predicted_index[classifier_available] = selected_global_index

        scores[sl] = batch_scores.cpu()
        max_score[sl] = batch_max_score.cpu()
        predicted_class_index[sl] = batch_predicted_index.cpu()
        kmer_ids_cpu[sl] = kmer_ids.cpu()

    return {
        'scores': scores,
        'score': max_score,
        'predicted_class_index': predicted_class_index,
        'kmer_ids': kmer_ids_cpu,
        'read_positions': read_positions.cpu(),
        'ref_positions': ref_positions.cpu(),
        'pad_base_seq': pad_base_seq.cpu(),
        'labels_idx': labels_idx.cpu(),
        'readnames': readnames,
        'refnames': refnames,
        'strands': strands,
    }


def compute_lda_batch(*args, **kwargs):
    """Apply an LDA state through the generic prototype scoring path."""
    return compute_prototype_classifier_batch(*args, **kwargs)


def classifier_score_column(class_label):
    """Return the parquet score-column name for one class label."""
    return f'score_{class_label}'


def empty_classifier_rows(class_labels):
    """Create empty core and per-class prototype-classifier output columns."""
    rows = {
        'readname': [],
        'chrom': [],
        'strand': [],
        'read_pos': [],
        'position': [],
        'kmer': [],
        'score': [],
        'predicted_class_index': [],
        'predicted_class': [],
    }
    for class_label in class_labels:
        rows[classifier_score_column(class_label)] = []
    return rows


def stitch_classifier_batch_rows(batch_result, kmer_size, class_labels):
    """Stitch overlapping chunks and emit valid per-position classifier calls.

    Overlaps between consecutive chunks of a read are divided at their
    midpoint, preventing duplicate rows while retaining continuous coverage.
    """
    rows = empty_classifier_rows(class_labels)
    readnames = np.asarray(batch_result['readnames'])
    refnames = np.asarray(batch_result['refnames'])
    strands = np.asarray(batch_result['strands'])
    _, chunks_per_read = query_pipeline.unique_read_boundaries(readnames)

    scores = batch_result['scores']
    max_score = batch_result['score']
    predicted_index = batch_result['predicted_class_index']
    kmer_ids = batch_result['kmer_ids']
    read_positions = batch_result['read_positions']
    ref_positions = batch_result['ref_positions']
    pad_base_seq = batch_result['pad_base_seq']
    labels_idx = batch_result['labels_idx']

    for read_indices in chunks_per_read:
        read_length = int(read_positions[read_indices].max().item()) + 1
        accum_scores = torch.full(
            (read_length, len(class_labels)),
            float('nan'),
            dtype=torch.float32,
        )
        accum_max_score = torch.full(
            (read_length,),
            float('nan'),
            dtype=torch.float32,
        )
        accum_predicted_index = torch.full((read_length,), -1, dtype=torch.long)
        accum_ref = torch.full((read_length,), -1, dtype=ref_positions.dtype)
        accum_kmer = torch.full((read_length,), -1, dtype=torch.long)
        accum_label = torch.zeros((read_length,), dtype=torch.bool)

        last_written_pos = -1
        for idx in read_indices:
            mask = pad_base_seq[idx]
            curr_pos = read_positions[idx][mask]
            if curr_pos.numel() == 0:
                continue

            start_pos = curr_pos[0].item()
            if start_pos <= last_written_pos:
                overlap_len = last_written_pos - start_pos + 1
                split_offset = (overlap_len + 1) // 2
            else:
                split_offset = 0

            write_pos = curr_pos[split_offset:]
            if write_pos.numel() == 0:
                continue

            write_start = write_pos[0].item()
            write_end = write_pos[-1].item() + 1
            chunk_masked = mask.nonzero(as_tuple=True)[0][split_offset:]

            accum_scores[write_start:write_end] = scores[idx][chunk_masked]
            accum_max_score[write_start:write_end] = max_score[idx][chunk_masked]
            accum_predicted_index[write_start:write_end] = predicted_index[idx][chunk_masked]
            accum_ref[write_start:write_end] = ref_positions[idx][chunk_masked]
            accum_kmer[write_start:write_end] = kmer_ids[idx][chunk_masked]
            accum_label[write_start:write_end] = labels_idx[idx][chunk_masked]
            last_written_pos = write_end - 1

        keep = (
            accum_label
            & (accum_kmer >= 0)
            & (accum_predicted_index >= 0)
            & torch.isfinite(accum_max_score)
        )
        if not keep.any():
            continue

        read_idx = read_indices[0]
        kept_read_pos = keep.nonzero(as_tuple=True)[0]
        kept_kmers = accum_kmer[keep].numpy()
        kept_predicted_index = accum_predicted_index[keep].numpy()
        rows['readname'].extend([str(readnames[read_idx])] * len(kept_read_pos))
        rows['chrom'].extend([str(refnames[read_idx])] * len(kept_read_pos))
        rows['strand'].extend([int(strands[read_idx])] * len(kept_read_pos))
        rows['read_pos'].extend(kept_read_pos.numpy().astype(np.int64).tolist())
        rows['position'].extend(accum_ref[keep].numpy().astype(np.int64).tolist())
        rows['kmer'].extend([
            query_pipeline.kmer_index_to_string(kmer_id, kmer_size)
            for kmer_id in kept_kmers
        ])
        rows['score'].extend(
            accum_max_score[keep].numpy().astype(np.float32).tolist()
        )
        rows['predicted_class_index'].extend(
            kept_predicted_index.astype(np.int32).tolist()
        )
        rows['predicted_class'].extend([
            class_labels[index] for index in kept_predicted_index
        ])
        for class_index, class_label in enumerate(class_labels):
            rows[classifier_score_column(class_label)].extend(
                accum_scores[keep, class_index].numpy().astype(np.float32).tolist()
            )

    return rows


def write_classifier_rows(rows, class_labels, output_dir, part_idx):
    """Write a Snappy parquet part, returning its path or ``None`` if empty."""
    if len(rows['score']) == 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    columns = {
        'readname': pa.array(rows['readname'], pa.string()),
        'chrom': pa.array(rows['chrom'], pa.dictionary(pa.int32(), pa.string())),
        'strand': pa.array(rows['strand'], pa.int8()),
        'read_pos': pa.array(rows['read_pos'], pa.int64()),
        'position': pa.array(rows['position'], pa.int64()),
        'kmer': pa.array(rows['kmer'], pa.dictionary(pa.int32(), pa.string())),
        'score': pa.array(rows['score'], pa.float32()),
        'predicted_class_index': pa.array(rows['predicted_class_index'], pa.int32()),
        'predicted_class': pa.array(
            rows['predicted_class'],
            pa.dictionary(pa.int32(), pa.string()),
        ),
    }
    for class_label in class_labels:
        score_column = classifier_score_column(class_label)
        columns[score_column] = pa.array(rows[score_column], pa.float32())
    table = pa.table(columns)
    path = os.path.join(output_dir, f"part_{part_idx:06d}.parquet")
    pq.write_table(table, path, compression='snappy')
    return path


def write_nearest_prototype_rows(rows, class_labels, output_dir, part_idx):
    """Write stitched classifier rows as raw distances and nearest class calls.

    Internal negative scores are converted back to distances, preserving NaN
    for classes outside the query's center-base family. No score/probability
    columns are written. Return the Parquet path, or None for empty rows.
    """
    distances = {key: rows[key] for key in
                 ('readname', 'chrom', 'strand', 'read_pos', 'position', 'kmer')}
    distances['distance'] = [-value for value in rows['score']]
    distances['closest_prototype_index'] = rows['predicted_class_index']
    distances['closest_prototype'] = rows['predicted_class']
    for label in class_labels:
        distances[query_pipeline.prototype_distance_column(label)] = [
            -value for value in rows[classifier_score_column(label)]
        ]
    return query_pipeline.write_prototype_rows(
        distances, class_labels, output_dir, part_idx,
    )


@use_model_bundle('prototype')
def run(args):
    """Classify reference or read-space targets with prototypes or fitted effects.

    Residual-effects mode delegates to the fitted-effect detection workflow.
    Other modes fit a prototype classifier whose artifact may be saved separately;
    query results are streamed as
    stitched parquet parts with per-class scores (raw distances for nearest-prototype).
    Read-based mode selects motifs in the called sequence. Optional contigs
    restrict indexed input; otherwise it streams unindexed BAMs. A supplied
    reference retains genomic coordinates; negative positions indicate absent
    reference coordinates (-1 for insertions/unmapped, -2 for soft clips).
    Optional BED exclusions discard whole overlapping mapped reads before
    signal decoding in either mode; unmapped reads cannot overlap a region.
    With ``--pod5``, the query BAM must be one file; startup prepares its shared
    index before model loading and workers partition whole POD5 files.
    """
    resolve_cli_preprocessing(args)
    if args.scorer == 'residual-effects':
        from .effects_pipeline import run as run_effects
        return run_effects(args)
    from porefm.data.excluded_regions import get_excluded_regions
    if args.exclude_bed:
        args.exclude_bed_sha256 = get_excluded_regions(args.exclude_bed).sha256
    from porefm.common.run_outputs import prepare_run_outputs
    classifier_output = args.classifier_output or os.path.join(args.output_dir, 'prototype_classifier.pt')
    prepare_run_outputs(
        args, command="prototype classify", directories={"output_dir": args.output_dir},
        files={"classifier_output": classifier_output},
    )
    pod5_options = query_pipeline.prepare_cli_pod5_input(args)
    lazy_import_runtime_dependencies()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_bases = query_pipeline.is_all_base_motif(args.motif)
    label_all_aligned_bases = all_bases and not args.read_based
    sample_motif = args.motif
    if all_bases:
        sample_motif = 'A 0;C 0;G 0;T 0' if args.read_based else None
    args.step = resolve_downstream_step(
        args.size,
        args.seq_type,
        args.step,
        context="k-mer prototype classification",
    )

    device = torch.device(args.device)
    classifier = fit_prototype_classifier(
        args.prototype_embeddings,
        args.class_labels,
        scorer_type=args.scorer,
        metric=args.metric,
        min_prototype_count=args.min_prototype_count,
        regularization=args.regularization,
        ridge=args.ridge,
        class_priors=args.class_priors,
        min_reservoir_count=args.min_reservoir_count,
        fit_device=args.fit_device,
        covariance_batch_size=args.covariance_batch_size,
        local_shrinkage=args.local_shrinkage,
        score_temperature=args.score_temperature,
        rbf_bandwidth=args.rbf_bandwidth,
        diagonal_shrinkage=args.diagonal_shrinkage,
        variance_floor=args.variance_floor,
        requested_embedding_key=args.embedding_key,
    )
    for family in classifier['base_classifiers']:
        summary = (
            f"Using {len(family['kmer_ids'])} shared "
            f"{family['kmer_size']}-mers for center {family['center_base']} "
            f"classes={family['class_labels']}; "
            f"embedding={family['embedding_key']}; scorer={args.scorer}"
        )
        if args.scorer == 'per-kmer-lda':
            shrinkage_values = family['local_shrinkage_values']
            summary += (
                f"; global_regularization="
                f"{family['metadata']['global_regularization']} "
                f"({family['metadata']['global_regularization_value']:.8g}); "
                f"local_shrinkage={float(shrinkage_values.min()):.6g}-"
                f"{float(shrinkage_values.max()):.6g} "
                f"(mean={float(shrinkage_values.mean()):.6g})"
            )
        elif args.scorer == 'diagonal-gaussian':
            summary += (
                f"; diagonal_shrinkage={args.diagonal_shrinkage}; "
                f"variance_floor={args.variance_floor}"
            )
        print(summary, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    classifier_output = args.classifier_output
    if classifier_output is None:
        classifier_output = os.path.join(args.output_dir, 'prototype_classifier.pt')
    os.makedirs(os.path.dirname(classifier_output) or '.', exist_ok=True)
    validate_artifact_preprocessing(classifier, args.preprocessing)
    classifier["metadata"]["preprocessing"] = args.preprocessing
    torch.save(classifier, classifier_output)
    print(f"Wrote {classifier_output}", flush=True)

    pretrain_model, _pretrain_config = query_pipeline.load_pretrain_model(
        args.model_path,
        device,
    )
    if args.disable_compile is False:
        pretrain_model = torch.compile(pretrain_model, dynamic=True)

    kmer_table = resolve_cli_preprocessing(args).kmer_table
    chrom_info = (
        query_pipeline.get_read_based_chrom_info(args.ref, args.chrom)
        if args.read_based else query_pipeline.get_chrom_info(args.ref, args.chrom)
    )
    bam_paths = query_pipeline.expand_bam_paths(args.bam_path)
    bam_samples = [
        query_pipeline.LabelledBamSample(
            bam_path=bam_path,
            chrom_info=chrom_info,
            ref=args.ref,
            # Query labels only select positions; they are independent of the
            # K output classes. Preserve the comparison pipeline's binary
            # position/motif label contract.
            num_classes=2,
            pos_file=args.pos_file,
            motif=sample_motif,
            motif_label=args.motif_label,
            flag=args.flag,
            read_based=args.read_based,
            **pod5_options,
            polish=args.use_polished_sequence,
            polish_indel_limit=args.polish_indel_limit,
            polish_vcf_path=(
                args.polish_vcf_path if args.use_polished_sequence else None
            ),
        )
        for bam_path in bam_paths
    ]

    dataset = query_pipeline.CompleteLabelledReadIterableDataset(
        bam_list=bam_samples,
        size=args.size,
        step=args.step,
        move_downsample_factor=1,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        max_read_length=args.max_read_length,
        max_alignment_length=args.max_alignment_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=0,
        world_size=1,
        discard_empty_labels=True,
        return_pysam=False,
        return_scale_features=False,
        label_all_aligned_bases=label_all_aligned_bases,
        exclude_bed=args.exclude_bed,
        allow_pod5=True,
    )
    num_workers = query_pipeline.inference_worker_count(dataset, args.num_workers)
    loader_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    if num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )
    loader = DataLoader(dataset, **loader_kwargs)
    classifier_state = prepare_classifier_for_prediction(classifier, device)

    write_rows = (write_nearest_prototype_rows if args.scorer == 'nearest-prototype'
                  else write_classifier_rows)
    rows = empty_classifier_rows(args.class_labels)
    part_idx = 0
    total_rows = 0
    start = time.time()
    batch_idx = 0
    loader_iter = iter(loader)
    try:
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            batch_idx += 1
            batch_result = compute_prototype_classifier_batch(
                batch,
                pretrain_model,
                classifier_state,
                classifier['kmer_size'],
                classifier['embedding_dim'],
                device,
                classifier['embedding_key'],
                args.num_trim,
                args.batch_size,
            )
            batch_rows = stitch_classifier_batch_rows(
                batch_result,
                classifier['kmer_size'],
                args.class_labels,
            )
            query_pipeline.append_rows(rows, batch_rows)
            total_rows += len(batch_rows['score'])

            if batch_idx % args.log_every == 0:
                print(
                    f"[{time.time() - start:.2f}s] batches={batch_idx} "
                    f"rows={total_rows}",
                    flush=True,
                )

            if batch_idx % args.checkpoint_batch_step == 0:
                path = write_rows(
                    rows,
                    args.class_labels,
                    args.output_dir,
                    part_idx,
                )
                if path is not None:
                    print(f"Wrote {path}", flush=True)
                    part_idx += 1
                rows = empty_classifier_rows(args.class_labels)

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
    finally:
        query_pipeline.shutdown_dataloader_iterator(loader_iter)

    path = write_rows(
        rows,
        args.class_labels,
        args.output_dir,
        part_idx,
    )
    if path is not None:
        print(f"Wrote {path}", flush=True)

    print(
        {
            'total_rows': total_rows,
            'output_dir': args.output_dir,
            'classifier_output': classifier_output,
        },
        flush=True,
    )


def parse_args(argv=None):
    """Parse prototype-classifier fitting and read-scoring options."""
    parser = PoreFMArgumentParser(
        formatter_class=DefaultsFormatter,
        description=(
            "Classify base-aligned read embeddings using nearest-prototype "
            "distances, a scorer fitted from per-k-mer prototypes, or a saved "
            "residual-effect model."
        ),
    )
    parser.add_argument('--bam_path', required=True, help="Input BAM file or directory containing BAM files.")
    query_pipeline.add_pod5_arguments(parser)
    add_exclude_bed_argument(parser)
    add_model_arguments(parser, local_flags=('--model_path', '--model-path'),
                        local_help="Pretrained checkpoint or complete inference bundle. A bundle overrides separate model paths and selects its scorer.")
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', required=True)
    parser.add_argument('--ref', default=None, help='Reference FASTA; optional with --read_based to retain genomic coordinates.')
    parser.add_argument('--output_dir', required=True)
    from porefm.common.run_outputs import add_output_safety_argument
    add_output_safety_argument(parser)
    parser.add_argument('--classifier_output', default=None, help="Saved fitted classifier artifact; defaults to OUTPUT_DIR/prototype_classifier.pt.")
    parser.add_argument('--chrom', nargs='+', default=None,
                        help='Contig selection (requires indexed BAMs); optional with --read_based, independently of --ref.')
    parser.add_argument(
        '--read_based', '--read-based', action='store_true',
        help='Select motifs in the read sequence; optionally restrict --chrom and retain genomic coordinates with --ref. Without --chrom, stream all records including unmapped reads.',
    )
    parser.add_argument('--pos_file', default=None)
    parser.add_argument(
        '--motif',
        default=None,
        help=(
            "Motif specification supplied as one quoted shell argument, for "
            "example 'CG 0;CH 0'; use 'N 0' to score every aligned base. "
            "With --read_based, motifs use the read sequence and default to 'N 0'."
        ),
    )
    parser.add_argument('--motif_label', default='1', help='Semicolon-delimited class labels corresponding to --motif.')
    parser.add_argument(
        '--embedding_key',
        choices=EMBEDDING_KEY_CHOICES,
        default=None,
        help=(
            "Optional validation override. By default, infer the raw post-LayerNorm "
            "base embedding from prototype metadata."
        ),
    )
    parser.add_argument('--seq_type', default='dna', choices=['dna', 'rna'])
    parser.add_argument('--pore_type', default='r10')
    parser.add_argument('--device', default='cuda',
                        help='Backbone execution device; auto explicitly enables CPU fallback when CUDA is unavailable.')
    parser.add_argument('--fit_device', default='auto', help="Device used to accumulate full LDA covariance moments.")
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--reads_per_chunk', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64, help="Chunk sub-batch size for model forward passes.")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--max_read_length', type=float, default=5e5)
    parser.add_argument('--max_alignment_length', type=float, default=1e6)
    parser.add_argument('--min_qscore', type=float, default=10)
    parser.add_argument('--use_polished_sequence', default=False, action='store_true', help='Use reference-guided polished base sequences for query extraction.')
    parser.add_argument('--polish_indel_limit', type=int, default=5, help='Maximum unmasked insertion/deletion length corrected during polishing.')
    parser.add_argument('--polish_vcf_path', default=None, help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.')
    parser.add_argument('--num_trim', type=int, default=16)
    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_batch_step', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--disable_compile', action='store_true')
    parser.add_argument(
        '--prototype_embeddings',
        '--reservoir_embeddings',
        dest='prototype_embeddings',
        nargs='+',
        required=False,
        help=(
            "Per-class prototype .pt files. Nearest-prototype and centroid scorers accept mean-only "
            "files; Gaussian and LDA scorers require saved reservoirs."
        ),
    )
    parser.add_argument(
        '--class_labels',
        nargs='+',
        required=False,
        help="Class names in the same order as --prototype_embeddings.",
    )
    parser.add_argument(
        '--scorer',
        choices=SCORER_TYPES + ('residual-effects',),
        default='per-kmer-lda',
        help="Per-k-mer scoring model.",
    )
    parser.add_argument(
        '--metric', choices=['cosine', 'l2'], default=None,
        help='Nearest-prototype only: raw distance metric (default: cosine).',
    )
    parser.add_argument(
        '--class_priors',
        nargs='+',
        type=float,
        default=None,
        help="Optional positive class priors. Values are normalized to sum to one; defaults to uniform priors. Nearest-prototype requires uniform priors.",
    )
    parser.add_argument(
        '--regularization',
        choices=['ledoit-wolf', 'ridge'],
        default='ledoit-wolf',
        help="LDA-only regularization used for the global covariance target.",
    )
    parser.add_argument('--ridge', type=float, default=1e-4, help="LDA-only absolute diagonal ridge used with ridge regularization.")
    parser.add_argument(
        '--local_shrinkage',
        type=float,
        default=None,
        help=(
            "LDA-only fixed weight in [0, 1] on the global covariance target "
            "for every 5-mer. By default, estimate a separate Ledoit-Wolf-style "
            "local-to-global shrinkage weight for each 5-mer."
        ),
    )
    parser.add_argument('--min_prototype_count', type=int, default=1, help="Minimum exact source observations required per retained k-mer/class.")
    parser.add_argument('--min_reservoir_count', type=int, default=2, help="Minimum sampled observations required per retained k-mer/class for reservoir-based scorers.")
    parser.add_argument('--covariance_batch_size', type=int, default=16384, help="Residual rows processed per diagonal-Gaussian or LDA covariance update.")
    parser.add_argument('--score_temperature', type=float, default=1.0, help="Positive temperature for L2 and cosine centroid scores.")
    parser.add_argument('--rbf_bandwidth', type=float, default=1.0, help="Positive isotropic bandwidth for centroid RBF scores.")
    parser.add_argument('--diagonal_shrinkage', type=float, default=0.1, help="Weight in [0, 1] on the pooled feature-wise variance target.")
    parser.add_argument('--variance_floor', type=float, default=1e-6, help="Positive lower bound applied to fitted diagonal variances.")
    parser.add_argument('--effect_model', help='Artifact from fit-effect-prototypes; required for residual-effects.')
    parser.add_argument('--output_format', choices=['parquet', 'bam', 'both'], default='parquet',
                        help='Residual-effects output format; BAM writes output.bam with RNA MM/ML codes.')
    parser.add_argument('--bam_threads', type=int, default=4, help='Compression threads for residual-effects BAM output.')
    parser.add_argument('--effect_probability_temperature', type=float, default=1.0,
                        help='Positive temperature for BAM conditional softmax probabilities; does not change scores or decoding.')
    parser.add_argument('--effect_neighbors', choices=['on', 'off'], default='on', help='Disable footprints for a center-only ablation.')
    parser.add_argument('--effect_penalty', type=float, default=None, help='Override the artifact modification penalty; not a calibrated FDR threshold.')
    parser.add_argument('--effect_beam_width', type=int, default=32)
    parser.add_argument('--effect_max_sweeps', type=int, default=8)
    parser.add_argument('--effect_decoder', choices=['auto', 'numba', 'python'], default='auto',
                        help='CPU beam decoder: auto uses optional Numba; python retains the reference implementation. '
                             'Numba compiles/loads its cache before input processing; scoring rules are unchanged.')
    parser.add_argument('--effect_threads', type=int, default=4, help='CPU threads for downstream effect scoring.')
    from .effects import DEFAULT_EFFECT_CACHE_MB
    parser.add_argument('--effect_cache_mb', type=int, default=DEFAULT_EFFECT_CACHE_MB,
                        help='Persistent effect-cache tensor budget in MiB, excluding model and Python overhead; 0 disables retention.')
    parser.add_argument('--effect_device', choices=['auto', 'cpu', 'cuda'], default='cuda',
                        help='Dense effect scoring device; auto follows the backbone. Beam decoding stays on CPU.')
    parser.add_argument('--effect_block_size', type=int, default=4096,
                        help='Centers per device scoring tile, plus a left overlap halo; whole reads are decoded jointly.')
    parser.add_argument('--effect_batch_reads', type=int, default=16,
                        help='Maximum independent reads grouped for dense effect scoring; decoding stays per read.')
    parser.add_argument('--effect_batch_bases', type=int, default=65536,
                        help='Maximum grouped read bases; longer single reads are scored alone in bounded tiles.')
    parser.add_argument('--effect_write_rows', type=int, default=4_000_000,
                        help='Target numeric output rows per paired data/metadata part; flushes at read boundaries.')
    parser.add_argument('--effect_write_mb', type=int, default=512,
                        help='MiB bound for the output buffer and, separately, all outstanding writer payloads.')
    parser.add_argument('--effect_writer_threads', type=int, default=2,
                        help='Concurrent writers for columnar part pairs; decoding does not wait for each disk write.')
    parser.add_argument('--effect_queue_size', type=int, default=8,
                        help='Maximum outstanding CPU score packets, including the one being decoded.')
    parser.add_argument('--effect_queue_mb', type=int, default=256,
                        help='Outstanding CPU score-packet payload budget in MiB; excludes producer working memory and Python overhead.')

    add_preprocessing_arguments(parser)
    args = parser.parse_args(argv)
    resolve_model_arguments(args, roles=('prototype',), parser=parser)
    try:
        apply_bundle_arguments(args, 'prototype')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    query_pipeline.validate_pod5_arguments(parser, args)
    if args.scorer == 'nearest-prototype':
        args.metric = args.metric or 'cosine'
    elif args.metric is not None:
        parser.error('--metric requires --scorer nearest-prototype')
    if args.output_format != 'parquet' and args.scorer != 'residual-effects':
        parser.error('--output_format bam/both requires --scorer residual-effects')
    if args.bam_threads < 1:
        parser.error('--bam_threads must be positive')
    if not math.isfinite(args.effect_probability_temperature) or args.effect_probability_temperature <= 0:
        parser.error('--effect_probability_temperature must be finite and positive')
    if args.read_based:
        if args.pos_file or args.use_polished_sequence or args.polish_vcf_path:
            parser.error('--read_based cannot be combined with --pos_file or polishing options')
        if args.motif is None:
            args.motif = 'N 0'
    elif not args.ref or not args.chrom:
        parser.error('--ref and --chrom are required unless --read_based is used')
    if args.scorer == 'residual-effects':
        if not args.effect_model:
            parser.error('--scorer residual-effects requires --effect_model')
        if any(x is not None for x in (args.prototype_embeddings, args.class_labels, args.class_priors, args.classifier_output)):
            parser.error('residual-effects uses the fitted artifact, not prototype/class/prior/classifier-output arguments')
    else:
        if args.effect_model is not None:
            parser.error('--effect_model requires --scorer residual-effects')
        if args.prototype_embeddings is None or args.class_labels is None:
            parser.error('--prototype_embeddings and --class_labels are required for this scorer')
        validate_class_inputs(args.prototype_embeddings, args.class_labels)
        normalize_class_priors(args.class_priors, len(args.class_labels))
        if args.scorer == 'nearest-prototype':
            try:
                validate_nearest_options(args.metric, args.class_priors, len(args.class_labels))
            except ValueError as exc:
                parser.error(str(exc))
    if args.effect_penalty is not None and (not math.isfinite(args.effect_penalty) or args.effect_penalty < 0):
        parser.error('--effect_penalty must be finite and nonnegative')
    if min(args.effect_beam_width, args.effect_max_sweeps, args.effect_threads) < 1:
        parser.error('effect beam width, sweeps, and threads must be positive')
    if args.effect_cache_mb < 0:
        parser.error('--effect_cache_mb must be nonnegative')
    if min(args.effect_block_size, args.effect_queue_size, args.effect_queue_mb) < 1:
        parser.error('effect block size and queue limits must be positive')
    if min(args.effect_batch_reads, args.effect_batch_bases, args.effect_write_rows,
           args.effect_write_mb, args.effect_writer_threads) < 1:
        parser.error('effect scoring batch and output limits must be positive')
    if args.scorer == 'residual-effects' and (args.num_trim < 0 or args.num_workers < 0 or min(args.batch_size, args.reads_per_chunk, args.prefetch_factor) < 1):
        parser.error('invalid effect extraction counts/trim')
    if args.scorer == 'residual-effects' and args.max_batches is not None and args.max_batches < 1:
        parser.error('--max_batches must be positive')
    if args.ridge < 0 or not math.isfinite(args.ridge):
        parser.error("--ridge must be finite and non-negative")
    if args.local_shrinkage is not None and (
        not math.isfinite(args.local_shrinkage)
        or not 0 <= args.local_shrinkage <= 1
    ):
        parser.error("--local_shrinkage must be between zero and one")
    if args.min_reservoir_count < 1:
        parser.error("--min_reservoir_count must be at least one")
    if args.min_prototype_count < 1:
        parser.error("--min_prototype_count must be at least one")
    if args.covariance_batch_size < 1:
        parser.error("--covariance_batch_size must be positive")
    if not math.isfinite(args.score_temperature) or args.score_temperature <= 0:
        parser.error("--score_temperature must be finite and positive")
    if not math.isfinite(args.rbf_bandwidth) or args.rbf_bandwidth <= 0:
        parser.error("--rbf_bandwidth must be finite and positive")
    if not math.isfinite(args.diagonal_shrinkage) or not 0 <= args.diagonal_shrinkage <= 1:
        parser.error("--diagonal_shrinkage must be between zero and one")
    if not math.isfinite(args.variance_floor) or args.variance_floor <= 0:
        parser.error("--variance_floor must be finite and positive")
    if args.checkpoint_batch_step < 1:
        parser.error("--checkpoint_batch_step must be positive")
    if args.log_every < 1:
        parser.error("--log_every must be positive")
    return args


if __name__ == '__main__':
    query_pipeline.require_single_process_environment(
        "porefm prototype classify",
        "Classifier fitting and locus/read-level output accumulation run in one process.",
    )
    print(
        '\n%s:\nCommand: python %s\n'
        % (str(datetime.datetime.now()), ' '.join(sys.argv)),
        flush=True,
    )
    run(parse_args())
