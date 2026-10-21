"""Canonical-only 5-mer covariance pooling after within-9-mer centering."""

import math

import torch


def centered_id(kmer_id, source_k, target_k):
    """Extract a centered shorter two-bit k-mer ID without changing orientation."""
    if source_k < target_k or (source_k - target_k) % 2:
        raise ValueError('target context must fit symmetrically inside source context')
    return (int(kmer_id) // 4 ** ((source_k - target_k) // 2)) % (4 ** target_k)


def row_whitening(covariance):
    """Return float32 W with W.T @ covariance @ W = I, using float64 Cholesky."""
    covariance = covariance.double()
    chol = torch.linalg.cholesky(covariance)
    return torch.linalg.solve_triangular(chol, torch.eye(len(chol), dtype=torch.double), upper=False).T.float()


def fit_hierarchical_noise(groups, dim, *, shrinkage=.1, floor=1e-6, strength=512., local_shrinkage=None):
    """Fit sparse receiver-5-mer metrics with an independently centered 9-mer null.

    Parameters
    ----------
    groups : iterable of (int, int, Tensor)
        (9-mer ID, k=9, observations[N,D]), already restricted to one central
        base family. Each group/source is centered on its own sample mean.
        Scatter matrices, not raw means, are then pooled by central 5-mer.
    dim : int
        Embedding width D.
    shrinkage, floor : float
        Isotropic shrinkage and absolute diagonal floor for the pooled base
        covariance. Each local empirical covariance also receives this floor.
    strength : float
        Positive pseudodegrees of freedom. Automatic local-to-base weight is
        strength/(strength + local_df); this is a heuristic, not Ledoit-Wolf.
    local_shrinkage : float or None
        Optional fixed local-to-base weight. Zero gives local+floor; one gives
        exactly the pooled base metric. Unsupported complete 5-mers always use
        that base metric. No modified data or query reads enter estimation.

    Returns
    -------
    dict
        Base covariance/whitening, sparse sorted 5-mer IDs and whitening
        matrices, pooling weights, and sample/degree counts. Temporary scatter
        storage is O(observed_5mers*D^2); scoring need not expand matrices per
        read position. Full local covariance matrices are not saved twice.
    """
    if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError('noise shrinkage must be in [0, 1]')
    if not math.isfinite(floor) or floor <= 0 or not math.isfinite(strength) or strength <= 0:
        raise ValueError('noise floor and local strength must be finite and positive')
    if local_shrinkage is not None and (not math.isfinite(local_shrinkage) or not 0 <= local_shrinkage <= 1):
        raise ValueError('local noise shrinkage must be in [0, 1]')
    scatter = {}
    total = torch.zeros(dim, dim, dtype=torch.double)
    total_df, observations, contexts = 0, 0, 0
    center_base = None
    for kid, k, values in groups:
        if k != 9 or not 0 <= int(kid) < 4 ** 9:
            raise ValueError('hierarchical-5mer noise requires canonical 9-mer reservoirs (kmer_size=9)')
        base = centered_id(kid, 9, 1)
        if center_base is None:
            center_base = base
        if base != center_base:
            raise ValueError('hierarchical noise groups must belong to one center-base family')
        x = values.to(device='cpu', dtype=torch.double)
        if x.ndim != 2 or x.shape[1] != dim or not torch.isfinite(x).all():
            raise ValueError('invalid canonical noise observations')
        if len(x) < 2:
            continue
        residual = x - x.mean(0)
        cross, df = residual.T @ residual, len(x) - 1
        key = centered_id(kid, 9, 5)
        if key not in scatter:
            scatter[key] = [torch.zeros_like(total), 0, 0, 0]
        row = scatter[key]
        row[0] += cross
        row[1] += df
        row[2] += len(x)
        row[3] += 1
        total += cross
        total_df += df
        observations += len(x)
        contexts += 1
    if not total_df:
        raise ValueError('canonical noise needs at least two observations in a context')
    empirical = total / total_df
    identity = torch.eye(dim, dtype=torch.double)
    base_cov = (1 - shrinkage) * empirical + (shrinkage * empirical.diag().mean() + floor) * identity
    ids = sorted(scatter)
    transforms = torch.empty(len(ids), dim, dim)
    weights, degrees, counts, group_counts = [], [], [], []
    for index, kid in enumerate(ids):
        cross, df, n, groups_n = scatter.pop(kid)
        weight = strength / (strength + df) if local_shrinkage is None else local_shrinkage
        local_cov = (1 - weight) * (cross / df + floor * identity) + weight * base_cov
        transforms[index] = row_whitening(local_cov)
        weights.append(weight)
        degrees.append(df)
        counts.append(n)
        group_counts.append(groups_n)
    return dict(covariance=base_cov.float(), whitening=row_whitening(base_cov),
                observations=observations, contexts=contexts, degrees_of_freedom=total_df,
                context_ids=torch.tensor(ids, dtype=torch.long), context_whitening=transforms,
                local_weights=torch.tensor(weights), local_degrees=torch.tensor(degrees),
                local_observations=torch.tensor(counts), local_contexts=torch.tensor(group_counts),
                centering_k=9, receiving_k=5)


def receiver_whitening(family, context_id):
    """Return a shared matrix view, or the base fallback for an unseen 5-mer.

    Incomplete IDs (<0) return None in hierarchical mode, making the observation
    unavailable. Other modes need only the center base. This never allocates
    an L-by-D-by-D array while preparing a read.
    """
    if family['noise_mode'] != 'hierarchical-5mer':
        return family['whitening']
    if int(context_id) < 0:
        return None
    keys = family['noise_context_ids']
    idx = int(torch.searchsorted(keys, int(context_id)))
    if idx < len(keys) and int(keys[idx]) == int(context_id):
        return family['noise_context_whitening'][idx]
    return family['whitening']
