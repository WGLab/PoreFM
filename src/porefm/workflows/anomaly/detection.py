"""Shared anomaly detection implementation for BAM-stream and stored-embedding workflows.

Both the online BAM-processing script and the offline stored-embedding script should
import process_block_anomaly_detection_vectorized from this module. Keeping the
math in one place avoids drift between the two workflows.
"""

from __future__ import annotations

import numpy as np
import torch

from porefm.workflows.anomaly.scoring import PVALUE_METHODS, calculate_pvalues, squared_mahalanobis


def get_intra_group_ranks(counts: torch.Tensor) -> torch.Tensor:
    """Expand consecutive group sizes into zero-based ranks within each group.

    ``counts`` is a one-dimensional integer tensor of group sizes. For example,
    ``[3, 2]`` produces ``[0, 1, 2, 0, 1]``. The output is a CPU ``long``
    tensor, matching this module's CPU-side reservoir construction.
    """
    n = counts.sum().item()
    ranks = torch.ones(n, dtype=torch.long)
    splits = torch.cumsum(counts, dim=0)[:-1]
    ranks[0] = 0
    ranks[splits] = -counts[:-1] + 1
    return torch.cumsum(ranks, dim=0)


def process_block_anomaly_detection_vectorized(
    control_dict,
    test_dict,
    min_control_cov: int = 15,
    min_test_cov: int = 15,
    max_fit_cov: int = 256,
    n_comp: int = 4,
    batch_size: int = 2000,
    score_batch_size: int | None = None,
    max_score_cov: int | None = None,
    pvalue_method: str = "chi2",
    calibration_fraction: float = 0.2,
    device: str | torch.device = "cuda",
    return_projection_details: bool = False,
):
    """Score test-read embeddings against per-locus control distributions.

    The calculation is performed independently by strand and genomic position:
    coverage filtering, random within-locus reservoir sampling, control-based
    standardization, joint PCA, and Mahalanobis scoring. Test reads are all
    scored unless ``max_score_cov`` limits their random per-locus reservoir.

    Parameters
    ----------
    control_dict, test_dict : dict
        Block data containing CPU ``positions`` and ``strands`` tensors,
        ``embeddings`` of shape ``(n_reads, embedding_dim)``, and, for test
        data, a NumPy ``readnames`` array aligned row-for-row.
    min_control_cov, min_test_cov : int
        Minimum per-position coverage required in each population. For
        empirical scoring, ``min_control_cov`` applies after reserving the
        calibration subset.
    max_fit_cov : int
        Maximum number of randomly selected rows per position and population
        used to fit standardization, PCA, and control null statistics.
    n_comp : int
        Number of leading joint-PCA components used for Mahalanobis distances.
    batch_size : int
        Number of genomic positions processed in each dense linear-algebra
        batch.
    score_batch_size : int or None
        Optional limit on read embeddings moved to ``device`` for each scoring
        operation. This does not change model fitting or which reads are scored.
    max_score_cov : int or None
        Optional random cap on test reads scored per position.
    pvalue_method : {``"chi2"``, ``"empirical"``}
        Chi-square approximation or finite-sample empirical upper-tail p-values.
        Empirical mode splits controls independently within each locus; fit
        controls estimate the null, while held-out controls calibrate p-values.
    calibration_fraction : float
        Fraction of each locus's controls reserved in empirical mode. Must lie
        strictly between zero and one.
    device : str or torch.device
        Device used for dense linear algebra and read scoring.
    return_projection_details : bool
        Return each scored test row's coordinates in the retained PCA space
        and the fitted per-locus control-distribution parameters. This is
        disabled by default so callers that only need anomaly scores do not
        retain additional arrays.

    Returns
    -------
    dict
        NumPy arrays ``readnames``, ``positions``, ``p_vals``, ``strands``, and
        ``squared_mahalanobis`` for scored test rows. ``row_indices`` contains
        their zero-based indices in the original ``test_dict`` arrays, in
        exactly the same randomized output order. When projection details
        are requested, ``principal_components`` has shape
        ``(n_scored_reads, n_comp)`` and ``site_statistics`` contains coverage
        and fitted control mean/covariance arrays aligned by site. Arrays are
        empty when no locus passes filtering.

    Notes
    -----
    Reservoir selection uses PyTorch and NumPy random draws and is therefore
    stochastic unless callers seed both generators.
    """
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available")
    if score_batch_size is not None and score_batch_size < 1:
        raise ValueError(f"score_batch_size must be >= 1 when set, got {score_batch_size}")
    if max_score_cov is not None and max_score_cov < 1:
        raise ValueError(f"max_score_cov must be >= 1 when set, got {max_score_cov}")
    if pvalue_method not in PVALUE_METHODS:
        raise ValueError(
            f"pvalue_method must be one of {PVALUE_METHODS}, got {pvalue_method!r}"
        )
    if pvalue_method == "empirical" and not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            "calibration_fraction must be strictly between 0 and 1 for "
            f"empirical p-values, got {calibration_fraction}"
        )

    out_readnames, out_positions, out_pvals, out_strands = [], [], [], []
    out_row_indices = []
    out_squared_mahalanobis = []
    out_principal_components = []
    site_statistics = {
        "positions": [],
        "strands": [],
        "control_coverage": [],
        "test_coverage": [],
        "control_fit_coverage": [],
        "test_fit_coverage": [],
        "scored_test_coverage": [],
        "calibration_coverage": [],
        "calibration_pca_coverage": [],
        "control_pca_mean": [],
        "control_pca_covariance": [],
    }

    def randomize_and_rank(pos, row_indices):
        """Order integer row references using the original random draws."""
        rands = torch.rand(len(pos)).numpy()
        order = torch.tensor(np.lexsort((rands, pos.numpy())))

        pos_sorted = pos[order]
        indices_sorted = row_indices[order]

        unique_pos, counts = torch.unique_consecutive(
            pos_sorted,
            return_counts=True,
        )
        ranks = get_intra_group_ranks(counts)
        return pos_sorted, indices_sorted, unique_pos, counts, ranks

    for strand_val in [0, 1]:
        # Model forward and reverse observations independently so genomic
        # positions with different strand-specific backgrounds do not mix.
        c_mask = control_dict["strands"] == strand_val
        t_mask = test_dict["strands"] == strand_val
        if not c_mask.any() or not t_mask.any():
            continue

        c_pos = control_dict["positions"][c_mask]
        c_indices = c_mask.nonzero(as_tuple=True)[0]

        t_pos = test_dict["positions"][t_mask]
        t_indices = t_mask.nonzero(as_tuple=True)[0]

        # Coverage filtering precedes dense allocation; empirical mode also
        # reserves held-out controls separately within every locus.
        c_unq, c_cnt = torch.unique(c_pos, return_counts=True)
        t_unq, t_cnt = torch.unique(t_pos, return_counts=True)

        if pvalue_method == "empirical":
            (
                c_sorted_pos,
                c_sorted_indices,
                c_split_unq,
                c_split_counts,
                c_split_ranks,
            ) = randomize_and_rank(c_pos, c_indices)

            calibration_counts = torch.floor(
                c_split_counts.to(torch.float64) * calibration_fraction
            ).to(torch.long)
            calibration_counts = calibration_counts.clamp(min=1)
            calibration_counts = torch.minimum(
                calibration_counts,
                (c_split_counts - 1).clamp(min=0),
            )
            control_fit_counts = c_split_counts - calibration_counts
            valid_c = c_split_unq[
                (control_fit_counts >= min_control_cov)
                & (calibration_counts >= 1)
            ].numpy()
        else:
            valid_c = c_unq[c_cnt >= min_control_cov].numpy()

        valid_t = t_unq[t_cnt >= min_test_cov].numpy()
        valid_pos = torch.tensor(np.intersect1d(valid_c, valid_t), dtype=c_pos.dtype)

        if len(valid_pos) == 0:
            continue

        # Preserve uncapped population coverage separately from the random
        # fitting reservoirs. These counts are site metadata, not per-read
        # measurements.
        full_control_counts = c_cnt[torch.searchsorted(c_unq, valid_pos)]
        full_test_counts = t_cnt[torch.searchsorted(t_unq, valid_pos)]

        # Random ranks provide fixed-size per-locus fitting reservoirs while
        # preserving flat arrays for scoring all eligible test reads.
        def sample_and_rank(pos, row_indices):
            """Select the same reservoirs without copying embedding matrices."""
            valid_mask = torch.isin(pos, valid_pos)
            pos_v = pos[valid_mask]
            indices_v = row_indices[valid_mask]

            pos_sorted, indices_sorted, _, counts, ranks = randomize_and_rank(
                pos_v,
                indices_v,
            )

            fit_mask = ranks < max_fit_cov

            return (
                pos_sorted[fit_mask],
                indices_sorted[fit_mask],
                ranks[fit_mask],
                counts.clamp(max=max_fit_cov),
                pos_sorted,
                indices_sorted,
                ranks,
            )

        if pvalue_method == "empirical":
            repeated_control_fit_counts = torch.repeat_interleave(
                control_fit_counts,
                c_split_counts,
            )
            valid_control_mask = torch.isin(c_sorted_pos, valid_pos)
            control_fit_mask = c_split_ranks < repeated_control_fit_counts
            calibration_mask = ~control_fit_mask

            c_fit_mask = (
                valid_control_mask
                & control_fit_mask
                & (c_split_ranks < max_fit_cov)
            )
            c_fit_pos = c_sorted_pos[c_fit_mask]
            c_fit_indices = c_sorted_indices[c_fit_mask]
            c_fit_ranks = c_split_ranks[c_fit_mask]

            calibration_ranks = c_split_ranks - repeated_control_fit_counts
            calibration_valid_mask = valid_control_mask & calibration_mask
            calibration_pca_mask = (
                calibration_valid_mask & (calibration_ranks < max_fit_cov)
            )
            calibration_fit_pos = c_sorted_pos[calibration_pca_mask]
            calibration_fit_indices = c_sorted_indices[calibration_pca_mask]
            calibration_fit_ranks = calibration_ranks[calibration_pca_mask]
            calibration_all_pos = c_sorted_pos[calibration_valid_mask]
            calibration_all_indices = c_sorted_indices[calibration_valid_mask]

            valid_control_groups = torch.isin(c_split_unq, valid_pos)
            c_fit_counts = control_fit_counts[valid_control_groups].clamp(
                max=max_fit_cov
            )
            calibration_fit_counts = calibration_counts[
                valid_control_groups
            ].clamp(max=max_fit_cov)
        else:
            (
                c_fit_pos,
                c_fit_indices,
                c_fit_ranks,
                c_fit_counts,
                _,
                _,
                _,
            ) = sample_and_rank(c_pos, c_indices)

        t_fit_pos, t_fit_indices, t_fit_ranks, t_fit_counts, t_all_pos, t_all_indices, t_all_ranks = sample_and_rank(
            t_pos, t_indices
        )

        # Batch loci before scattering their ragged reservoirs into dense
        # tensors used by the device-side eigendecomposition.
        n_valid = len(valid_pos)
        dim = control_dict["embeddings"].shape[1]
        fit_arange = torch.arange(max_fit_cov, device=device)
        eye_dim = torch.eye(dim, device=device)
        eye_ncomp = torch.eye(n_comp, device=device)

        for start_idx in range(0, n_valid, batch_size):
            end_idx = min(start_idx + batch_size, n_valid)
            batch_pos = valid_pos[start_idx:end_idx]
            bsz = len(batch_pos)

            # Extract chunk data from the flat sorted arrays using searchsorted.
            c_s = torch.searchsorted(c_fit_pos, batch_pos[0])
            c_e = torch.searchsorted(c_fit_pos, batch_pos[-1], right=True)
            c_b_pos = c_fit_pos[c_s:c_e]
            c_b_emb = control_dict["embeddings"][c_fit_indices[c_s:c_e]].to(torch.float32)
            c_b_ranks = c_fit_ranks[c_s:c_e]

            t_s = torch.searchsorted(t_fit_pos, batch_pos[0])
            t_e = torch.searchsorted(t_fit_pos, batch_pos[-1], right=True)
            t_b_pos = t_fit_pos[t_s:t_e]
            t_b_emb = test_dict["embeddings"][t_fit_indices[t_s:t_e]].to(torch.float32)
            t_b_ranks = t_fit_ranks[t_s:t_e]

            ta_s = torch.searchsorted(t_all_pos, batch_pos[0])
            ta_e = torch.searchsorted(t_all_pos, batch_pos[-1], right=True)
            ta_b_pos = t_all_pos[ta_s:ta_e]
            ta_b_indices = t_all_indices[ta_s:ta_e]
            ta_b_ranks = t_all_ranks[ta_s:ta_e]

            if pvalue_method == "empirical":
                calibration_fit_start = torch.searchsorted(
                    calibration_fit_pos,
                    batch_pos[0],
                )
                calibration_fit_end = torch.searchsorted(
                    calibration_fit_pos,
                    batch_pos[-1],
                    right=True,
                )
                calibration_b_pos = calibration_fit_pos[
                    calibration_fit_start:calibration_fit_end
                ]
                calibration_b_emb = control_dict["embeddings"][
                    calibration_fit_indices[calibration_fit_start:calibration_fit_end]
                ].to(torch.float32)
                calibration_b_ranks = calibration_fit_ranks[
                    calibration_fit_start:calibration_fit_end
                ]

                calibration_all_start = torch.searchsorted(
                    calibration_all_pos,
                    batch_pos[0],
                )
                calibration_all_end = torch.searchsorted(
                    calibration_all_pos,
                    batch_pos[-1],
                    right=True,
                )
                calibration_score_pos = calibration_all_pos[
                    calibration_all_start:calibration_all_end
                ]
                calibration_score_indices = calibration_all_indices[
                    calibration_all_start:calibration_all_end
                ]

            if max_score_cov is not None:
                score_mask = ta_b_ranks < max_score_cov
                ta_b_pos = ta_b_pos[score_mask]
                ta_b_indices = ta_b_indices[score_mask]

            # Scatter into dense tensors for batched linear algebra.
            c_res = torch.zeros((bsz, max_fit_cov, dim), dtype=torch.float32, device=device)
            t_res = torch.zeros((bsz, max_fit_cov, dim), dtype=torch.float32, device=device)

            c_row_idx = torch.searchsorted(batch_pos, c_b_pos)
            t_row_idx = torch.searchsorted(batch_pos, t_b_pos)

            c_res[c_row_idx, c_b_ranks] = c_b_emb.to(device)
            t_res[t_row_idx, t_b_ranks] = t_b_emb.to(device)

            c_counts = c_fit_counts[start_idx:end_idx].to(device)
            t_counts = t_fit_counts[start_idx:end_idx].to(device)

            if pvalue_method == "empirical":
                calibration_res = torch.zeros(
                    (bsz, max_fit_cov, dim),
                    dtype=torch.float32,
                    device=device,
                )
                calibration_row_idx = torch.searchsorted(
                    batch_pos,
                    calibration_b_pos,
                )
                calibration_res[
                    calibration_row_idx,
                    calibration_b_ranks,
                ] = calibration_b_emb.to(device)
                calibration_counts_batch = calibration_fit_counts[
                    start_idx:end_idx
                ].to(device)

            # Masked per-locus standardization.
            c_mask_bool = fit_arange.expand(bsz, max_fit_cov) < c_counts.unsqueeze(1)
            t_mask_bool = fit_arange.expand(bsz, max_fit_cov) < t_counts.unsqueeze(1)

            c_mask_f = c_mask_bool.unsqueeze(-1).float()
            c_mean = (c_res * c_mask_f).sum(dim=1) / c_counts.unsqueeze(1)
            c_var = (((c_res - c_mean.unsqueeze(1)) * c_mask_f) ** 2).sum(dim=1) / (
                c_counts.unsqueeze(1) - 1
            ).clamp(min=1)
            c_std = torch.sqrt(c_var).clamp(min=1e-8)

            c_res_std = ((c_res - c_mean.unsqueeze(1)) / c_std.unsqueeze(1)) * c_mask_f
            t_res_std = ((t_res - c_mean.unsqueeze(1)) / c_std.unsqueeze(1)) * t_mask_bool.unsqueeze(-1).float()

            # Batched joint PCA via eigendecomposition of X^T X.
            joint_res_parts = [c_res_std, t_res_std]
            joint_mask_parts = [c_mask_bool, t_mask_bool]
            joint_counts = c_counts + t_counts

            if pvalue_method == "empirical":
                calibration_mask_bool = (
                    fit_arange.expand(bsz, max_fit_cov)
                    < calibration_counts_batch.unsqueeze(1)
                )
                calibration_res_std = (
                    (calibration_res - c_mean.unsqueeze(1))
                    / c_std.unsqueeze(1)
                ) * calibration_mask_bool.unsqueeze(-1).float()
                joint_res_parts.append(calibration_res_std)
                joint_mask_parts.append(calibration_mask_bool)
                joint_counts = joint_counts + calibration_counts_batch

            joint_res = torch.cat(joint_res_parts, dim=1)
            joint_mask = torch.cat(joint_mask_parts, dim=1).unsqueeze(-1).float()

            joint_mean = (joint_res * joint_mask).sum(dim=1) / joint_counts.unsqueeze(1)
            joint_centered = (joint_res - joint_mean.unsqueeze(1)) * joint_mask

            c_joint = torch.bmm(joint_centered.transpose(1, 2), joint_centered)
            c_joint += eye_dim * 1e-6

            _, evecs = torch.linalg.eigh(c_joint)
            top_evecs = evecs[:, :, -n_comp:]
            vh_k = top_evecs.transpose(1, 2).flip(dims=[1])

            # Control PCA statistics.
            c_pca = torch.bmm((c_res_std - joint_mean.unsqueeze(1)) * c_mask_f, vh_k.transpose(1, 2))
            c_pca_mean = (c_pca * c_mask_bool.unsqueeze(-1).float()).sum(dim=1) / c_counts.unsqueeze(1)
            c_pca_centered = (c_pca - c_pca_mean.unsqueeze(1)) * c_mask_bool.unsqueeze(-1).float()

            c_cov = torch.bmm(c_pca_centered.transpose(1, 2), c_pca_centered) / (
                c_counts.unsqueeze(1).unsqueeze(1).float() - 1.0
            )
            c_cov += eye_ncomp * 1e-6
            c_cov_inv = torch.linalg.inv(c_cov)

            # score_batch_size only chunks independent per-read scoring. It does
            # not change PCA/null fitting or subsample the scored populations.
            def score_embeddings(source_embeddings, source_indices, flat_row_idx, keep_projections=False):
                """Gather original rows and retain the existing scoring operations."""
                effective_batch_size = score_batch_size or max(
                    1,
                    len(source_indices),
                )
                squared_distance_chunks = []
                projection_chunks = []
                for score_start in range(
                    0,
                    len(source_indices),
                    effective_batch_size,
                ):
                    score_end = min(
                        score_start + effective_batch_size,
                        len(source_indices),
                    )
                    # Convert on CPU before transfer, as in the original scorer.
                    embeddings_gpu = source_embeddings[
                        source_indices[score_start:score_end]
                    ].to(torch.float32).to(device)
                    row_idx_chunk = flat_row_idx[score_start:score_end]

                    expanded_c_mean = c_mean[row_idx_chunk]
                    expanded_c_std = c_std[row_idx_chunk]
                    expanded_joint_mean = joint_mean[row_idx_chunk]
                    expanded_vh_k = vh_k[row_idx_chunk]
                    expanded_pca_mean = c_pca_mean[row_idx_chunk]
                    expanded_cov_inv = c_cov_inv[row_idx_chunk]

                    standardized = (
                        embeddings_gpu - expanded_c_mean
                    ) / expanded_c_std
                    centered = standardized - expanded_joint_mean
                    projected = torch.bmm(
                        centered.unsqueeze(1),
                        expanded_vh_k.transpose(1, 2),
                    ).squeeze(1)
                    if keep_projections:
                        projection_chunks.append(projected.cpu().numpy())
                    squared_distance_chunks.append(
                        squared_mahalanobis(
                            projected,
                            expanded_pca_mean,
                            expanded_cov_inv,
                        ).cpu().numpy()
                    )

                if not squared_distance_chunks:
                    empty_projections = (
                        np.empty((0, n_comp), dtype=np.float32)
                        if keep_projections
                        else None
                    )
                    return np.array([], dtype=np.float64), empty_projections
                projections = (
                    np.concatenate(projection_chunks)
                    if keep_projections
                    else None
                )
                return np.concatenate(squared_distance_chunks), projections

            ta_row_idx = torch.searchsorted(batch_pos, ta_b_pos)
            test_squared_distances, test_principal_components = score_embeddings(
                test_dict["embeddings"],
                ta_b_indices,
                ta_row_idx,
                keep_projections=return_projection_details,
            )

            if pvalue_method == "empirical":
                calibration_score_row_idx = torch.searchsorted(
                    batch_pos,
                    calibration_score_pos,
                )
                calibration_squared_distances, _ = score_embeddings(
                    control_dict["embeddings"],
                    calibration_score_indices,
                    calibration_score_row_idx,
                )
                pvalues = calculate_pvalues(
                    "empirical",
                    test_squared_distances,
                    groups=ta_b_pos.numpy(),
                    calibration_distances=calibration_squared_distances,
                    calibration_groups=calibration_score_pos.numpy(),
                )
            else:
                pvalues = calculate_pvalues(
                    "chi2",
                    test_squared_distances,
                    degrees_of_freedom=n_comp,
                )

            if return_projection_details:
                scored_test_counts = torch.bincount(
                    ta_row_idx,
                    minlength=bsz,
                )
                site_statistics["positions"].append(batch_pos.numpy())
                site_statistics["strands"].append(
                    np.full(bsz, strand_val, dtype=np.int8)
                )
                site_statistics["control_coverage"].append(
                    full_control_counts[start_idx:end_idx].numpy()
                )
                site_statistics["test_coverage"].append(
                    full_test_counts[start_idx:end_idx].numpy()
                )
                site_statistics["control_fit_coverage"].append(
                    c_counts.cpu().numpy()
                )
                site_statistics["test_fit_coverage"].append(
                    t_counts.cpu().numpy()
                )
                site_statistics["scored_test_coverage"].append(
                    scored_test_counts.numpy()
                )
                if pvalue_method == "empirical":
                    site_statistics["calibration_coverage"].append(
                        calibration_counts[valid_control_groups][
                            start_idx:end_idx
                        ].numpy()
                    )
                    site_statistics["calibration_pca_coverage"].append(
                        calibration_counts_batch.cpu().numpy()
                    )
                else:
                    site_statistics["calibration_coverage"].append(
                        np.zeros(bsz, dtype=np.int64)
                    )
                    site_statistics["calibration_pca_coverage"].append(
                        np.zeros(bsz, dtype=np.int64)
                    )
                site_statistics["control_pca_mean"].append(
                    c_pca_mean.cpu().numpy()
                )
                # This is the regularized covariance actually inverted for
                # Mahalanobis scoring, including the 1e-6 diagonal ridge.
                site_statistics["control_pca_covariance"].append(
                    c_cov.cpu().numpy()
                )
                out_principal_components.append(test_principal_components)

            out_row_indices.append(ta_b_indices.numpy())
            out_readnames.append(test_dict["readnames"][ta_b_indices.numpy()])
            out_positions.append(ta_b_pos.numpy())
            out_pvals.append(pvalues)
            out_squared_mahalanobis.append(test_squared_distances)
            out_strands.append(
                np.full(len(ta_b_pos), strand_val, dtype=np.int8)
            )

    if len(out_readnames) == 0:
        result = {
            "readnames": np.array([]),
            "positions": np.array([]),
            "p_vals": np.array([]),
            "strands": np.array([]),
            "squared_mahalanobis": np.array([]),
        }
    else:
        result = {
            "readnames": np.concatenate(out_readnames),
            "positions": np.concatenate(out_positions),
            "p_vals": np.concatenate(out_pvals),
            "strands": np.concatenate(out_strands),
            "squared_mahalanobis": np.concatenate(out_squared_mahalanobis),
        }

    if return_projection_details:
        result["principal_components"] = (
            np.concatenate(out_principal_components)
            if out_principal_components
            else np.empty((0, n_comp), dtype=np.float32)
        )
        result["site_statistics"] = {
            name: (
                np.concatenate(parts, axis=0)
                if parts
                else np.empty(
                    (0, n_comp, n_comp)
                    if name == "control_pca_covariance"
                    else (0, n_comp)
                    if name == "control_pca_mean"
                    else (0,),
                    dtype=np.float32
                    if name.startswith("control_pca_")
                    else np.int64,
                )
            )
            for name, parts in site_statistics.items()
        }
    result["row_indices"] = (
        np.concatenate(out_row_indices)
        if out_row_indices
        else np.empty(0, dtype=np.int64)
    )
    return result
