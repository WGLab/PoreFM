"""Selectable anomaly-score to p-value conversions.

The embedding transformation and joint-PCA fit live in
``anomaly_detection_shared.py``.  This module keeps the final statistical
scoring choices small and explicit so streaming and stored-embedding workflows
use the same implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import chdtrc


PVALUE_METHODS = ("chi2", "empirical")


def squared_mahalanobis(
    values: torch.Tensor,
    means: torch.Tensor,
    precision_matrices: torch.Tensor,
) -> torch.Tensor:
    """Compute row-wise squared Mahalanobis distances for batched null models.

    Parameters
    ----------
    values, means : torch.Tensor
        Tensors of shape ``(n, d)`` containing observations and the matching
        group-specific means.
    precision_matrices : torch.Tensor
        Inverse covariance matrices with shape ``(n, d, d)``.

    Returns
    -------
    torch.Tensor
        Squared distances with shape ``(n,)`` on the input device.
    """
    differences = values - means
    left_terms = torch.bmm(
        differences.unsqueeze(1),
        precision_matrices,
    ).squeeze(1)
    return (left_terms * differences).sum(dim=-1)


def chi_square_pvalues(
    squared_distances: np.ndarray,
    *,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Map squared Mahalanobis distances to chi-square survival probabilities.

    ``degrees_of_freedom`` normally equals the dimensionality of the projected
    score space. The returned array has the same broadcasted shape as
    ``squared_distances`` and ``float64`` dtype.
    """
    if degrees_of_freedom < 1:
        raise ValueError(
            f"degrees_of_freedom must be >= 1, got {degrees_of_freedom}"
        )
    return np.asarray(
        chdtrc(degrees_of_freedom, np.asarray(squared_distances)),
        dtype=np.float64,
    )


def empirical_survival_pvalues(
    squared_distances: np.ndarray,
    *,
    groups: np.ndarray,
    calibration_distances: np.ndarray,
    calibration_groups: np.ndarray,
) -> np.ndarray:
    """Calculate group-conditional empirical upper-tail p-values.

    For test score ``s`` and the ``m`` calibration scores at the same group,
    the finite-sample corrected p-value is::

        (1 + number of calibration scores >= s) / (m + 1)

    In the read-tagging workflow, a group is a genomic position within the
    strand currently being processed.

    Parameters
    ----------
    squared_distances, groups : np.ndarray
        One-dimensional test scores and their group identifiers.
    calibration_distances, calibration_groups : np.ndarray
        One-dimensional calibration scores and group identifiers. Every test
        group must have at least one calibration observation.

    Returns
    -------
    np.ndarray
        ``float64`` p-values in the original test-row order.
    """
    squared_distances = np.asarray(squared_distances)
    groups = np.asarray(groups)
    calibration_distances = np.asarray(calibration_distances)
    calibration_groups = np.asarray(calibration_groups)

    if squared_distances.ndim != 1 or groups.ndim != 1:
        raise ValueError("squared_distances and groups must be one-dimensional")
    if calibration_distances.ndim != 1 or calibration_groups.ndim != 1:
        raise ValueError(
            "calibration_distances and calibration_groups must be one-dimensional"
        )
    if len(squared_distances) != len(groups):
        raise ValueError("squared_distances and groups must have the same length")
    if len(calibration_distances) != len(calibration_groups):
        raise ValueError(
            "calibration_distances and calibration_groups must have the same length"
        )

    # Sort once by group and then score. Avoid repeatedly scanning every
    # calibration row for each genomic position in a large anomaly batch.
    calibration_order = np.lexsort(
        (calibration_distances, calibration_groups)
    )
    sorted_calibration_groups = calibration_groups[calibration_order]
    sorted_calibration_distances = calibration_distances[calibration_order]

    pvalues = np.empty(len(squared_distances), dtype=np.float64)
    for group in np.unique(groups):
        test_mask = groups == group
        calibration_start = np.searchsorted(
            sorted_calibration_groups,
            group,
            side="left",
        )
        calibration_end = np.searchsorted(
            sorted_calibration_groups,
            group,
            side="right",
        )
        group_calibration = sorted_calibration_distances[
            calibration_start:calibration_end
        ]
        if len(group_calibration) == 0:
            raise ValueError(f"No calibration scores are available for group {group!r}")

        test_scores = squared_distances[test_mask]
        first_greater_or_equal = np.searchsorted(
            group_calibration,
            test_scores,
            side="left",
        )
        greater_or_equal_count = len(group_calibration) - first_greater_or_equal
        pvalues[test_mask] = (1.0 + greater_or_equal_count) / (
            len(group_calibration) + 1.0
        )

    return pvalues


def calculate_pvalues(
    method: str,
    squared_distances: np.ndarray,
    *,
    degrees_of_freedom: int | None = None,
    groups: np.ndarray | None = None,
    calibration_distances: np.ndarray | None = None,
    calibration_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Convert anomaly distances to p-values using the selected null model.

    Parameters specific to the unselected method may be omitted. ``chi2``
    requires ``degrees_of_freedom``; ``empirical`` requires test ``groups`` and
    the two calibration arrays.
    """
    if method == "chi2":
        if degrees_of_freedom is None:
            raise ValueError("degrees_of_freedom is required for chi2 p-values")
        return chi_square_pvalues(
            squared_distances,
            degrees_of_freedom=degrees_of_freedom,
        )

    if method == "empirical":
        if groups is None:
            raise ValueError("groups are required for empirical p-values")
        if calibration_distances is None or calibration_groups is None:
            raise ValueError(
                "calibration distances and groups are required for empirical p-values"
            )
        return empirical_survival_pvalues(
            squared_distances,
            groups=groups,
            calibration_distances=calibration_distances,
            calibration_groups=calibration_groups,
        )

    raise ValueError(
        f"Unknown p-value method {method!r}; expected one of {PVALUE_METHODS}"
    )
