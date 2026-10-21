"""Shared downstream chunk-stride policy.

Pretraining may deliberately use a small stride to upsample overlapping
windows. Downstream training, inference, and embedding analysis instead use a
modality-aware overlap unless the caller explicitly supplies a stride.
"""

from __future__ import annotations

from numbers import Integral


DEFAULT_OVERLAP = {
    "dna": 128,
    "rna": 256,
}


def resolve_downstream_step(
    size: int,
    seq_type: str,
    step: int | None = None,
    *,
    minimum_overlap: int = 0,
    context: str = "downstream chunking",
    announce: bool = True,
) -> int:
    """Resolve and optionally announce a downstream chunk step.

    Parameters
    ----------
    size : int
        Positive model-level window length.
    seq_type : {"dna", "rna"}
        Sequencing modality controlling the automatic overlap. DNA reserves
        128 time steps and RNA reserves 256.
    step : int or None, default=None
        Explicit positive stride. When omitted, the modality-specific overlap
        is subtracted from ``size``. If that would produce a non-positive
        stride, up to three quarters of ``size`` is used while still honoring
        ``minimum_overlap``.
    minimum_overlap : int, default=0
        Additional lower bound on ``size - step``. Automatic selection expands
        its overlap to this value. An explicit step that violates the bound is
        rejected rather than silently leaving context-invalid gaps.
    context : str, default="downstream chunking"
        Human-readable operation included in the announcement.
    announce : bool, default=True
        Print the selected size, step, overlap, modality, and selection source.
        Distributed callers should normally enable this only on rank zero.

    Returns
    -------
    int
        Validated step in the same model-level units as ``size``.

    Raises
    ------
    TypeError
        If ``size`` or an explicit ``step`` is not an integer.
    ValueError
        If dimensions are non-positive, ``step`` exceeds ``size``, or the
        sequencing modality is unsupported.
    """
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError(f"size must be an integer, got {type(size).__name__}")
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    normalized_seq_type = str(seq_type).lower()
    if normalized_seq_type not in DEFAULT_OVERLAP:
        raise ValueError(
            "seq_type must be one of "
            f"{sorted(DEFAULT_OVERLAP)}, got {seq_type!r}"
        )

    if isinstance(minimum_overlap, bool) or not isinstance(minimum_overlap, Integral):
        raise TypeError(
            "minimum_overlap must be an integer, got "
            f"{type(minimum_overlap).__name__}"
        )
    minimum_overlap = int(minimum_overlap)
    if minimum_overlap < 0:
        raise ValueError(
            f"minimum_overlap must be non-negative, got {minimum_overlap}"
        )
    if minimum_overlap >= size and minimum_overlap > 0:
        raise ValueError(
            f"minimum_overlap must be smaller than size, got "
            f"minimum_overlap={minimum_overlap}, size={size}"
        )

    if step is None:
        default_overlap = DEFAULT_OVERLAP[normalized_seq_type]
        desired_overlap = max(default_overlap, minimum_overlap)
        resolved = size - desired_overlap
        if resolved > 0:
            policy = (
                "automatic minimum context overlap"
                if minimum_overlap > default_overlap
                else f"automatic {normalized_seq_type.upper()} overlap"
            )
        else:
            quarter_window_fallback = max(1, (3 * size) // 4)
            maximum_context_step = size - minimum_overlap
            resolved = min(quarter_window_fallback, maximum_context_step)
            policy = (
                "automatic minimum context overlap fallback"
                if resolved < quarter_window_fallback
                else "automatic quarter-window overlap fallback"
            )
    else:
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError(
                f"explicit step must be an integer, got {type(step).__name__}"
            )
        resolved = int(step)
        policy = "explicit override"

    if resolved < 1:
        raise ValueError(f"step must be >= 1, got {resolved}")
    if resolved > size:
        raise ValueError(
            f"step must be <= size to avoid uncovered gaps, got step={resolved}, "
            f"size={size}"
        )
    actual_overlap = size - resolved
    if actual_overlap < minimum_overlap:
        raise ValueError(
            f"step={resolved} leaves overlap={actual_overlap}, but {context} "
            f"requires minimum_overlap={minimum_overlap}; use step <= "
            f"{size - minimum_overlap}"
        )

    if announce:
        print(
            f"{context}: selected chunk step={resolved} for size={size} "
            f"(overlap={actual_overlap}, seq_type={normalized_seq_type}, "
            f"source={policy}).",
            flush=True,
        )
    return resolved
