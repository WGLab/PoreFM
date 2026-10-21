"""Context-specific full-space whitening for canonical-null residuals."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F


WHITENING_ARTIFACT_TYPE = "base_specific_full_space_whitening"
WHITENING_FORMAT_VERSION = 2
SUPPORTED_WHITENING_FORMAT_VERSIONS = {1, WHITENING_FORMAT_VERSION}

RAW_COSINE_SCORE_TYPE = "raw_cosine_distance"
STD_COSINE_SCORE_TYPE = "log_standardized_canonical_null_v1"
RAW_WHITENING_SCORE_TYPE = "base_specific_full_space_whitening"
STD_WHITENING_SCORE_TYPE = "log_standardized_base_specific_full_space_whitening_v1"

DEFAULT_EMBEDDING_NORM_EPS = 1e-12
DEFAULT_SCORE_LOG_EPS = 1e-8


def _config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    """Convert common configuration containers into a shallow plain dictionary."""
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return None


def normalize_score_type(score_type: Optional[str]) -> str:
    """Normalize score aliases and standardized labels to a raw score type.

    ``None`` selects raw cosine distance. Standardized artifact labels map back
    to the raw family whose values they represent.
    """
    if score_type is None:
        return RAW_COSINE_SCORE_TYPE
    score_type = str(score_type).strip().lower()
    aliases = {
        "cosine": RAW_COSINE_SCORE_TYPE,
        "raw_cosine": RAW_COSINE_SCORE_TYPE,
        RAW_COSINE_SCORE_TYPE: RAW_COSINE_SCORE_TYPE,
        STD_COSINE_SCORE_TYPE: RAW_COSINE_SCORE_TYPE,
        "whitening": RAW_WHITENING_SCORE_TYPE,
        "whiten": RAW_WHITENING_SCORE_TYPE,
        "base_specific_whitening": RAW_WHITENING_SCORE_TYPE,
        RAW_WHITENING_SCORE_TYPE: RAW_WHITENING_SCORE_TYPE,
        STD_WHITENING_SCORE_TYPE: RAW_WHITENING_SCORE_TYPE,
    }
    if score_type not in aliases:
        raise ValueError(f"Unsupported score_type={score_type!r}")
    return aliases[score_type]


def standardized_score_type(raw_score_type: Optional[str]) -> str:
    """Return the persisted standardized-label name for a raw score family."""
    raw_score_type = normalize_score_type(raw_score_type)
    if raw_score_type == RAW_COSINE_SCORE_TYPE:
        return STD_COSINE_SCORE_TYPE
    if raw_score_type == RAW_WHITENING_SCORE_TYPE:
        return STD_WHITENING_SCORE_TYPE
    raise ValueError(f"Unsupported score_type={raw_score_type!r}")


def is_whitening_score_type(score_type: Optional[str]) -> bool:
    """Return whether a score label belongs to the whitening score family."""
    return normalize_score_type(score_type) == RAW_WHITENING_SCORE_TYPE


def canonical_base_groups(seq_type: str = "dna") -> Tuple[List[str], List[List[int]]]:
    """Return canonical central-base groups in repository token IDs.

    The repository encodes pad as 0, A/C/G/T as 1/2/3/4, and mask as 5. Direct
    RNA currently flows through the same numeric token IDs; the final token is
    reported as U in metadata while retaining token id 4.
    """

    seq_type = str(seq_type).lower()
    if seq_type not in {"dna", "rna"}:
        raise ValueError(f"seq_type must be 'dna' or 'rna', got {seq_type!r}")
    base_names = ["A", "C", "G", "U" if seq_type == "rna" else "T"]
    return base_names, [[1], [2], [3], [4]]


def validate_centered_kmer_size(kmer_size: int, argument_name: str = "kmer_size") -> int:
    """Validate and return a positive odd centered-context k-mer size."""
    kmer_size = int(kmer_size)
    if kmer_size <= 0 or kmer_size % 2 != 1:
        raise ValueError(f"{argument_name} must be a positive odd integer, got {kmer_size}")
    return kmer_size


def canonical_kmer_contexts(kmer_size: int, seq_type: str = "dna") -> Tuple[List[str], torch.Tensor]:
    """Return lexicographic canonical k-mers and their central-base parents.

    Context IDs use base-4 digits A=0, C=1, G=2, T/U=3. Consequently the
    returned list index is also the centered-k-mer ID computed by
    :func:`centered_kmer_indices`. The parent tensor maps each context to the
    index of its central canonical base.
    """
    kmer_size = validate_centered_kmer_size(kmer_size)
    base_names, _ = canonical_base_groups(seq_type)
    num_contexts = 4 ** kmer_size
    names: List[str] = []
    for context_id in range(num_contexts):
        value = context_id
        digits = [0] * kmer_size
        for position in range(kmer_size - 1, -1, -1):
            digits[position] = value % 4
            value //= 4
        names.append("".join(base_names[digit] for digit in digits))
    center_divisor = 4 ** (kmer_size // 2)
    parent = (torch.arange(num_contexts, dtype=torch.long) // center_divisor) % 4
    return names, parent


def centered_kmer_indices(base_seq_input: torch.Tensor, kmer_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode centered canonical k-mers with invalid positions set to ``-1``.

    ``base_seq_input`` must have shape ``(batch, bases)`` and use repository
    token IDs 1..4 for A/C/G/T(U). The returned ID and validity tensors have
    the same shape. Boundary positions without a complete centered context,
    pads, masks, and other noncanonical tokens are invalid.
    """
    kmer_size = validate_centered_kmer_size(kmer_size)
    if base_seq_input.ndim != 2:
        raise ValueError(
            f"base_seq_input must have shape (batch, bases), got {tuple(base_seq_input.shape)}"
        )
    bases = base_seq_input.long() - 1
    canonical = (bases >= 0) & (bases < 4)
    ids = torch.full_like(bases, -1, dtype=torch.long)
    valid = torch.zeros_like(canonical, dtype=torch.bool)
    if base_seq_input.shape[1] < kmer_size:
        return ids, valid

    windows = bases.unfold(dimension=1, size=kmer_size, step=1)
    valid_windows = canonical.unfold(dimension=1, size=kmer_size, step=1).all(dim=-1)
    powers = torch.tensor(
        [4 ** power for power in range(kmer_size - 1, -1, -1)],
        dtype=torch.long,
        device=base_seq_input.device,
    )
    encoded = torch.sum(windows * powers, dim=-1)
    half = kmer_size // 2
    target = ids[:, half:base_seq_input.shape[1] - half if half else None]
    target_valid = valid[:, half:base_seq_input.shape[1] - half if half else None]
    target[valid_windows] = encoded[valid_windows]
    target_valid.copy_(valid_windows)
    return ids, valid


def build_token_to_base_index(
    base_token_ids: Sequence[Sequence[int]] | Sequence[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a dense lookup from token ID to canonical-base group index.

    ``base_token_ids`` may contain individual integer IDs or groups of IDs that
    represent the same base. Unassigned IDs map to ``-1``. Duplicate IDs are
    assigned to the last group in input order.
    """
    normalized: List[List[int]] = []
    for item in base_token_ids:
        if isinstance(item, int):
            normalized.append([int(item)])
        else:
            normalized.append([int(x) for x in item])
    max_token_id = max((token_id for group in normalized for token_id in group), default=-1)
    token_to_base = torch.full((max_token_id + 1,), -1, dtype=torch.long, device=device)
    for base_index, group in enumerate(normalized):
        for token_id in group:
            if token_id < 0:
                raise ValueError(f"Negative token id in base_token_ids: {token_id}")
            token_to_base[token_id] = int(base_index)
    return token_to_base


def central_base_index(
    central_base_id: torch.Tensor,
    base_token_ids: Sequence[Sequence[int]] | Sequence[int],
) -> torch.Tensor:
    """Map central sequence tokens to canonical-base groups, using ``-1`` otherwise.

    The result has the same shape and device as ``central_base_id`` and integer
    dtype.
    """
    token_to_base = build_token_to_base_index(base_token_ids, device=central_base_id.device)
    out = torch.full_like(central_base_id.long(), -1)
    in_range = (central_base_id >= 0) & (central_base_id < token_to_base.numel())
    out[in_range] = token_to_base[central_base_id[in_range].long()]
    return out


def canonical_central_base_mask(
    central_base_id: torch.Tensor,
    seq_type: str = "dna",
    base_token_ids: Optional[Sequence[Sequence[int]]] = None,
) -> torch.Tensor:
    """Identify central tokens that belong to a configured canonical base group."""
    if base_token_ids is None:
        _, base_token_ids = canonical_base_groups(seq_type)
    return central_base_index(central_base_id, base_token_ids) >= 0


def canonical_embedding_residuals(
    contextual_base_seq: torch.Tensor,
    predicted_canonical_embedding: torch.Tensor,
    norm_eps: float = DEFAULT_EMBEDDING_NORM_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize observed/null embeddings in FP32 and return their residual.

    Both inputs must be broadcast-compatible tensors whose final dimension is
    embedding width. Returns ``(h_obs, h_null, h_obs - h_null)`` with matching
    shapes and ``float32`` dtype.
    """

    h_obs = F.normalize(contextual_base_seq.float(), dim=-1, eps=float(norm_eps))
    h_null = F.normalize(predicted_canonical_embedding.float(), dim=-1, eps=float(norm_eps))
    residual = h_obs - h_null
    return h_obs, h_null, residual.float()


def canonical_cosine_scores_from_normalized(h_obs: torch.Tensor, h_null: torch.Tensor) -> torch.Tensor:
    """Compute ``1 - cosine_similarity`` from already normalized embeddings."""
    return 1.0 - (h_obs.float() * h_null.float()).sum(dim=-1)


def canonical_cosine_scores(
    contextual_base_seq: torch.Tensor,
    predicted_canonical_embedding: torch.Tensor,
    norm_eps: float = DEFAULT_EMBEDDING_NORM_EPS,
) -> torch.Tensor:
    """Compute per-position cosine distance between observed and null embeddings."""
    h_obs, h_null, _ = canonical_embedding_residuals(
        contextual_base_seq,
        predicted_canonical_embedding,
        norm_eps=norm_eps,
    )
    return canonical_cosine_scores_from_normalized(h_obs, h_null)


def canonical_score_log_target(
    scores: torch.Tensor,
    score_type: Optional[str] = RAW_COSINE_SCORE_TYPE,
    hidden_dim: Optional[int] = None,
    eps: float = DEFAULT_SCORE_LOG_EPS,
) -> torch.Tensor:
    """Transform non-negative raw scores to the log target used for calibration.

    Whitening scores are first divided by ``hidden_dim`` so their scale is
    comparable across embedding widths. Negative values are clipped to zero
    before adding ``eps`` and taking the natural logarithm.
    """
    raw_score_type = normalize_score_type(score_type)
    safe_scores = torch.clamp(scores.float(), min=0.0)
    if raw_score_type == RAW_WHITENING_SCORE_TYPE:
        if hidden_dim is None or int(hidden_dim) <= 0:
            raise ValueError("hidden_dim is required when standardizing whitening scores")
        safe_scores = safe_scores / float(hidden_dim)
    return torch.log(safe_scores + float(eps))


class StreamingCovariance:
    """Mergeable CPU FP64 covariance accumulator with optional sample weights.

    The accumulator stores a weighted mean and centered second-moment matrix,
    allowing batches or independently accumulated shards to be combined without
    retaining observations. ``count`` records accepted rows, while
    ``effective_count`` accounts for unequal positive weights.

    Parameters
    ----------
    dim : int
        Feature dimension of every observation.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.count = 0
        self.total_weight = torch.tensor(0.0, dtype=torch.float64)
        self.total_weight_sq = torch.tensor(0.0, dtype=torch.float64)
        self.mean = torch.zeros(self.dim, dtype=torch.float64)
        self.M2 = torch.zeros((self.dim, self.dim), dtype=torch.float64)
        self.norm_sum = torch.tensor(0.0, dtype=torch.float64)
        self.downweighted_count = 0

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "StreamingCovariance":
        """Reconstruct an accumulator from ``state_dict`` output."""
        acc = cls(int(state["dim"]))
        acc.load_state_dict(state)
        return acc

    @property
    def weighted(self) -> bool:
        """Whether accepted observations used weights other than all ones."""
        return bool(self.count > 0 and abs(float(self.total_weight_sq.item()) - float(self.count)) > 1e-8)

    @property
    def effective_count(self) -> float:
        """Kish effective sample size implied by accumulated weights."""
        if self.total_weight_sq.item() <= 0.0:
            return 0.0
        return float((self.total_weight * self.total_weight / self.total_weight_sq).item())

    @property
    def unbiased_denominator(self) -> float:
        """Denominator for the unbiased reliability-weighted covariance."""
        if self.total_weight.item() <= 0.0:
            return 0.0
        return float((self.total_weight - self.total_weight_sq / self.total_weight).item())

    def update(self, batch: torch.Tensor, weights: Optional[torch.Tensor] = None) -> None:
        """Accumulate finite observations and optional positive scalar weights.

        ``batch`` is flattened to ``(-1, dim)`` and copied to CPU FP64. Invalid
        rows, and rows with missing/non-positive weights, are silently skipped.
        """
        if batch.numel() == 0:
            return
        x = batch.detach().reshape(-1, self.dim).to(device="cpu", dtype=torch.float64)
        finite = torch.isfinite(x).all(dim=-1)
        if weights is None:
            w = torch.ones(x.shape[0], dtype=torch.float64)
        else:
            w = weights.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
            finite = finite & torch.isfinite(w) & (w > 0.0)
        if not finite.any():
            return

        x = x[finite]
        w = w[finite]
        W = w.sum()
        if W.item() <= 0.0:
            return
        W2 = (w * w).sum()
        mean = (x * w[:, None]).sum(dim=0) / W
        centered = x - mean
        M2 = (centered * w[:, None]).T.matmul(centered)

        other = StreamingCovariance(self.dim)
        other.count = int(x.shape[0])
        other.total_weight = W
        other.total_weight_sq = W2
        other.mean = mean
        other.M2 = M2
        other.norm_sum = torch.linalg.vector_norm(x, dim=-1).sum()
        other.downweighted_count = int((w < 1.0).sum().item())
        self.merge(other)

    def merge(self, other: "StreamingCovariance") -> None:
        """Merge another same-dimension accumulator into this instance in place."""
        if self.dim != other.dim:
            raise ValueError(f"Cannot merge dim={other.dim} into dim={self.dim}")
        if other.total_weight.item() <= 0.0:
            return
        if self.total_weight.item() <= 0.0:
            self.count = int(other.count)
            self.total_weight = other.total_weight.clone()
            self.total_weight_sq = other.total_weight_sq.clone()
            self.mean = other.mean.clone()
            self.M2 = other.M2.clone()
            self.norm_sum = other.norm_sum.clone()
            self.downweighted_count = int(other.downweighted_count)
            return

        Wa = self.total_weight
        Wb = other.total_weight
        W = Wa + Wb
        delta = other.mean - self.mean
        self.mean = self.mean + delta * (Wb / W)
        self.M2 = self.M2 + other.M2 + torch.outer(delta, delta) * (Wa * Wb / W)
        self.total_weight = W
        self.total_weight_sq = self.total_weight_sq + other.total_weight_sq
        self.count += int(other.count)
        self.norm_sum = self.norm_sum + other.norm_sum
        self.downweighted_count += int(other.downweighted_count)

    def finalize(self) -> torch.Tensor:
        """Return the symmetric unbiased covariance matrix with shape ``(dim, dim)``."""
        denom = self.unbiased_denominator
        if denom <= 0.0:
            raise ValueError(
                f"Cannot finalize covariance with count={self.count}, "
                f"total_weight={self.total_weight.item():.6g}, "
                f"total_weight_sq={self.total_weight_sq.item():.6g}."
            )
        cov = self.M2 / denom
        return 0.5 * (cov + cov.T)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize sufficient statistics as independent CPU tensors."""
        return {
            "dim": self.dim,
            "count": int(self.count),
            "total_weight": self.total_weight.clone(),
            "total_weight_sq": self.total_weight_sq.clone(),
            "mean": self.mean.clone(),
            "M2": self.M2.clone(),
            "norm_sum": self.norm_sum.clone(),
            "downweighted_count": int(self.downweighted_count),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Replace accumulated statistics from a matching-dimension state."""
        dim = int(state["dim"])
        if dim != self.dim:
            raise ValueError(f"State dim={dim} does not match accumulator dim={self.dim}")
        self.count = int(state["count"])
        self.total_weight = torch.as_tensor(state["total_weight"], dtype=torch.float64).cpu().clone()
        self.total_weight_sq = torch.as_tensor(state["total_weight_sq"], dtype=torch.float64).cpu().clone()
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float64).cpu().reshape(self.dim).clone()
        self.M2 = torch.as_tensor(state["M2"], dtype=torch.float64).cpu().reshape(self.dim, self.dim).clone()
        self.norm_sum = torch.as_tensor(state.get("norm_sum", 0.0), dtype=torch.float64).cpu().clone()
        self.downweighted_count = int(state.get("downweighted_count", 0))


class PerBaseScoreReservoir:
    """Bounded-memory CPU reservoir for per-base pilot score quantiles.

    Standard reservoir sampling gives each finite score observed for a base an
    equal chance of occupying one of ``capacity_per_base`` slots. ``seed``
    initializes a private CPU generator, making update order reproducible.
    """

    def __init__(self, num_bases: int, capacity_per_base: int, seed: int = 0):
        self.num_bases = int(num_bases)
        self.capacity_per_base = int(capacity_per_base)
        if self.capacity_per_base <= 0:
            raise ValueError("capacity_per_base must be positive")
        self.values = torch.full(
            (self.num_bases, self.capacity_per_base),
            float("nan"),
            dtype=torch.float32,
        )
        self.seen_counts = torch.zeros(self.num_bases, dtype=torch.long)
        self.retained_counts = torch.zeros(self.num_bases, dtype=torch.long)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def update(self, base_indices: torch.Tensor, scores: torch.Tensor) -> None:
        """Consume aligned base-group indices and scalar scores in place."""
        if scores.numel() == 0:
            return
        base_indices = base_indices.detach().reshape(-1).cpu().long()
        scores = scores.detach().reshape(-1).cpu().float()
        valid = (
            torch.isfinite(scores)
            & (base_indices >= 0)
            & (base_indices < self.num_bases)
        )
        if not valid.any():
            return
        for base_index, score in zip(base_indices[valid].tolist(), scores[valid].tolist()):
            seen = int(self.seen_counts[base_index].item())
            self.seen_counts[base_index] += 1
            if seen < self.capacity_per_base:
                self.values[base_index, seen] = float(score)
                self.retained_counts[base_index] += 1
                continue
            slot = int(torch.randint(seen + 1, (1,), generator=self.generator).item())
            if slot < self.capacity_per_base:
                self.values[base_index, slot] = float(score)

    def merge(self, other: "PerBaseScoreReservoir") -> None:
        """Merge independent uniform reservoirs without rank-volume bias.

        For each group, the number of rows selected from ``other`` is drawn
        from the exact hypergeometric distribution implied by the two stream
        sizes. Uniform subsets of the already-uniform local reservoirs then
        form a uniform reservoir of the combined stream. This is associative
        in distribution and lets distributed robust-cutoff fitting account for
        unequal token counts across ranks without gathering complete streams.
        """
        if self.num_bases != other.num_bases:
            raise ValueError("Cannot merge reservoirs with different num_bases")
        if self.capacity_per_base != other.capacity_per_base:
            raise ValueError("Cannot merge reservoirs with different capacities")

        for base_index in range(other.num_bases):
            self_seen = int(self.seen_counts[base_index].item())
            other_seen = int(other.seen_counts[base_index].item())
            combined_seen = self_seen + other_seen
            combined_retained = min(self.capacity_per_base, combined_seen)

            if combined_retained == 0:
                continue

            # Draw how many elements the global reservoir receives from the
            # other rank. The local retained sets are already uniform subsets,
            # so selecting a uniform subset from each is sufficient.
            hypergeom_seed = int(
                torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (1,),
                    generator=self.generator,
                    dtype=torch.int64,
                ).item()
            )
            rng = np.random.default_rng(hypergeom_seed)
            from_other = int(
                rng.hypergeometric(
                    ngood=other_seen,
                    nbad=self_seen,
                    nsample=combined_retained,
                )
            )
            from_self = combined_retained - from_other

            self_retained = int(self.retained_counts[base_index].item())
            other_retained = int(other.retained_counts[base_index].item())
            if from_self > self_retained or from_other > other_retained:
                raise RuntimeError(
                    "Reservoir state cannot supply the sampled distributed merge counts"
                )

            pieces = []
            if from_self:
                self_indices = torch.randperm(
                    self_retained,
                    generator=self.generator,
                )[:from_self]
                pieces.append(self.values[base_index, self_indices].clone())
            if from_other:
                other_indices = torch.randperm(
                    other_retained,
                    generator=self.generator,
                )[:from_other]
                pieces.append(other.values[base_index, other_indices])

            merged_values = torch.cat(pieces, dim=0)
            self.values[base_index].fill_(float("nan"))
            self.values[base_index, :combined_retained] = merged_values
            self.seen_counts[base_index] = combined_seen
            self.retained_counts[base_index] = combined_retained

    def quantiles(self, q: float) -> torch.Tensor:
        """Estimate one quantile per base, returning NaN for empty groups."""
        q = min(max(float(q), 0.0), 1.0)
        out = torch.full((self.num_bases,), float("nan"), dtype=torch.float32)
        for base_index in range(self.num_bases):
            n = int(self.retained_counts[base_index].item())
            if n <= 0:
                continue
            out[base_index] = torch.quantile(self.values[base_index, :n], q)
        return out

    def state_dict(self) -> Dict[str, Any]:
        """Return reservoir contents and counters for checkpointing."""
        return {
            "num_bases": self.num_bases,
            "capacity_per_base": self.capacity_per_base,
            "values": self.values.clone(),
            "seen_counts": self.seen_counts.clone(),
            "retained_counts": self.retained_counts.clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], seed: int = 0) -> "PerBaseScoreReservoir":
        """Restore a score reservoir and initialize future sampling with ``seed``."""
        obj = cls(int(state["num_bases"]), int(state["capacity_per_base"]), seed=seed)
        obj.values = torch.as_tensor(state["values"], dtype=torch.float32).cpu().clone()
        obj.seen_counts = torch.as_tensor(state["seen_counts"], dtype=torch.long).cpu().clone()
        obj.retained_counts = torch.as_tensor(state["retained_counts"], dtype=torch.long).cpu().clone()
        return obj


class PerBaseTensorReservoir:
    """Bounded CPU reservoir of per-base residual vectors and optional weights.

    Parameters
    ----------
    num_bases : int
        Number of canonical-base groups.
    dim : int
        Residual-vector width.
    capacity_per_base : int
        Maximum retained vectors for each base group.
    seed : int
        Seed for the private reservoir-sampling generator.
    store_weights : bool
        Allocate a parallel scalar-weight reservoir. Missing update weights are
        stored as one when this is enabled.
    """

    def __init__(
        self,
        num_bases: int,
        dim: int,
        capacity_per_base: int,
        seed: int = 0,
        store_weights: bool = False,
    ):
        self.num_bases = int(num_bases)
        self.dim = int(dim)
        self.capacity_per_base = int(capacity_per_base)
        if self.capacity_per_base <= 0:
            raise ValueError("capacity_per_base must be positive")
        self.values = torch.empty(
            (self.num_bases, self.capacity_per_base, self.dim),
            dtype=torch.float32,
        )
        self.weights = (
            torch.empty((self.num_bases, self.capacity_per_base), dtype=torch.float32)
            if store_weights
            else None
        )
        self.seen_counts = torch.zeros(self.num_bases, dtype=torch.long)
        self.retained_counts = torch.zeros(self.num_bases, dtype=torch.long)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def update(
        self,
        base_indices: torch.Tensor,
        residuals: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Consume aligned base indices, residuals, and optional positive weights."""
        if residuals.numel() == 0:
            return
        base_indices = base_indices.detach().reshape(-1).cpu().long()
        residuals = residuals.detach().reshape(-1, self.dim).cpu().float()
        if weights is not None:
            weights = weights.detach().reshape(-1).cpu().float()
        valid = (
            torch.isfinite(residuals).all(dim=-1)
            & (base_indices >= 0)
            & (base_indices < self.num_bases)
        )
        if weights is not None:
            valid = valid & torch.isfinite(weights) & (weights > 0.0)
        if not valid.any():
            return

        selected_bases = base_indices[valid]
        selected_residuals = residuals[valid]
        selected_weights = weights[valid] if weights is not None else None
        for row_idx, base_index in enumerate(selected_bases.tolist()):
            seen = int(self.seen_counts[base_index].item())
            self.seen_counts[base_index] += 1
            if seen < self.capacity_per_base:
                slot = seen
                self.retained_counts[base_index] += 1
            else:
                slot = int(torch.randint(seen + 1, (1,), generator=self.generator).item())
                if slot >= self.capacity_per_base:
                    continue
            self.values[base_index, slot] = selected_residuals[row_idx]
            if self.weights is not None:
                if selected_weights is None:
                    self.weights[base_index, slot] = 1.0
                else:
                    self.weights[base_index, slot] = selected_weights[row_idx]

    def to_accumulators(self) -> List[StreamingCovariance]:
        """Convert retained samples into one covariance accumulator per base."""
        accumulators = [StreamingCovariance(self.dim) for _ in range(self.num_bases)]
        for base_index in range(self.num_bases):
            n = int(self.retained_counts[base_index].item())
            if n <= 0:
                continue
            weights = self.weights[base_index, :n] if self.weights is not None else None
            accumulators[base_index].update(self.values[base_index, :n], weights=weights)
        return accumulators

    def state_dict(self) -> Dict[str, Any]:
        """Return retained tensors and counters for checkpointing."""
        return {
            "num_bases": self.num_bases,
            "dim": self.dim,
            "capacity_per_base": self.capacity_per_base,
            "values": self.values.clone(),
            "weights": None if self.weights is None else self.weights.clone(),
            "seen_counts": self.seen_counts.clone(),
            "retained_counts": self.retained_counts.clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], seed: int = 0) -> "PerBaseTensorReservoir":
        """Restore a tensor reservoir and initialize future sampling with ``seed``."""
        weights = state.get("weights")
        obj = cls(
            int(state["num_bases"]),
            int(state["dim"]),
            int(state["capacity_per_base"]),
            seed=seed,
            store_weights=weights is not None,
        )
        obj.values = torch.as_tensor(state["values"], dtype=torch.float32).cpu().clone()
        if weights is not None:
            obj.weights = torch.as_tensor(weights, dtype=torch.float32).cpu().clone()
        obj.seen_counts = torch.as_tensor(state["seen_counts"], dtype=torch.long).cpu().clone()
        obj.retained_counts = torch.as_tensor(state["retained_counts"], dtype=torch.long).cpu().clone()
        return obj


def oas_shrinkage_from_covariance(sample_covariance: torch.Tensor, n_samples: float) -> float:
    """Compute a scikit-learn-compatible OAS covariance shrinkage coefficient.

    ``sample_covariance`` must be a square unbiased covariance matrix and
    ``n_samples`` its effective sample count. The returned coefficient lies in
    ``[0, 1]`` and is zero for one-dimensional covariance.
    """

    n_samples = float(n_samples)
    if n_samples <= 1:
        raise ValueError("OAS shrinkage requires more than one sample")
    S = sample_covariance.to(dtype=torch.float64)
    p = S.shape[0]
    if p != S.shape[1]:
        raise ValueError("sample_covariance must be square")
    if p == 1:
        return 0.0
    # Sklearn computes the coefficient from the ML covariance X'X / n. Scaling
    # does not change the coefficient, but converting makes this explicit.
    emp_cov = S * ((n_samples - 1.0) / n_samples)
    alpha = torch.mean(emp_cov * emp_cov).item()
    mu = torch.trace(emp_cov).item() / float(p)
    mu_sq = mu * mu
    den = (n_samples + 1.0) * (alpha - mu_sq / float(p))
    if den == 0.0:
        return 1.0
    return float(min((alpha + mu_sq) / den, 1.0))


def _validate_covariance_mode(mode: str) -> str:
    """Normalize and validate the supported covariance structure name."""
    mode = str(mode).lower()
    if mode not in {"full", "diagonal", "isotropic"}:
        raise ValueError("covariance_mode must be one of: full, diagonal, isotropic")
    return mode


def _regularize_one_covariance(
    sample_covariance: torch.Tensor,
    count: int,
    effective_count: float,
    mode: str = "full",
    shrinkage_mode: str = "fixed",
    shrinkage_alpha: float = 0.03,
    relative_eigenvalue_floor: float = 1e-3,
    max_condition_number: float = 1e4,
    weighted: bool = False,
) -> Dict[str, Any]:
    """Shrink, floor, and factor one covariance into a whitening transform.

    ``mode`` retains the full covariance, only its diagonal, or only its mean
    variance. Fixed or OAS shrinkage pulls that structure toward an isotropic
    covariance, after which eigenvalues are floored to satisfy the requested
    relative scale and maximum condition number. Returned diagnostics include
    raw/regularized spectra and a matrix ``whitening`` satisfying approximately
    ``W @ W.T == regularized_covariance^-1``.
    """
    mode = _validate_covariance_mode(mode)
    shrinkage_mode = str(shrinkage_mode).lower()
    if shrinkage_mode not in {"fixed", "oas"}:
        raise ValueError("shrinkage_mode must be 'fixed' or 'oas'")

    S = sample_covariance.to(dtype=torch.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("sample_covariance must be a square matrix")
    dim = S.shape[0]
    S = 0.5 * (S + S.T)
    if not torch.isfinite(S).all():
        raise ValueError("sample_covariance contains non-finite values")

    tau = torch.trace(S) / float(dim)
    if not torch.isfinite(tau) or tau.item() <= 0.0:
        raise ValueError(f"Covariance trace is not positive finite: tau={tau.item()}")

    if mode == "isotropic":
        alpha = 1.0
        sigma = torch.eye(dim, dtype=torch.float64) * tau
    else:
        if shrinkage_mode == "fixed":
            alpha = float(shrinkage_alpha)
        else:
            if weighted:
                raise ValueError("OAS shrinkage with weighted covariance is not supported; use fixed shrinkage")
            alpha = oas_shrinkage_from_covariance(S, n_samples=count)
        alpha = min(max(float(alpha), 0.0), 1.0)
        if mode == "diagonal":
            diag = torch.diag(S).clone()
            diag = (1.0 - alpha) * diag + alpha * tau
            sigma = torch.diag(diag)
        else:
            sigma = (1.0 - alpha) * S
            sigma = sigma.clone()
            sigma.diagonal().add_(alpha * tau)
    sigma = 0.5 * (sigma + sigma.T)

    raw_eigvals = torch.linalg.eigvalsh(S)
    if mode == "full":
        eigvals, eigvecs = torch.linalg.eigh(sigma)
    elif mode == "diagonal":
        eigvals = torch.diag(sigma)
        eigvecs = torch.eye(dim, dtype=torch.float64)
    else:
        eigvals = torch.full((dim,), tau.item(), dtype=torch.float64)
        eigvecs = torch.eye(dim, dtype=torch.float64)

    if not torch.isfinite(eigvals).all():
        raise ValueError("Covariance eigendecomposition produced non-finite eigenvalues")

    lambda_max = eigvals.max()
    floor = torch.maximum(
        torch.as_tensor(float(relative_eigenvalue_floor), dtype=torch.float64) * tau,
        lambda_max / float(max_condition_number),
    )
    if not torch.isfinite(floor) or floor.item() <= 0.0:
        raise ValueError(f"Invalid eigenvalue floor: {floor.item()}")
    regularized_eigvals = torch.maximum(eigvals, floor)
    if not torch.isfinite(regularized_eigvals).all() or (regularized_eigvals <= 0.0).any():
        raise ValueError("Regularized eigenvalues must be positive finite")

    whitening = eigvecs.matmul(torch.diag(torch.rsqrt(regularized_eigvals)))
    regularized_covariance = eigvecs.matmul(torch.diag(regularized_eigvals)).matmul(eigvecs.T)
    if not torch.isfinite(whitening).all():
        raise ValueError("Whitening matrix contains non-finite values")

    sigma_inv = eigvecs.matmul(torch.diag(1.0 / regularized_eigvals)).matmul(eigvecs.T)
    reconstruction_error = torch.linalg.matrix_norm(whitening.matmul(whitening.T) - sigma_inv).item()

    final_condition_number = (
        regularized_eigvals.max() / regularized_eigvals.min().clamp_min(torch.finfo(torch.float64).tiny)
    ).item()

    return {
        "covariance_mode": mode,
        "tau": tau,
        "shrinkage": torch.tensor(alpha, dtype=torch.float64),
        "eigenvalue_floor": floor,
        "raw_eigenvalues": raw_eigvals,
        "regularized_eigenvalues": regularized_eigvals,
        "regularized_covariance": regularized_covariance,
        "whitening": whitening,
        "raw_min_eigenvalue": raw_eigvals.min(),
        "raw_max_eigenvalue": raw_eigvals.max(),
        "regularized_min_eigenvalue": regularized_eigvals.min(),
        "final_condition_number": torch.tensor(final_condition_number, dtype=torch.float64),
        "num_floored_eigenvalues": torch.sum(eigvals < floor).long(),
        "reconstruction_error": torch.tensor(reconstruction_error, dtype=torch.float64),
        "effective_count": torch.tensor(float(effective_count), dtype=torch.float64),
    }


def fit_whitening_from_accumulators(
    accumulators: Sequence[StreamingCovariance],
    base_names: Sequence[str],
    base_token_ids: Sequence[Sequence[int]] | Sequence[int],
    hidden_dim: int,
    covariance_mode: str = "full",
    shrinkage_mode: str = "fixed",
    shrinkage_alpha: float = 0.03,
    relative_eigenvalue_floor: float = 1e-3,
    max_condition_number: float = 1e4,
    robust_settings: Optional[Dict[str, Any]] = None,
    fit_configuration: Optional[Dict[str, Any]] = None,
    model_checkpoint_identifier: Optional[str] = None,
    null_predictor_identifier: Optional[str] = None,
    chemistry: Optional[str] = None,
    molecule_type: Optional[str] = None,
    normalization_configuration: Optional[Dict[str, Any]] = None,
    residual_definition: Optional[Dict[str, Any]] = None,
    pretrain_config: Any = None,
    include_raw_covariance: bool = False,
) -> Dict[str, Any]:
    """Fit and package one full-space whitening transform per canonical base.

    Every accumulator must have dimension ``hidden_dim`` and both its raw and
    effective sample counts must exceed that dimension. Covariance structure,
    shrinkage, and eigenvalue flooring are applied independently per base.
    Remaining mapping, model, chemistry, normalization, and residual arguments
    are stored as provenance rather than used in fitting.

    Returns
    -------
    dict
        A validated, CPU-resident artifact containing stacked base means,
        whitening matrices, regularized covariances and spectra, observation
        counts, diagnostics, token mappings, and provenance. Raw covariances
        are included only when ``include_raw_covariance`` is true.
    """
    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if len(accumulators) != len(base_names):
        raise ValueError("accumulators and base_names lengths differ")
    if len(accumulators) == 0:
        raise ValueError("At least one base accumulator is required")

    base_token_ids_normalized = [
        [int(x) for x in group] if not isinstance(group, int) else [int(group)]
        for group in base_token_ids
    ]
    if len(base_token_ids_normalized) != len(base_names):
        raise ValueError("base_token_ids and base_names lengths differ")

    means = []
    whitenings = []
    reg_covs = []
    raw_covs = []
    counts = []
    effective_counts = []
    taus = []
    shrinkages = []
    floors = []
    raw_eigvals = []
    reg_eigvals = []
    diagnostics = {}

    for base_index, (base_name, acc) in enumerate(zip(base_names, accumulators)):
        if acc.dim != hidden_dim:
            raise ValueError(f"Accumulator for base {base_name} has dim={acc.dim}, expected {hidden_dim}")
        if acc.count <= hidden_dim:
            raise ValueError(
                f"Base {base_name} has only {acc.count} residuals; need more than hidden_dim={hidden_dim}"
            )
        if acc.effective_count <= hidden_dim:
            raise ValueError(
                f"Base {base_name} effective count {acc.effective_count:.3f} is too small for hidden_dim={hidden_dim}"
            )
        if acc.count < 10 * hidden_dim:
            warnings.warn(
                f"Base {base_name} has {acc.count} residuals for hidden_dim={hidden_dim}; covariance may be noisy.",
                RuntimeWarning,
                stacklevel=2,
            )

        cov = acc.finalize()
        reg = _regularize_one_covariance(
            cov,
            count=acc.count,
            effective_count=acc.effective_count,
            mode=covariance_mode,
            shrinkage_mode=shrinkage_mode,
            shrinkage_alpha=shrinkage_alpha,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            max_condition_number=max_condition_number,
            weighted=acc.weighted,
        )

        means.append(acc.mean.float())
        whitenings.append(reg["whitening"].float())
        reg_covs.append(reg["regularized_covariance"].float())
        if include_raw_covariance:
            raw_covs.append(cov.float())
        counts.append(int(acc.count))
        effective_counts.append(float(reg["effective_count"].item()))
        taus.append(reg["tau"].float())
        shrinkages.append(reg["shrinkage"].float())
        floors.append(reg["eigenvalue_floor"].float())
        raw_eigvals.append(reg["raw_eigenvalues"].float())
        reg_eigvals.append(reg["regularized_eigenvalues"].float())
        diagnostics[str(base_name)] = {
            "count": int(acc.count),
            "effective_count": float(reg["effective_count"].item()),
            "mean_residual_norm": float((acc.norm_sum / max(acc.count, 1)).item()),
            "trace_raw_covariance": float(torch.trace(cov).item()),
            "raw_min_eigenvalue": float(reg["raw_min_eigenvalue"].item()),
            "raw_max_eigenvalue": float(reg["raw_max_eigenvalue"].item()),
            "regularized_min_eigenvalue": float(reg["regularized_min_eigenvalue"].item()),
            "shrinkage": float(reg["shrinkage"].item()),
            "eigenvalue_floor": float(reg["eigenvalue_floor"].item()),
            "num_floored_eigenvalues": int(reg["num_floored_eigenvalues"].item()),
            "final_condition_number": float(reg["final_condition_number"].item()),
            "downweighted_fraction": float(acc.downweighted_count / max(acc.count, 1)),
            "reconstruction_error": float(reg["reconstruction_error"].item()),
        }

    artifact = {
        "artifact_type": WHITENING_ARTIFACT_TYPE,
        # Keep the legacy base-specific constructor on its original wire
        # format; new centered-k-mer artifacts use format version 2 below.
        "format_version": 1,
        "hidden_dim": hidden_dim,
        "base_names": [str(x) for x in base_names],
        "base_token_ids": base_token_ids_normalized,
        "token_to_base_index": build_token_to_base_index(base_token_ids_normalized).cpu(),
        "mean": torch.stack(means, dim=0).contiguous(),
        "whitening": torch.stack(whitenings, dim=0).contiguous(),
        "regularized_covariance": torch.stack(reg_covs, dim=0).contiguous(),
        "counts": torch.tensor(counts, dtype=torch.long),
        "effective_counts": torch.tensor(effective_counts, dtype=torch.float64),
        "tau": torch.stack(taus, dim=0),
        "shrinkage": torch.stack(shrinkages, dim=0),
        "eigenvalue_floor": torch.stack(floors, dim=0),
        "raw_eigenvalues": torch.stack(raw_eigvals, dim=0),
        "regularized_eigenvalues": torch.stack(reg_eigvals, dim=0),
        "robust_settings": dict(robust_settings or {}),
        "fit_configuration": dict(fit_configuration or {}),
        "model_checkpoint_identifier": model_checkpoint_identifier,
        "null_predictor_identifier": null_predictor_identifier,
        "chemistry": chemistry,
        "molecule_type": molecule_type,
        "normalization_configuration": dict(normalization_configuration or {}),
        "residual_definition": dict(
            residual_definition
            or {
                "h_obs": "F.normalize(contextual_base_seq.float(), dim=-1, eps=norm_eps)",
                "h_null": "F.normalize(predicted_canonical_embedding.float(), dim=-1, eps=norm_eps)",
                "residual": "h_obs - h_null",
            }
        ),
        "score_type": RAW_WHITENING_SCORE_TYPE,
        "diagnostics": diagnostics,
        "pretrain_config": _config_to_dict(pretrain_config),
    }
    if include_raw_covariance:
        artifact["raw_covariance"] = torch.stack(raw_covs, dim=0).contiguous()
    validate_whitening_artifact(artifact, hidden_dim=hidden_dim, strict=True)
    return artifact


def fit_context_whitening_from_accumulators(
    accumulators: Sequence[StreamingCovariance],
    parent_accumulators: Sequence[StreamingCovariance],
    whitening_kmer_size: int,
    seq_type: str,
    hidden_dim: int,
    covariance_mode: str = "full",
    shrinkage_mode: str = "fixed",
    shrinkage_alpha: float = 0.03,
    parent_shrinkage_strength: float = 0.0,
    relative_eigenvalue_floor: float = 1e-3,
    max_condition_number: float = 1e4,
    robust_settings: Optional[Dict[str, Any]] = None,
    fit_configuration: Optional[Dict[str, Any]] = None,
    model_checkpoint_identifier: Optional[str] = None,
    null_predictor_identifier: Optional[str] = None,
    chemistry: Optional[str] = None,
    molecule_type: Optional[str] = None,
    normalization_configuration: Optional[Dict[str, Any]] = None,
    residual_definition: Optional[Dict[str, Any]] = None,
    pretrain_config: Any = None,
    include_raw_covariance: bool = False,
    include_regularized_covariance: bool = False,
) -> Dict[str, Any]:
    """Fit one full-space whitening transform per centered k-mer context.

    Each context covariance and mean are shrunk toward estimates from its
    central-base parent before the ordinary isotropic shrinkage and eigenvalue
    floor are applied. The parent weight is
    ``strength / (effective_count + strength)``. A non-positive strength uses
    ``10 * hidden_dim`` for k-mers larger than one, and zero for one-mers.

    The whitening matrices retain the complete regularized covariance
    geometry. Large duplicate raw/regularized covariance tensors are omitted
    by default because a 5-mer, 512-dimensional artifact would otherwise
    require roughly two additional GiB.
    """
    whitening_kmer_size = validate_centered_kmer_size(
        whitening_kmer_size,
        argument_name="whitening_kmer_size",
    )
    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")

    context_names, context_parent = canonical_kmer_contexts(whitening_kmer_size, seq_type)
    base_names, base_token_ids = canonical_base_groups(seq_type)
    if len(accumulators) != len(context_names):
        raise ValueError(
            f"Expected {len(context_names)} context accumulators for {whitening_kmer_size}-mers, "
            f"got {len(accumulators)}"
        )
    if len(parent_accumulators) != len(base_names):
        raise ValueError(
            f"Expected {len(base_names)} central-base parent accumulators, got {len(parent_accumulators)}"
        )
    for parent_name, parent_acc in zip(base_names, parent_accumulators):
        if parent_acc.dim != hidden_dim or parent_acc.effective_count <= hidden_dim:
            raise ValueError(
                f"Parent base {parent_name} needs more than hidden_dim={hidden_dim} effective observations; "
                f"got {parent_acc.effective_count:.3f}"
            )

    resolved_parent_strength = float(parent_shrinkage_strength)
    if whitening_kmer_size == 1:
        resolved_parent_strength = 0.0
    elif resolved_parent_strength <= 0.0:
        resolved_parent_strength = 10.0 * hidden_dim
    if not math.isfinite(resolved_parent_strength) or resolved_parent_strength < 0.0:
        raise ValueError(
            f"parent_shrinkage_strength must be finite and non-negative, got {parent_shrinkage_strength}"
        )

    parent_covariances = [acc.finalize() for acc in parent_accumulators]
    means: List[torch.Tensor] = []
    whitenings: List[torch.Tensor] = []
    regularized_covariances: List[torch.Tensor] = []
    raw_covariances: List[torch.Tensor] = []
    counts: List[int] = []
    effective_counts: List[float] = []
    taus: List[torch.Tensor] = []
    isotropic_shrinkages: List[torch.Tensor] = []
    parent_shrinkages: List[float] = []
    floors: List[torch.Tensor] = []
    raw_eigenvalues: List[torch.Tensor] = []
    regularized_eigenvalues: List[torch.Tensor] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for context_index, (context_name, acc) in enumerate(zip(context_names, accumulators)):
        if acc.dim != hidden_dim:
            raise ValueError(
                f"Accumulator for context {context_name} has dim={acc.dim}, expected {hidden_dim}"
            )
        if acc.count <= 1 or acc.effective_count <= 1.0:
            raise ValueError(
                f"Context {context_name} needs at least two effective residuals; "
                f"got count={acc.count}, effective_count={acc.effective_count:.3f}"
            )
        if whitening_kmer_size == 1 and acc.effective_count <= hidden_dim:
            raise ValueError(
                f"Context {context_name} needs more than hidden_dim={hidden_dim} effective residuals"
            )

        parent_index = int(context_parent[context_index].item())
        parent_acc = parent_accumulators[parent_index]
        context_covariance = acc.finalize()
        parent_weight = (
            resolved_parent_strength / (acc.effective_count + resolved_parent_strength)
            if resolved_parent_strength > 0.0
            else 0.0
        )
        fitted_covariance = (
            (1.0 - parent_weight) * context_covariance
            + parent_weight * parent_covariances[parent_index]
        )
        fitted_mean = (
            (1.0 - parent_weight) * acc.mean
            + parent_weight * parent_acc.mean
        )
        reg = _regularize_one_covariance(
            fitted_covariance,
            count=acc.count,
            effective_count=acc.effective_count,
            mode=covariance_mode,
            shrinkage_mode=shrinkage_mode,
            shrinkage_alpha=shrinkage_alpha,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            max_condition_number=max_condition_number,
            weighted=acc.weighted,
        )

        means.append(fitted_mean.float())
        whitenings.append(reg["whitening"].float())
        if include_regularized_covariance:
            regularized_covariances.append(reg["regularized_covariance"].float())
        if include_raw_covariance:
            raw_covariances.append(context_covariance.float())
        counts.append(int(acc.count))
        effective_counts.append(float(reg["effective_count"].item()))
        taus.append(reg["tau"].float())
        isotropic_shrinkages.append(reg["shrinkage"].float())
        parent_shrinkages.append(float(parent_weight))
        floors.append(reg["eigenvalue_floor"].float())
        raw_eigenvalues.append(reg["raw_eigenvalues"].float())
        regularized_eigenvalues.append(reg["regularized_eigenvalues"].float())
        diagnostics[context_name] = {
            "context_id": context_index,
            "parent_base": base_names[parent_index],
            "count": int(acc.count),
            "effective_count": float(reg["effective_count"].item()),
            "mean_residual_norm": float((acc.norm_sum / max(acc.count, 1)).item()),
            "trace_context_covariance": float(torch.trace(context_covariance).item()),
            "trace_parent_shrunk_covariance": float(torch.trace(fitted_covariance).item()),
            "parent_shrinkage": float(parent_weight),
            "isotropic_shrinkage": float(reg["shrinkage"].item()),
            "eigenvalue_floor": float(reg["eigenvalue_floor"].item()),
            "num_floored_eigenvalues": int(reg["num_floored_eigenvalues"].item()),
            "final_condition_number": float(reg["final_condition_number"].item()),
            "downweighted_fraction": float(acc.downweighted_count / max(acc.count, 1)),
            "reconstruction_error": float(reg["reconstruction_error"].item()),
        }

    artifact: Dict[str, Any] = {
        "artifact_type": WHITENING_ARTIFACT_TYPE,
        "format_version": WHITENING_FORMAT_VERSION,
        "context_type": "centered_kmer",
        "whitening_kmer_size": whitening_kmer_size,
        "num_contexts": len(context_names),
        "context_names": context_names,
        "context_parent_base_index": context_parent,
        "hidden_dim": hidden_dim,
        "base_names": base_names,
        "base_token_ids": base_token_ids,
        "token_to_base_index": build_token_to_base_index(base_token_ids).cpu(),
        "mean": torch.stack(means, dim=0).contiguous(),
        "whitening": torch.stack(whitenings, dim=0).contiguous(),
        "counts": torch.tensor(counts, dtype=torch.long),
        "effective_counts": torch.tensor(effective_counts, dtype=torch.float64),
        "tau": torch.stack(taus, dim=0),
        "shrinkage": torch.stack(isotropic_shrinkages, dim=0),
        "parent_shrinkage": torch.tensor(parent_shrinkages, dtype=torch.float32),
        "parent_shrinkage_strength": resolved_parent_strength,
        "eigenvalue_floor": torch.stack(floors, dim=0),
        "raw_eigenvalues": torch.stack(raw_eigenvalues, dim=0),
        "regularized_eigenvalues": torch.stack(regularized_eigenvalues, dim=0),
        "parent_counts": torch.tensor([acc.count for acc in parent_accumulators], dtype=torch.long),
        "parent_effective_counts": torch.tensor(
            [acc.effective_count for acc in parent_accumulators], dtype=torch.float64
        ),
        "parent_mean": torch.stack([acc.mean.float() for acc in parent_accumulators], dim=0),
        "parent_covariance": torch.stack(
            [cov.float() for cov in parent_covariances], dim=0
        ).contiguous(),
        "robust_settings": dict(robust_settings or {}),
        "fit_configuration": dict(fit_configuration or {}),
        "model_checkpoint_identifier": model_checkpoint_identifier,
        "null_predictor_identifier": null_predictor_identifier,
        "chemistry": chemistry,
        "molecule_type": molecule_type or seq_type,
        "normalization_configuration": dict(normalization_configuration or {}),
        "residual_definition": dict(
            residual_definition
            or {
                "h_obs": "F.normalize(contextual_base_seq.float(), dim=-1, eps=norm_eps)",
                "h_null": "F.normalize(predicted_canonical_embedding.float(), dim=-1, eps=norm_eps)",
                "residual": "h_obs - h_null",
            }
        ),
        "score_type": RAW_WHITENING_SCORE_TYPE,
        "diagnostics": diagnostics,
        "pretrain_config": _config_to_dict(pretrain_config),
    }
    if include_regularized_covariance:
        artifact["regularized_covariance"] = torch.stack(
            regularized_covariances, dim=0
        ).contiguous()
    if include_raw_covariance:
        artifact["raw_covariance"] = torch.stack(raw_covariances, dim=0).contiguous()
    validate_whitening_artifact(artifact, hidden_dim=hidden_dim, seq_type=seq_type, strict=True)
    return artifact


def validate_whitening_artifact(
    artifact: Dict[str, Any],
    hidden_dim: Optional[int] = None,
    seq_type: Optional[str] = None,
    base_token_ids: Optional[Sequence[Sequence[int]]] = None,
    model_checkpoint_identifier: Optional[str] = None,
    null_predictor_identifier: Optional[str] = None,
    strict: bool = True,
    allow_checkpoint_mismatch: bool = False,
) -> Dict[str, Any]:
    """Validate whitening artifact identity, compatibility, shapes, and values.

    Expected dimensions, sequence type, token groups, and checkpoint identifiers
    are optional compatibility constraints. Checkpoint identifiers and molecule
    type are enforced only in ``strict`` mode; identifier checks may be disabled
    explicitly with ``allow_checkpoint_mismatch``. The same artifact object is
    returned after successful validation.
    """
    if artifact.get("artifact_type") != WHITENING_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported whitening artifact_type={artifact.get('artifact_type')!r}")
    format_version = int(artifact.get("format_version", -1))
    if format_version not in SUPPORTED_WHITENING_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported whitening format_version={artifact.get('format_version')!r}")
    artifact_hidden_dim = int(artifact.get("hidden_dim", -1))
    if hidden_dim is not None and artifact_hidden_dim != int(hidden_dim):
        raise ValueError(f"Whitening hidden_dim={artifact_hidden_dim}, expected {int(hidden_dim)}")
    if artifact_hidden_dim <= 0:
        raise ValueError("Whitening artifact has invalid hidden_dim")

    base_names = artifact.get("base_names")
    stored_base_token_ids = artifact.get("base_token_ids")
    if not isinstance(base_names, (list, tuple)) or not base_names:
        raise ValueError("Whitening artifact must contain non-empty base_names")
    if not isinstance(stored_base_token_ids, (list, tuple)) or len(stored_base_token_ids) != len(base_names):
        raise ValueError("Whitening artifact base_token_ids do not match base_names")
    stored_base_token_ids = [
        [int(x) for x in group] if not isinstance(group, int) else [int(group)]
        for group in stored_base_token_ids
    ]

    if seq_type is not None:
        expected_names, expected_token_ids = canonical_base_groups(seq_type)
        if list(base_names) != expected_names:
            raise ValueError(f"Whitening base_names={list(base_names)}, expected {expected_names} for seq_type={seq_type}")
        if stored_base_token_ids != expected_token_ids:
            raise ValueError(
                f"Whitening base_token_ids={stored_base_token_ids}, expected {expected_token_ids} for seq_type={seq_type}"
            )
        molecule_type = artifact.get("molecule_type")
        if strict and molecule_type is not None and str(molecule_type).lower() != str(seq_type).lower():
            raise ValueError(f"Whitening molecule_type={molecule_type!r}, expected {seq_type!r}")

    if base_token_ids is not None:
        expected_token_ids = [
            [int(x) for x in group] if not isinstance(group, int) else [int(group)]
            for group in base_token_ids
        ]
        if stored_base_token_ids != expected_token_ids:
            raise ValueError(f"Whitening token mapping mismatch: {stored_base_token_ids} != {expected_token_ids}")

    if not allow_checkpoint_mismatch:
        if model_checkpoint_identifier is not None:
            stored = artifact.get("model_checkpoint_identifier")
            if strict and stored is not None and str(stored) != str(model_checkpoint_identifier):
                raise ValueError(
                    f"Whitening model checkpoint mismatch: artifact={stored!r}, expected={model_checkpoint_identifier!r}"
                )
        if null_predictor_identifier is not None:
            stored = artifact.get("null_predictor_identifier")
            if strict and stored is not None and str(stored) != str(null_predictor_identifier):
                raise ValueError(
                    f"Whitening null predictor mismatch: artifact={stored!r}, expected={null_predictor_identifier!r}"
                )

    if format_version == 1:
        whitening_kmer_size = 1
        num_contexts = len(base_names)
    else:
        whitening_kmer_size = validate_centered_kmer_size(
            artifact.get("whitening_kmer_size", -1),
            argument_name="artifact whitening_kmer_size",
        )
        num_contexts = int(artifact.get("num_contexts", -1))
        expected_num_contexts = 4 ** whitening_kmer_size
        if num_contexts != expected_num_contexts:
            raise ValueError(
                f"Whitening artifact num_contexts={num_contexts}, expected {expected_num_contexts} "
                f"for k={whitening_kmer_size}"
            )
        context_names = artifact.get("context_names")
        if not isinstance(context_names, (list, tuple)) or len(context_names) != num_contexts:
            raise ValueError("Whitening artifact context_names do not match num_contexts")
        if seq_type is not None:
            expected_context_names, expected_parent_index = canonical_kmer_contexts(
                whitening_kmer_size,
                seq_type,
            )
            if list(context_names) != expected_context_names:
                raise ValueError(
                    f"Whitening context_names do not match canonical {seq_type} "
                    f"{whitening_kmer_size}-mer order"
                )
        parent_index = torch.as_tensor(artifact.get("context_parent_base_index"), dtype=torch.long)
        if tuple(parent_index.shape) != (num_contexts,):
            raise ValueError(
                "Whitening artifact context_parent_base_index must have shape "
                f"({num_contexts},)"
            )
        if ((parent_index < 0) | (parent_index >= len(base_names))).any():
            raise ValueError("Whitening artifact contains an invalid context parent-base index")
        if seq_type is not None and not torch.equal(parent_index.cpu(), expected_parent_index):
            raise ValueError("Whitening artifact context parent mapping is inconsistent with its k-mer order")

    expected_shapes = {
        "mean": (num_contexts, artifact_hidden_dim),
        "whitening": (num_contexts, artifact_hidden_dim, artifact_hidden_dim),
        "regularized_eigenvalues": (num_contexts, artifact_hidden_dim),
        "counts": (num_contexts,),
        "tau": (num_contexts,),
        "shrinkage": (num_contexts,),
        "eigenvalue_floor": (num_contexts,),
    }
    for key, shape in expected_shapes.items():
        if key not in artifact:
            raise ValueError(f"Whitening artifact missing required key {key!r}")
        tensor = torch.as_tensor(artifact[key])
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Whitening artifact key {key!r} has shape {tuple(tensor.shape)}, expected {shape}")
        if key != "counts" and not torch.isfinite(tensor).all():
            raise ValueError(f"Whitening artifact key {key!r} contains non-finite values")

    counts = torch.as_tensor(artifact["counts"])
    if format_version == 1 and (counts <= artifact_hidden_dim).any():
        raise ValueError("Whitening artifact contains a base group with too few observations")
    if format_version >= 2 and (counts <= 1).any():
        raise ValueError("Whitening artifact contains a context with fewer than two observations")
    if (torch.as_tensor(artifact["regularized_eigenvalues"]) <= 0.0).any():
        raise ValueError("Whitening artifact regularized eigenvalues must be positive")
    if torch.as_tensor(artifact["whitening"]).dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError("Whitening matrix must be floating point")

    for optional_key, optional_shape in (
        ("raw_covariance", (num_contexts, artifact_hidden_dim, artifact_hidden_dim)),
        ("regularized_covariance", (num_contexts, artifact_hidden_dim, artifact_hidden_dim)),
    ):
        if optional_key in artifact:
            tensor = torch.as_tensor(artifact[optional_key])
            if tuple(tensor.shape) != optional_shape:
                raise ValueError(
                    f"Whitening artifact key {optional_key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {optional_shape}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Whitening artifact key {optional_key!r} contains non-finite values")
    return artifact


def save_whitening_artifact(artifact: Dict[str, Any], path: str) -> None:
    """Validate and serialize a whitening artifact with ``torch.save``."""
    validate_whitening_artifact(artifact, hidden_dim=int(artifact["hidden_dim"]), strict=True)
    torch.save(artifact, path)


def load_whitening_artifact(
    path: str,
    device: Optional[torch.device | str] = None,
    hidden_dim: Optional[int] = None,
    seq_type: Optional[str] = None,
    base_token_ids: Optional[Sequence[Sequence[int]]] = None,
    model_checkpoint_identifier: Optional[str] = None,
    null_predictor_identifier: Optional[str] = None,
    strict: bool = True,
    allow_checkpoint_mismatch: bool = False,
) -> Dict[str, Any]:
    """Load, validate, and optionally move whitening artifact tensors to a device.

    The file is always deserialized through CPU first. Only known numeric tensor
    fields are transferred; provenance and diagnostic structures remain on CPU.
    """
    artifact = torch.load(path, map_location="cpu")
    validate_whitening_artifact(
        artifact,
        hidden_dim=hidden_dim,
        seq_type=seq_type,
        base_token_ids=base_token_ids,
        model_checkpoint_identifier=model_checkpoint_identifier,
        null_predictor_identifier=null_predictor_identifier,
        strict=strict,
        allow_checkpoint_mismatch=allow_checkpoint_mismatch,
    )
    if device is not None:
        device = torch.device(device)
        for key in (
            "mean",
            "whitening",
            "regularized_covariance",
            "raw_covariance",
            "regularized_eigenvalues",
            "raw_eigenvalues",
            "tau",
            "shrinkage",
            "eigenvalue_floor",
            "counts",
            "effective_counts",
            "token_to_base_index",
        ):
            if key in artifact and torch.is_tensor(artifact[key]):
                artifact[key] = artifact[key].to(device=device)
    return artifact


class BaseSpecificWhiteningScorer:
    """Apply a fitted base- or centered-k-mer-specific Mahalanobis score.

    Format-v1 artifacts select one transform from the central base. Format-v2
    artifacts select one transform from a complete centered k-mer. The scorer
    owns the device-local means and whitening matrices; calling ``to`` mutates
    those tensors and the scorer's target device.
    """

    def __init__(
        self,
        artifact: Dict[str, Any],
        device: Optional[torch.device | str] = None,
        strict: bool = True,
        expected_hidden_dim: Optional[int] = None,
        expected_seq_type: Optional[str] = None,
    ):
        """Initialize from an in-memory artifact with optional compatibility checks."""
        validate_whitening_artifact(
            artifact,
            hidden_dim=expected_hidden_dim,
            seq_type=expected_seq_type,
            strict=strict,
        )
        self.artifact = artifact
        self.format_version = int(artifact["format_version"])
        self.hidden_dim = int(artifact["hidden_dim"])
        self.base_names = list(artifact["base_names"])
        self.base_token_ids = [
            [int(x) for x in group] if not isinstance(group, int) else [int(group)]
            for group in artifact["base_token_ids"]
        ]
        self.whitening_kmer_size = int(artifact.get("whitening_kmer_size", 1))
        self.context_names = list(artifact.get("context_names", self.base_names))
        self.num_contexts = len(self.context_names)
        # Retain this attribute for callers that inspect legacy scorer metadata.
        self.num_bases = len(self.base_names)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mean = torch.as_tensor(artifact["mean"], dtype=torch.float32, device=self.device)
        self.whitening = torch.as_tensor(artifact["whitening"], dtype=torch.float32, device=self.device)
        self.token_to_base_index = build_token_to_base_index(self.base_token_ids, device=self.device)

    @classmethod
    def from_path(
        cls,
        path: str,
        device: Optional[torch.device | str] = None,
        strict: bool = True,
        expected_hidden_dim: Optional[int] = None,
        expected_seq_type: Optional[str] = None,
        allow_checkpoint_mismatch: bool = False,
    ) -> "BaseSpecificWhiteningScorer":
        """Load a whitening artifact and construct a scorer on ``device``."""
        artifact = load_whitening_artifact(
            path,
            device=None,
            hidden_dim=expected_hidden_dim,
            seq_type=expected_seq_type,
            strict=strict,
            allow_checkpoint_mismatch=allow_checkpoint_mismatch,
        )
        return cls(
            artifact,
            device=device,
            strict=strict,
            expected_hidden_dim=expected_hidden_dim,
            expected_seq_type=expected_seq_type,
        )

    def to(self, device: torch.device | str) -> "BaseSpecificWhiteningScorer":
        """Move scoring tensors to ``device`` in place and return ``self``."""
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.whitening = self.whitening.to(self.device)
        self.token_to_base_index = self.token_to_base_index.to(self.device)
        return self

    def _flat_context_indices(self, base_seq_input: torch.Tensor) -> torch.Tensor:
        """Map a base-token sequence to the artifact's transform indices."""
        ids = base_seq_input.to(device=self.device, dtype=torch.long, non_blocking=True)
        if self.format_version >= 2:
            context_ids, _valid = centered_kmer_indices(ids, self.whitening_kmer_size)
            return context_ids.reshape(-1)

        ids = ids.reshape(-1)
        base_indices = torch.full(ids.shape, -1, dtype=torch.long, device=self.device)
        in_range = (ids >= 0) & (ids < self.token_to_base_index.numel())
        if in_range.any():
            base_indices[in_range] = self.token_to_base_index[ids[in_range]]
        return base_indices

    def score(
        self,
        residual: torch.Tensor,
        base_seq_input: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        return_coordinates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Score residuals with the matching base or centered-k-mer transform.

        ``residual`` has shape ``(..., hidden_dim)``. ``base_seq_input`` and the
        optional ``valid_mask`` must have shape ``...``; centered-k-mer
        artifacts currently require the usual two-dimensional ``(batch,
        bases)`` sequence shape. Incomplete/noncanonical contexts, masked
        entries, and non-finite residuals receive NaN. ``chunk_size`` limits
        temporary device work per context without changing results.

        Returns the Mahalanobis-squared score tensor with shape ``...``. When
        ``return_coordinates`` is true, also returns whitened coordinates with
        shape ``(..., hidden_dim)``; invalid rows are NaN in both outputs.
        Whitening arithmetic and both outputs use float32 even under caller
        autocast, matching artifact fitting and calibration. The caller's
        autocast and global matmul policies are preserved.
        """
        if residual.shape[-1] != self.hidden_dim:
            raise ValueError(f"Residual dim={residual.shape[-1]}, expected {self.hidden_dim}")
        leading_shape = residual.shape[:-1]
        flat_residual = residual.reshape(-1, self.hidden_dim).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        if base_seq_input.shape != leading_shape:
            raise ValueError(
                f"base_seq_input shape {tuple(base_seq_input.shape)} does not match residual leading shape {tuple(leading_shape)}"
            )
        context_indices = self._flat_context_indices(base_seq_input)
        valid = context_indices >= 0
        if valid_mask is not None:
            if valid_mask.shape != leading_shape:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match residual leading shape {tuple(leading_shape)}"
                )
            valid = valid & valid_mask.reshape(-1).to(device=self.device, dtype=torch.bool, non_blocking=True)
        valid = valid & torch.isfinite(flat_residual).all(dim=-1)

        scores = torch.full((flat_residual.shape[0],), float("nan"), dtype=torch.float32, device=self.device)
        coordinates = None
        if return_coordinates:
            coordinates = torch.full_like(flat_residual, float("nan"), dtype=torch.float32)

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() > 0:
            order = torch.argsort(context_indices[valid_indices])
            sorted_indices = valid_indices[order]
            sorted_contexts = context_indices[sorted_indices]
            active_contexts, context_counts = torch.unique_consecutive(
                sorted_contexts,
                return_counts=True,
            )
        else:
            sorted_indices = valid_indices
            active_contexts = torch.empty(0, dtype=torch.long, device=self.device)
            context_counts = torch.empty(0, dtype=torch.long, device=self.device)

        offset = 0
        for context_index, context_count in zip(
            active_contexts.tolist(),
            context_counts.tolist(),
        ):
            context_index = int(context_index)
            idx = sorted_indices[offset:offset + int(context_count)]
            offset += int(context_count)
            step = int(chunk_size) if chunk_size is not None and int(chunk_size) > 0 else int(idx.numel())
            for start in range(0, idx.numel(), step):
                chunk_idx = idx[start:start + step]
                # FP32 operands/output buffers do not prevent autocast from
                # rounding the matrix product and its squares to BF16.
                with torch.autocast(device_type=self.device.type, enabled=False):
                    x = flat_residual[chunk_idx].float() - self.mean[context_index]
                    y = x.matmul(self.whitening[context_index])
                    q = torch.sum(y * y, dim=-1, dtype=torch.float32)
                tiny_negative = (q < 0.0) & (q > -1e-5)
                if tiny_negative.any():
                    q = torch.where(tiny_negative, torch.zeros_like(q), q)
                scores[chunk_idx] = q
                if coordinates is not None:
                    coordinates[chunk_idx] = y

        scores = scores.reshape(leading_shape)
        if coordinates is None:
            return scores
        return scores, coordinates.reshape(*leading_shape, self.hidden_dim)


def empirical_score_summary(scores: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Summarize finite scores with count, mean, standard deviation, and quantiles.

    The optional mask is applied before non-finite values are discarded. Empty
    inputs return only ``{"count": 0}``.
    """
    values = scores.detach().float()
    if valid_mask is not None:
        values = values[valid_mask.bool()]
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0}
    quantile_levels = torch.tensor([0.5, 0.9, 0.95, 0.99, 0.995], dtype=torch.float32)
    quantiles = torch.quantile(values.cpu(), quantile_levels)
    out = {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0,
    }
    for level, value in zip(quantile_levels.tolist(), quantiles.tolist()):
        out[f"q{level:g}"] = float(value)
    return out
