"""Shared decoding, alignment, scheduling, loss, and normalization utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from typing import Callable, Tuple

import math
from torch.optim.lr_scheduler import LambdaLR

from numba import njit

import torch
import torch.nn.functional as F

@njit
def mm_ml_coords_jit(mm, ml, seq):
    """Decode byte-level SAM MM deltas and paired ML values.

    ``mm`` and ``seq`` are uint8 ASCII arrays and ``ml`` is a uint8 probability
    array. Coordinates are zero-based in the supplied sequence orientation.
    Multi-code MM groups consume one ML value per modification code at each
    site; ``N`` treats every sequence position as a candidate.
    """
    coords = np.empty(len(ml), np.int32)
    vals = np.empty(len(ml), np.uint8)

    m = 0
    ml_i = 0
    n = len(mm)

    while m < n:
        if mm[m] == 59:  # ;
            m += 1
            continue

        base = mm[m]     # e.g. ord("C")
        m += 2           # skip base and strand

        # count modification codes until comma / ; / . / ?
        n_mods = 0
        while m < n and mm[m] not in (44, 59):  # , ;
            if mm[m] in (46, 63):              # . ?
                m += 1
                continue
            if 48 <= mm[m] <= 57:              # numeric ChEBI code
                n_mods = 1
                while m < n and mm[m] not in (44, 59):
                    m += 1
                break
            n_mods += 1
            m += 1

        if m >= n or mm[m] == 59:
            while m < n and mm[m] != 59:
                m += 1
            continue

        m += 1  # skip first comma

        seq_i = -1

        while m < n and mm[m] != 59:
            delta = 0
            while m < n and 48 <= mm[m] <= 57:
                delta = delta * 10 + (mm[m] - 48)
                m += 1

            skip = delta + 1
            while skip > 0:
                seq_i += 1
                if seq[seq_i] == base or base == 78:  # N means any base
                    skip -= 1

            for _ in range(n_mods):
                coords[ml_i] = seq_i
                vals[ml_i] = ml[ml_i]
                ml_i += 1

            if m < n and mm[m] == 44:  # ,
                m += 1

        if m < n and mm[m] == 59:
            m += 1

    return coords[:ml_i], vals[:ml_i]


def normalize_mm_code(code):
    """Validate one SAM modification code, returning a letter or integer ID.

    Accept one ASCII letter or a positive numeric ChEBI ID (including its
    decimal string representation). Names such as ``m6A`` and whole MM group
    headers such as ``A+a`` are not codes.
    """
    if isinstance(code, str):
        if len(code) == 1 and code.isascii() and code.isalpha():
            return code
        if code.isascii() and code.isdecimal() and int(code) > 0:
            return int(code)
    elif isinstance(code, int) and not isinstance(code, bool) and code > 0:
        return code
    raise ValueError('MM_code must be one ASCII letter or a positive numeric modification code')


def mm_ml_coords(read, mod_code=None):
    """Decode a BAM record's MM/ML tags in original read orientation.

    Returns zero-based ``int32`` coordinates and ``float32`` probabilities
    reconstructed as ML-bin midpoints ``(value + 0.5) / 256``. The record must
    expose MM, ML, and a forward sequence through pysam. ``mod_code`` selects
    one exact SAM code (a letter or numeric ChEBI ID); other codes contribute
    neither positive nor negative targets. Selected calls are sorted by read
    coordinate, including interleaved multi-code MM groups. An absent code
    returns empty arrays. Unknown probabilities are ignored; malformed tags
    and duplicate selected coordinates raise ``ValueError``.

    With ``mod_code=None``, retain the existing pooled decoder and its tag
    order. Codes are matched exactly, without translating equivalent letter
    and numeric codes or inferring canonical calls at unlisted positions.
    """
    if mod_code is not None:
        code = normalize_mm_code(mod_code)
        # Require both tags as on the pooled path. Pysam handles MM code
        # interleaving and reverse-alignment orientation without losing code IDs.
        read.get_tag('MM')
        read.get_tag('ML')
        decoded = read.modified_bases_forward
        if decoded is None:
            raise ValueError(f'Cannot decode MM/ML tags for read {read.query_name!r}')
        calls = sorted(
            (pos, quality)
            for (_, _, candidate), positions in decoded.items()
            if candidate == code
            for pos, quality in positions
            if quality >= 0
        )
        coords = np.asarray([pos for pos, _ in calls], dtype=np.int32)
        if np.any(coords[1:] == coords[:-1]):
            raise ValueError(
                f'Duplicate coordinates for MM_code={mod_code!r} in read {read.query_name!r}'
            )
        vals = np.asarray([quality for _, quality in calls], dtype=np.float32)
        return coords, (vals + 0.5) / 256.0

    mm = np.frombuffer(read.get_tag("MM").encode("ascii"), dtype=np.uint8)
    ml = np.asarray(read.get_tag("ML"), dtype=np.uint8)
    
    # Use forward/original read orientation.
    seq = np.frombuffer(read.get_forward_sequence().upper().encode("ascii"),
                        dtype=np.uint8)

    coords, vals = mm_ml_coords_jit(mm, ml, seq)
    vals = (vals.astype(np.float32) + 0.5) / 256.0
    
    return coords, vals

    

def focal_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | list[float] | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute multiclass focal loss for hard-distribution or soft targets.

    Parameters
    ----------
    logits, targets : torch.Tensor
        Matching tensors shaped ``(N, C, ...)`` with classes on dimension one.
        Target vectors normally sum to one.
    gamma : float, default=2.0
        Nonnegative focusing exponent on ``1 - p(class)``.
    alpha : torch.Tensor, list of float, or None
        Optional length-``C`` per-class multipliers.
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction after summing the class dimension.
    eps : float, default=1e-8
        Lower bound applied to the focal base before exponentiation.
    label_smoothing : float, default=0.0
        Mix targets with a uniform distribution before focal weighting.

    Returns
    -------
    torch.Tensor
        Scalar for mean/sum reduction or shape ``(N, ...)`` for ``none``.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape. "
            f"Got logits={logits.shape}, targets={targets.shape}."
        )
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}.")
    if not 0 <= label_smoothing < 1:
        raise ValueError(
            f"label_smoothing must be in the range [0, 1), got {label_smoothing}."
        )

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if label_smoothing:
        targets = targets * (1.0 - label_smoothing)
        targets = targets + label_smoothing / logits.shape[1]

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    focal_weight = (1.0 - probs).clamp(min=eps) ** gamma

    loss = -targets * focal_weight * log_probs

    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
        else:
            alpha = alpha.to(device=logits.device, dtype=logits.dtype)

        if alpha.ndim != 1 or alpha.numel() != logits.shape[1]:
            raise ValueError(
                f"alpha must have shape [C], where C={logits.shape[1]}. "
                f"Got alpha shape={alpha.shape}."
            )

        view_shape = [1, -1] + [1] * (logits.ndim - 2)
        loss = loss * alpha.view(*view_shape)

    loss = loss.sum(dim=1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

def _to_numpy_float64_1d(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Expected a 1D signal.")
    return np.ascontiguousarray(x)


def make_sloped_diagonal_band(n, m, half_width):
    """Construct row bounds around a short-to-long sloped diagonal.

    For an ``n``-element short sequence and ``m``-element long sequence, row
    ``i`` permits long indices within ``half_width`` of the length-ratio
    diagonal. Endpoints ``(0, 0)`` and ``(n-1, m-1)`` are forced admissible.

    Returns
    -------
    tuple of numpy.ndarray
        Inclusive ``int32`` start/end indices of length ``n``.
    """
    if n < 1 or m < 1:
        raise ValueError("n and m must be >= 1")
    if half_width < 0:
        raise ValueError("half_width must be >= 0")
    if n > m:
        raise ValueError(
            f"This one-to-many formulation requires len(s1) <= len(s2), got n={n}, m={m}"
        )

    if n == 1:
        centers = np.array([0], dtype=np.int32)
    else:
        centers = np.rint(np.linspace(0, m - 1, n)).astype(np.int32)

    starts = np.maximum(0, centers - half_width).astype(np.int32)
    ends   = np.minimum(m - 1, centers + half_width).astype(np.int32)

    # Force endpoints admissible
    starts[0] = 0
    ends[-1] = m - 1

    return starts, ends


@njit(cache=True)
def _banded_onetomany_path_core(x, y, starts, ends):
    """
    Exact monotone one-to-many alignment inside a row-wise contiguous band.

    Allowed moves:
      diag: (i-1, j-1) -> (i, j)
      left: (i,   j-1) -> (i, j)

    Forbidden:
      up:   (i-1, j)   -> (i, j)

    Interpretation:
      - diagonal starts the chunk for the next short index
      - left extends the current short index over more long samples

    This enforces:
      one short -> one contiguous chunk of one or more long samples
      never many short -> same long sample

    Returns
    -------
    total_cost : float64
        Sum of squared differences along the path.
    path : int32 array of shape (L, 2)
        Path in (i, j) order.
    """
    n = x.shape[0]
    m = y.shape[0]

    max_bw = 0
    for i in range(n):
        bw = ends[i] - starts[i] + 1
        if bw > max_bw:
            max_bw = bw

    inf = 1e300

    dp = np.full((n, max_bw), inf, dtype=np.float64)
    ptr = np.full((n, max_bw), -1, dtype=np.int8)
    # ptr codes: 0=diag, 2=left, -1=start/unreachable

    for i in range(n):
        lo = starts[i]
        hi = ends[i]

        for j in range(lo, hi + 1):
            k = j - lo
            c = (x[i] - y[j]) * (x[i] - y[j])

            # Start cell
            if i == 0 and j == 0:
                dp[i, k] = c
                ptr[i, k] = -1
                continue

            best = inf
            best_ptr = -1

            # left: stay on same short index, consume another long sample
            if j > lo:
                cand = dp[i, k - 1] + c
                if cand < best:
                    best = cand
                    best_ptr = 2

            # diag: move to next short index, must also consume next long sample
            if i > 0 and j > 0:
                plo = starts[i - 1]
                phi = ends[i - 1]
                if plo <= (j - 1) <= phi:
                    cand = dp[i - 1, (j - 1) - plo] + c
                    if cand < best:
                        best = cand
                        best_ptr = 0

            dp[i, k] = best
            ptr[i, k] = best_ptr

    # Must end at (n-1, m-1)
    if not (starts[n - 1] <= (m - 1) <= ends[n - 1]):
        return inf, np.empty((0, 2), dtype=np.int32)

    final_k = (m - 1) - starts[n - 1]
    total_cost = dp[n - 1, final_k]

    if total_cost >= inf / 10:
        return inf, np.empty((0, 2), dtype=np.int32)

    # Backtrack
    path_rev = np.empty((n + m - 1, 2), dtype=np.int32)
    t = 0

    i = n - 1
    j = m - 1

    while True:
        path_rev[t, 0] = i
        path_rev[t, 1] = j
        t += 1

        if i == 0 and j == 0:
            break

        k = j - starts[i]
        p = ptr[i, k]

        if p == 0:      # diag
            i -= 1
            j -= 1
        elif p == 2:    # left
            j -= 1
        else:
            return inf, np.empty((0, 2), dtype=np.int32)

    path = np.empty((t, 2), dtype=np.int32)
    for u in range(t):
        path[u, 0] = path_rev[t - 1 - u, 0]
        path[u, 1] = path_rev[t - 1 - u, 1]

    return total_cost, path


def warping_path_sloped_fast(s1, s2, window):
    """
    Fast exact one-to-many monotone alignment in a narrow band around the sloped diagonal.

    Parameters
    ----------
    s1 : 1D numpy array or torch tensor
        Shorter sequence.
    s2 : 1D numpy array or torch tensor
        Longer sequence.
    window : int
        Half-width of the band in s2 / long-sequence index units.

    Returns
    -------
    path : list[(i, j)]
        i indexes s1, j indexes s2
    cost : float
        Sum of squared differences along the path
        Raises ``ValueError`` if ``s1`` is longer than ``s2``. An unreachable
        endpoint yields an empty path and infinite cost.
    """
    x = _to_numpy_float64_1d(s1)
    y = _to_numpy_float64_1d(s2)

    if len(x) > len(y):
        raise ValueError(
            f"Expected s1 to be the shorter sequence and s2 the longer one, got {len(x)} > {len(y)}"
        )

    starts, ends = make_sloped_diagonal_band(len(x), len(y), window)
    cost, path = _banded_onetomany_path_core(x, y, starts, ends)

    return path, cost

class CurriculumManager:
    """Interpolate MASKLM/SUBLM masking parameters from a CSV schedule.

    Each non-empty line contains task name, one-based start checkpoint, end
    checkpoint, start/end mask probabilities, and for ``masklm`` start/end
    integer spans. Before the first applicable range, defaults are mask
    probability 0.25 and MASKLM span 5. Values persist after a range ends.
    """

    def __init__(self, config_path=None):
        self.schedules = {"masklm": [], "sublm": []}
        self.current_state = {
            "masklm": {"mask_prob": 0.25, "mask_span": 5},
            "sublm": {"mask_prob": 0.25}
        }
        if config_path:
            self._load_config(config_path)

    def _load_config(self, path):
        with open(path, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if not parts: continue
                
                task_type = parts[0]
                entry = {
                    "start_cp": int(parts[1])-1,
                    "end_cp": int(parts[2]),
                    "start_prob": float(parts[3]),
                    "end_prob": float(parts[4]),
                }
                if task_type == "masklm":
                    entry["start_span"] = int(parts[5])
                    entry["end_span"] = int(parts[6])
                
                self.schedules[task_type].append(entry)
        
        for task in self.schedules:
            self.schedules[task].sort(key=lambda x: x["start_cp"])

    def _interpolate(self, cp, start_cp, end_cp, start_val, end_val, is_int=False):
        if cp <= start_cp: return start_val
        if cp >= end_cp: return end_val
        fraction = (cp - start_cp) / (end_cp - start_cp)
        val = start_val + fraction * (end_val - start_val)
        return math.floor(val) if is_int else val

    def get_params(self, checkpoint):
        """Update and return task parameters interpolated at ``checkpoint``.

        The returned mapping is the manager's mutable current state, not a
        defensive copy.
        """
        for task_type in ["masklm", "sublm"]:
            active_range = None
            for schedule in self.schedules[task_type]:
                if schedule["start_cp"] <= checkpoint:
                    active_range = schedule
                if schedule["start_cp"] <= checkpoint <= schedule["end_cp"]:
                    break # Found the exact active range
            
            if active_range:
                self.current_state[task_type]["mask_prob"] = self._interpolate(
                    checkpoint, active_range["start_cp"], active_range["end_cp"],
                    active_range["start_prob"], active_range["end_prob"]
                )
                if task_type == "masklm":
                    self.current_state[task_type]["mask_span"] = self._interpolate(
                        checkpoint, active_range["start_cp"], active_range["end_cp"],
                        active_range["start_span"], active_range["end_span"], is_int=True
                    )
        return self.current_state
    
def warmup_only_scheduler(optimizer, warmup_steps):
    """Return a LambdaLR that ramps linearly from zero to the base rates."""
    def lr_lambda(step):
        """Map optimizer step to the linear warmup multiplier."""
        return min(1.0, step / max(1, warmup_steps))
    return LambdaLR(optimizer, lr_lambda)

def rewarmup_then_constant(optimizer, warmup_steps, start_factor=0.2):
    """Linearly rewarm from ``start_factor`` to the optimizer base rates.

    After ``warmup_steps`` the multiplier remains one. The scheduler uses the
    learning rates present when it is constructed as its base rates.
    """
    def lr_lambda(step):
        """Map optimizer step to a rewarmup multiplier capped at one."""
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def get_warmup_cosine_schedule(optimizer, warmup_steps, decay_start_steps, total_steps, start_factor=0.01, min_factor=0.1):
    """Create linear-warmup, constant-plateau, cosine-decay scheduling.

    The multiplier rises from ``start_factor`` to one over ``warmup_steps``,
    stays at one until ``decay_start_steps``, then cosine-decays to
    ``min_factor`` at ``total_steps`` and remains there afterward.
    """
    def lr_lambda(step):
        """Map optimizer step to its warmup, plateau, or cosine multiplier."""
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))

        if step < decay_start_steps:
            return 1.0

        # Clamp progress after the terminal step so the cosine cannot rise on
        # later optimizer updates.
        step = min(step, total_steps)
        
        decay_steps = total_steps - decay_start_steps
        progress = (step - decay_start_steps) / max(1, decay_steps)

        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def align_to_power_of_two(value: int) -> int:
    """Return the largest power of two not exceeding a positive integer.

    Values below one return zero.
    """
    if value < 1:
        return 0

    return 1 << (value.bit_length() - 1)


def get_dummy_loss(model: nn.Module) -> torch.Tensor:
    """Connect every trainable parameter to autograd through a zero-valued loss.

    DDP can then observe parameters from task-specific modules that the current
    forward branch skipped. Wrapped models exposing ``module`` are unwrapped.
    """
    dummy_loss = 0.0

    real_model = model.module if hasattr(model, "module") else model

    for param in real_model.parameters():
        if param.requires_grad:
            dummy_loss += param.sum() * 0.0

    return dummy_loss


def require_single_process_execution(script_name: str, reason: str):
    """Reject multi-rank execution and select the requested local CUDA device.

    Returns ``(rank, world_size, local_rank)`` from the environment when
    ``world_size`` is one. Otherwise raises ``RuntimeError`` including the
    caller-supplied ``script_name`` and ``reason``.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != 1:
        raise RuntimeError(
            f"{script_name} must run with a single process/GPU. {reason} "
            "Launch it without torchrun/srun multi-rank environment variables."
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

class TorchSeqToSignal:
    """Vectorized GPU lookup of centered expected current for 9-mer sequences.

    ``lookup_table_np`` must be indexed by base-4 encoded 9-mers. The table and
    encoding powers are copied to ``device`` at construction.
    """

    def __init__(self, lookup_table_np, device='cuda'):
        self.lookup = torch.tensor(lookup_table_np, device=device, dtype=torch.float32)
        
        k = 9
        powers = 4 ** torch.arange(k - 1, -1, -1, device=device)
        self.powers = powers.long()  # Integers for indexing
        self.k = k
        self.half_k = 4

    def __call__(self, seq_ints_torch):
        """Return expected levels for integer sequences shaped ``(B, L)``.

        Inputs use ``A/C/G/T = 0/1/2/3`` and must have at least nine bases.
        Complete 9-mers populate positions ``4:L-4``; four zero values pad each
        edge so the output remains ``(B, L)``.
        """
        B, L = seq_ints_torch.shape
        
        windows = seq_ints_torch.unfold(dimension=1, size=self.k, step=1)
        
        indices = (windows * self.powers).sum(dim=2)
        
        signals = self.lookup[indices]
        
        out = torch.zeros((B, L), device=seq_ints_torch.device, dtype=torch.float32)

        out[:, self.half_k : -self.half_k] = signals
        
        return out
    
def estimate_per_sample_memory(
    model: nn.Module,
    device: torch.device,
    max_base_seq_len: int,
    max_ts_len: int,
    ts_dim: int,
    add_percentile: bool,
    mask_fn: Callable[[torch.Tensor, torch.Tensor, float, int], Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Measure peak CUDA memory for one synthetic masked-language sample.

    The model is temporarily put in training mode, run under bfloat16 autocast,
    and backpropagated through MASKLM plus dummy loss. Gradients and CUDA cache
    are cleared afterward and the original evaluation mode is restored. CPU
    devices return ``0.0`` without running the model.

    Returns
    -------
    float
        Increase in peak allocated CUDA bytes for the one-sample probe.
    """
    if device.type != "cuda":
        return 0.0

    batch_size = 1
    base_seq_input = torch.randint(1, 5, (batch_size, max_base_seq_len), device=device)
    exp_signal = torch.randn(batch_size, max_base_seq_len).to(device)
    ts_input = torch.randn(batch_size, max_ts_len, ts_dim, device=device)
    percentiles_input = torch.randn(batch_size, max_ts_len, ts_dim, device=device)
    pad_base_seq = torch.ones(batch_size, max_base_seq_len, device=device, dtype=torch.bool)
    pad_ts = torch.ones(batch_size, max_ts_len, device=device, dtype=torch.bool)

    task_base_seq, task_exp_signal, mask_mlm = mask_fn(base_seq_input, exp_signal, pad_base_seq, mask_prob=0.15, span=1)

    was_training = model.training
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(
            task_base_seq,
            task_exp_signal,
            ts_input,
            percentiles_input=percentiles_input,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
        )

    logits = out["masklm_logits"].float()[mask_mlm]
    targets = base_seq_input[mask_mlm] - 1
    loss = F.cross_entropy(logits, targets)
    loss = loss + get_dummy_loss(model)

    loss.backward()
    torch.cuda.synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    per_sample_mem = max(peak_allocated - start_allocated, 0) / batch_size

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    torch.cuda.empty_cache()

    return float(per_sample_mem)


def quantiles(in_data, quants):
    """Compute linearly interpolated quantiles of a one-dimensional array.

    ``quants`` contains fractions conventionally in ``[0, 1]``. Empty input
    returns an empty array.
    """
    if in_data.size == 0:
        return np.array([])
    
    
    data = np.sort(in_data)
    n = len(data)
    
    pos_indices = quants * (n - 1)
    
    idx_left = np.floor(pos_indices).astype(np.int64)
    idx_right = np.ceil(pos_indices).astype(np.int64)
    
    np.clip(idx_left, 0, n - 1, out=idx_left)
    np.clip(idx_right, 0, n - 1, out=idx_right)
    
    data_left = data[idx_left]
    data_right = data[idx_right]
    
    fraction = pos_indices - idx_left
    
    results = data_left * (1 - fraction) + data_right * fraction
    
    return results

class DegenerateQuantileNormalizationError(ValueError):
    """Raised when an affine quantile-normalization fit cannot be estimated."""


def linear_regression(x, y):
    """Fit ``y = m*x + b`` and return slope, intercept, and correlation.

    Inputs are flattened to ``float64``. Unequal lengths raise ``ValueError``;
    fewer than two finite values, effectively constant inputs, or a non-finite
    fit raise ``DegenerateQuantileNormalizationError``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if len(y) != n:
        raise ValueError(
            f"Linear regression inputs must have equal lengths; got {n} and {len(y)}."
        )

    # An affine quantile fit is undefined when either side is empty, non-finite,
    # or effectively constant. Let the per-read iterable skip this read rather
    # than collapsing its signal to a constant expected level.
    if n < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DegenerateQuantileNormalizationError(
            "Quantile regression requires at least two finite values per input."
        )

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    centered_x = x - mean_x
    centered_y = y - mean_y
    sum_sq_x = np.dot(centered_x, centered_x)
    sum_sq_y = np.dot(centered_y, centered_y)

    eps = np.finfo(np.float64).eps
    tolerance_x = eps * n * max(float(np.max(np.abs(x))) ** 2, 1.0)
    tolerance_y = eps * n * max(float(np.max(np.abs(y))) ** 2, 1.0)
    if sum_sq_x <= tolerance_x:
        raise DegenerateQuantileNormalizationError(
            "Observed signal quantiles have insufficient variation."
        )
    if sum_sq_y <= tolerance_y:
        raise DegenerateQuantileNormalizationError(
            "Expected k-mer signal quantiles have insufficient variation."
        )

    covariance = np.dot(centered_x, centered_y)
    m = covariance / sum_sq_x
    b = mean_y - m * mean_x
    r = covariance / np.sqrt(sum_sq_x * sum_sq_y)

    if not np.all(np.isfinite((m, b, r))):
        raise DegenerateQuantileNormalizationError(
            "Quantile regression produced a non-finite fit."
        )

    return float(m), float(b), float(np.clip(r, -1.0, 1.0))


def quantile_normalize(A, B, n_quantiles=1000):
    """Affine-calibrate observed signal quantiles to expected-current quantiles.

    Parameters
    ----------
    A, B : array-like
        Finite observed signal and expected k-mer levels. Inputs are flattened
        only for quantile fitting; calibrated ``A`` retains its original shape.
    n_quantiles : int, default=1000
        Number of evenly spaced quantile levels used for the fit.

    Returns
    -------
    tuple
        ``(normalized_A, scale, shift, correlation)`` from an affine regression
        of expected quantiles on observed quantiles. The method follows the
        quantile-scaling approach used by ONT Dorado's ModBaseScaler.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if (
        A.size == 0
        or B.size == 0
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(B))
    ):
        raise DegenerateQuantileNormalizationError(
            "Quantile normalization requires non-empty, finite observed and expected signals."
        )

    quants_vec = np.linspace(0, 1.0, n_quantiles)
    

    q_A = quantiles(A.reshape(-1), quants_vec)
    q_B = quantiles(B.reshape(-1), quants_vec)
    
    m, c, r = linear_regression(q_A, q_B)

    A_normalized = A * m + c
    
    return A_normalized, m, c, r
