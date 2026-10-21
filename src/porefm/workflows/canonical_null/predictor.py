"""Train canonical references and calibrate embedding deviation scores.

Canonical-null models predict the contextual embedding expected for an
unmodified base from the backbone's fixed sequence representation. This module
also learns per-token score location/scale, fits base-specific whitening, and
stores empirical per-k-mer null distributions for p-value lookup.
"""

from porefm.preprocessing import (
    attach_preprocessing, preprocessing_from_checkpoint, resolve_preprocessing,
    validate_artifact_preprocessing,
)
from porefm.cli.preprocessing_args import resolve_cli_preprocessing
from porefm.cli.chunking_args import resolve_chunking_mode
import sys

from porefm.cli.canonical_null_args import build_canonical_null_parser, parse_canonical_null_args


if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    parse_canonical_null_args()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
import random, time, os, datetime, csv, math
import copy
import json

from collections import defaultdict
from functools import partial

from sklearn.metrics import precision_recall_fscore_support
from dataclasses import dataclass, asdict

from contextlib import nullcontext

import torch.distributed as dist
from porefm.data.chunk_boundaries import validate_chunking_mode
import numpy as np
import pysam

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)

from porefm.models.backbone.model import ONTBackbone, ONTBackboneConfig
from porefm.models.backbone.self_attention import SelfAttnBlock
from porefm.models.backbone.utils import LM_Head, strip_module_prefix
from porefm.models.checkpointing import (
    get_pretraining_state_dict_from_checkpoint,
    modular_canonical_null_payload,
    split_pretraining_state_dict,
)
from porefm.models.configuration import get_backbone_config_from_checkpoint
from porefm.workflows.canonical_null.rna_structure import (
    RNA_STRUCTURE_FEATURE_DIM,
    RNA_STRUCTURE_SPAN,
    append_rna_structure_features,
    require_rnalocal_extension,
)
from porefm.workflows.canonical_null.whitening import (
    DEFAULT_EMBEDDING_NORM_EPS,
    DEFAULT_SCORE_LOG_EPS,
    RAW_COSINE_SCORE_TYPE,
    RAW_WHITENING_SCORE_TYPE,
    BaseSpecificWhiteningScorer,
    PerBaseScoreReservoir,
    PerBaseTensorReservoir,
    StreamingCovariance,
    canonical_base_groups,
    canonical_kmer_contexts,
    canonical_cosine_scores,
    canonical_cosine_scores_from_normalized,
    canonical_embedding_residuals,
    canonical_score_log_target,
    centered_kmer_indices,
    fit_context_whitening_from_accumulators,
    normalize_score_type,
    save_whitening_artifact,
    standardized_score_type,
    validate_centered_kmer_size,
)
from porefm.data.inferred_chunks import has_inferred_boundaries, inferred_chunks, slice_base_values
from porefm.data.bam import Read_Lite, build_lookup_table, get_read_info_lite, passes_min_qscore
from porefm.data.reference_match import (
    build_md_exact_match_kmer_mask,
    validate_match_kmer_size,
)
from porefm.data.reader import (
    BamIterableDataset,
    collate_modal_batch,
    format_iterable_bam_log,
    get_percentiles,
    iter_file_cycles,
    normalize_read_signal_with_stats,
    pad_read_components,
)
from porefm.common.chunking import resolve_downstream_step
from porefm.common.utils import DegenerateQuantileNormalizationError, align_to_power_of_two, estimate_per_sample_memory, get_dummy_loss, get_warmup_cosine_schedule
from porefm.common.tensor_masks import ignore_edges

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

CANONICAL_SCALE_FEATURE_SCHEMA_VERSION = 5
READ_POSITION_FEATURE_DIM = 7
CANONICAL_SCALE_TOKEN_FEATURE_DIM_NO_QUALITY = 12
CANONICAL_SCALE_GLOBAL_METADATA_DIM_NO_QUALITY = 8
DEFAULT_SCALE_KMER_EMBEDDING_DIM = 64
DEFAULT_SCALE_CONTEXT_EMBEDDING_DIM = 32
DEFAULT_SCALE_CONTEXT_CHANNELS = 128
DEFAULT_SCALE_CONTEXT_KERNEL_SIZE = 5
DEFAULT_SCALE_CONTEXT_DILATIONS = (1, 1, 2, 4, 8, 16)
DEFAULT_SCALE_CONTEXT_EXPANSION = 4
DEFAULT_SCALE_CONTEXT_DROPOUT = 0.05
DEFAULT_SCALE_H_NULL_FEATURE_DIM = 128

@dataclass
class CanonicalNullConfig:
    """Architecture settings for the canonical embedding prediction network.

    ``external_hidden_dim`` is the width of backbone inputs and predicted
    embeddings; ``hidden_dim`` is the predictor's internal width. Attention
    fields follow the backbone self-attention configuration contract.
    """

    hidden_dim: int
    external_hidden_dim: int
    num_heads: int
    kv_num_heads: int
    num_layers: int
    dim_feedforward: int
    base_seq_window: tuple 
    self_attn_temp: float
    dropout: float
    self_attn_alibi_half_life_min: float = None
    self_attn_alibi_half_life_max: float = None
    # Disabled when loading older checkpoints; no backbone parameters change.
    use_sequence_context: bool = False
    base_seq_vocab: int = 6
    context_embedding_dim: int = 32
    context_channels: int = 128
    context_kernel_size: int = 5
    context_dilations: tuple = (1, 1, 2, 4, 8, 16)
    context_expansion: int = 4
    context_dropout: float = 0.05
    use_rna_structure_features: bool = False

    def __post_init__(self):
        if self.use_rna_structure_features and not self.use_sequence_context:
            raise ValueError('Null RNA structure features require --null_sequence_context')

    @property
    def context_radius(self):
        """CNN radius plus input folding span, used for chunk overlap only.

        Structure enters before the CNN, so paired-energy context can extend
        beyond its sequence radius. Exterior delta-f3 can depend on more
        distant bases; this overlap is not a full-read equivalence guarantee.
        """
        cnn_radius = (self.context_kernel_size // 2) * sum(self.context_dilations) if self.use_sequence_context else 0
        return cnn_radius + (RNA_STRUCTURE_SPAN if self.use_rna_structure_features else 0)

@dataclass
class CanonicalScaleConfig:
    """Feature schema and bounds for the canonical score scale model.

    ``fixed_base_dim``, ``token_feature_dim``, and ``global_feature_dim`` must
    match tensors constructed by :func:`build_scale_model_inputs`. The model
    predicts Gaussian log-score location and clamped log standard deviation.
    ``kmer_embedding_dim`` is deliberately independent of the backbone width:
    it controls a categorical lookup rather than a backbone representation.
    """

    hidden_dim: int
    fixed_base_dim: int
    token_feature_dim: int
    global_feature_dim: int
    kmer_size: int = 9
    kmer_embedding_dim: int = DEFAULT_SCALE_KMER_EMBEDDING_DIM
    dropout: float = 0.0
    eps: float = 1e-6
    log_sigma_min: float = -4.0
    log_sigma_max: float = 3.0
    use_base_quality_features: bool = False
    feature_schema_version: int = CANONICAL_SCALE_FEATURE_SCHEMA_VERSION
    score_type: str = RAW_COSINE_SCORE_TYPE

    # Schema-v3 sequence-context conditioning. Schema-v2 checkpoints are
    # loaded with both conditioning branches disabled.
    base_seq_vocab: int = 6
    use_sequence_context: bool = True
    context_embedding_dim: int = DEFAULT_SCALE_CONTEXT_EMBEDDING_DIM
    context_channels: int = DEFAULT_SCALE_CONTEXT_CHANNELS
    context_kernel_size: int = DEFAULT_SCALE_CONTEXT_KERNEL_SIZE
    context_dilations: tuple = DEFAULT_SCALE_CONTEXT_DILATIONS
    context_expansion: int = DEFAULT_SCALE_CONTEXT_EXPANSION
    context_dropout: float = DEFAULT_SCALE_CONTEXT_DROPOUT
    use_h_null_features: bool = True
    h_null_feature_dim: int = DEFAULT_SCALE_H_NULL_FEATURE_DIM
    use_read_position_features: bool = False
    # Explicit opt-in; existing checkpoints keep sequence-only CNN inputs.
    use_context_quality: bool = False
    # Schema v5 appends the prototype's 109 RNALfold channels after all
    # existing token features. Missing fields in old checkpoints stay off.
    use_rna_structure_features: bool = False

    def __post_init__(self):
        if self.use_context_quality and not (self.use_base_quality_features and self.use_sequence_context):
            raise ValueError("Context quality requires base-quality features and the sequence-context CNN")

    @property
    def context_radius(self):
        """CNN radius or RNA pair span used to estimate chunk overlap.

        RNALfold's exterior delta-f3 can depend on more distant structures;
        this overlap does not guarantee equality with full-read folding.
        """
        cnn_radius = ((self.context_kernel_size // 2) * sum(self.context_dilations)
                      if self.use_sequence_context else 0)
        return max(cnn_radius, RNA_STRUCTURE_SPAN if self.use_rna_structure_features else 0)

class CanonicalNullPredictionNetwork(nn.Module):
    """Predict canonical contextual embeddings from fixed base embeddings.

    A projection, alternating residual convolutions/self-attention blocks, and
    an MLP head preserve base length while mapping ``external_hidden_dim``
    inputs back to the same external embedding width.
    """

    def __init__(self, config):
        super().__init__()
        self.mask_padding = config.use_sequence_context
                    
        if config.external_hidden_dim == config.hidden_dim:
            self.project = nn.Identity()
        else:
            self.project = nn.Linear(config.external_hidden_dim, config.hidden_dim, bias=False)
        
        kernel_size = 7
        pad = kernel_size // 2
        
        self.cnn0 = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=True),
            nn.GELU(),
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GELU(),
        )
        
        self.attn1 = SelfAttnBlock(config, query_modality='seq')
        
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=True),
            nn.GELU(),
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GELU(),
        )
        
        self.attn2 = SelfAttnBlock(config, query_modality='seq')
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=True),
            nn.GELU(),
            nn.Conv1d(in_channels=config.hidden_dim, out_channels=config.hidden_dim, kernel_size=kernel_size, padding=pad, bias=False),
            nn.GELU(),
        )
                
        self.final_norm_base_seq = nn.LayerNorm(config.hidden_dim)
        
        self.mean_emb_head = LM_Head(config.hidden_dim, config.hidden_dim, config.external_hidden_dim)     

    def forward(self, base_seq_in, pad_base_seq):
        """Predict per-base null means.

        ``base_seq_in`` has shape ``(B, L, external_hidden_dim)`` and
        ``pad_base_seq`` is a boolean ``(B, L)`` validity mask. Returns aliases
        ``mean_emb`` and ``h_null`` for a ``(B, L, external_hidden_dim)`` tensor.
        """
        base_seq = self.project(base_seq_in)
        
        base_seq = base_seq + self._convolve(self.cnn0, base_seq, pad_base_seq)
        
        base_seq = self.attn1(base_seq, pad_base_seq)
        base_seq = base_seq + self._convolve(self.cnn1, base_seq, pad_base_seq)
        
        base_seq = self.attn2(base_seq, pad_base_seq)
        base_seq = self._convolve(self.cnn2, base_seq, pad_base_seq)

        base_seq = self.final_norm_base_seq(base_seq)
        
        mean_emb=self.mean_emb_head(base_seq)
        if self.mask_padding:
            mean_emb = mean_emb * pad_base_seq.unsqueeze(-1)
        
        return {'mean_emb': mean_emb, 'h_null': mean_emb}

    def _convolve(self, block, x, mask):
        """For new models, suppress padded bias activations between convolutions."""
        x = x.transpose(1, 2)
        if not self.mask_padding:
            return block(x).transpose(1, 2)
        mask = mask.unsqueeze(1).to(x.dtype)
        x = x * mask
        for layer in block:
            x = layer(x) * mask
        return x.transpose(1, 2)
    
class CanonicalNullPredictor(nn.Module):
    """Thin wrapper exposing the canonical-null prediction network.

    ``embedding``, ``prenorm_embedding``, and ``delta`` are retained in the
    constructor for legacy call compatibility; current behavior always predicts
    a full embedding from ``fixed_sa_base``.
    """

    def __init__(self, config, embedding=True, prenorm_embedding=False, delta=False):
        super().__init__()
        self.config=config
        
        self.embedding_predictor = CanonicalNullPredictionNetwork(self.config)
        if config.use_sequence_context:
            self.sequence_context_encoder = CanonicalScaleSequenceContextEncoder(
                config, use_structure=config.use_rna_structure_features,
            )
            self.context_projection = nn.Linear(config.context_channels + 2, config.external_hidden_dim, bias=False)
            self.context_gate = nn.Parameter(torch.full((config.external_hidden_dim,), 0.01))
        
    def forward(self, fixed_sa_base, pad_base_seq, base_seq_input=None, rna_structure_features=None):
        """Predict null embeddings, optionally conditioning on raw base IDs.

        New checkpoints fuse an independent masked sequence CNN and two
        capped chunk-context distances into the fixed representation. Base
        IDs are required for this branch. Structure-enabled checkpoints also
        require worker-precomputed float32 (B,L,109) RNALfold channels, fused
        with base embeddings before the CNN. No observed signal embedding is
        supplied to this branch. Older checkpoints retain their computation.
        """
        if self.config.use_sequence_context:
            if base_seq_input is None:
                raise ValueError("The sequence-conditioned null predictor requires base_seq_input")
            context = self.sequence_context_encoder(
                base_seq_input, pad_base_seq, rna_structure_features=rna_structure_features,
            )
            availability = chunk_context_features(pad_base_seq, self.config.context_radius)
            delta = self.context_projection(torch.cat((context, availability.to(context.dtype)), dim=-1))
            fixed_sa_base = (fixed_sa_base + self.context_gate * delta) * pad_base_seq.unsqueeze(-1)
        
        embedding_pred = self.embedding_predictor(fixed_sa_base, pad_base_seq)
        
        return embedding_pred


class MaskedDepthwiseSeparableConv1dBlock(nn.Module):
    """Residual 1-D context block that cannot propagate padded activations.

    The depthwise convolution learns offset-specific sequence filters, while
    the pointwise expansion and contraction mix channels so the stack can
    represent interactions between bases at different offsets. Inputs and
    outputs use ``(batch, bases, channels)`` layout.
    """

    def __init__(self, channels, kernel_size, dilation, expansion, dropout):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        expanded_channels = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pointwise_expand = nn.Linear(channels, expanded_channels)
        self.pointwise_contract = nn.Linear(expanded_channels, channels)
        self.dropout = nn.Dropout(dropout)
        # A small initial residual contribution keeps the new context tower
        # stable while its scale-specific representation is learned.
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, x, pad_base_seq):
        """Apply one masked residual context block."""
        mask = pad_base_seq.unsqueeze(-1).to(x.dtype)
        residual = x
        y = self.norm(x) * mask
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        y = self.pointwise_expand(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.pointwise_contract(y)
        y = self.dropout(y)
        return (residual + self.layer_scale * y) * mask


class CanonicalScaleSequenceContextEncoder(nn.Module):
    """Encode raw bases and optional quality/structure with a masked CNN.

    The encoder has a symmetric receptive field whose learned convolution
    weights remain directional. With the default kernel and dilations, each
    output can depend on bases at offsets ``-64..+64``. Token zero is padding;
    nonzero canonical/unknown tokens follow the backbone vocabulary. Optional
    quality is concatenated to the base embedding before channel projection,
    so every convolution can learn joint sequence/quality context. Null models
    can instead append RNALfold channels here. Scale structure features retain
    their existing direct-token route, independent of this encoder option.
    """

    def __init__(self, config, use_quality=False, use_structure=False):
        super().__init__()
        self.use_quality = bool(use_quality)
        self.use_structure = bool(use_structure)
        self.embedding = nn.Embedding(
            config.base_seq_vocab,
            config.context_embedding_dim,
            padding_idx=0,
        )
        self.input_projection = nn.Linear(
            config.context_embedding_dim + int(self.use_quality) + RNA_STRUCTURE_FEATURE_DIM * int(self.use_structure),
            config.context_channels,
        )
        self.blocks = nn.ModuleList(
            [
                MaskedDepthwiseSeparableConv1dBlock(
                    channels=config.context_channels,
                    kernel_size=config.context_kernel_size,
                    dilation=dilation,
                    expansion=config.context_expansion,
                    dropout=config.context_dropout,
                )
                for dilation in config.context_dilations
            ]
        )
        self.final_norm = nn.LayerNorm(config.context_channels)

    def forward(self, base_seq_input, pad_base_seq, scaled_quality=None, rna_structure_features=None):
        """Return masked ``(B,L,C)`` context from base IDs and optional Q/40.

        ``scaled_quality`` must have shape ``(B,L,1)`` when enabled and use
        the same clipped [0,2] Phred/40 units as the direct token input.
        Padding is removed before mixing, including nonfinite padded values.
        Optional structure has shape (B,L,109), dtype float32, and the native
        energy/mask units; it is not re-folded or inferred from hidden states.
        """
        mask = pad_base_seq.unsqueeze(-1)
        x = self.embedding(base_seq_input.long())
        if self.use_quality:
            if scaled_quality is None or scaled_quality.shape != (*base_seq_input.shape, 1):
                raise ValueError("Quality CNN requires scaled_quality with shape (B,L,1)")
            quality = torch.where(mask, scaled_quality.to(x.dtype), 0.0)
            x = torch.cat((x, quality), dim=-1)
        if self.use_structure:
            if (rna_structure_features is None
                    or rna_structure_features.shape != (*base_seq_input.shape, RNA_STRUCTURE_FEATURE_DIM)
                    or rna_structure_features.dtype != torch.float32):
                raise ValueError('Structure null CNN requires float32 RNA structure features with shape (B,L,109) from batch preparation')
            structure = torch.where(mask, rna_structure_features.to(device=x.device, dtype=x.dtype), 0.0)
            x = torch.cat((x, structure), dim=-1)
        x = self.input_projection(x)
        x = x * mask.to(x.dtype)
        for block in self.blocks:
            x = block(x, pad_base_seq)
        return self.final_norm(x) * mask.to(x.dtype)


class CanonicalNullScaleModel(nn.Module):
    """Predict per-token Gaussian location and scale for log anomaly scores.

    Fixed base embeddings, centered k-mer embeddings, raw-sequence CNN context,
    projected canonical-null predictions, engineered token features, and
    read-global features are fused with FiLM conditioning. Optional RNALfold
    energies/masks/delta-f3 are appended to the engineered token features.
    Invalid k-mer IDs share a padding embedding. Outputs ``m``, ``log_sigma``, and ``sigma`` each
    have shape ``(B, L)``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_kmers = 4 ** config.kmer_size
        self.invalid_kmer_id = self.num_kmers
        if int(config.kmer_embedding_dim) <= 0:
            raise ValueError("kmer_embedding_dim must be positive")

        self.kmer_embedding = nn.Embedding(
            self.num_kmers + 1,
            config.kmer_embedding_dim,
            padding_idx=self.invalid_kmer_id,
        )

        self.sequence_context_encoder = (
            CanonicalScaleSequenceContextEncoder(config, use_quality=config.use_context_quality)
            if config.use_sequence_context
            else None
        )
        self.h_null_encoder = (
            nn.Sequential(
                nn.LayerNorm(config.fixed_base_dim),
                nn.Linear(config.fixed_base_dim, config.h_null_feature_dim),
                nn.GELU(),
            )
            if config.use_h_null_features
            else None
        )

        token_in_dim = (
            config.fixed_base_dim
            + config.kmer_embedding_dim
            + config.token_feature_dim
        )
        if config.use_sequence_context:
            token_in_dim += config.context_channels
        if config.use_h_null_features:
            token_in_dim += config.h_null_feature_dim
        self.token_encoder = nn.Sequential(
            nn.Linear(token_in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(config.global_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        self.token_norm = nn.LayerNorm(config.hidden_dim)
        self.film = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(
        self,
        fixed_sa_base,
        kmer_ids,
        token_features,
        global_features,
        pad_base_seq=None,
        base_seq_input=None,
        h_null=None,
    ):
        """Predict log-score distribution parameters for each base token.

        Token inputs have leading shape ``(B, L)``; ``global_features`` has
        shape ``(B, global_feature_dim)``. Schema-v3 models also require raw
        ``base_seq_input`` and/or ``h_null`` according to their configuration.
        Both backbone-derived tensors are detached here so scale-model loss can
        never update the pretrained backbone or null predictor. When supplied,
        ``pad_base_seq`` zeros ``m`` and ``log_sigma`` at invalid positions
        (so ``sigma`` is one there).

        The entire scale network runs in float32, matching scale training and
        calibration even when its caller uses autocast for backbone inference.
        Parameters must retain their normal float32 storage dtype. This local
        boundary preserves the caller's autocast and global matmul policies.
        """
        # Casting inputs alone cannot stop autocast from rounding intermediate
        # convolutions/linear layers before the final distribution parameters.
        with torch.autocast(device_type=fixed_sa_base.device.type, enabled=False):
            return self._forward_float32(
                fixed_sa_base, kmer_ids, token_features, global_features,
                pad_base_seq=pad_base_seq, base_seq_input=base_seq_input,
                h_null=h_null,
            )

    def _forward_float32(
        self, fixed_sa_base, kmer_ids, token_features, global_features,
        pad_base_seq=None, base_seq_input=None, h_null=None,
    ):
        """Evaluate scale features and heads inside ``forward``'s FP32 boundary."""
        fixed_sa_base = fixed_sa_base.detach().float()
        token_features = token_features.float()
        global_features = global_features.float()
        if token_features.shape[-1] != self.config.token_feature_dim:
            raise ValueError(
                f"Scale token feature width mismatch: expected {self.config.token_feature_dim}, "
                f"got {token_features.shape[-1]}."
            )
        if global_features.shape[-1] != self.config.global_feature_dim:
            raise ValueError(
                f"Scale global feature width mismatch: expected {self.config.global_feature_dim}, "
                f"got {global_features.shape[-1]}."
            )

        safe_kmer_ids = torch.where(
            (kmer_ids >= 0) & (kmer_ids < self.num_kmers),
            kmer_ids,
            torch.full_like(kmer_ids, self.invalid_kmer_id),
        )
        kmer_emb = self.kmer_embedding(safe_kmer_ids.long())

        h_null_feature = None
        if self.h_null_encoder is not None:
            if h_null is None:
                raise ValueError("Scale model requires h_null when use_h_null_features=True")
            if h_null.shape != fixed_sa_base.shape:
                raise ValueError(
                    f"h_null shape {tuple(h_null.shape)} does not match fixed_sa_base "
                    f"shape {tuple(fixed_sa_base.shape)}"
                )
            h_null_feature = self.h_null_encoder(h_null.detach().float())

        sequence_context_feature = None
        if self.sequence_context_encoder is not None:
            if base_seq_input is None:
                raise ValueError(
                    "Scale model requires base_seq_input when use_sequence_context=True"
                )
            if pad_base_seq is None:
                raise ValueError(
                    "Scale model requires pad_base_seq when use_sequence_context=True"
                )
            if base_seq_input.shape != fixed_sa_base.shape[:2]:
                raise ValueError(
                    f"base_seq_input shape {tuple(base_seq_input.shape)} does not match "
                    f"token shape {tuple(fixed_sa_base.shape[:2])}"
                )
            sequence_context_feature = self.sequence_context_encoder(
                base_seq_input,
                pad_base_seq.bool(),
                # Quality follows the 12 legacy position/Fourier channels,
                # before optional full-read-position channels. Reuse the
                # already normalized input across train/calibrate/inference.
                scaled_quality=(token_features[..., CANONICAL_SCALE_TOKEN_FEATURE_DIM_NO_QUALITY:
                                               CANONICAL_SCALE_TOKEN_FEATURE_DIM_NO_QUALITY + 1]
                                if self.config.use_context_quality else None),
            )

        if h_null_feature is not None and sequence_context_feature is not None:
            token_x = torch.cat(
                (
                    fixed_sa_base,
                    h_null_feature,
                    kmer_emb,
                    sequence_context_feature,
                    token_features,
                ),
                dim=-1,
            )
        elif h_null_feature is not None:
            token_x = torch.cat(
                (fixed_sa_base, h_null_feature, kmer_emb, token_features),
                dim=-1,
            )
        elif sequence_context_feature is not None:
            token_x = torch.cat(
                (fixed_sa_base, kmer_emb, sequence_context_feature, token_features),
                dim=-1,
            )
        else:
            token_x = torch.cat(
                (fixed_sa_base, kmer_emb, token_features),
                dim=-1,
            )
        t = self.token_encoder(token_x)

        g = self.global_encoder(global_features)
        gamma, beta = self.film(g).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * beta

        t = self.token_norm(t)
        t = t * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        g_expanded = g[:, None, :].expand(-1, t.shape[1], -1)
        pred = self.head(torch.cat([t, g_expanded], dim=-1))
        m, log_sigma = pred.chunk(2, dim=-1)

        m = m.squeeze(-1)
        log_sigma = log_sigma.squeeze(-1).clamp(
            min=self.config.log_sigma_min,
            max=self.config.log_sigma_max,
        )

        if pad_base_seq is not None:
            m = torch.where(pad_base_seq, m, torch.zeros_like(m))
            log_sigma = torch.where(pad_base_seq, log_sigma, torch.zeros_like(log_sigma))

        return {
            'm': m,
            'log_sigma': log_sigma,
            'sigma': torch.exp(log_sigma),
        }

class OntModelForCanonicalNull(nn.Module):
    """Compose a frozen backbone, null predictor, and optional score calibrators.

    Backbone execution is always under ``no_grad``. The returned dictionary
    extends backbone embeddings with the predicted null, raw cosine distance,
    optional standardized score, and optional base-specific whitening scores.
    """

    def __init__(self, backbone, null_predictor, scale_model=None, whitening_scorer=None):
        super().__init__()
        self.backbone = backbone
        self.null_predictor = null_predictor
        self.scale_model = scale_model
        self.whitening_scorer = whitening_scorer
        self.config = null_predictor.config
        self.backbone_config = backbone.config

    def forward(self, base_seq_input, ts_input, percentiles_input=None,
                pad_base_seq=None, pad_ts=None, embeddings_only=True,
                base_quality=None, global_features=None, read_positions=None, read_lengths=None,
                rna_structure_features=None):
        """Compute canonical-null outputs for one padded multimodal batch.

        Base IDs and masks have shape ``(B, L_base)``; signal inputs/masks use
        ``(B, L_signal, channels)`` and ``(B, L_signal)``. Missing masks are
        inferred. ``base_quality`` and ``global_features`` are required only by
        scale models configured for those features. ``embeddings_only`` is a
        compatibility argument; the backbone is always requested in embedding
        mode. Read-position scale models also require (B,L) full-read
        positions and (B,) lengths, in raw/polished sequence coordinates.
        Structure-enabled null or scale heads require precomputed (B,L,109) float32
        ``rna_structure_features`` from batch preparation.
        """
        if pad_base_seq is None:
            pad_base_seq = base_seq_input != 0
        if pad_ts is None:
            pad_ts = torch.ones(
                ts_input.shape[:2],
                dtype=torch.bool,
                device=ts_input.device,
            )

        with torch.no_grad():
            backbone_out = self.backbone(
                base_seq_input,
                ts_input,
                percentiles_input=percentiles_input,
                pad_base_seq=pad_base_seq,
                pad_ts=pad_ts,
                embeddings_only=True,
            )

        null_out = self.null_predictor(backbone_out['fixed_sa_base'], pad_base_seq, base_seq_input=base_seq_input,
                                       rna_structure_features=rna_structure_features)
        h_null = null_out['mean_emb']
        out = dict(backbone_out)
        out['h_null'] = h_null
        h_obs_norm, h_null_norm, residual = canonical_null_residuals(
            backbone_out['contextual_base_seq'],
            h_null,
        )
        raw_score = canonical_cosine_scores_from_normalized(h_obs_norm, h_null_norm)
        out['canonical_null_score_raw'] = raw_score
        out['s_cos'] = raw_score
        out['canonical_null_score'] = raw_score

        scale_score_type = (
            normalize_score_type(getattr(self.scale_model.config, 'score_type', RAW_COSINE_SCORE_TYPE))
            if self.scale_model is not None
            else None
        )

        if self.scale_model is not None and scale_score_type == RAW_COSINE_SCORE_TYPE:
            scale_inputs = build_scale_model_inputs(
                self.backbone,
                base_seq_input,
                pad_base_seq,
                pad_ts,
                self.scale_model.config.kmer_size,
                base_quality=base_quality,
                read_global_features=global_features,
                use_base_quality_features=self.scale_model.config.use_base_quality_features,
                scale_config=self.scale_model.config,
                read_positions=read_positions, read_lengths=read_lengths,
                rna_structure_features=rna_structure_features,
            )
            scale_out = self.scale_model(
                backbone_out['fixed_sa_base'],
                scale_inputs['kmer_ids'],
                scale_inputs['token_features'],
                scale_inputs['global_features'],
                pad_base_seq=pad_base_seq,
                base_seq_input=base_seq_input,
                h_null=h_null,
            )
            standardized = standardize_canonical_scores(
                raw_score,
                scale_out,
                eps=self.scale_model.config.eps,
                score_type=scale_score_type,
                hidden_dim=self.backbone.config.hidden_dim,
            )
            out.update(scale_out)
            out['canonical_null_score_std'] = standardized
            out['z_cos'] = standardized
            out['canonical_null_score'] = standardized

        if self.whitening_scorer is not None:
            whitening_score = self.whitening_scorer.score(
                residual,
                base_seq_input,
                valid_mask=pad_base_seq.bool(),
            )
            out['s_whiten'] = whitening_score
            if self.scale_model is not None and scale_score_type == RAW_WHITENING_SCORE_TYPE:
                scale_inputs = build_scale_model_inputs(
                    self.backbone,
                    base_seq_input,
                    pad_base_seq,
                    pad_ts,
                    self.scale_model.config.kmer_size,
                    base_quality=base_quality,
                    read_global_features=global_features,
                    use_base_quality_features=self.scale_model.config.use_base_quality_features,
                    scale_config=self.scale_model.config,
                    read_positions=read_positions, read_lengths=read_lengths,
                    rna_structure_features=rna_structure_features,
                )
                scale_out = self.scale_model(
                    backbone_out['fixed_sa_base'],
                    scale_inputs['kmer_ids'],
                    scale_inputs['token_features'],
                    scale_inputs['global_features'],
                    pad_base_seq=pad_base_seq,
                    base_seq_input=base_seq_input,
                    h_null=h_null,
                )
                standardized = standardize_canonical_scores(
                    whitening_score,
                    scale_out,
                    eps=self.scale_model.config.eps,
                    score_type=scale_score_type,
                    hidden_dim=self.backbone.config.hidden_dim,
                )
                out['whiten_m'] = scale_out['m']
                out['whiten_log_sigma'] = scale_out['log_sigma']
                out['whiten_sigma'] = scale_out['sigma']
                out['z_whiten'] = standardized
        return out
    
def canonical_null_residuals(contextual_base_seq, h_null, norm_eps=DEFAULT_EMBEDDING_NORM_EPS):
    """Return normalized observed/null embeddings and their FP32 residual."""
    return canonical_embedding_residuals(
        contextual_base_seq,
        h_null,
        norm_eps=norm_eps,
    )

def canonical_null_scores(contextual_base_seq, h_null, norm_eps=DEFAULT_EMBEDDING_NORM_EPS):
    """Return per-base cosine distance between contextual and null embeddings."""
    return canonical_cosine_scores(
        contextual_base_seq,
        h_null,
        norm_eps=norm_eps,
    )

def scaled_phred_from_error_prob(error_prob):
    """Convert error probabilities to Phred scores divided by 40 and clipped to 2."""
    error_prob = torch.clamp(error_prob.float(), min=1e-12, max=1.0)
    phred = -10.0 * torch.log(error_prob) / math.log(10.0)
    return torch.clamp(phred / 40.0, min=0.0, max=2.0)

def build_scale_config(
    pretrain_config,
    kmer_size=9,
    kmer_embedding_dim=DEFAULT_SCALE_KMER_EMBEDDING_DIM,
    dropout=0.0,
    use_base_quality_features=False,
    score_type=RAW_COSINE_SCORE_TYPE,
    eps=None,
    use_sequence_context=True,
    context_embedding_dim=DEFAULT_SCALE_CONTEXT_EMBEDDING_DIM,
    context_channels=DEFAULT_SCALE_CONTEXT_CHANNELS,
    context_kernel_size=DEFAULT_SCALE_CONTEXT_KERNEL_SIZE,
    context_dilations=DEFAULT_SCALE_CONTEXT_DILATIONS,
    context_expansion=DEFAULT_SCALE_CONTEXT_EXPANSION,
    context_dropout=DEFAULT_SCALE_CONTEXT_DROPOUT,
    use_h_null_features=True,
    h_null_feature_dim=DEFAULT_SCALE_H_NULL_FEATURE_DIM,
    use_read_position_features=False,
    use_context_quality=False,
    use_rna_structure_features=False,
):
    """Derive a scale-model feature schema from a backbone configuration.

    Whitening targets default to ``DEFAULT_SCORE_LOG_EPS``; cosine targets use
    ``1e-6``. Quality-enabled schemas add one token and one global metadata
    feature. ``use_context_quality`` additionally routes that token quality
    through the sequence CNN; it does not add another token feature. RNA
    structure conditioning appends 109 channels in the prototype's layout.
    """
    base_embed_dim = pretrain_config.hidden_dim // 8
    token_feature_dim = (CANONICAL_SCALE_TOKEN_FEATURE_DIM_NO_QUALITY + int(use_base_quality_features)
                         + READ_POSITION_FEATURE_DIM * int(use_read_position_features)
                         + RNA_STRUCTURE_FEATURE_DIM * int(use_rna_structure_features))
    global_metadata_dim = CANONICAL_SCALE_GLOBAL_METADATA_DIM_NO_QUALITY + int(use_base_quality_features)
    score_type = normalize_score_type(score_type)
    if eps is None:
        eps = DEFAULT_SCORE_LOG_EPS if score_type == RAW_WHITENING_SCORE_TYPE else 1e-6
    return CanonicalScaleConfig(
        hidden_dim=pretrain_config.hidden_dim,
        fixed_base_dim=pretrain_config.hidden_dim,
        token_feature_dim=token_feature_dim,
        global_feature_dim=global_metadata_dim + base_embed_dim,
        kmer_size=kmer_size,
        kmer_embedding_dim=kmer_embedding_dim,
        dropout=dropout,
        eps=float(eps),
        use_base_quality_features=use_base_quality_features,
        feature_schema_version=CANONICAL_SCALE_FEATURE_SCHEMA_VERSION,
        score_type=score_type,
        base_seq_vocab=pretrain_config.base_seq_vocab,
        use_sequence_context=use_sequence_context,
        context_embedding_dim=context_embedding_dim,
        context_channels=context_channels,
        context_kernel_size=context_kernel_size,
        context_dilations=tuple(context_dilations),
        context_expansion=context_expansion,
        context_dropout=context_dropout,
        use_h_null_features=use_h_null_features,
        h_null_feature_dim=h_null_feature_dim,
        use_read_position_features=use_read_position_features,
        use_context_quality=use_context_quality,
        use_rna_structure_features=use_rna_structure_features,
    )


def build_scale_config_from_args(pretrain_config, args, score_type):
    """Build the requested scale configuration, restricting RNA folding to RNA."""
    if getattr(args, 'scale_rna_structure_features', False) and args.seq_type != 'rna':
        raise ValueError('--scale_rna_structure_features requires --seq_type rna')
    return build_scale_config(
        pretrain_config,
        use_read_position_features=getattr(args, 'scale_read_position_features', False),
        use_context_quality=getattr(args, 'scale_context_quality', False),
        use_rna_structure_features=getattr(args, 'scale_rna_structure_features', False),
        kmer_size=resolved_scale_kmer_size(args),
        kmer_embedding_dim=getattr(
            args,
            'scale_kmer_embedding_dim',
            DEFAULT_SCALE_KMER_EMBEDDING_DIM,
        ),
        dropout=args.scale_dropout,
        use_base_quality_features=getattr(
            args,
            'use_base_quality_scale_features',
            False,
        ),
        score_type=score_type,
        eps=(
            args.score_log_eps
            if score_type == RAW_WHITENING_SCORE_TYPE
            else None
        ),
        use_sequence_context=not getattr(
            args,
            'disable_scale_sequence_context',
            False,
        ),
        context_embedding_dim=getattr(
            args,
            'scale_context_embedding_dim',
            DEFAULT_SCALE_CONTEXT_EMBEDDING_DIM,
        ),
        context_channels=getattr(
            args,
            'scale_context_channels',
            DEFAULT_SCALE_CONTEXT_CHANNELS,
        ),
        context_kernel_size=getattr(
            args,
            'scale_context_kernel_size',
            DEFAULT_SCALE_CONTEXT_KERNEL_SIZE,
        ),
        context_dilations=getattr(
            args,
            'scale_context_dilations',
            DEFAULT_SCALE_CONTEXT_DILATIONS,
        ),
        context_expansion=getattr(
            args,
            'scale_context_expansion',
            DEFAULT_SCALE_CONTEXT_EXPANSION,
        ),
        context_dropout=getattr(
            args,
            'scale_context_dropout',
            DEFAULT_SCALE_CONTEXT_DROPOUT,
        ),
        use_h_null_features=not getattr(
            args,
            'disable_scale_h_null_features',
            False,
        ),
        h_null_feature_dim=getattr(
            args,
            'scale_h_null_feature_dim',
            DEFAULT_SCALE_H_NULL_FEATURE_DIM,
        ),
    )


def validate_rna_structure_request(args, scale_config=None, null_config=None):
    """Validate RNA-only opt-ins and the stored null/scale feature settings.

    Calibration/evaluation may omit the flag and use the checkpoint setting,
    but an explicit opt-in must not silently apply a head lacking features.
    Changing a null head's input channels requires training a fresh null model.
    """
    requested = bool(getattr(args, 'scale_rna_structure_features', False))
    enabled = scale_config is not None and scale_config.use_rna_structure_features
    if (requested or enabled) and args.seq_type != 'rna':
        raise ValueError('--scale_rna_structure_features requires --seq_type rna')
    applying = args.mode == 'calibrate' or (args.mode == 'train-scale' and getattr(args, 'eval_only', False))
    if scale_config is not None and requested and applying and not enabled:
        raise ValueError('The loaded scale checkpoint does not contain RNA structure features; train a new scaler first')
    if enabled:
        require_rnalocal_extension()
    requested_null = bool(getattr(args, 'null_rna_structure_features', False))
    enabled_null = bool(getattr(null_config, 'use_rna_structure_features', False))
    if (requested_null or enabled_null) and args.seq_type != 'rna':
        raise ValueError('--null_rna_structure_features requires --seq_type rna')
    if null_config is not None and requested_null and not enabled_null:
        raise ValueError('The loaded null checkpoint does not contain RNA structure features; train a new null from a pretrained backbone first')
    if enabled_null:
        require_rnalocal_extension()


def requires_rna_structure_features(null_predictor=None, scale_model=None):
    """Whether either checkpoint needs the single shared batch feature tensor."""
    return any(model is not None and bool(getattr(unwrap_model(model).config, 'use_rna_structure_features', False))
               for model in (null_predictor, scale_model))


def scale_context_radius(scale_model_or_config):
    """Return a null/scale configuration's context radius in bases."""
    config = getattr(scale_model_or_config, 'config', scale_model_or_config)
    return int(getattr(config, 'context_radius', 0))


def required_scale_signal_overlap(pretrain_config, scale_model_or_config=None):
    """Estimate overlap from a null/scale head's sequence/structure radius.

    Adjacent chunks need twice the symmetric base-context radius so every
    internal base is fully represented by at least one chunk. The backbone's
    expected signal/base ratio converts that span into signal-step units.
    For structure this is a local-context heuristic; exterior delta-f3 can
    depend on farther sequence and is not bounded by the pairing span.
    """
    if scale_model_or_config is None:
        return 0
    base_overlap = 2 * scale_context_radius(scale_model_or_config)
    expected_ratio = float(getattr(pretrain_config, 'expected_ratio', 1.0))
    if not math.isfinite(expected_ratio) or expected_ratio <= 0.0:
        raise ValueError(
            f"pretrain expected_ratio must be finite and positive, got {expected_ratio}"
        )
    return int(math.ceil(base_overlap * expected_ratio))


def scale_context_config_matches(left, right):
    """Compare effective scale architecture/features, not serialization versions.

    Schemas v4/v5 add optional read-position and RNA structure channels; with
    those disabled a matching v3 model remains resumable, including its optimizer state. Input
    dimensions and feature flags still prevent reuse across real changes.
    """
    if left.feature_schema_version not in {2, 3, 4, 5} or right.feature_schema_version not in {2, 3, 4, 5}:
        return False
    fields = (
        'hidden_dim',
        'fixed_base_dim',
        'token_feature_dim',
        'global_feature_dim',
        'kmer_size',
        'use_base_quality_features',
        'use_read_position_features',
        'use_context_quality',
        'use_rna_structure_features',
        'kmer_embedding_dim',
        'base_seq_vocab',
        'use_sequence_context',
        'context_embedding_dim',
        'context_channels',
        'context_kernel_size',
        'context_dilations',
        'context_expansion',
        'context_dropout',
        'use_h_null_features',
        'h_null_feature_dim',
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def canonical_scale_config_from_dict(scale_config_dict):
    """Load a scale config dictionary with legacy architecture fallbacks.

    Schema-v2 checkpoints retain their original local-only architecture. New
    schema-v3/v4/v5 checkpoints serialize all sequence-context and null-feature
    settings so their tensor shapes and receptive field are unambiguous.
    Checkpoints predating ``kmer_embedding_dim`` used ``hidden_dim`` for that
    lookup, so the missing field must resolve to the checkpoint's full width.
    """
    scale_config_dict = dict(scale_config_dict)
    if 'feature_schema_version' not in scale_config_dict:
        scale_config_dict['feature_schema_version'] = 1
    if 'use_base_quality_features' not in scale_config_dict:
        scale_config_dict['use_base_quality_features'] = int(scale_config_dict.get('token_feature_dim', 0)) >= 14
    if 'score_type' not in scale_config_dict:
        scale_config_dict['score_type'] = RAW_COSINE_SCORE_TYPE
    if 'kmer_embedding_dim' not in scale_config_dict:
        scale_config_dict['kmer_embedding_dim'] = int(scale_config_dict['hidden_dim'])
    if int(scale_config_dict['feature_schema_version']) < 3:
        scale_config_dict.setdefault('use_sequence_context', False)
        scale_config_dict.setdefault('use_h_null_features', False)
    if int(scale_config_dict['feature_schema_version']) < 4:
        scale_config_dict.setdefault('context_dilations', (1, 2, 4, 8, 16))
    if 'context_dilations' in scale_config_dict:
        scale_config_dict['context_dilations'] = tuple(
            scale_config_dict['context_dilations']
        )
    scale_config_dict['score_type'] = normalize_score_type(scale_config_dict['score_type'])
    return CanonicalScaleConfig(**scale_config_dict)

def expected_scale_feature_dims(hidden_dim, use_base_quality_features, use_read_position_features=False,
                                use_rna_structure_features=False):
    """Return expected ``(token_feature_dim, global_feature_dim)`` for a backbone."""
    token_feature_dim = (CANONICAL_SCALE_TOKEN_FEATURE_DIM_NO_QUALITY + int(use_base_quality_features)
                         + READ_POSITION_FEATURE_DIM * int(use_read_position_features)
                         + RNA_STRUCTURE_FEATURE_DIM * int(use_rna_structure_features))
    global_metadata_dim = CANONICAL_SCALE_GLOBAL_METADATA_DIM_NO_QUALITY + int(use_base_quality_features)
    global_feature_dim = global_metadata_dim + hidden_dim // 8
    return token_feature_dim, global_feature_dim

def validate_null_config_for_backbone(null_config, pretrain_config, context):
    """Require the null predictor's external embedding width to match the backbone."""
    expected_external_dim = int(pretrain_config.hidden_dim)
    if int(null_config.external_hidden_dim) != expected_external_dim:
        raise ValueError(
            f"{context} is incompatible with the backbone: "
            f"null_config.external_hidden_dim={null_config.external_hidden_dim}, "
            f"but backbone hidden_dim={expected_external_dim}."
        )

def validate_scale_config_for_backbone(scale_config, pretrain_config, context,
                                       expected_use_base_quality_features=None,
                                       expected_score_type=None):
    """Validate scale feature schema, optional quality mode, and score family.

    All incompatibilities are collected and raised together as ``ValueError``
    with ``context`` identifying the checkpoint or current run.
    """
    errors = []
    schema_version = int(getattr(scale_config, 'feature_schema_version', 1))
    if schema_version not in {2, 3, 4, CANONICAL_SCALE_FEATURE_SCHEMA_VERSION}:
        errors.append(
            f"feature_schema_version={schema_version}, supported versions are "
            f"2, 3, 4 and {CANONICAL_SCALE_FEATURE_SCHEMA_VERSION}; legacy schema v1 kept "
            "duplicate pos_norm and left_norm scalar channels"
        )

    if expected_use_base_quality_features is not None:
        actual_quality = bool(getattr(scale_config, 'use_base_quality_features', False))
        expected_quality = bool(expected_use_base_quality_features)
        if actual_quality != expected_quality:
            errors.append(
                f"use_base_quality_features={actual_quality}, but current run requested "
                f"use_base_quality_scale_features={expected_quality}"
            )

    if expected_score_type is not None:
        actual_score_type = normalize_score_type(getattr(scale_config, 'score_type', RAW_COSINE_SCORE_TYPE))
        expected_score_type = normalize_score_type(expected_score_type)
        if actual_score_type != expected_score_type:
            errors.append(
                f"score_type={actual_score_type}, but current run expected {expected_score_type}"
            )

    expected_fixed_base_dim = int(pretrain_config.hidden_dim)
    if int(scale_config.fixed_base_dim) != expected_fixed_base_dim:
        errors.append(
            f"fixed_base_dim={scale_config.fixed_base_dim}, expected backbone hidden_dim={expected_fixed_base_dim}"
        )

    expected_token_dim, expected_global_dim = expected_scale_feature_dims(
        expected_fixed_base_dim,
        bool(getattr(scale_config, 'use_base_quality_features', False)),
        scale_config.use_read_position_features,
        scale_config.use_rna_structure_features,
    )
    if int(scale_config.token_feature_dim) != expected_token_dim:
        errors.append(
            f"token_feature_dim={scale_config.token_feature_dim}, expected {expected_token_dim}"
        )
    if int(scale_config.global_feature_dim) != expected_global_dim:
        errors.append(
            f"global_feature_dim={scale_config.global_feature_dim}, expected {expected_global_dim}"
        )
    if int(scale_config.kmer_embedding_dim) <= 0:
        errors.append("kmer_embedding_dim must be positive")

    if schema_version >= 3:
        if int(scale_config.base_seq_vocab) != int(pretrain_config.base_seq_vocab):
            errors.append(
                f"base_seq_vocab={scale_config.base_seq_vocab}, expected "
                f"backbone base_seq_vocab={pretrain_config.base_seq_vocab}"
            )
        if bool(scale_config.use_sequence_context):
            if int(scale_config.context_embedding_dim) <= 0:
                errors.append("context_embedding_dim must be positive")
            if int(scale_config.context_channels) <= 0:
                errors.append("context_channels must be positive")
            if int(scale_config.context_kernel_size) <= 0 or int(scale_config.context_kernel_size) % 2 == 0:
                errors.append("context_kernel_size must be a positive odd integer")
            if not scale_config.context_dilations or any(
                int(dilation) <= 0 for dilation in scale_config.context_dilations
            ):
                errors.append("context_dilations must contain positive integers")
            if int(scale_config.context_expansion) <= 0:
                errors.append("context_expansion must be positive")
            if not 0.0 <= float(scale_config.context_dropout) < 1.0:
                errors.append("context_dropout must be in [0, 1)")
        if bool(scale_config.use_h_null_features) and int(scale_config.h_null_feature_dim) <= 0:
            errors.append("h_null_feature_dim must be positive")

    if errors:
        raise ValueError(f"{context} has an incompatible scale feature schema:\n - " + "\n - ".join(errors))

def normalize_read_global_features(read_global_features, batch_size, device, feature_dim=6):
    """Create a finite fixed-width batch of read metadata features.

    Missing input becomes zeros, short rows are right-padded, long rows are
    truncated, and NaN/Inf values become zero.
    """
    if read_global_features is None:
        return torch.zeros(batch_size, feature_dim, dtype=torch.float32, device=device)

    read_global_features = read_global_features.to(device=device, dtype=torch.float32)
    if read_global_features.ndim == 1:
        read_global_features = read_global_features.unsqueeze(0)

    if read_global_features.shape[0] != batch_size:
        raise ValueError(
            f"read_global_features batch mismatch: expected {batch_size}, "
            f"got {read_global_features.shape[0]}"
        )

    if read_global_features.shape[1] < feature_dim:
        pad = torch.zeros(
            batch_size,
            feature_dim - read_global_features.shape[1],
            dtype=read_global_features.dtype,
            device=device,
        )
        read_global_features = torch.cat([read_global_features, pad], dim=-1)
    elif read_global_features.shape[1] > feature_dim:
        read_global_features = read_global_features[:, :feature_dim]

    return torch.nan_to_num(read_global_features, nan=0.0, posinf=0.0, neginf=0.0)

def average_base_seq_embedding(pretrained_model, base_seq_input, pad_base_seq):
    """Mean-pool the backbone's fixed base embedding over valid tokens."""
    backbone = unwrap_model(pretrained_model)
    with torch.no_grad():
        base_emb = backbone.base_seq_projector.base_seq_embed(base_seq_input).float()
        mask = pad_base_seq.unsqueeze(-1).to(base_emb.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (base_emb * mask).sum(dim=1) / denom

def chunk_context_features(pad_base_seq, radius):
    """Return available left/right chunk bases, capped at R and divided by R.

    Masks must be prefix-valid (right padded), as produced by the canonical
    loaders. Values are absolute-base distances in fixed CNN-radius units,
    not fractions of variable chunk length. R=0 produces zeros.
    """
    idx = torch.arange(pad_base_seq.shape[1], device=pad_base_seq.device).float()[None, :]
    right = (pad_base_seq.sum(1, keepdim=True) - 1 - idx).clamp_min(0)
    distances = torch.stack((idx.expand_as(right), right), dim=-1)
    return distances.clamp(max=radius) / max(radius, 1) * pad_base_seq.unsqueeze(-1)


def read_position_features(pad_base_seq, read_positions, read_lengths, radius):
    """Build seven boundary/read-position inputs in model sequence order.

    ``read_positions`` is (B,L), zero-based in the full raw or polished read;
    ``read_lengths`` is (B,), including bases absent from retained chunks.
    Channels are capped left/right chunk context / R, log1p distances to
    full-read ends / log(1001), fractional read position, and two indicators
    that the chunk reaches the corresponding true read end. Read-position
    transforms are not genomic coordinates and are not inferred from labels.
    """
    if read_positions is None or read_lengths is None:
        raise ValueError("Read-position scale features require full-read positions and lengths")
    if read_positions.shape != pad_base_seq.shape or read_lengths.shape != (pad_base_seq.shape[0],):
        raise ValueError("Read-position metadata must have shapes (B,L) and (B,)")
    pos = read_positions.to(device=pad_base_seq.device, dtype=torch.float32)
    length = read_lengths.to(device=pad_base_seq.device, dtype=torch.float32)[:, None]
    left = pos.clamp_min(0)
    right = (length - 1 - pos).clamp_min(0)
    true_left = ((pos == 0) & pad_base_seq).any(1, keepdim=True).expand_as(pos)
    true_right = ((pos == length - 1) & pad_base_seq).any(1, keepdim=True).expand_as(pos)
    features = torch.cat((chunk_context_features(pad_base_seq, radius), torch.stack((
        torch.log1p(left) / math.log(1001), torch.log1p(right) / math.log(1001),
        left / (length - 1).clamp_min(1), true_left.float(), true_right.float(),
    ), dim=-1)), dim=-1)
    return torch.where(pad_base_seq.unsqueeze(-1), features, 0.0)


def build_token_position_features(pad_base_seq, base_quality=None, use_base_quality_features=False,
                                  use_read_position_features=False, read_positions=None,
                                  read_lengths=None, context_radius=0):
    """Build normalized edge, Fourier-position, and optional quality features.

    Returns ``(B, L, token_feature_dim)`` with padded positions set to zero.
    ``base_quality`` contains error probabilities, converted to scaled Phred.
    """
    device = pad_base_seq.device
    batch_size, seq_len = pad_base_seq.shape
    idx = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    base_lens = pad_base_seq.sum(dim=1, keepdim=True).float()
    denom = (base_lens - 1.0).clamp_min(1.0)

    left_norm = idx / denom
    right_norm = (base_lens - 1.0 - idx).clamp_min(0.0) / denom
    edge_dist = torch.minimum(idx, (base_lens - 1.0 - idx).clamp_min(0.0))
    edge_norm = edge_dist / denom
    edge_log_norm = torch.log1p(edge_dist) / torch.log1p(denom)

    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device).view(1, 1, -1)
    phase = 2.0 * math.pi * left_norm.unsqueeze(-1) * freqs
    pos_fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

    feature_parts = [
        left_norm.unsqueeze(-1),
        right_norm.unsqueeze(-1),
        edge_norm.unsqueeze(-1),
        edge_log_norm.unsqueeze(-1),
        pos_fourier,
    ]
    if use_base_quality_features:
        if base_quality is None:
            quality_feature = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=device)
        else:
            quality_feature = scaled_phred_from_error_prob(base_quality.to(device)).unsqueeze(-1)
        feature_parts.append(quality_feature)
    if use_read_position_features:
        feature_parts.append(read_position_features(pad_base_seq, read_positions, read_lengths, context_radius))

    token_features = torch.cat(feature_parts, dim=-1)
    token_features = torch.where(
        pad_base_seq.unsqueeze(-1),
        token_features,
        torch.zeros_like(token_features),
    )
    return token_features

def build_global_scale_features(pretrained_model, base_seq_input, pad_base_seq, pad_ts, read_global_features=None,
                                use_base_quality_features=False):
    """Combine length ratios, read metadata, and mean fixed-base embeddings.

    The result is a finite ``(B, global_feature_dim)`` tensor. Read metadata is
    normalized to five fields, or six when median quality is enabled.
    """
    device = base_seq_input.device
    batch_size = base_seq_input.shape[0]
    base_lens = pad_base_seq.sum(dim=1).float().clamp_min(1.0)
    signal_lens = pad_ts.sum(dim=1).float().clamp_min(1.0)
    max_base_len = float(max(base_seq_input.shape[1], 1))
    max_signal_len = float(max(pad_ts.shape[1], 1))

    derived_global = torch.stack(
        [
            torch.log(signal_lens / base_lens),
            base_lens / max_base_len,
            signal_lens / max_signal_len,
        ],
        dim=-1,
    )

    read_metadata_dim = 5 + int(use_base_quality_features)
    read_meta = normalize_read_global_features(read_global_features, batch_size, device, feature_dim=read_metadata_dim)
    base_embed_mean = average_base_seq_embedding(pretrained_model, base_seq_input, pad_base_seq)
    global_features = torch.cat([derived_global, read_meta, base_embed_mean], dim=-1)
    return torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

def build_scale_model_inputs(pretrained_model, base_seq_input, pad_base_seq, pad_ts, kmer_size,
                             base_quality=None, read_global_features=None, use_base_quality_features=False,
                             scale_config=None, read_positions=None, read_lengths=None,
                             rna_structure_features=None):
    """Construct centered k-mer, token, and read-global scale-model features.

    RNA-enabled checkpoints append precomputed (B,L,109) float32 folding
    channels after position/quality/read-position features. Batch preparation
    computes them on CPU; this builder only transfers them to the device.
    Missing or incorrectly shaped features are errors for enabled checkpoints,
    so folding cannot silently move back into the GPU execution path.
    """
    kmer_ids, kmer_valid = compute_chunk_kmer_ids(base_seq_input, kmer_size)
    token_features = build_token_position_features(
        pad_base_seq,
        base_quality=base_quality,
        use_base_quality_features=use_base_quality_features,
        use_read_position_features=scale_config.use_read_position_features if scale_config is not None else False,
        context_radius=scale_config.context_radius if scale_config is not None else 0,
        read_positions=read_positions,
        read_lengths=read_lengths,
    )
    if scale_config is not None and scale_config.use_rna_structure_features:
        if rna_structure_features is None:
            raise ValueError('Structure-enabled scale requires RNA structure features from batch preparation')
        if (rna_structure_features.shape != (*base_seq_input.shape, RNA_STRUCTURE_FEATURE_DIM)
                or rna_structure_features.dtype != torch.float32):
            raise ValueError('RNA structure features must be float32 with shape (B,L,109)')
        token_features = torch.cat((
            token_features, rna_structure_features.to(base_seq_input.device, non_blocking=True),
        ), dim=-1)
    global_features = build_global_scale_features(
        pretrained_model,
        base_seq_input,
        pad_base_seq,
        pad_ts,
        read_global_features=read_global_features,
        use_base_quality_features=use_base_quality_features,
    )
    return {
        'kmer_ids': kmer_ids,
        'kmer_valid': kmer_valid,
        'token_features': token_features,
        'global_features': global_features,
    }

def standardize_canonical_scores(scores, scale_out, eps=1e-6, score_type=RAW_COSINE_SCORE_TYPE, hidden_dim=None):
    """Convert raw scores to predicted Gaussian z-scores in log-score space."""
    y = canonical_score_log_target(
        scores,
        score_type=score_type,
        hidden_dim=hidden_dim,
        eps=eps,
    )
    sigma = torch.exp(scale_out['log_sigma'].float())
    return (y - scale_out['m'].float()) / sigma

def get_canonical_scale_loss(scores, scale_out, trimmed_pad_base_seq, eps=1e-6,
                             score_type=RAW_COSINE_SCORE_TYPE, hidden_dim=None):
    """Compute Gaussian NLL for log scores over valid trimmed base positions.

    Non-finite score targets are excluded before indexing. Non-finite model
    predictions at otherwise valid targets raise immediately so one bad batch
    cannot silently poison the optimizer. Returns total loss, mean squared
    standardized residual, mean predicted log scale, and the scalar integer
    tensor count of targets actually used. Empty masks return differentiable
    zeros with a zero count.
    """
    y = canonical_score_log_target(
        scores,
        score_type=score_type,
        hidden_dim=hidden_dim,
        eps=eps,
    )
    m = scale_out['m'].float()
    log_sigma = scale_out['log_sigma'].float()
    valid_targets = trimmed_pad_base_seq.bool() & torch.isfinite(y)
    valid_target_count = valid_targets.sum()

    if not valid_targets.any():
        # Keep a differentiable zero even if an irrelevant padded prediction
        # is non-finite; no target-bearing position is being optimized here.
        zero = torch.nan_to_num(m).sum() * 0.0
        return zero, zero.detach(), zero.detach(), valid_target_count

    valid_y = y[valid_targets]
    valid_m = m[valid_targets]
    valid_log_sigma = log_sigma[valid_targets]
    if not torch.isfinite(valid_m).all() or not torch.isfinite(valid_log_sigma).all():
        raise FloatingPointError(
            "Canonical scale model produced non-finite location/scale predictions "
            "at valid score targets. Do not resume a checkpoint containing NaN weights."
        )

    valid_residual = (valid_y - valid_m) / torch.exp(valid_log_sigma)
    if not torch.isfinite(valid_residual).all():
        raise FloatingPointError("Canonical scale standardized residuals became non-finite")
    loss = (valid_log_sigma + 0.5 * valid_residual.pow(2)).mean()
    return loss, valid_residual.pow(2).mean().detach(), valid_log_sigma.mean().detach(), valid_target_count

def get_canonical_null_loss(pred, target, trimmed_pad_base_seq):
    """Combine cosine and weighted Smooth-L1 loss for null embeddings.

    The loss is ``cosine_distance + 0.02 * smooth_l1`` over valid positions and
    returns all three scalar tensors. Targets are detached from the backbone.
    """
    mean = pred['mean_emb']
    target = target.detach()

    valid_mean = mean[trimmed_pad_base_seq].float()
    valid_target = target[trimmed_pad_base_seq].float()

    if valid_mean.numel() == 0:
        zero = mean.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    loss_cos = 1.0 - (
        F.normalize(valid_mean, dim=-1) * F.normalize(valid_target, dim=-1)
    ).sum(dim=-1).mean()
    loss_mse = F.smooth_l1_loss(valid_mean, valid_target)
    loss = loss_cos + 0.02 * loss_mse
    return loss, loss_cos.detach(), loss_mse.detach()

def compute_chunk_kmer_ids(base_seq_input, kmer_size):
    """Encode every valid centered canonical k-mer as a base-4 integer.

    Base IDs must use 1--4 for A/C/G/T(U). Invalid or edge positions receive
    ID ``-1`` and a false validity mask. ``kmer_size`` must be positive and
    odd, including the supported one-mer case.
    """
    return centered_kmer_indices(base_seq_input, kmer_size)

def build_tail_enriched_quantile_levels(body_count=201, tail_count=200, tail_min_prob=1e-4):
    """Build sorted unique CDF levels dense in the upper one-percent tail."""
    body_count = max(int(body_count), 2)
    tail_count = max(int(tail_count), 0)
    tail_min_prob = min(max(float(tail_min_prob), 1e-8), 1e-2)

    body = torch.linspace(0.0, 0.99, body_count, dtype=torch.float32)
    if tail_count > 0:
        tail_survival = torch.logspace(
            math.log10(1e-2),
            math.log10(tail_min_prob),
            tail_count,
            dtype=torch.float32,
        )
        tail = 1.0 - tail_survival
        levels = torch.cat([body, tail, torch.ones(1, dtype=torch.float32)])
    else:
        levels = torch.cat([body, torch.ones(1, dtype=torch.float32)])
    return torch.unique(torch.clamp(levels, 0.0, 1.0))

class CanonicalCalibrationAccumulator:
    """GPU accumulator for per-k-mer canonical score distributions.

    ``reservoir`` and ``quantile`` modes retain a bounded uniform sample per
    base-4 k-mer; ``histogram`` stores fixed-bin counts plus under/overflow.
    Quantile mode compresses the final reservoir onto tail-enriched CDF levels.
    The accumulator is intentionally CUDA-only because its dense state and
    updates are designed for calibration during GPU inference.
    """

    def __init__(
        self,
        kmer_size=9,
        max_scores_per_kmer=20000,
        seed=0,
        distribution_type='reservoir',
        hist_min=-8.0,
        hist_max=8.0,
        hist_bins=256,
        quantile_body_count=201,
        quantile_tail_count=200,
        quantile_tail_min_prob=1e-4,
        device=None,
    ):
        if distribution_type in {'reservoir', 'quantile'} and int(max_scores_per_kmer) <= 0:
            raise ValueError(
                f"max_scores_per_kmer must be positive for distribution_type={distribution_type}"
            )
        if distribution_type not in {'reservoir', 'histogram', 'quantile'}:
            raise ValueError(f"Unsupported calibration distribution_type: {distribution_type}")

        self.kmer_size = kmer_size
        self.num_kmers = 4 ** kmer_size
        self.max_scores_per_kmer = int(max_scores_per_kmer)
        self.distribution_type = distribution_type
        self.device = torch.device(device) if device is not None else torch.device('cuda')
        if self.device.type != 'cuda':
            raise RuntimeError("Canonical calibration accumulation is GPU-only; run calibration on CUDA.")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        self.seen_counts = torch.zeros(self.num_kmers, dtype=torch.long, device=self.device)

        if distribution_type == 'histogram':
            if hist_bins <= 0:
                raise ValueError("hist_bins must be positive")
            if not hist_max > hist_min:
                raise ValueError("hist_max must be greater than hist_min")
            self.hist_min = float(hist_min)
            self.hist_max = float(hist_max)
            self.hist_bins = int(hist_bins)
            self.hist_width = (self.hist_max - self.hist_min) / self.hist_bins
            self.hist_counts = torch.zeros(
                (self.num_kmers, self.hist_bins),
                dtype=torch.long,
                device=self.device,
            )
            self.hist_underflow_counts = torch.zeros(self.num_kmers, dtype=torch.long, device=self.device)
            self.hist_overflow_counts = torch.zeros(self.num_kmers, dtype=torch.long, device=self.device)
            self.reservoir_scores = None
        else:
            self.hist_min = None
            self.hist_max = None
            self.hist_bins = None
            self.hist_width = None
            self.hist_counts = None
            self.hist_underflow_counts = None
            self.hist_overflow_counts = None
            self.reservoir_scores = torch.full(
                (self.num_kmers, self.max_scores_per_kmer),
                float('inf'),
                dtype=torch.float32,
                device=self.device,
            )

        if distribution_type == 'quantile':
            self.quantile_levels = build_tail_enriched_quantile_levels(
                body_count=quantile_body_count,
                tail_count=quantile_tail_count,
                tail_min_prob=quantile_tail_min_prob,
            ).to(self.device)
        else:
            self.quantile_levels = None

    def update(self, kmer_ids, scores):
        """Accumulate finite scores whose integer k-mer IDs are in range."""
        if kmer_ids.numel() == 0:
            return

        ids = kmer_ids.detach().to(device=self.device, dtype=torch.long, non_blocking=True).reshape(-1)
        values = scores.detach().to(device=self.device, dtype=torch.float32, non_blocking=True).reshape(-1)
        valid = (ids >= 0) & (ids < self.num_kmers) & torch.isfinite(values)
        if not valid.any():
            return

        ids = ids[valid]
        values = values[valid]

        if self.distribution_type == 'histogram':
            self._update_histogram(ids, values)
            return

        self._update_reservoir(ids, values)

    def _update_reservoir(self, ids, values):
        if ids.numel() == 0:
            return

        order = torch.argsort(ids, stable=True)
        ids = ids[order]
        values = values[order]
        n = ids.numel()

        new_group = torch.ones(n, dtype=torch.bool, device=self.device)
        if n > 1:
            new_group[1:] = ids[1:] != ids[:-1]
        group_starts = torch.nonzero(new_group, as_tuple=False).flatten()
        group_ids = torch.cumsum(new_group.to(torch.long), dim=0) - 1
        ranks = torch.arange(n, device=self.device, dtype=torch.long) - group_starts[group_ids]

        previous_seen = self.seen_counts[ids]
        stream_indices = previous_seen + ranks + 1

        group_ends = torch.cat(
            [
                group_starts[1:],
                torch.tensor([n], device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        unique_ids = ids[group_starts]
        group_counts = group_ends - group_starts
        self.seen_counts[unique_ids] += group_counts

        slots = torch.empty(n, dtype=torch.long, device=self.device)
        fill = stream_indices <= self.max_scores_per_kmer
        slots[fill] = stream_indices[fill] - 1

        replace_candidates = ~fill
        if replace_candidates.any():
            candidate_stream_indices = stream_indices[replace_candidates]
            random_slots = torch.floor(
                torch.rand(
                    candidate_stream_indices.shape,
                    device=self.device,
                    dtype=torch.float64,
                    generator=self.generator,
                )
                * candidate_stream_indices.to(torch.float64)
            ).to(torch.long)
            slots[replace_candidates] = random_slots

        keep = slots < self.max_scores_per_kmer
        if not keep.any():
            return

        linear_slots = ids[keep] * self.max_scores_per_kmer + slots[keep]
        kept_values = values[keep]

        slot_order = torch.argsort(linear_slots, stable=True)
        linear_slots = linear_slots[slot_order]
        kept_values = kept_values[slot_order]

        last_for_slot = torch.ones(linear_slots.numel(), dtype=torch.bool, device=self.device)
        if linear_slots.numel() > 1:
            last_for_slot[:-1] = linear_slots[:-1] != linear_slots[1:]

        self.reservoir_scores.view(-1)[linear_slots[last_for_slot]] = kept_values[last_for_slot]

    def _update_histogram(self, ids, values):
        ones = torch.ones_like(ids, dtype=torch.long)
        self.seen_counts.scatter_add_(0, ids, ones)

        under = values < self.hist_min
        if under.any():
            self.hist_underflow_counts.scatter_add_(0, ids[under], torch.ones_like(ids[under], dtype=torch.long))

        over = values >= self.hist_max
        if over.any():
            self.hist_overflow_counts.scatter_add_(0, ids[over], torch.ones_like(ids[over], dtype=torch.long))

        in_range = ~(under | over)
        if in_range.any():
            bins = torch.floor((values[in_range] - self.hist_min) / self.hist_width).to(torch.long)
            bins = bins.clamp(0, self.hist_bins - 1)
            flat_bins = ids[in_range] * self.hist_bins + bins
            self.hist_counts.view(-1).scatter_add_(
                0,
                flat_bins,
                torch.ones_like(flat_bins, dtype=torch.long),
            )

    def _stored_counts(self):
        return torch.minimum(
            self.seen_counts,
            torch.full_like(self.seen_counts, self.max_scores_per_kmer),
        )

    def finalize(self, metadata=None):
        """Materialize a CPU-serializable calibration payload.

        Reservoir rows are sorted and truncated to their retained count;
        quantile rows are sampled at configured CDF levels. ``metadata`` is
        copied into the payload unchanged.
        """
        if self.distribution_type == 'histogram':
            stored_counts = self.hist_counts.long().sum(dim=1) + self.hist_underflow_counts + self.hist_overflow_counts
            return {
                'format': 'canonical_null_calibration_v1',
                'distribution_type': 'histogram',
                'kmer_size': self.kmer_size,
                'hist_counts': self.hist_counts.cpu(),
                'hist_min': self.hist_min,
                'hist_max': self.hist_max,
                'hist_bins': self.hist_bins,
                'hist_underflow_counts': self.hist_underflow_counts.cpu(),
                'hist_overflow_counts': self.hist_overflow_counts.cpu(),
                'stored_counts': stored_counts.cpu(),
                'seen_counts': self.seen_counts.cpu(),
                'metadata': metadata or {},
            }

        stored_counts = self._stored_counts()
        observed_ids = torch.nonzero(stored_counts > 0, as_tuple=False).flatten()
        sorted_scores = {}
        quantile_scores = {}

        chunk_keys = 512
        for start in range(0, observed_ids.numel(), chunk_keys):
            key_ids = observed_ids[start:start + chunk_keys]
            rows = self.reservoir_scores[key_ids]
            sorted_rows = torch.sort(rows, dim=1).values
            counts = stored_counts[key_ids]

            if self.distribution_type == 'quantile':
                idx = torch.round(self.quantile_levels.unsqueeze(0) * (counts.unsqueeze(1).float() - 1.0)).long()
                idx = idx.clamp_min(0)
                q_rows = torch.gather(sorted_rows, 1, idx)
                q_rows = q_rows.cpu()
                for row_idx, kmer_id in enumerate(key_ids.cpu().tolist()):
                    quantile_scores[int(kmer_id)] = q_rows[row_idx].clone()
            else:
                sorted_rows = sorted_rows.cpu()
                counts_cpu = counts.cpu()
                for row_idx, kmer_id in enumerate(key_ids.cpu().tolist()):
                    count = int(counts_cpu[row_idx].item())
                    sorted_scores[int(kmer_id)] = sorted_rows[row_idx, :count].clone()

        payload = {
            'format': 'canonical_null_calibration_v1',
            'distribution_type': self.distribution_type,
            'kmer_size': self.kmer_size,
            'stored_counts': stored_counts.cpu(),
            'seen_counts': self.seen_counts.cpu(),
            'metadata': metadata or {},
        }

        if self.distribution_type == 'quantile':
            payload.update(
                {
                    'quantile_levels': self.quantile_levels.cpu(),
                    'quantile_scores_by_kmer_id': quantile_scores,
                }
            )
        else:
            payload['scores_by_kmer_id'] = sorted_scores

        return payload

def lookup_scores_table_pvalues(scores, kmer_ids, calibration):
    """Compute finite-sample upper-tail p-values from per-k-mer reservoirs.

    Returns p-values and calibration counts with the input score shape. Missing
    or invalid k-mers receive NaN and count zero. A prepared dense CUDA table is
    used when memory permits; otherwise lookup falls back to CPU dictionaries.
    """
    device = scores.device
    prepare_canonical_calibration(calibration, device)
    cache = calibration.get('_lookup_cache')
    if cache is not None and cache.get('distribution_type') == 'reservoir_dense':
        flat_scores = scores.detach().float().reshape(-1)
        flat_kmers = kmer_ids.detach().to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
        pvalues = torch.full(flat_scores.shape, float('nan'), dtype=torch.float32, device=device)
        counts = torch.zeros(flat_scores.shape, dtype=torch.long, device=device)

        score_table = cache['score_table']
        stored_counts = cache['stored_counts']
        num_kmers = score_table.shape[0]
        valid = torch.isfinite(flat_scores) & (flat_kmers >= 0) & (flat_kmers < num_kmers)
        if not valid.any():
            return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        selected_scores = flat_scores[valid_idx]
        selected_kmers = flat_kmers[valid_idx]
        selected_counts = stored_counts[selected_kmers]
        has_calibration = selected_counts > 0
        if not has_calibration.any():
            return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

        valid_idx = valid_idx[has_calibration]
        selected_scores = selected_scores[has_calibration]
        selected_kmers = selected_kmers[has_calibration]
        selected_counts = selected_counts[has_calibration]

        chunk_size = int(cache.get('lookup_chunk_size', 1024))
        for start in range(0, selected_scores.numel(), chunk_size):
            end = min(start + chunk_size, selected_scores.numel())
            rows = score_table[selected_kmers[start:end]]
            lower_bound = torch.searchsorted(rows, selected_scores[start:end, None], right=False).squeeze(1)
            lower_bound = torch.minimum(lower_bound, selected_counts[start:end])
            n_ge = selected_counts[start:end] - lower_bound
            pvalues[valid_idx[start:end]] = (1.0 + n_ge.float()) / (1.0 + selected_counts[start:end].float())
            counts[valid_idx[start:end]] = selected_counts[start:end]

        return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

    flat_scores = scores.detach().cpu().float().reshape(-1)
    flat_kmers = kmer_ids.detach().cpu().long().reshape(-1)
    pvalues = torch.full(flat_scores.shape, float('nan'), dtype=torch.float32)
    counts = torch.zeros(flat_scores.shape, dtype=torch.long)

    scores_by_kmer = calibration['scores_by_kmer_id']
    for kmer_id in flat_kmers[flat_kmers >= 0].unique().tolist():
        null_scores = scores_by_kmer.get(int(kmer_id))
        if null_scores is None:
            null_scores = scores_by_kmer.get(str(int(kmer_id)))
        if null_scores is None or null_scores.numel() == 0:
            continue

        null_scores = null_scores.cpu().float()
        mask = flat_kmers == int(kmer_id)
        selected = flat_scores[mask]
        lower_bound = torch.searchsorted(null_scores, selected, right=False)
        n_ge = null_scores.numel() - lower_bound
        pvalues[mask] = (1.0 + n_ge.float()) / (1.0 + float(null_scores.numel()))
        counts[mask] = null_scores.numel()

    return pvalues.reshape(scores.shape).to(device), counts.reshape(scores.shape).to(device)

def lookup_histogram_pvalues(scores, kmer_ids, calibration):
    """Estimate upper-tail p-values from per-k-mer histogram calibrations.

    Counts within the query's bin are linearly interpolated; explicit overflow
    counts handle scores at or above the stored range.
    """
    device = scores.device
    prepare_canonical_calibration(calibration, device)
    cache = calibration['_lookup_cache']

    flat_scores = scores.detach().float().reshape(-1)
    flat_kmers = kmer_ids.detach().to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    pvalues = torch.full(flat_scores.shape, float('nan'), dtype=torch.float32, device=device)
    counts = torch.zeros(flat_scores.shape, dtype=torch.long, device=device)

    suffix_counts = cache['suffix_counts']
    seen_counts = cache['seen_counts']
    overflow_counts = cache['overflow_counts']
    hist_min = float(calibration['hist_min'])
    hist_max = float(calibration['hist_max'])
    hist_bins = int(calibration['hist_bins'])
    hist_width = (hist_max - hist_min) / hist_bins
    num_kmers = suffix_counts.shape[0]

    valid = torch.isfinite(flat_scores) & (flat_kmers >= 0) & (flat_kmers < num_kmers)
    if not valid.any():
        return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    selected_scores = flat_scores[valid_idx]
    selected_kmers = flat_kmers[valid_idx]
    selected_seen = seen_counts[selected_kmers]
    has_calibration = selected_seen > 0
    if not has_calibration.any():
        return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

    valid_idx = valid_idx[has_calibration]
    selected_scores = selected_scores[has_calibration]
    selected_kmers = selected_kmers[has_calibration]
    selected_seen = selected_seen[has_calibration]
    selected_seen_float = selected_seen.float()
    selected_overflow = overflow_counts[selected_kmers]

    below = selected_scores < hist_min
    above = selected_scores >= hist_max
    inside = ~(below | above)
    n_ge = torch.empty_like(selected_scores, dtype=torch.float32)
    n_ge[below] = selected_seen_float[below]
    n_ge[above] = selected_overflow[above]

    if inside.any():
        inside_scores = selected_scores[inside]
        inside_kmers = selected_kmers[inside]
        bin_idx = torch.floor((inside_scores - hist_min) / hist_width).long().clamp(0, hist_bins - 1)
        bin_lo = hist_min + bin_idx.float() * hist_width
        bin_hi = bin_lo + hist_width
        within_frac = ((bin_hi - inside_scores) / hist_width).clamp(0.0, 1.0)

        current_plus_higher = suffix_counts[inside_kmers, bin_idx]
        has_next = bin_idx + 1 < hist_bins
        next_idx = (bin_idx + 1).clamp(max=hist_bins - 1)
        higher_bins = torch.zeros_like(current_plus_higher)
        higher_bins[has_next] = suffix_counts[inside_kmers[has_next], next_idx[has_next]]
        current_bin = current_plus_higher - higher_bins
        n_ge[inside] = selected_overflow[inside] + higher_bins + current_bin * within_frac

    pvalues[valid_idx] = (1.0 + n_ge) / (1.0 + selected_seen_float)
    counts[valid_idx] = selected_seen

    return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

def lookup_quantile_pvalues(scores, kmer_ids, calibration):
    """Interpolate per-k-mer empirical CDFs stored at quantile levels."""
    device = scores.device
    prepare_canonical_calibration(calibration, device)
    cache = calibration['_lookup_cache']

    flat_scores = scores.detach().float().reshape(-1)
    flat_kmers = kmer_ids.detach().to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    pvalues = torch.full(flat_scores.shape, float('nan'), dtype=torch.float32, device=device)
    counts = torch.zeros(flat_scores.shape, dtype=torch.long, device=device)

    quantile_scores = cache['quantile_scores']
    quantile_levels = cache['quantile_levels']
    effective_counts = cache['effective_counts']
    has_quantiles = cache['has_quantiles']
    num_kmers = quantile_scores.shape[0]

    valid = torch.isfinite(flat_scores) & (flat_kmers >= 0) & (flat_kmers < num_kmers)
    if not valid.any():
        return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    selected_scores = flat_scores[valid_idx]
    selected_kmers = flat_kmers[valid_idx]
    selected_counts = effective_counts[selected_kmers]
    has_calibration = (selected_counts > 0) & has_quantiles[selected_kmers]
    if not has_calibration.any():
        return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

    valid_idx = valid_idx[has_calibration]
    selected_scores = selected_scores[has_calibration]
    selected_kmers = selected_kmers[has_calibration]
    selected_counts = selected_counts[has_calibration]

    q_rows = quantile_scores[selected_kmers]
    right = torch.searchsorted(q_rows, selected_scores[:, None], right=False).squeeze(1)
    num_levels = quantile_scores.shape[1]

    cdf = torch.empty_like(selected_scores, dtype=torch.float32)
    below = right <= 0
    above = right >= num_levels
    middle = ~(below | above)
    cdf[below] = 0.0
    cdf[above] = 1.0
    if middle.any():
        hi_idx = right[middle]
        lo_idx = hi_idx - 1
        middle_rows = q_rows[middle]
        row_idx = torch.arange(middle_rows.shape[0], device=device)
        x0 = middle_rows[row_idx, lo_idx]
        x1 = middle_rows[row_idx, hi_idx]
        q0 = quantile_levels[lo_idx]
        q1 = quantile_levels[hi_idx]
        denom = (x1 - x0).clamp_min(1e-12)
        frac = ((selected_scores[middle] - x0) / denom).clamp(0.0, 1.0)
        cdf[middle] = q0 + frac * (q1 - q0)

    n_ge = (1.0 - cdf).clamp(0.0, 1.0) * selected_counts.float()
    pvalues[valid_idx] = (1.0 + n_ge) / (1.0 + selected_counts.float())
    counts[valid_idx] = selected_counts

    return pvalues.reshape(scores.shape), counts.reshape(scores.shape)

def lookup_canonical_pvalues(scores, kmer_ids, calibration):
    """Dispatch p-value lookup according to calibration ``distribution_type``."""
    distribution_type = calibration.get('distribution_type')
    if distribution_type == 'reservoir':
        return lookup_scores_table_pvalues(scores, kmer_ids, calibration)
    if distribution_type == 'histogram':
        return lookup_histogram_pvalues(scores, kmer_ids, calibration)
    if distribution_type == 'quantile':
        return lookup_quantile_pvalues(scores, kmer_ids, calibration)
    raise ValueError(f"Unsupported calibration distribution_type: {distribution_type}")

def unwrap_model(model):
    """Remove DDP and ``torch.compile`` wrappers when present."""
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "_orig_mod", model)

def _key_list_summary(keys, max_keys=20):
    keys = list(keys)
    preview = keys[:max_keys]
    suffix = "" if len(keys) <= max_keys else f"\n    ... and {len(keys) - max_keys} more"
    return "\n    " + "\n    ".join(preview) + suffix if preview else ""

def _allowed_by_prefix(key, prefixes):
    return any(key.startswith(prefix) for prefix in prefixes)

def checked_load_state_dict(module, state_dict, component_name, source_path,
                            allowed_missing_prefixes=(), allowed_unexpected_prefixes=()):
    """Load a state dict while enforcing all non-allowlisted key mismatches.

    Module prefixes are stripped first. Returns PyTorch's incompatibility
    record, or raises a detailed ``RuntimeError`` listing missing/unexpected
    keys outside the supplied prefixes.
    """
    state_dict = strip_module_prefix(state_dict)
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not _allowed_by_prefix(key, allowed_missing_prefixes)
    ]
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not _allowed_by_prefix(key, allowed_unexpected_prefixes)
    ]
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f"missing keys ({len(missing)}):{_key_list_summary(missing)}"
            )
        if unexpected:
            details.append(
                f"unexpected keys ({len(unexpected)}):{_key_list_summary(unexpected)}"
            )
        raise RuntimeError(
            f"State-dict mismatch while loading {component_name} from {source_path}:\n"
            + "\n".join(details)
        )
    return incompatible

def optimizer_target_for_mode(mode):
    """Return the component optimized by a canonical-null training mode."""
    if mode == 'train':
        return 'null_predictor'
    if mode == 'train-scale':
        return 'scale_model'
    return None

def checkpoint_optimizer_target(checkpoint):
    """Recover optimizer component identity from current or legacy metadata."""
    target = checkpoint.get('optimizer_target')
    if target:
        return target
    saved_args = checkpoint.get('meta', {}).get('args', {})
    if isinstance(saved_args, dict):
        return optimizer_target_for_mode(saved_args.get('mode'))
    return None

def move_optimizer_state_to_device(optimizer, device):
    """Move every tensor in an optimizer state mapping to ``device`` in place."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)

def restore_training_state_from_checkpoint(args, checkpoint, optimizer, scheduler, device, is_main_process):
    """Restore compatible optimizer/scheduler state and return the saved step.

    State is skipped when absent or when the checkpoint optimized a different
    component than the current mode. Optimizer incompatibility raises rather
    than silently starting over.
    """
    if checkpoint is None or optimizer is None:
        return 0

    expected_target = optimizer_target_for_mode(args.mode)
    saved_target = checkpoint_optimizer_target(checkpoint)
    optimizer_state = checkpoint.get('optimizer_state_dict')
    if optimizer_state is None:
        if is_main_process:
            print("Checkpoint has no optimizer_state_dict; starting optimizer/scheduler from step 0.", flush=True)
        return 0

    if expected_target is None or saved_target != expected_target:
        if is_main_process:
            print(
                "Skipping optimizer/scheduler restore because checkpoint optimizer target "
                f"is {saved_target!r}, but current mode expects {expected_target!r}.",
                flush=True,
            )
        return 0

    try:
        optimizer.load_state_dict(optimizer_state)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Optimizer state from checkpoint is incompatible with current {expected_target} optimizer."
        ) from exc
    move_optimizer_state_to_device(optimizer, device)

    global_step = int(checkpoint.get('global_step') or 0)
    scheduler_state = checkpoint.get('scheduler_state_dict')
    if scheduler is not None:
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        elif global_step > 0:
            scheduler.base_lrs = [
                group.get('initial_lr', base_lr)
                for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs)
            ]
            scheduler.last_epoch = global_step
            scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]

    if is_main_process:
        print(
            f"Restored {expected_target} optimizer state from step {global_step}.",
            flush=True,
        )
    if getattr(args, 'lr_schedule', 'legacy') == 'constant':
        # A requested constant LR must not inherit a historical decayed LR.
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] = float(args.lr)
        if scheduler is not None:
            scheduler.base_lrs = [float(args.lr)] * len(optimizer.param_groups)
            scheduler._last_lr = list(scheduler.base_lrs)
    return global_step

def build_null_config(pretrain_config, args=None):
    """Construct the default two-layer null predictor for a backbone width."""
    hidden_dim = pretrain_config.hidden_dim
    num_heads = 16
    if hidden_dim % num_heads != 0:
        for candidate in range(min(num_heads, hidden_dim), 0, -1):
            if hidden_dim % candidate == 0:
                num_heads = candidate
                break

    if getattr(args, 'null_rna_structure_features', False) and getattr(args, 'seq_type', None) != 'rna':
        raise ValueError('--null_rna_structure_features requires --seq_type rna')
    return CanonicalNullConfig(
        hidden_dim=hidden_dim,
        external_hidden_dim=hidden_dim,
        num_heads=num_heads,
        kv_num_heads=pretrain_config.kv_num_heads,
        num_layers=2,
        dim_feedforward=hidden_dim * 4,
        base_seq_window=(-1, -1),
        self_attn_temp=1.0,
        dropout=0.0,
        self_attn_alibi_half_life_min=16,
        self_attn_alibi_half_life_max=64,
        use_sequence_context=getattr(args, 'null_sequence_context', False),
        base_seq_vocab=pretrain_config.base_seq_vocab,
        context_embedding_dim=getattr(args, 'null_context_embedding_dim', 32),
        context_channels=getattr(args, 'null_context_channels', 128),
        context_kernel_size=getattr(args, 'null_context_kernel_size', 5),
        context_dilations=tuple(getattr(args, 'null_context_dilations', (1, 1, 2, 4, 8, 16))),
        context_expansion=getattr(args, 'null_context_expansion', 4),
        context_dropout=getattr(args, 'null_context_dropout', 0.05),
        use_rna_structure_features=getattr(args, 'null_rna_structure_features', False),
    )

def load_pretrained_backbone(pretrain_load_path, device):
    """Load and freeze a backbone checkpoint for canonical-null workflows.

    Signal-to-base cross-attention is disabled in the constructed configuration.
    Signal-modality dropout remains recorded in that configuration but is
    inactive because inference forwards use the zero default. Returns
    ``(backbone, config)`` in evaluation mode on ``device``.
    """
    print(f'Loading pretrained model  from:{pretrain_load_path}', flush=True)
    checkpoint = torch.load(pretrain_load_path, map_location="cpu")
    pretrain_config_dict = get_backbone_config_from_checkpoint(
        checkpoint,
        source=pretrain_load_path,
    )
    pretrain_config_dict['add_signal_cross_attn'] = False
    pretrain_config = ONTBackboneConfig(**pretrain_config_dict)
    pretrained_model = ONTBackbone(pretrain_config)
    state_dict = strip_module_prefix(get_pretraining_state_dict_from_checkpoint(checkpoint))
    backbone_state_dict, _ = split_pretraining_state_dict(state_dict)
    checked_load_state_dict(
        pretrained_model,
        backbone_state_dict,
        'pretrained backbone',
        pretrain_load_path,
        allowed_unexpected_prefixes=(
            'cross_attn.signal_to_base_cross_tower.',
            'final_norm_ts.',
        ),
    )

    for param in pretrained_model.parameters():
        param.requires_grad = False

    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    pretrained_model.preprocessing = preprocessing_from_checkpoint(checkpoint, required=False)
    return pretrained_model, pretrain_config

def load_null_predictor_from_checkpoint(model_path, pretrain_config, device):
    """Load a compatible canonical-null predictor in evaluation mode."""
    checkpoint = torch.load(model_path, map_location="cpu")
    null_config_dict = checkpoint.get('null_config') or checkpoint.get('config')
    if null_config_dict is None:
        null_config = build_null_config(pretrain_config)
    else:
        null_config = CanonicalNullConfig(**null_config_dict)
    validate_null_config_for_backbone(
        null_config,
        pretrain_config,
        f"Null predictor in {model_path}",
    )

    model = CanonicalNullPredictor(null_config)
    state_dict = checkpoint.get('null_model_state_dict') or checkpoint.get('model_state_dict')
    if state_dict is None and checkpoint.get('head_state_dicts') is not None:
        state_dict = checkpoint['head_state_dicts'].get('null_predictor')
    if state_dict is None:
        raise KeyError(f"No null predictor state dict found in {model_path}")
    checked_load_state_dict(model, state_dict, 'canonical null predictor', model_path)
    model.eval()
    model = model.to(device)
    return model

def load_scale_model_from_checkpoint(model_path, pretrain_config, device, kmer_size=9,
                                     expected_use_base_quality_features=None,
                                     expected_score_type=None):
    """Load and validate an optional scale model from a canonical checkpoint.

    Returns ``None`` when no scale configuration is stored. ``kmer_size`` is a
    retained compatibility argument; the serialized scale config is authoritative.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    scale_config_dict = checkpoint.get('scale_config')
    if scale_config_dict is None:
        return None

    scale_config = canonical_scale_config_from_dict(scale_config_dict)
    validate_scale_config_for_backbone(
        scale_config,
        pretrain_config,
        f"Scale model in {model_path}",
        expected_use_base_quality_features=expected_use_base_quality_features,
        expected_score_type=expected_score_type,
    )
    scale_model = CanonicalNullScaleModel(scale_config)
    state_dict = checkpoint.get('scale_model_state_dict')
    if state_dict is None and checkpoint.get('head_state_dicts') is not None:
        state_dict = checkpoint['head_state_dicts'].get('scale_model')
    if state_dict is None:
        raise KeyError(f"scale_config is present but scale_model_state_dict is missing in {model_path}")
    checked_load_state_dict(scale_model, state_dict, 'canonical null scale model', model_path)
    scale_model.eval()
    scale_model = scale_model.to(device)
    return scale_model

def save_canonical_null_checkpoint(model, pretrained_model, optimizer, scheduler, args, model_path, global_step, scale_model=None):
    """Serialize modular backbone/null/scale state plus training provenance."""
    model_to_save = unwrap_model(model)
    backbone_to_save = unwrap_model(pretrained_model)
    modular_payload = modular_canonical_null_payload(backbone_to_save, model_to_save, scale_model=unwrap_model(scale_model) if scale_model is not None else None)
    ckpt = {
        'checkpoint_type': 'ont_canonical_null_v2',
        'model_class': 'OntModelForCanonicalNull',
        'null_config': asdict(model_to_save.config),
        'config': asdict(model_to_save.config),
        'backbone_config': asdict(backbone_to_save.config),
        'pretrain_config': asdict(backbone_to_save.config),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'optimizer_target': optimizer_target_for_mode(args.mode),
        'global_step': global_step,
        'meta': {
            'pretrain_load_path': args.pretrain_load_path,
            'loss': 'cosine_normalized_plus_0.02_smooth_l1',
            'args': vars(args),
        },
    }
    ckpt.update(modular_payload)
    if scale_model is not None:
        scale_config = unwrap_model(scale_model).config
        ckpt['scale_config'] = asdict(scale_config)
        ckpt['score_type'] = standardized_score_type(scale_config.score_type)
        if normalize_score_type(scale_config.score_type) == RAW_WHITENING_SCORE_TYPE:
            ckpt['meta']['scale_loss'] = 'gaussian_nll_on_log_base_specific_full_space_whitening_div_hidden_dim'
            ckpt['meta']['whitening_artifact'] = getattr(args, 'whitening_artifact', '')
        else:
            ckpt['meta']['scale_loss'] = 'gaussian_nll_on_log_raw_cosine_distance'
        ckpt['meta']['scale_feature_schema_version'] = int(scale_config.feature_schema_version)
    attach_preprocessing(ckpt, args.preprocessing)
    torch.save(ckpt, model_path)

def load_canonical_null_checkpoint(model_path, device, expected_use_base_quality_features=None,
                                   expected_scale_score_type=None,
                                   whitening_artifact_path=None,
                                   expected_whitening_seq_type=None,
                                   allow_whitening_checkpoint_mismatch=False):
    """Load a unified canonical-null checkpoint and optional whitening scorer.

    The backbone is frozen, all components enter evaluation mode, and the
    composed model is moved to ``device``. Returns ``(model, raw_checkpoint)``.
    Optional expectations enforce scale schema and whitening compatibility.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    if 'backbone_state_dict' not in checkpoint:
        raise KeyError(f"{model_path} is not a unified canonical-null checkpoint")

    pretrain_config = ONTBackboneConfig(
        **get_backbone_config_from_checkpoint(checkpoint, source=model_path)
    )
    backbone = ONTBackbone(pretrain_config)
    checked_load_state_dict(
        backbone,
        checkpoint['backbone_state_dict'],
        'canonical-null backbone',
        model_path,
    )
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    null_config = CanonicalNullConfig(**checkpoint['null_config'])
    validate_null_config_for_backbone(
        null_config,
        pretrain_config,
        f"Null predictor in {model_path}",
    )
    null_predictor = CanonicalNullPredictor(null_config)
    null_state_dict = checkpoint.get('null_model_state_dict')
    if null_state_dict is None and checkpoint.get('head_state_dicts') is not None:
        null_state_dict = checkpoint['head_state_dicts'].get('null_predictor')
    if null_state_dict is None:
        raise KeyError(f"No null predictor state dict found in {model_path}")
    checked_load_state_dict(null_predictor, null_state_dict, 'canonical null predictor', model_path)
    null_predictor.eval()

    scale_model = None
    scale_state_dict = checkpoint.get('scale_model_state_dict')
    if scale_state_dict is None and checkpoint.get('head_state_dicts') is not None:
        scale_state_dict = checkpoint['head_state_dicts'].get('scale_model')
    if 'scale_config' in checkpoint and scale_state_dict is not None:
        scale_config = canonical_scale_config_from_dict(checkpoint['scale_config'])
        validate_scale_config_for_backbone(
            scale_config,
            pretrain_config,
            f"Scale model in {model_path}",
            expected_use_base_quality_features=expected_use_base_quality_features,
            expected_score_type=expected_scale_score_type,
        )
        scale_model = CanonicalNullScaleModel(scale_config)
        checked_load_state_dict(scale_model, scale_state_dict, 'canonical null scale model', model_path)
        scale_model.eval()

    whitening_scorer = None
    if whitening_artifact_path:
        whitening_scorer = BaseSpecificWhiteningScorer.from_path(
            whitening_artifact_path,
            device=device,
            expected_hidden_dim=pretrain_config.hidden_dim,
            expected_seq_type=expected_whitening_seq_type,
            allow_checkpoint_mismatch=allow_whitening_checkpoint_mismatch,
        )

    canonical_model = OntModelForCanonicalNull(
        backbone,
        null_predictor,
        scale_model=scale_model,
        whitening_scorer=whitening_scorer,
    )
    canonical_model.preprocessing = preprocessing_from_checkpoint(checkpoint, required=False)
    backbone.preprocessing = canonical_model.preprocessing
    if whitening_scorer is not None:
        validate_artifact_preprocessing(whitening_scorer.artifact, canonical_model.preprocessing)
    canonical_model.to(device)
    return canonical_model, checkpoint

CanonicalNullModel = OntModelForCanonicalNull

def load_canonical_calibration(calibration_path, expected_score_type=None):
    """Load a v1 calibration payload and optionally validate its score family."""
    calibration = torch.load(calibration_path, map_location='cpu')
    if calibration.get('format') != 'canonical_null_calibration_v1':
        raise ValueError(f"Unsupported calibration format in {calibration_path}")
    if expected_score_type is not None:
        metadata = calibration.get('metadata', {})
        stored_score_type = metadata.get('raw_score_type') or metadata.get('score_type')
        if stored_score_type is not None:
            stored_raw = normalize_score_type(stored_score_type)
            expected_raw = normalize_score_type(expected_score_type)
            if stored_raw != expected_raw:
                raise ValueError(
                    f"Calibration score type mismatch in {calibration_path}: "
                    f"stored={stored_raw}, expected={expected_raw}"
                )
    return calibration

def prepare_canonical_calibration(calibration, device):
    """Build and cache device-local structures used by calibration lookup.

    Histogram suffix counts and quantile tables are dense. Reservoirs use a
    dense CUDA table only when it consumes at most 60% of currently free memory;
    otherwise callers use the CPU dictionary fallback. Mutates and returns the
    calibration dictionary.
    """
    distribution_type = calibration.get('distribution_type')
    device = torch.device(device)
    cache = calibration.get('_lookup_cache')
    if cache is not None and cache.get('device') == device:
        return calibration

    if distribution_type == 'histogram':
        hist_counts = calibration['hist_counts'].to(device=device, dtype=torch.float32, non_blocking=True)
        suffix_counts = torch.flip(torch.cumsum(torch.flip(hist_counts, dims=[1]), dim=1), dims=[1])
        del hist_counts

        seen_counts = calibration.get('seen_counts')
        if seen_counts is None:
            seen_counts = calibration.get('stored_counts')
        calibration['_lookup_cache'] = {
            'distribution_type': 'histogram',
            'device': device,
            'suffix_counts': suffix_counts,
            'seen_counts': seen_counts.to(device=device, dtype=torch.long, non_blocking=True),
            'overflow_counts': calibration['hist_overflow_counts'].to(device=device, dtype=torch.float32, non_blocking=True),
        }
        return calibration

    if distribution_type == 'quantile':
        effective_counts = calibration.get('stored_counts')
        if effective_counts is None:
            effective_counts = calibration.get('seen_counts')
        effective_counts = effective_counts.to(device=device, dtype=torch.long, non_blocking=True)
        quantile_levels = calibration['quantile_levels'].to(device=device, dtype=torch.float32, non_blocking=True)
        num_kmers = effective_counts.numel()
        num_levels = quantile_levels.numel()
        quantile_scores = torch.full((num_kmers, num_levels), float('inf'), dtype=torch.float32, device=device)
        has_quantiles = torch.zeros((num_kmers,), dtype=torch.bool, device=device)
        for kmer_id, scores in calibration['quantile_scores_by_kmer_id'].items():
            kmer_id = int(kmer_id)
            if 0 <= kmer_id < num_kmers and scores is not None and scores.numel() > 0:
                quantile_scores[kmer_id] = scores.to(device=device, dtype=torch.float32, non_blocking=True)
                has_quantiles[kmer_id] = True
        calibration['_lookup_cache'] = {
            'distribution_type': 'quantile',
            'device': device,
            'quantile_levels': quantile_levels,
            'quantile_scores': quantile_scores,
            'effective_counts': effective_counts,
            'has_quantiles': has_quantiles,
        }
        return calibration

    if distribution_type == 'reservoir':
        scores_by_kmer = calibration.get('scores_by_kmer_id', {})
        stored_counts = calibration.get('stored_counts')
        if stored_counts is None:
            stored_counts = calibration.get('seen_counts')
        if stored_counts is None or not torch.cuda.is_available() or device.type != 'cuda':
            calibration['_lookup_cache'] = {
                'distribution_type': 'reservoir_fallback',
                'device': device,
            }
            return calibration

        stored_counts = stored_counts.to(device=device, dtype=torch.long, non_blocking=True)
        num_kmers = stored_counts.numel()
        max_count = int(stored_counts.max().item()) if stored_counts.numel() > 0 else 0
        if max_count == 0:
            calibration['_lookup_cache'] = {
                'distribution_type': 'reservoir_fallback',
                'device': device,
            }
            return calibration

        bytes_needed = num_kmers * max_count * torch.finfo(torch.float32).bits // 8
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        if bytes_needed > int(0.60 * free_bytes):
            calibration['_lookup_cache'] = {
                'distribution_type': 'reservoir_fallback',
                'device': device,
            }
            return calibration

        score_table = torch.full((num_kmers, max_count), float('inf'), dtype=torch.float32, device=device)
        for kmer_id, null_scores in scores_by_kmer.items():
            kmer_id = int(kmer_id)
            if not (0 <= kmer_id < num_kmers) or null_scores is None or null_scores.numel() == 0:
                continue
            n = min(int(null_scores.numel()), max_count)
            score_table[kmer_id, :n] = null_scores[:n].to(device=device, dtype=torch.float32, non_blocking=True)
        calibration['_lookup_cache'] = {
            'distribution_type': 'reservoir_dense',
            'device': device,
            'score_table': score_table,
            'stored_counts': stored_counts.clamp(max=max_count),
            'lookup_chunk_size': 1024,
        }
    return calibration

def training_step(model, pretrained_model, batch, optimizer, scheduler, device, eval_only, num_base_trim=16,
                  num_signal_trim=32, loss_mask=None):
    """Run one null-mean predictor update against frozen contextual embeddings.

    Base positions are trimmed by ``num_base_trim``; ``num_signal_trim`` is
    retained for interface symmetry and is unused. Training clips gradients to
    one and advances optimizer/scheduler. Returns total, cosine, and Smooth-L1
    scalar losses followed by currently unused prediction/target placeholders.
    """
    
    module = model.module if isinstance(model, DDP) else model
    base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch[:5]
    rna_structure_features = (batch[11].to(device, non_blocking=True)
                              if requires_rna_structure_features(model) else None)
    
    base_seq_input = base_seq_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_input = percentiles_input.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    
    trimmed_pad_base_seq=ignore_edges(pad_base_seq, num_base_trim)
    if loss_mask is not None:
        trimmed_pad_base_seq = trimmed_pad_base_seq & loss_mask.to(device)
    
    preds, targets = None, None
    
    if not eval_only:
        optimizer.zero_grad()
        
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        with torch.no_grad():
            pretrain_out= pretrained_model(base_seq_input, ts_input, percentiles_input= percentiles_input,
                            pad_base_seq=pad_base_seq, pad_ts=pad_ts, embeddings_only=True)
            
            embedding_target = pretrain_out['contextual_base_seq']
            
        with torch.no_grad() if eval_only else nullcontext():
            embedding_pred = model(pretrain_out['fixed_sa_base'], pad_base_seq, base_seq_input=base_seq_input,
                                   rna_structure_features=rna_structure_features)
        
    if embedding_pred:
        embedding_loss, loss_cos, loss_mse = get_canonical_null_loss(
            embedding_pred,
            embedding_target,
            trimmed_pad_base_seq,
        )
    else:
        embedding_loss = torch.tensor(0, device=device)
        loss_cos = torch.tensor(0, device=device)
        loss_mse = torch.tensor(0, device=device)
    
    loss=embedding_loss
    
    if not eval_only:
#         loss += get_dummy_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    return embedding_loss.item(), \
           loss_cos.item(), \
           loss_mse.item(), \
           preds.cpu() if preds is not None else None, \
           targets.cpu() if targets is not None else None

def compute_residual_and_score_for_scale(
    scale_module,
    whitening_scorer,
    pretrain_out,
    null_out,
    base_seq_input,
    valid_mask=None,
):
    """Compute raw cosine or whitening scores from frozen model outputs.

    Returns ``(scores, normalized_embedding_residual)``. Whitening mode requires
    a scorer compatible with the scale model's configured score family.
    """
    score_type = normalize_score_type(getattr(scale_module.config, 'score_type', RAW_COSINE_SCORE_TYPE))
    h_obs, h_null, residual = canonical_null_residuals(
        pretrain_out['contextual_base_seq'],
        null_out['mean_emb'],
    )
    raw_cosine = canonical_cosine_scores_from_normalized(h_obs, h_null)
    if score_type == RAW_COSINE_SCORE_TYPE:
        return raw_cosine, residual
    if score_type == RAW_WHITENING_SCORE_TYPE:
        if whitening_scorer is None:
            raise ValueError("A whitening artifact is required for whitening scale training/calibration")
        with torch.autocast(device_type=residual.device.type, enabled=False):
            whitening_scores = whitening_scorer.score(
                residual.float(),
                base_seq_input,
                valid_mask=valid_mask,
            )
        return whitening_scores, residual
    raise ValueError(f"Unsupported scale score_type={score_type!r}")


def scale_training_step(scale_model, null_predictor, pretrained_model, batch, optimizer, scheduler, device,
                        eval_only, num_base_trim=16, whitening_scorer=None, loss_mask=None):
    """Run one scale-model update with frozen backbone and null predictions.

    Centered k-mer validity is combined with the trimmed base mask. Returns the
    Gaussian NLL, standardized-residual MSE, and mean log scale as floats,
    followed by the integer count of targets actually included in the loss.
    """
    base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch[:5]
    base_quality = batch[5] if len(batch) > 5 else None
    read_global_features = batch[6] if len(batch) > 6 else None
    reference_match_mask = batch[8] if len(batch) > 8 else None

    rna_structure_features = batch[11]
    if rna_structure_features is not None:
        rna_structure_features = rna_structure_features.to(device, non_blocking=True)

    base_seq_input = base_seq_input.to(device, non_blocking=True)
    ts_input = ts_input.to(device, non_blocking=True)
    percentiles_input = percentiles_input.to(device, non_blocking=True)
    pad_base_seq = pad_base_seq.to(device, non_blocking=True)
    pad_ts = pad_ts.to(device, non_blocking=True)
    if base_quality is not None:
        base_quality = base_quality.to(device, non_blocking=True)
    if read_global_features is not None:
        read_global_features = read_global_features.to(device, non_blocking=True)
    if reference_match_mask is not None:
        reference_match_mask = reference_match_mask.to(device, non_blocking=True)

    scale_module = unwrap_model(scale_model)
    # The context radius controls chunk overlap, not token eligibility. The
    # context encoder is explicitly masked and the scale model receives edge
    # position features, so true read-end positions are valid training cases.
    # Artificial chunk boundaries are handled by overlap/stitching downstream.
    trimmed_pad_base_seq = ignore_edges(pad_base_seq.bool(), int(num_base_trim))
    if loss_mask is not None:
        trimmed_pad_base_seq = trimmed_pad_base_seq & loss_mask.to(device)
    if reference_match_mask is not None:
        trimmed_pad_base_seq = trimmed_pad_base_seq & reference_match_mask.bool()

    if not eval_only:
        optimizer.zero_grad()

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        pretrain_out = pretrained_model(
            base_seq_input,
            ts_input,
            percentiles_input=percentiles_input,
            pad_base_seq=pad_base_seq,
            pad_ts=pad_ts,
            embeddings_only=True,
        )
        null_out = null_predictor(pretrain_out['fixed_sa_base'], pad_base_seq, base_seq_input=base_seq_input,
                                  rna_structure_features=rna_structure_features)
        scores, _residual = compute_residual_and_score_for_scale(
            scale_module,
            whitening_scorer,
            pretrain_out,
            null_out,
            base_seq_input,
            valid_mask=trimmed_pad_base_seq,
        )

    # A separately compiled, trainable scale head cannot save inference
    # tensors for backward. Materialize its two learned upstream features at
    # the boundary, outside both inference mode and the compiled forward. The
    # forced copy also covers an upstream feature that is already FP32.
    scale_fixed_sa_base = pretrain_out['fixed_sa_base'].detach().to(
        dtype=torch.float32,
        copy=True,
    )
    scale_h_null = null_out['mean_emb'].detach().to(
        dtype=torch.float32,
        copy=True,
    )

    scale_inputs = build_scale_model_inputs(
        pretrained_model,
        base_seq_input,
        pad_base_seq,
        pad_ts,
        scale_module.config.kmer_size,
        base_quality=base_quality,
        read_global_features=read_global_features,
        use_base_quality_features=scale_module.config.use_base_quality_features,
        scale_config=scale_module.config,
        read_positions=batch[9], read_lengths=batch[10],
        rna_structure_features=rna_structure_features,
    )
    # Whitening emits NaN outside its valid base/residual domain. Such values
    # must not enter the NLL even when another positional mask admits them.
    trimmed_pad_base_seq = (
        trimmed_pad_base_seq
        & scale_inputs['kmer_valid']
        & torch.isfinite(scores)
    )

    with torch.no_grad() if eval_only else nullcontext():
        scale_out = scale_model(
            scale_fixed_sa_base,
            scale_inputs['kmer_ids'],
            scale_inputs['token_features'],
            scale_inputs['global_features'],
            pad_base_seq=pad_base_seq,
            base_seq_input=base_seq_input,
            h_null=scale_h_null,
        )

    loss, residual_mse, mean_log_sigma, valid_target_count = get_canonical_scale_loss(
        scores.detach(),
        scale_out,
        trimmed_pad_base_seq,
        eps=scale_module.config.eps,
        score_type=scale_module.config.score_type,
        hidden_dim=unwrap_model(pretrained_model).config.hidden_dim,
    )

    if not eval_only:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scale_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return loss.item(), residual_mse.item(), mean_log_sigma.item(), int(valid_target_count.item())

def custom_collate_fn(batch):
    """Collate base/signal/percentile triples and omit shared-collator metadata."""
    seqs_batch, signals_batch, percentiles_batch = [], [], []
    for item_seqs, item_signals, item_percentiles in batch:
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)

    return collate_modal_batch(seqs_batch, signals_batch, percentiles_batch)[:-1]

def base_error_prob_from_bam_read(bam_read):
    """Decode BAM ASCII qualities into original-orientation error probabilities."""
    if bam_read.qual is None:
        return np.ones(len(bam_read.seq), dtype=np.float32)
    qual = bam_read.qual if bam_read.is_forward else bam_read.qual[::-1]
    qual_ord = np.fromiter((ord(x) for x in qual), dtype=np.float32, count=len(qual))
    return np.power(10.0, (33.0 - qual_ord) / 10.0).astype(np.float32)

def read_scale_metadata(norm_stats, base_error_prob=None, use_base_quality_features=False):
    """Combine signal-normalization statistics with optional median scaled Phred."""
    metadata = [norm_stats.astype(np.float32)]
    if use_base_quality_features:
        if base_error_prob is None:
            median_q = 0.0
        else:
            base_error_prob = np.clip(base_error_prob.astype(np.float32), 1e-12, 1.0)
            phred = -10.0 * np.log10(base_error_prob)
            median_q = float(np.median(phred) / 40.0)
        metadata.append(np.array([median_q], dtype=np.float32))
    return torch.tensor(
        np.concatenate(metadata),
        dtype=torch.float32,
    )

def build_read_position_dedup_masks(read_pos_chunks, min_valid_margin=0, min_chunk_base_len=64):
    """Select one well-centered chunk occurrence for each original read position.

    Candidates in short chunks, near chunk edges, or with negative read offsets
    are excluded. Ties retain the first occurrence. Returns one boolean mask per
    input chunk.
    """
    best_by_pos = {}
    candidates = []

    for chunk_idx, read_pos in enumerate(read_pos_chunks):
        seq_len = int(read_pos.numel())
        if seq_len == 0:
            candidates.append((read_pos, torch.empty(0, dtype=torch.long)))
            continue

        token_idx = torch.arange(seq_len, dtype=torch.long)
        margin = torch.minimum(token_idx, (seq_len - 1 - token_idx).clamp_min(0))
        candidates.append((read_pos, margin))
        if seq_len < int(min_chunk_base_len):
            continue

        valid = (read_pos >= 0) & (margin >= int(min_valid_margin))
        for token_pos in valid.nonzero(as_tuple=True)[0].tolist():
            pos = int(read_pos[token_pos].item())
            token_margin = int(margin[token_pos].item())
            current_best = best_by_pos.get(pos)
            if current_best is None or token_margin > current_best[0]:
                best_by_pos[pos] = (token_margin, chunk_idx, token_pos)

    dedup_masks = [torch.zeros_like(read_pos, dtype=torch.bool) for read_pos, _ in candidates]
    for _, chunk_idx, token_pos in best_by_pos.values():
        dedup_masks[chunk_idx][token_pos] = True
    return dedup_masks


def get_read_chunks_with_quality(
    read,
    size,
    step,
    deduplicate_read_positions=False,
    dedup_min_valid_margin=0,
    reference_match_mask=None,
):
    """Chunk a normalized read with quality and optional eligibility masks.

    ``size`` and ``step`` are signal-sample units before the caller's later
    move-resolution reshape. Base quality, original read indices, and an
    optional read-coordinate reference-match mask follow emitted bases.
    Returns signal chunks transposed channel-last-to-first, ragged base/quality
    lists, optional deduplication/reference-match masks, and read-position
    lists. Read indices are 5'-to-3' for DNA and RNA (RNA signal is reversed
    upstream); padded indices are -1.
    """
    if has_inferred_boundaries(read):
        signal_chunks, base_seq_chunks, _, read_pos_chunks = inferred_chunks(read, size, step)
        base_quality_chunks = [x.float() for x in slice_base_values(read, read.base_qual, size, step)]
    else:
        signal, base_seq, move_table, padding_needed = pad_read_components(read, size, step)
        base_quality = torch.from_numpy(read.base_qual[read.indexes]).float()
        base_quality = F.pad(base_quality, (0, padding_needed), mode='constant', value=1.0)
        base_seq_idx = torch.arange(read.length)[read.indexes]
        base_seq_idx = F.pad(
            base_seq_idx,
            (0, padding_needed),
            mode='constant',
            value=-1,
        )

        signal_chunks = signal.unfold(dimension=0, size=size, step=step)
        base_seq_chunks = base_seq.unfold(dimension=0, size=size, step=step)
        base_quality_chunks = base_quality.unfold(dimension=0, size=size, step=step)
        base_seq_idx_chunks = base_seq_idx.unfold(dimension=0, size=size, step=step)
        init_move_table_chunks = move_table.unfold(dimension=0, size=size, step=step)

        move_table_chunks = init_move_table_chunks.clone()
        move_table_chunks[:, 0] += 1
        move_table_chunks = move_table_chunks.bool()

        base_seq_chunks = [base_seq_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
        base_quality_chunks = [base_quality_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
        read_pos_chunks = [base_seq_idx_chunks[i][move_table_chunks[i]] for i in range(len(move_table_chunks))]
    if deduplicate_read_positions:
        dedup_masks = build_read_position_dedup_masks(
            read_pos_chunks,
            min_valid_margin=dedup_min_valid_margin,
            min_chunk_base_len=64,
        )
    else:
        dedup_masks = None
    if reference_match_mask is not None:
        reference_match_mask = np.asarray(reference_match_mask, dtype=np.bool_)
        if reference_match_mask.shape != (read.length,):
            raise ValueError(
                "reference_match_mask must have one value per read-oriented base"
            )
        match_by_read = torch.from_numpy(reference_match_mask)
        reference_match_chunks = []
        for read_positions in read_pos_chunks:
            chunk_mask = torch.zeros_like(read_positions, dtype=torch.bool)
            valid_positions = (read_positions >= 0) & (read_positions < read.length)
            chunk_mask[valid_positions] = match_by_read[read_positions[valid_positions]]
            reference_match_chunks.append(chunk_mask)
    else:
        reference_match_chunks = None
    return (
        signal_chunks.transpose(1, 2),
        base_seq_chunks,
        base_quality_chunks,
        dedup_masks,
        reference_match_chunks,
        read_pos_chunks,
    )

class CanonicalScaleBamIterableDataset(IterableDataset):
    """Stream finite or cycling BAM chunks with scale-model quality metadata.

    BAM files are deterministically sharded across distributed ranks and loader
    workers. Reads are sampled and quality-filtered, normalized against expected
    k-mer current, chunked by the selected boundary mode, and buffered by read before random
    chunk order. Output items contain base IDs, signal, percentiles, base error
    probabilities, read-level scale metadata, overlap-deduplication masks, and
    exact-reference-match masks. The optional reference mask is derived from
    MD+CIGAR while leaving the read sequence and signal unchanged.
    """

    def __init__(
        self,
        bam_list,
        size,
        step,
        move_downsample_factor,
        seq_type=None,
        kmer_table=None,
        reads_per_chunk=1000,
        min_read_length=100,
        min_qscore=10,
        pore_type=None,
        base_seed=1234,
        rank=0,
        world_size=1,
        deduplicate_read_positions=False,
        dedup_min_valid_margin=0,
        cycle_files=True,
        expected_ts_dim=None,
        use_base_quality_features=False,
        reference_match_kmer_size=0,
        chunking_mode="auto",
        *, preprocessing=None, chemistry=None,
    ):
        super().__init__()
        reference_match_kmer_size = validate_match_kmer_size(
            reference_match_kmer_size,
            allow_zero=True,
        )
        self.bam_list = bam_list
        self.size = size * move_downsample_factor
        self.move_downsample_factor = move_downsample_factor
        self.step = step * move_downsample_factor
        self.preprocessing = resolve_preprocessing(
            preprocessing=preprocessing, chemistry=chemistry, seq_type=seq_type,
            pore_type=pore_type, kmer_table=kmer_table,
        )
        seq_type, pore_type, kmer_table = (
            self.preprocessing.seq_type, self.preprocessing.pore_type, self.preprocessing.kmer_table
        )
        self.seq_type = seq_type
        self.reads_per_chunk = reads_per_chunk
        self.min_read_length = min_read_length
        self.min_qscore = min_qscore
        self.pore_type = pore_type
        self.base_seed = base_seed
        self.rank = rank
        self.world_size = world_size
        self.kmer_table = kmer_table
        self.deduplicate_read_positions = deduplicate_read_positions
        self.dedup_min_valid_margin = int(dedup_min_valid_margin)
        self.cycle_files = cycle_files
        self.chunking_mode = validate_chunking_mode(chunking_mode)
        self.expected_ts_dim = expected_ts_dim
        self.use_base_quality_features = use_base_quality_features
        self.reference_match_kmer_size = reference_match_kmer_size
        # ``max_steps`` can end a pass while worker processes still hold an
        # open BAM and a partially yielded read buffer.  Let those workers
        # close their generators normally instead of terminating them during
        # DataLoader teardown.  The event is shared by spawn-based workers.
        self._stop_event = torch.multiprocessing.Event()

    def request_stop(self):
        """Ask this loader's worker replicas to end their current BAM pass."""
        self._stop_event.set()

    def _stop_requested(self):
        """Return whether the parent process requested cooperative shutdown."""
        return self._stop_event.is_set()

    def _empty_buffer(self):
        data = {'signal': [], 'seq': [], 'base_quality': [], 'global_features': []}
        data['dedup_mask'] = []
        data['reference_match_mask'] = []
        data['read_positions'] = []
        data['read_lengths'] = []
        return data

    def _yield_buffer(self, data):
        signals = torch.cat(data['signal'], dim=0)
        percentiles = get_percentiles(signals)

        signals = signals.reshape(
            signals.size(0),
            signals.size(1) // self.move_downsample_factor,
            signals.size(2) * self.move_downsample_factor,
        )
        percentiles = percentiles.reshape(
            percentiles.size(0),
            percentiles.size(1) // self.move_downsample_factor,
            percentiles.size(2) * self.move_downsample_factor,
        )

        idxs = list(range(len(data['seq'])))
        random.shuffle(idxs)

        for i in idxs:
            if self._stop_requested():
                return
            if len(data['seq'][i]) >= 64:
                item = (
                    data['seq'][i],
                    signals[i].to(dtype=torch.float16),
                    percentiles[i].to(dtype=torch.float16),
                    data['base_quality'][i],
                    data['global_features'][i],
                )
                item = item + (
                    data['dedup_mask'][i],
                    data['reference_match_mask'][i],
                    data['read_positions'][i],
                    data['read_lengths'][i],
                )
                yield item

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        poly_gaa = 'GAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAAGAA'
        poly_tgg = 'TGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGGTGG'

        global_rank = self.rank
        global_world_size = self.world_size
        global_worker_id = global_rank * num_workers + worker_id
        global_worker_count = max(global_world_size * num_workers, 1)

        seed = self.base_seed + 1000003 * (global_worker_id + 1)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)

        if self._stop_requested():
            return

        my_files = []
        for bam_idx in range(global_worker_id, len(self.bam_list), global_worker_count):
            my_files.append(self.bam_list[bam_idx])

        if len(my_files) == 0:
            return

        data = self._empty_buffer()
        for cycle_counter, (bam_path, sample) in iter_file_cycles(my_files, self.cycle_files):
            if self._stop_requested():
                return
            with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
                print(
                    f'{str(datetime.datetime.now())}: '
                    f'{format_iterable_bam_log(self, global_rank, worker_id, bam_path, cycle_counter=cycle_counter)}',
                    flush=True,
                )

                for bam_read in bam.fetch(until_eof=True):
                    if self._stop_requested():
                        return
                    if (
                        np.random.binomial(n=1, p=sample)
                        and bam_read.flag & 0x900 == 0
                        and passes_min_qscore(bam_read, self.min_qscore)
                        and len(bam_read.seq) >= self.min_read_length
                    ):
                        if self.seq_type == 'rna' and (poly_gaa in bam_read.seq or poly_tgg in bam_read.seq):
                            continue
                        if self.reference_match_kmer_size and bam_read.is_unmapped:
                            continue

                        read_info = get_read_info_lite(
                            bam_read, self.seq_type, self.kmer_table,
                            expected_ts_dim=self.expected_ts_dim, chunking_mode=self.chunking_mode,
                        )
                        read = Read_Lite(*read_info)
                        if self.use_base_quality_features:
                            read.base_qual = base_error_prob_from_bam_read(bam_read)
                        else:
                            read.base_qual = np.ones(len(bam_read.seq), dtype=np.float32)
                        try:
                            read.signal, norm_stats = normalize_read_signal_with_stats(read, self.pore_type)
                        except DegenerateQuantileNormalizationError:
                            continue
                        global_features = read_scale_metadata(
                            norm_stats,
                            read.base_qual if self.use_base_quality_features else None,
                            use_base_quality_features=self.use_base_quality_features,
                        )

                        reference_match_mask = None
                        if self.reference_match_kmer_size:
                            reference_match_mask = build_md_exact_match_kmer_mask(
                                bam_read,
                                read.base_seq,
                                k=self.reference_match_kmer_size,
                            )

                        (
                            signal_chunks,
                            base_seq_chunks,
                            base_quality_chunks,
                            dedup_masks,
                            reference_match_chunks,
                            read_pos_chunks,
                        ) = get_read_chunks_with_quality(
                            read,
                            self.size,
                            self.step,
                            deduplicate_read_positions=self.deduplicate_read_positions,
                            dedup_min_valid_margin=self.dedup_min_valid_margin,
                            reference_match_mask=reference_match_mask,
                        )

                        data['signal'].append(signal_chunks)
                        data['seq'].extend(base_seq_chunks)
                        data['base_quality'].extend(base_quality_chunks)
                        data['read_positions'].extend(read_pos_chunks)
                        data['read_lengths'].extend([read.length] * len(base_seq_chunks))
                        data['global_features'].extend([global_features] * len(base_seq_chunks))
                        data['dedup_mask'].extend(
                            dedup_masks
                            if dedup_masks is not None
                            else [torch.ones_like(chunk, dtype=torch.bool) for chunk in base_seq_chunks]
                        )
                        data['reference_match_mask'].extend(
                            reference_match_chunks
                            if reference_match_chunks is not None
                            else [torch.ones_like(chunk, dtype=torch.bool) for chunk in base_seq_chunks]
                        )

                        if len(data['signal']) == self.reads_per_chunk:
                            yield from self._yield_buffer(data)
                            if self._stop_requested():
                                return
                            data = self._empty_buffer()

        if len(data['signal']) > 0 and not self._stop_requested():
            yield from self._yield_buffer(data)

def custom_scale_collate_fn(batch, *, use_rna_structure_features=False):
    """Pad canonical-scale samples, metadata, and token eligibility masks.

    Returns twelve fields. The first seven follow the scale-model multimodal
    contract; indices seven and eight are respectively overlap-deduplication
    and exact-reference-match masks. Indices nine/ten are (B,L) full-read
    positions and (B,) full-read lengths in read-oriented base coordinates.
    Index eleven is (B,L,109) float32 RNA structure features, or None when
    disabled. Folding uses final padded chunk coordinates in loader workers.
    """
    seqs_batch, signals_batch, percentiles_batch = [], [], []
    base_quality_batch, global_features_batch = [], []
    dedup_mask_batch, reference_match_batch = [], []
    read_positions_batch, read_lengths_batch = [], []

    for item in batch:
        (
            item_seqs,
            item_signals,
            item_percentiles,
            item_base_quality,
            item_global_features,
            item_dedup_mask,
            item_reference_match,
            item_read_positions,
            item_read_length,
        ) = item
        seqs_batch.append(item_seqs)
        signals_batch.append(item_signals)
        percentiles_batch.append(item_percentiles)
        base_quality_batch.append(item_base_quality)
        global_features_batch.append(item_global_features)
        dedup_mask_batch.append(item_dedup_mask)
        reference_match_batch.append(item_reference_match)
        read_positions_batch.append(item_read_positions)
        read_lengths_batch.append(item_read_length)

    seqs_batch_padded, signals_batch, percentiles_batch, pad_base_seq, pad_ts, pad_needed = collate_modal_batch(
        seqs_batch,
        signals_batch,
        percentiles_batch,
    )
    base_quality_padded = pad_sequence(base_quality_batch, batch_first=True, padding_value=1.0)
    if pad_needed > 0:
        base_quality_padded = F.pad(base_quality_padded, (0, pad_needed), value=1.0)
    base_quality_padded = torch.where(
        pad_base_seq,
        base_quality_padded.float(),
        torch.ones_like(base_quality_padded, dtype=torch.float32),
    )

    dedup_mask_padded = pad_sequence(dedup_mask_batch, batch_first=True, padding_value=False)
    reference_match_padded = pad_sequence(
        reference_match_batch,
        batch_first=True,
        padding_value=False,
    )
    if pad_needed > 0:
        dedup_mask_padded = F.pad(dedup_mask_padded, (0, pad_needed), value=False)
        reference_match_padded = F.pad(
            reference_match_padded,
            (0, pad_needed),
            value=False,
        )

    read_positions_padded = pad_sequence(read_positions_batch, batch_first=True, padding_value=-1)
    read_positions_padded = F.pad(read_positions_padded, (0, pad_needed), value=-1)
    return append_rna_structure_features((
        seqs_batch_padded,
        signals_batch,
        percentiles_batch,
        pad_base_seq,
        pad_ts,
        base_quality_padded,
        torch.stack(global_features_batch, dim=0).float(),
        dedup_mask_padded.bool() & pad_base_seq,
        reference_match_padded.bool() & pad_base_seq,
        read_positions_padded,
        torch.tensor(read_lengths_batch, dtype=torch.long),
    ), enabled=use_rna_structure_features)
            
def load_pretraining_bam_list(bam_config):
    """Load a pretraining JSON config as ``(BAM, probability)`` tuples."""
    from porefm.config.validation import load_pretraining_config

    return load_pretraining_config(bam_config).bam_list

def setup_distributed():
    """Initialize NCCL from torchrun variables, or return single-process ranks."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank

    return 0, 1, 0

def distributed_mean(value: float, device: torch.device, world_size: int) -> float:
    """Average one scalar across ranks, avoiding collectives for one process."""
    if world_size == 1:
        return value

    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


class NoOpWriter:
    """TensorBoard-compatible sink used by non-main processes."""

    def add_scalar(self, *args, **kwargs):
        """Discard a scalar log event."""
        return None

    def flush(self):
        """Provide the writer flush interface without side effects."""
        return None

def reduce_loss_stats(total_loss: float, count: int, device: torch.device, world_size: int):
    """Sum loss and observation accumulators across distributed ranks."""
    if world_size == 1:
        return total_loss, count

    tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor[0].item(), int(tensor[1].item())

def get_global_counts(local_instances, local_base_tokens, local_signal_tokens, device):
    """Sums instances and tokens across all GPUs."""
    if not dist.is_initialized():
        return local_instances, local_base_tokens, local_signal_tokens
        
    # Pack into a tensor for efficient transfer
    tensor = torch.tensor([local_instances, local_base_tokens, local_signal_tokens], device=device, dtype=torch.long)
    
    # Sums the values from all ranks in-place
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return tensor[0].item(), tensor[1].item(), tensor[2].item()

def make_bam_dataloader(args, pretrain_config, kmer_table, rank, world_size, drop_last):
    """Construct the standard canonical-null mean-prediction BAM DataLoader."""
    bam_list = load_pretraining_bam_list(args.bam_config)

    ds = BamIterableDataset(
        bam_list=bam_list,
        size=args.size,
        move_downsample_factor=pretrain_config.move_downsample_factor,
        step=args.step,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=rank,
        world_size=world_size,
        expected_ts_dim=pretrain_config.ts_dim,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )

    if args.dataloader_workers > 0:
        dataloader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )

    return DataLoader(ds, **dataloader_kwargs)

def make_scale_bam_dataloader(args, pretrain_config, kmer_table, rank, world_size, drop_last,
                            use_rna_structure_features=False):
    """Construct the quality-aware finite/cycling scale-workflow DataLoader."""
    bam_list = load_pretraining_bam_list(args.bam_config)
    from porefm.workflows.canonical_null.splits import balance_worker_order
    bam_list = balance_worker_order(bam_list, world_size * max(args.dataloader_workers, 1), args.seed)

    if args.mode == 'fit-whitening':
        context_kmer_size = resolved_whitening_kmer_size(args)
    elif args.mode == 'calibrate':
        context_kmer_size = resolved_calibration_kmer_size(args)
    else:
        context_kmer_size = resolved_scale_kmer_size(args)
    ds = CanonicalScaleBamIterableDataset(
        bam_list=bam_list,
        size=args.size,
        move_downsample_factor=pretrain_config.move_downsample_factor,
        step=args.step,
        seq_type=args.seq_type,
        chunking_mode=resolve_chunking_mode(args), kmer_table=kmer_table,
        reads_per_chunk=args.reads_per_chunk,
        min_read_length=args.min_read_length,
        min_qscore=args.min_qscore,
        pore_type=args.pore_type,
        base_seed=args.seed,
        rank=rank,
        world_size=world_size,
        deduplicate_read_positions=args.mode in {'calibrate', 'fit-whitening'} or getattr(args, 'eval_only', False),
        dedup_min_valid_margin=max(
            args.num_loss_trim[0],
            context_kmer_size // 2,
        ),
        cycle_files=args.mode not in {'calibrate', 'fit-whitening'} and not getattr(args, 'eval_only', False),
        expected_ts_dim=pretrain_config.ts_dim,
        use_base_quality_features=getattr(args, 'use_base_quality_scale_features', False),
        reference_match_kmer_size=getattr(args, 'reference_match_kmer_size', 0),
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=partial(custom_scale_collate_fn, use_rna_structure_features=use_rna_structure_features),
        num_workers=args.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )

    if args.dataloader_workers > 0:
        dataloader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            # Finite whitening/calibration loaders are constructed once per
            # pass, so persistent workers only complicate early-cap teardown.
            persistent_workers=args.mode not in {'calibrate', 'fit-whitening'} and not getattr(args, 'eval_only', False),
        )

    return DataLoader(ds, **dataloader_kwargs)


def resolved_scale_kmer_size(args):
    """Return the validated scale-model context size with legacy fallback."""
    value = getattr(args, 'scale_kmer_size', None)
    if value is None:
        value = getattr(args, 'kmer_size', 9)
    return validate_centered_kmer_size(value, argument_name='scale_kmer_size')


def resolved_calibration_kmer_size(args):
    """Return the calibration context size, defaulting to the scale context."""
    value = getattr(args, 'calibration_kmer_size', None)
    if value is None:
        value = resolved_scale_kmer_size(args)
    return validate_centered_kmer_size(value, argument_name='calibration_kmer_size')


def resolved_whitening_kmer_size(args):
    """Return the validated full-covariance whitening context size."""
    return validate_centered_kmer_size(
        getattr(args, 'whitening_kmer_size', 1),
        argument_name='whitening_kmer_size',
    )

def requested_scale_score_type(args):
    """Normalize the raw scale score family requested by command-line arguments."""
    return normalize_score_type(getattr(args, 'scale_score_type', RAW_COSINE_SCORE_TYPE))

def make_whitening_scorer(args, pretrain_config, device, required=False):
    """Load a compatible whitening scorer, optionally requiring its artifact."""
    artifact_path = getattr(args, 'whitening_artifact', '') or ''
    if not artifact_path:
        if required:
            raise ValueError("--whitening_artifact is required for whitening score modes")
        return None
    scorer = BaseSpecificWhiteningScorer.from_path(
        artifact_path,
        device=device,
        expected_hidden_dim=pretrain_config.hidden_dim,
        expected_seq_type=getattr(args, 'seq_type', None),
        allow_checkpoint_mismatch=getattr(args, 'allow_whitening_checkpoint_mismatch', False),
    )
    requested_kmer_size = resolved_whitening_kmer_size(args)
    validate_artifact_preprocessing(scorer.artifact, getattr(args, 'preprocessing', None))
    if scorer.whitening_kmer_size != requested_kmer_size:
        raise ValueError(
            f"Whitening artifact uses k={scorer.whitening_kmer_size}, but the run requested "
            f"--whitening_kmer_size {requested_kmer_size}"
        )
    return scorer

def load_models_for_args(args, device):
    """Resolve backbone, null predictor, and scale model for the requested mode.

    Unified canonical checkpoints are preferred when no separate pretrained
    path is supplied. Scale training may initialize a new scale model when the
    loaded feature/score schema differs. Returns
    ``(backbone, null_predictor, scale_model_or_none, backbone_config)``.
    Evaluation never initializes a replacement scale model.
    """
    validate_rna_structure_request(args)
    scale_evaluation = args.mode == 'train-scale' and getattr(args, 'eval_only', False)
    expected_scale_quality = (
        bool(getattr(args, 'use_base_quality_scale_features', False))
        if args.mode in {'train-scale', 'calibrate'}
        else None
    )
    expected_scale_score = (
        requested_scale_score_type(args)
        if args.mode in {'train-scale', 'calibrate'}
        else None
    )
    if args.model_load_path and not args.pretrain_load_path:
        try:
            canonical_model, _ = load_canonical_null_checkpoint(
                args.model_load_path,
                device,
                expected_use_base_quality_features=(
                    expected_scale_quality if args.mode == 'calibrate' or scale_evaluation else None
                ),
                expected_scale_score_type=(
                    expected_scale_score if args.mode == 'calibrate' or scale_evaluation else None
                ),
                # This loader returns the three components rather than the
                # composed wrapper. Whitening is loaded once below by the
                # stage-specific scorer, avoiding a duplicate multi-GiB copy
                # for 5-mer full-covariance artifacts.
                whitening_artifact_path=None,
            )
            scale_model = canonical_model.scale_model
            validate_rna_structure_request(args, null_config=canonical_model.null_predictor.config)
            if scale_evaluation and scale_model is None:
                raise ValueError("Scale evaluation requires a checkpoint containing a trained scale model")
            if args.mode == 'train-scale' and not scale_evaluation:
                requested_scale_config = build_scale_config_from_args(
                    canonical_model.backbone.config,
                    args,
                    expected_scale_score,
                )
                if scale_model is not None:
                    loaded_score_type = normalize_score_type(
                        getattr(scale_model.config, 'score_type', RAW_COSINE_SCORE_TYPE)
                    )
                    loaded_quality = bool(getattr(scale_model.config, 'use_base_quality_features', False))
                    if (
                        loaded_score_type != expected_scale_score
                        or loaded_quality != bool(expected_scale_quality)
                        or not scale_context_config_matches(
                            scale_model.config,
                            requested_scale_config,
                        )
                    ):
                        scale_model = None
                        args._initialized_new_scale_model = True
                if scale_model is None:
                    scale_model = CanonicalNullScaleModel(
                        requested_scale_config
                    ).to(device)
            if scale_model is not None and args.mode in {'train-scale', 'calibrate'}:
                validate_rna_structure_request(args, scale_model.config)
            elif args.mode == 'calibrate' and getattr(args, 'scale_rna_structure_features', False):
                raise ValueError('RNA structure calibration requires a trained scale checkpoint')
            return canonical_model.backbone, canonical_model.null_predictor, scale_model, canonical_model.backbone.config
        except KeyError:
            if args.mode == 'calibrate' or scale_evaluation:
                raise

    if not args.pretrain_load_path:
        raise ValueError("--pretrain_load_path is required unless --model_load_path is a unified canonical-null checkpoint")

    pretrained_model, pretrain_config = load_pretrained_backbone(args.pretrain_load_path, device)
    if args.model_load_path:
        print(f'Loading null predictor weights from:{args.model_load_path}', flush=True)
        model = load_null_predictor_from_checkpoint(args.model_load_path, pretrain_config, device)
    else:
        model = CanonicalNullPredictor(build_null_config(pretrain_config, args)).to(device)

    scale_model = None
    if args.model_load_path:
        scale_model = load_scale_model_from_checkpoint(
            args.model_load_path,
            pretrain_config,
            device,
            expected_use_base_quality_features=expected_scale_quality,
            expected_score_type=expected_scale_score if args.mode == 'calibrate' or scale_evaluation else None,
        )
    if scale_evaluation and scale_model is None:
        raise ValueError("Scale evaluation requires a checkpoint containing a trained scale model")
    if args.mode == 'train-scale' and not scale_evaluation and scale_model is not None:
        requested_scale_config = build_scale_config_from_args(
            pretrain_config,
            args,
            expected_scale_score,
        )
        loaded_score_type = normalize_score_type(
            getattr(scale_model.config, 'score_type', RAW_COSINE_SCORE_TYPE)
        )
        loaded_quality = bool(getattr(scale_model.config, 'use_base_quality_features', False))
        if (
            loaded_score_type != expected_scale_score
            or loaded_quality != bool(expected_scale_quality)
            or not scale_context_config_matches(
                scale_model.config,
                requested_scale_config,
            )
        ):
            scale_model = None
            args._initialized_new_scale_model = True
    if args.mode == 'train-scale' and not scale_evaluation and scale_model is None:
        args._initialized_new_scale_model = True
        scale_model = CanonicalNullScaleModel(
            build_scale_config_from_args(
                pretrain_config,
                args,
                expected_scale_score,
            )
        ).to(device)

    validate_rna_structure_request(args, null_config=model.config)
    if scale_model is not None and args.mode in {'train-scale', 'calibrate'}:
        validate_rna_structure_request(args, scale_model.config)
    elif args.mode == 'calibrate' and getattr(args, 'scale_rna_structure_features', False):
        raise ValueError('RNA structure calibration requires a trained scale checkpoint')
    return pretrained_model, model, scale_model, pretrain_config

def _all_gather_object_if_distributed(obj, world_size):
    if world_size == 1:
        return [obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered

def iter_bounded_dataloader(dataloader, max_batches):
    """Yield at most ``max_batches`` and then close loader workers safely.

    A compatible iterable dataset may expose ``request_stop()``. At an early
    cap, this helper signals that method and drains the small prefetched tail,
    allowing each worker generator to close its open BAM and reach ordinary
    iterable exhaustion. Loaders without cooperative cancellation retain the
    explicit PyTorch iterator-shutdown fallback.
    """
    max_batches = int(max_batches)
    if max_batches < 0:
        raise ValueError(f"max_batches must be non-negative, got {max_batches}")

    loader_iter = iter(dataloader)
    exhausted = False
    try:
        for _ in range(max_batches):
            try:
                yield next(loader_iter)
            except StopIteration:
                exhausted = True
                return
    finally:
        if not exhausted:
            dataset = getattr(dataloader, 'dataset', None)
            request_stop = getattr(dataset, 'request_stop', None)
            if request_stop is not None:
                request_stop()
                # Consume already-prefetched batches and worker exhaustion
                # markers. No additional BAM work is admitted after the event
                # is observed, so this tail is bounded by loader prefetching
                # plus any sample already being assembled by a worker.
                for _ in loader_iter:
                    pass
            else:
                shutdown = getattr(loader_iter, '_shutdown_workers', None)
                if shutdown is not None:
                    shutdown()

def validate_distributed_calibration_mode(distribution_type, world_size):
    """Allow distributed calibration only for exactly mergeable histograms."""
    if int(world_size) > 1 and distribution_type != 'histogram':
        raise NotImplementedError(
            "Distributed calibration supports only --calibration_distribution_type histogram. "
            "Reservoir and quantile calibration remain single-process because naively merging "
            "their per-rank retained samples is not an exact global reservoir."
        )

def distributed_calibration_batch_limit(max_steps, rank, world_size):
    """Split a global calibration batch cap deterministically across ranks.

    The first ``max_steps % world_size`` ranks receive one extra batch. This
    preserves the single-process meaning of ``--max_steps`` instead of silently
    multiplying calibration volume by the GPU count.
    """
    max_steps = int(max_steps)
    rank = int(rank)
    world_size = int(world_size)
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    quotient, remainder = divmod(max_steps, world_size)
    return quotient + int(rank < remainder)

def _reduce_tensor_to_rank_zero(tensor, world_size, chunk_elements=4 * 1024 * 1024):
    """Sum one dense device tensor onto rank zero in bounded-size chunks."""
    if int(world_size) == 1:
        return
    chunk_elements = int(chunk_elements)
    if chunk_elements <= 0:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")

    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat.narrow(0, start, min(chunk_elements, flat.numel() - start))
        dist.reduce(chunk, dst=0, op=dist.ReduceOp.SUM)

def _wait_for_workflow_ranks(rank, world_size, device, *, phase):
    """Wait for every rank's local workflow phase without a collective.

    All ranks in the default process group must call this with the same
    ``phase`` label, in the same order. Call after completing local work,
    including loader shutdown or CPU fitting, and before its next collective.
    CUDA work on ``device`` is synchronized before publishing readiness. The
    rendezvous uses the group's existing store; checking for missing keys returns
    immediately, so rank completion skew consumes neither a collective nor
    a store key-wait timeout. Individual communication failures still raise.

    There is intentionally no deadline for peer computation. A failed rank
    must be handled by the process supervisor (torchrun in the CLI), which
    terminates the other ranks. Per-rank sequence counters keep completion
    keys from an earlier invocation of a phase from satisfying a later one.
    Different phase labels use independent store namespaces.
    """
    if int(world_size) == 1:
        return
    device = torch.device(device)
    if device.type == 'cuda':
        # A host-side completion marker must also cover queued GPU work.
        torch.cuda.synchronize(device)

    # PyTorch exposes Store/PrefixStore publicly, but retrieving the store
    # already owned by the default group requires this c10d accessor.
    store = dist.PrefixStore(
        f'porefm/workflow_completion/{phase}',
        dist.distributed_c10d._get_default_store(),
    )
    sequence = store.add(f'sequence/{rank}', 1)
    keys = [f'{sequence}/ready/{peer}' for peer in range(world_size)]
    store.set(keys[rank], '1')

    started = time.monotonic()
    next_log = started
    while not store.check(keys):
        now = time.monotonic()
        if now >= next_log:
            pending = [peer for peer, key in enumerate(keys) if not store.check([key])]
            print(
                f'{datetime.datetime.now()}: {phase} rank={rank} local work complete; '
                f'waiting for ranks={pending} before distributed communication '
                f'wait={now - started:.1f}s',
                flush=True,
            )
            next_log = now + 60.0
        time.sleep(0.1)
    if rank == 0:
        print(
            f'{datetime.datetime.now()}: {phase} all {world_size} ranks ready; '
            f'continuing after wait={time.monotonic() - started:.1f}s',
            flush=True,
        )

def reduce_histogram_calibration_distributed(accumulator, rank, world_size, chunk_elements=4 * 1024 * 1024):
    """Merge exact histogram/count state onto rank zero without CPU gathering.

    The dense histogram can exceed one GiB for 9-mers, so reduction is chunked
    to bound each NCCL operation. Non-root tensors are scratch state after this
    function and must not be finalized. All ranks rendezvous through the
    process-group store before the first reduction, allowing arbitrarily
    uneven local calibration workloads without an early collective timeout.
    """
    validate_distributed_calibration_mode(accumulator.distribution_type, world_size)
    if int(world_size) == 1:
        return accumulator if int(rank) == 0 else None

    _wait_for_workflow_ranks(
        rank, world_size, accumulator.hist_counts.device, phase='calibration-histogram',
    )
    for tensor in (
        accumulator.hist_counts,
        accumulator.seen_counts,
        accumulator.hist_underflow_counts,
        accumulator.hist_overflow_counts,
    ):
        _reduce_tensor_to_rank_zero(
            tensor,
            world_size,
            chunk_elements=chunk_elements,
        )
    return accumulator if int(rank) == 0 else None

def merge_accumulators_distributed(accumulators, rank, world_size, device, contexts_per_chunk=8):
    """Merge CPU covariance states onto rank zero in bounded context chunks.

    A 5-mer fit with 512-dimensional full covariance contains roughly two GiB
    of FP64 sufficient statistics per rank. Gathering the complete Python
    object at once can multiply that footprint by world size. Chunked gathers
    preserve the exact merge while bounding transient serialization memory.
    Before each gather, ranks rendezvous using their communication ``device``
    so rank-zero CPU merging cannot leave peers in an early collective.
    """
    if int(world_size) == 1:
        return accumulators
    contexts_per_chunk = int(contexts_per_chunk)
    if contexts_per_chunk <= 0:
        raise ValueError("contexts_per_chunk must be positive")

    for start in range(0, len(accumulators), contexts_per_chunk):
        stop = min(start + contexts_per_chunk, len(accumulators))
        local_states = [acc.state_dict() for acc in accumulators[start:stop]]
        _wait_for_workflow_ranks(
            rank, world_size, device, phase='whitening-covariance-merge',
        )
        gathered = _all_gather_object_if_distributed(local_states, world_size)
        if int(rank) == 0:
            # Rank zero already owns its local states. Merge only other ranks
            # into those accumulators to avoid allocating a second full copy.
            for rank_states in gathered[1:]:
                for offset, state in enumerate(rank_states):
                    accumulators[start + offset].merge(
                        StreamingCovariance.from_state_dict(state)
                    )
    return accumulators if int(rank) == 0 else None

def reduce_count_vector(counts, device, world_size):
    """Sum an integer count vector across ranks and return it on CPU."""
    if world_size == 1:
        return counts.cpu()
    tensor = counts.to(device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu()

def merge_score_reservoir_distributed(reservoir, rank, world_size, seed):
    """Gather and exactly merge per-context uniform score reservoirs on rank zero."""
    gathered = _all_gather_object_if_distributed(reservoir.state_dict(), world_size)
    if rank != 0:
        return None
    merged = PerBaseScoreReservoir(
        reservoir.num_bases,
        reservoir.capacity_per_base,
        seed=seed,
    )
    for state in gathered:
        merged.merge(PerBaseScoreReservoir.from_state_dict(state, seed=seed))
    return merged

def merge_tensor_reservoir_distributed(reservoir, rank, world_size, seed):
    """Gather residual reservoirs and merge retained samples on rank zero."""
    gathered = _all_gather_object_if_distributed(reservoir.state_dict(), world_size)
    if rank != 0:
        return None
    merged = PerBaseTensorReservoir(
        reservoir.num_bases,
        reservoir.dim,
        reservoir.capacity_per_base,
        seed=seed,
        store_weights=reservoir.weights is not None,
    )
    for state in gathered:
        other = PerBaseTensorReservoir.from_state_dict(state, seed=seed)
        for base_index in range(other.num_bases):
            n = int(other.retained_counts[base_index].item())
            if n <= 0:
                continue
            bases = torch.full((n,), base_index, dtype=torch.long)
            weights = other.weights[base_index, :n] if other.weights is not None else None
            merged.update(bases, other.values[base_index, :n], weights=weights)
            merged.seen_counts[base_index] += torch.clamp(other.seen_counts[base_index] - n, min=0)
    return merged

def broadcast_python_object(obj, src=0):
    """Broadcast one picklable object when distributed execution is initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]

def _sample_whitening_tokens(valid_flat, token_sample_rate, generator):
    idx = torch.nonzero(valid_flat, as_tuple=False).flatten()
    sample_rate = float(token_sample_rate)
    if sample_rate <= 0.0 or sample_rate > 1.0 or not math.isfinite(sample_rate):
        raise ValueError(f"whitening_token_sample_rate must be in (0, 1], got {sample_rate}")
    if sample_rate >= 1.0 or idx.numel() == 0:
        return idx
    keep = torch.rand(idx.numel(), device=idx.device, generator=generator) < sample_rate
    return idx[keep]

def _empty_whitening_accumulators(num_bases, hidden_dim):
    return [StreamingCovariance(hidden_dim) for _ in range(num_bases)]


class BufferedGroupedCovariance:
    """Buffer sparse per-context rows before updating full covariance moments.

    K-mer token sampling often contributes only a handful of rows to each of
    many contexts in one model batch. Accumulating those rows immediately would
    launch thousands of tiny 512-by-512 matrix products. This helper transfers
    one selected batch to CPU, groups it, and waits for a configurable number
    of rows before each covariance update.
    """

    def __init__(self, num_contexts, hidden_dim, buffer_rows=128):
        self.num_contexts = int(num_contexts)
        self.hidden_dim = int(hidden_dim)
        self.buffer_rows = int(buffer_rows)
        if self.num_contexts <= 0 or self.hidden_dim <= 0 or self.buffer_rows <= 0:
            raise ValueError("num_contexts, hidden_dim, and buffer_rows must be positive")
        self.accumulators = _empty_whitening_accumulators(
            self.num_contexts,
            self.hidden_dim,
        )
        self._values = [[] for _ in range(self.num_contexts)]
        self._weights = [[] for _ in range(self.num_contexts)]
        self._buffered_counts = [0 for _ in range(self.num_contexts)]

    def _flush_context(self, context_index):
        if self._buffered_counts[context_index] == 0:
            return
        values = torch.cat(self._values[context_index], dim=0)
        weights = None
        if self._weights[context_index]:
            weights = torch.cat(self._weights[context_index], dim=0)
        self.accumulators[context_index].update(values, weights=weights)
        self._values[context_index].clear()
        self._weights[context_index].clear()
        self._buffered_counts[context_index] = 0

    def update(self, context_indices, residuals, weights=None):
        """Append aligned rows and flush contexts reaching ``buffer_rows``."""
        if residuals.numel() == 0:
            return
        context_indices = context_indices.detach().reshape(-1).cpu().long()
        residuals = residuals.detach().reshape(-1, self.hidden_dim).cpu().float()
        if weights is not None:
            weights = weights.detach().reshape(-1).cpu().float()
        valid = (
            (context_indices >= 0)
            & (context_indices < self.num_contexts)
            & torch.isfinite(residuals).all(dim=-1)
        )
        if weights is not None:
            valid = valid & torch.isfinite(weights) & (weights > 0.0)
        if not valid.any():
            return
        context_indices = context_indices[valid]
        residuals = residuals[valid]
        weights = weights[valid] if weights is not None else None
        for context_index in torch.unique(context_indices).tolist():
            context_index = int(context_index)
            mask = context_indices == context_index
            values = residuals[mask]
            self._values[context_index].append(values)
            if weights is not None:
                self._weights[context_index].append(weights[mask])
            self._buffered_counts[context_index] += int(values.shape[0])
            if self._buffered_counts[context_index] >= self.buffer_rows:
                self._flush_context(context_index)

    def finalize(self):
        """Flush all pending rows and return the covariance accumulators."""
        for context_index in range(self.num_contexts):
            self._flush_context(context_index)
        return self.accumulators

def accumulate_whitening_pass(
    args,
    model,
    pretrained_model,
    dataloader,
    device,
    pretrain_config,
    context_names,
    base_token_ids,
    context_parent_base_index,
    rank,
    pass_name,
    pilot_scorer=None,
    cutoff_by_base=None,
    collect_quantiles=False,
):
    """Accumulate one finite pass of residual statistics for whitening.

    Valid tokens may be Bernoulli-sampled and capped by a per-context reservoir.
    With ``collect_quantiles``, pilot whitening scores populate a score
    reservoir; with ``cutoff_by_base``, pilot radii produce robust weights
    ``min(1, cutoff / radius)``. Returns local accumulators/reservoirs plus
    per-context seen, sampled, retained, and downweighted counts.
    """
    hidden_dim = int(pretrain_config.hidden_dim)
    whitening_kmer_size = resolved_whitening_kmer_size(args)
    num_contexts = len(context_names)
    expected_contexts = 4 ** whitening_kmer_size
    if num_contexts != expected_contexts:
        raise ValueError(
            f"Expected {expected_contexts} whitening contexts, got {num_contexts}"
        )
    context_parent_base_index = torch.as_tensor(
        context_parent_base_index,
        dtype=torch.long,
        device=device,
    )
    grouped_covariance = BufferedGroupedCovariance(
        num_contexts,
        hidden_dim,
        buffer_rows=args.whitening_covariance_buffer_rows,
    )
    accumulators = grouped_covariance.accumulators

    score_reservoir = None
    if collect_quantiles:
        score_reservoir = PerBaseScoreReservoir(
            num_contexts,
            capacity_per_base=int(args.robust_quantile_sample_size),
            seed=int(args.sampling_seed) + 7919 * (rank + 1),
        )

    tensor_reservoir = None
    if not collect_quantiles and int(args.max_samples_per_base) > 0:
        tensor_reservoir = PerBaseTensorReservoir(
            num_contexts,
            hidden_dim,
            capacity_per_base=int(args.max_samples_per_base),
            seed=int(args.sampling_seed) + 104729 * (rank + 1),
            store_weights=cutoff_by_base is not None,
        )

    chunk_generator = torch.Generator(device=device)
    chunk_generator.manual_seed(int(args.sampling_seed) + 1299709 * (rank + 1))
    cutoff_tensor = None
    if cutoff_by_base is not None:
        cutoff_tensor = torch.as_tensor(cutoff_by_base, dtype=torch.float32, device=device)

    seen_by_base = torch.zeros(num_contexts, dtype=torch.long)
    sampled_by_base = torch.zeros(num_contexts, dtype=torch.long)
    retained_by_base = torch.zeros(num_contexts, dtype=torch.long)
    downweighted_by_base = torch.zeros(num_contexts, dtype=torch.long)
    batch_num = 0
    t0 = time.time()

    model.eval()
    pretrained_model.eval()

    for batch in iter_bounded_dataloader(dataloader, args.max_steps):
        base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch[:5]
        rna_structure_features = (batch[11].to(device, non_blocking=True)
                                  if requires_rna_structure_features(model) else None)
        dedup_mask = batch[7] if len(batch) > 7 else None
        reference_match_mask = batch[8] if len(batch) > 8 else None
        base_seq_input = base_seq_input.to(device, non_blocking=True)
        ts_input = ts_input.to(device, non_blocking=True)
        percentiles_input = percentiles_input.to(device, non_blocking=True)
        pad_base_seq = pad_base_seq.to(device, non_blocking=True)
        pad_ts = pad_ts.to(device, non_blocking=True)
        if dedup_mask is not None:
            dedup_mask = dedup_mask.to(device, non_blocking=True)
        if reference_match_mask is not None:
            reference_match_mask = reference_match_mask.to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                pretrain_out = pretrained_model(
                    base_seq_input,
                    ts_input,
                    percentiles_input=percentiles_input,
                    pad_base_seq=pad_base_seq,
                    pad_ts=pad_ts,
                    embeddings_only=True,
                )
                null_out = model(pretrain_out['fixed_sa_base'], pad_base_seq, base_seq_input=base_seq_input,
                                  rna_structure_features=rna_structure_features)
            with torch.autocast(device_type=device.type, enabled=False):
                _h_obs, _h_null, residual = canonical_null_residuals(
                    pretrain_out['contextual_base_seq'],
                    null_out['mean_emb'],
                    norm_eps=args.norm_eps,
                )
        if residual.shape[-1] != hidden_dim:
            raise ValueError(f"Residual dim={residual.shape[-1]}, expected hidden_dim={hidden_dim}")

        kmer_ids, kmer_valid = compute_chunk_kmer_ids(
            base_seq_input,
            whitening_kmer_size,
        )
        valid = ignore_edges(pad_base_seq.bool(), args.num_loss_trim[0])
        # A context-specific transform cannot be selected without a complete
        # canonical centered k-mer, so this mask is mandatory for k > 1.
        if args.require_valid_kmer_for_whitening_fit or whitening_kmer_size > 1:
            valid = valid & kmer_valid
        if dedup_mask is not None:
            valid = valid & dedup_mask.bool()
        if reference_match_mask is not None:
            valid = valid & reference_match_mask.bool()

        flat_residual = residual.reshape(-1, hidden_dim)
        flat_valid = valid.reshape(-1)
        context_index_flat = kmer_ids.reshape(-1).long()
        flat_valid = (
            flat_valid
            & (context_index_flat >= 0)
            & (context_index_flat < num_contexts)
            & torch.isfinite(flat_residual).all(dim=-1)
        )
        valid_base_index = context_index_flat[flat_valid]
        if valid_base_index.numel() > 0:
            seen_by_base += torch.bincount(
                valid_base_index.detach().cpu(),
                minlength=num_contexts,
            ).long()

        selected_idx = _sample_whitening_tokens(
            flat_valid,
            args.whitening_token_sample_rate,
            chunk_generator,
        )
        if selected_idx.numel() == 0:
            batch_num += 1
            continue

        selected_base_index = context_index_flat[selected_idx]
        selected_residual = flat_residual[selected_idx].detach()
        sampled_by_base += torch.bincount(
            selected_base_index.detach().cpu(),
            minlength=num_contexts,
        ).long()

        weights = None
        if collect_quantiles:
            if pilot_scorer is None:
                raise ValueError("collect_quantiles requires pilot_scorer")
            with torch.autocast(device_type=device.type, enabled=False):
                pilot_scores = pilot_scorer.score(
                    residual.float(),
                    base_seq_input,
                    valid_mask=valid,
                    chunk_size=args.whitening_score_chunk_size,
                )
            score_reservoir.update(selected_base_index, pilot_scores.reshape(-1)[selected_idx])
        else:
            if cutoff_tensor is not None:
                if pilot_scorer is None:
                    raise ValueError("Weighted robust fitting requires a pilot_scorer")
                with torch.autocast(device_type=device.type, enabled=False):
                    pilot_scores = pilot_scorer.score(
                        residual.float(),
                        base_seq_input,
                        valid_mask=valid,
                        chunk_size=args.whitening_score_chunk_size,
                    )
                q = pilot_scores.reshape(-1)[selected_idx].float()
                cutoff = cutoff_tensor[selected_base_index]
                weights = torch.minimum(torch.ones_like(q), cutoff / q.clamp_min(args.robust_eps))
                downweighted_by_base += torch.bincount(
                    selected_base_index[weights < 1.0].detach().cpu(),
                    minlength=num_contexts,
                ).long()
            if tensor_reservoir is not None:
                tensor_reservoir.update(selected_base_index, selected_residual, weights=weights)
            else:
                grouped_covariance.update(
                    selected_base_index,
                    selected_residual,
                    weights=weights,
                )
                retained_by_base += torch.bincount(
                    selected_base_index.detach().cpu(),
                    minlength=num_contexts,
                ).long()

        batch_num += 1
        if rank == 0 and (batch_num == 1 or batch_num % args.log_every == 0):
            elapsed = time.time() - t0
            print(
                f"{str(datetime.datetime.now())}: {pass_name} batches={batch_num} "
                f"tokens_seen={int(seen_by_base.sum().item()):,} "
                f"tokens_sampled={int(sampled_by_base.sum().item()):,} "
                f"seen_tok/s={int(seen_by_base.sum().item()) / max(elapsed, 1e-9):,.0f}",
                flush=True,
            )
    if tensor_reservoir is not None:
        retained_by_base = tensor_reservoir.retained_counts.clone()
        accumulators = tensor_reservoir.to_accumulators()
    else:
        accumulators = grouped_covariance.finalize()

    parent_accumulators = _empty_whitening_accumulators(4, hidden_dim)
    context_parent_cpu = context_parent_base_index.cpu()
    for context_index, accumulator in enumerate(accumulators):
        parent_accumulators[int(context_parent_cpu[context_index].item())].merge(accumulator)

    return {
        "accumulators": accumulators,
        "parent_accumulators": parent_accumulators,
        "score_reservoir": score_reservoir,
        "tensor_reservoir": tensor_reservoir,
        "seen_by_base": seen_by_base,
        "sampled_by_base": sampled_by_base,
        "retained_by_base": retained_by_base,
        "downweighted_by_base": downweighted_by_base,
        "batches_seen": batch_num,
    }

def build_whitening_artifact_from_pass(
    args,
    pass_result,
    context_names,
    context_parent_base_index,
    pretrain_config,
    robust_settings,
):
    """Fit a whitening artifact from a completed pass and attach sampling counts."""
    whitening_kmer_size = resolved_whitening_kmer_size(args)
    fit_config = {
        'bam_config': args.bam_config,
        'seq_type': args.seq_type,
        'whitening_kmer_size': whitening_kmer_size,
        'scale_kmer_size': resolved_scale_kmer_size(args),
        'calibration_kmer_size': resolved_calibration_kmer_size(args),
        'num_trim': args.num_loss_trim[0],
        'covariance_mode': args.covariance_mode,
        'shrinkage_mode': args.shrinkage_mode,
        'shrinkage': args.shrinkage,
        'parent_shrinkage_strength_requested': args.whitening_parent_shrinkage_strength,
        'relative_eigenvalue_floor': args.relative_eigenvalue_floor,
        'max_condition_number': args.max_condition_number,
        'max_samples_per_base': args.max_samples_per_base,
        'whitening_token_sample_rate': args.whitening_token_sample_rate,
        'sampling_seed': args.sampling_seed,
        'reference_match_kmer_size': int(getattr(args, 'reference_match_kmer_size', 0)),
        'require_valid_kmer_for_whitening_fit': args.require_valid_kmer_for_whitening_fit,
        'covariance_buffer_rows': args.whitening_covariance_buffer_rows,
    }
    artifact = fit_context_whitening_from_accumulators(
        pass_result["accumulators"],
        parent_accumulators=pass_result["parent_accumulators"],
        whitening_kmer_size=whitening_kmer_size,
        seq_type=args.seq_type,
        hidden_dim=pretrain_config.hidden_dim,
        covariance_mode=args.covariance_mode,
        shrinkage_mode=args.shrinkage_mode,
        shrinkage_alpha=args.shrinkage,
        parent_shrinkage_strength=args.whitening_parent_shrinkage_strength,
        relative_eigenvalue_floor=args.relative_eigenvalue_floor,
        max_condition_number=args.max_condition_number,
        robust_settings=robust_settings,
        fit_configuration=fit_config,
        model_checkpoint_identifier=args.pretrain_load_path or args.model_load_path or '',
        null_predictor_identifier=args.model_load_path or '',
        chemistry=args.pore_type,
        molecule_type=args.seq_type,
        normalization_configuration={
            'norm_eps': args.norm_eps,
            'score_log_eps': args.score_log_eps,
        },
        residual_definition={
            'h_obs': 'F.normalize(contextual_base_seq.float(), dim=-1, eps=norm_eps)',
            'h_null': 'F.normalize(predicted_canonical_embedding.float(), dim=-1, eps=norm_eps)',
            'residual': 'h_obs - h_null',
            'hidden_dim_asserted': int(pretrain_config.hidden_dim),
        },
        pretrain_config=pretrain_config,
        include_raw_covariance=args.include_raw_covariance,
        include_regularized_covariance=whitening_kmer_size == 1,
    )
    artifact['preprocessing'] = getattr(args, 'preprocessing', None)
    artifact['seen_counts'] = pass_result["seen_by_base"].cpu()
    artifact['sampled_counts'] = pass_result["sampled_by_base"].cpu()
    artifact['retained_counts'] = pass_result["retained_by_base"].cpu()
    artifact['downweighted_counts'] = pass_result["downweighted_by_base"].cpu()
    for idx, context_name in enumerate(context_names):
        diag = artifact['diagnostics'][str(context_name)]
        diag['seen_count'] = int(pass_result["seen_by_base"][idx].item())
        diag['sampled_count'] = int(pass_result["sampled_by_base"][idx].item())
        diag['retained_count'] = int(pass_result["retained_by_base"][idx].item())
        diag['downweighted_count'] = int(pass_result["downweighted_by_base"][idx].item())
    return artifact

def run_fit_whitening(args, model, pretrained_model, device, pretrain_config, rank, world_size, is_main_process):
    """Run ordinary or robust multi-pass whitening fitting and save on rank zero.

    Robust fitting first constructs a pilot transform, optionally estimates
    base-specific radius cutoffs, and then performs a weighted covariance pass.
    Store rendezvous separate local passes and rank-zero fitting from the
    following collectives, without imposing a peer-computation timeout.
    """
    if not args.whitening_output:
        raise ValueError("--whitening_output is required for stage fit-whitening")
    if args.model_load_path is None:
        raise ValueError("--model_load_path is required for stage fit-whitening")
    if (
        float(args.whitening_token_sample_rate) <= 0.0
        or float(args.whitening_token_sample_rate) > 1.0
        or not math.isfinite(float(args.whitening_token_sample_rate))
    ):
        raise ValueError(
            f"--whitening_token_sample_rate must be in (0, 1], got {args.whitening_token_sample_rate}"
        )
    if args.robust_fit and args.shrinkage_mode == 'oas':
        raise ValueError("OAS shrinkage is not supported with robust weighted fitting; use --disable_robust_fit or --shrinkage_mode fixed.")

    whitening_kmer_size = resolved_whitening_kmer_size(args)
    if whitening_kmer_size > 1 and int(args.max_samples_per_base) > 0:
        raise ValueError(
            "--max_samples_per_base would allocate that reservoir independently for every "
            "whitening k-mer. Use --max_samples_per_base 0 and control volume with "
            "--whitening_token_sample_rate."
        )
    if whitening_kmer_size > 1 and not args.require_valid_kmer_for_whitening_fit:
        raise ValueError(
            "--allow_invalid_kmer_for_whitening_fit is incompatible with k-mer-specific whitening"
        )

    base_names, base_token_ids = canonical_base_groups(args.seq_type)
    context_names, context_parent_base_index = canonical_kmer_contexts(
        whitening_kmer_size,
        args.seq_type,
    )
    kmer_table = resolve_cli_preprocessing(args).kmer_table

    def make_loader():
        """Create a fresh finite loader for each whitening pass."""
        return make_scale_bam_dataloader(
            args,
            pretrain_config,
            kmer_table,
            rank,
            world_size,
            drop_last=False,
            use_rna_structure_features=requires_rna_structure_features(model),
        )

    if is_main_process:
        print(
            {
                'mode': 'fit-whitening',
                'bases': base_names,
                'whitening_kmer_size': whitening_kmer_size,
                'num_whitening_contexts': len(context_names),
                'covariance_mode': args.covariance_mode,
                'shrinkage_mode': args.shrinkage_mode,
                'shrinkage': args.shrinkage,
                'relative_eigenvalue_floor': args.relative_eigenvalue_floor,
                'max_condition_number': args.max_condition_number,
                'robust_fit': bool(args.robust_fit),
                'whitening_token_sample_rate': float(args.whitening_token_sample_rate),
            },
            flush=True,
        )

    first_pass = accumulate_whitening_pass(
        args,
        model,
        pretrained_model,
        make_loader(),
        device,
        pretrain_config,
        context_names,
        base_token_ids,
        context_parent_base_index,
        rank,
        pass_name='whitening-pass1',
    )
    _wait_for_workflow_ranks(rank, world_size, device, phase='whitening-pass1')
    first_pass["seen_by_base"] = reduce_count_vector(first_pass["seen_by_base"], device, world_size)
    first_pass["sampled_by_base"] = reduce_count_vector(first_pass["sampled_by_base"], device, world_size)
    first_pass["retained_by_base"] = reduce_count_vector(first_pass["retained_by_base"], device, world_size)
    first_pass["downweighted_by_base"] = reduce_count_vector(first_pass["downweighted_by_base"], device, world_size)
    first_pass["accumulators"] = merge_accumulators_distributed(first_pass["accumulators"], rank, world_size, device)
    first_pass["parent_accumulators"] = merge_accumulators_distributed(
        first_pass["parent_accumulators"], rank, world_size, device
    )
    if rank != 0:
        first_pass["accumulators"] = None

    robust_settings = {
        'enabled': bool(args.robust_fit),
        'robust_quantile': args.robust_quantile,
        'robust_cutoff': args.robust_cutoff,
        'robust_quantile_sample_size': args.robust_quantile_sample_size,
        'robust_eps': args.robust_eps,
    }

    if not args.robust_fit:
        artifact = None
        if is_main_process:
            artifact = build_whitening_artifact_from_pass(
                args,
                first_pass,
                context_names,
                context_parent_base_index,
                pretrain_config,
                robust_settings,
            )
            os.makedirs(os.path.dirname(args.whitening_output) or '.', exist_ok=True)
            save_whitening_artifact(artifact, args.whitening_output)
            counts = artifact['counts']
            print(
                {
                    'whitening_output': args.whitening_output,
                    'whitening_kmer_size': whitening_kmer_size,
                    'num_contexts': len(context_names),
                    'min_context_count': int(counts.min().item()),
                    'median_context_count': int(counts.median().item()),
                    'max_context_count': int(counts.max().item()),
                    'parent_shrinkage_strength': artifact['parent_shrinkage_strength'],
                    'max_parent_shrinkage': float(artifact['parent_shrinkage'].max().item()),
                },
                flush=True,
            )
        # Keep peers and their process groups alive through rank-zero fitting
        # and artifact I/O, which can outlast the communication timeout.
        _wait_for_workflow_ranks(
            rank, world_size, device, phase='whitening-artifact-saved',
        )
        return

    pilot_artifact = None
    if is_main_process:
        pilot_artifact = build_whitening_artifact_from_pass(
            args,
            first_pass,
            context_names,
            context_parent_base_index,
            pretrain_config,
            dict(robust_settings, pilot=True),
        )
    _wait_for_workflow_ranks(rank, world_size, device, phase='whitening-pilot-fit')
    pilot_artifact = broadcast_python_object(pilot_artifact, src=0)
    pilot_scorer = BaseSpecificWhiteningScorer(
        pilot_artifact,
        device=device,
        expected_hidden_dim=pretrain_config.hidden_dim,
    )

    if args.robust_cutoff > 0:
        cutoff_by_base = torch.full((len(context_names),), float(args.robust_cutoff), dtype=torch.float32)
    else:
        quantile_pass = accumulate_whitening_pass(
            args,
            model,
            pretrained_model,
            make_loader(),
            device,
            pretrain_config,
            context_names,
            base_token_ids,
            context_parent_base_index,
            rank,
            pass_name='whitening-robust-quantile',
            pilot_scorer=pilot_scorer,
            collect_quantiles=True,
        )
        _wait_for_workflow_ranks(
            rank, world_size, device, phase='whitening-robust-quantile',
        )
        merged_reservoir = merge_score_reservoir_distributed(
            quantile_pass["score_reservoir"],
            rank,
            world_size,
            seed=int(args.sampling_seed) + 31,
        )
        cutoff_by_base = None
        if is_main_process:
            cutoff_by_base = merged_reservoir.quantiles(args.robust_quantile)
            if not torch.isfinite(cutoff_by_base).all():
                raise ValueError("Could not estimate finite robust cutoffs for all bases")
        _wait_for_workflow_ranks(
            rank, world_size, device, phase='whitening-robust-cutoffs',
        )
        cutoff_by_base = broadcast_python_object(cutoff_by_base, src=0)

    robust_settings['cutoff_by_context'] = [
        float(x) for x in torch.as_tensor(cutoff_by_base).cpu().tolist()
    ]
    weighted_pass = accumulate_whitening_pass(
        args,
        model,
        pretrained_model,
        make_loader(),
        device,
        pretrain_config,
        context_names,
        base_token_ids,
        context_parent_base_index,
        rank,
        pass_name='whitening-robust-weighted',
        pilot_scorer=pilot_scorer,
        cutoff_by_base=cutoff_by_base,
    )
    _wait_for_workflow_ranks(
        rank, world_size, device, phase='whitening-robust-weighted',
    )
    weighted_pass["seen_by_base"] = reduce_count_vector(weighted_pass["seen_by_base"], device, world_size)
    weighted_pass["sampled_by_base"] = reduce_count_vector(weighted_pass["sampled_by_base"], device, world_size)
    weighted_pass["retained_by_base"] = reduce_count_vector(weighted_pass["retained_by_base"], device, world_size)
    weighted_pass["downweighted_by_base"] = reduce_count_vector(weighted_pass["downweighted_by_base"], device, world_size)
    weighted_pass["accumulators"] = merge_accumulators_distributed(weighted_pass["accumulators"], rank, world_size, device)
    weighted_pass["parent_accumulators"] = merge_accumulators_distributed(
        weighted_pass["parent_accumulators"], rank, world_size, device
    )
    if rank != 0:
        weighted_pass["accumulators"] = None

    if is_main_process:
        artifact = build_whitening_artifact_from_pass(
            args,
            weighted_pass,
            context_names,
            context_parent_base_index,
            pretrain_config,
            robust_settings,
        )
        os.makedirs(os.path.dirname(args.whitening_output) or '.', exist_ok=True)
        save_whitening_artifact(artifact, args.whitening_output)
        print({'whitening_output': args.whitening_output, 'num_contexts': len(context_names)}, flush=True)
    _wait_for_workflow_ranks(
        rank, world_size, device, phase='whitening-artifact-saved',
    )

def run_calibration(
    args,
    model,
    scale_model,
    pretrained_model,
    dataloader,
    device,
    pretrain_config,
    rank,
    world_size,
    is_main_process,
):
    """Accumulate a finite per-k-mer score calibration and save it on rank zero.

    Histogram state is exactly summed across ranks. ``args.max_steps`` is a
    global batch cap divided deterministically among ranks; reservoir and
    quantile modes are rejected for distributed execution. Ranks finish their
    local loaders independently and rendezvous before the histogram merge.
    """
    if not args.calibration_output:
        raise ValueError("--calibration_output is required for stage calibrate")
    validate_distributed_calibration_mode(
        args.calibration_distribution_type,
        world_size,
    )
    local_max_steps = distributed_calibration_batch_limit(
        args.max_steps,
        rank,
        world_size,
    )

    model.eval()
    if scale_model is not None:
        scale_model.eval()
    pretrained_model.eval()
    score_type = requested_scale_score_type(args)
    if scale_model is not None:
        score_type = normalize_score_type(getattr(unwrap_model(scale_model).config, 'score_type', RAW_COSINE_SCORE_TYPE))
        scale_module_for_score = unwrap_model(scale_model)
    else:
        score_config = type("_ScoreConfig", (), {"score_type": score_type})()
        scale_module_for_score = type("_ScoreConfigHolder", (), {"config": score_config})()
    whitening_scorer = make_whitening_scorer(
        args,
        pretrain_config,
        device,
        required=score_type == RAW_WHITENING_SCORE_TYPE,
    )
    calibration_kmer_size = resolved_calibration_kmer_size(args)
    accumulator = CanonicalCalibrationAccumulator(
        kmer_size=calibration_kmer_size,
        max_scores_per_kmer=args.max_scores_per_kmer,
        seed=args.seed,
        distribution_type=args.calibration_distribution_type,
        hist_min=args.hist_min,
        hist_max=args.hist_max,
        hist_bins=args.hist_bins,
        quantile_body_count=args.quantile_body_count,
        quantile_tail_count=args.quantile_tail_count,
        quantile_tail_min_prob=args.quantile_tail_min_prob,
        device=device,
    )
    calibration_score_type = score_type
    if scale_model is not None:
        calibration_score_type = standardized_score_type(score_type)
    # Calibration must sample the same boundary-conditioned score population
    # that inference emits. The CNN radius is accommodated through overlap and
    # best-margin deduplication rather than by deleting true read-end tokens.
    calibration_base_trim = int(args.num_loss_trim[0])

    total_tokens = 0
    batch_num = 0
    t = time.time()
    last_batch_end = t
    data_wait_time = 0.0
    forward_time = 0.0
    accumulator_time = 0.0

    for batch in iter_bounded_dataloader(dataloader, local_max_steps):
        batch_start = time.time()
        data_wait_time += batch_start - last_batch_end
        base_seq_input, ts_input, percentiles_input, pad_base_seq, pad_ts = batch[:5]
        base_quality = batch[5] if len(batch) > 5 else None
        read_global_features = batch[6] if len(batch) > 6 else None
        dedup_mask = batch[7] if len(batch) > 7 else None
        reference_match_mask = batch[8] if len(batch) > 8 else None
        rna_structure_features = batch[11]
        if rna_structure_features is not None:
            rna_structure_features = rna_structure_features.to(device, non_blocking=True)
        base_seq_input = base_seq_input.to(device, non_blocking=True)
        ts_input = ts_input.to(device, non_blocking=True)
        percentiles_input = percentiles_input.to(device, non_blocking=True)
        pad_base_seq = pad_base_seq.to(device, non_blocking=True)
        pad_ts = pad_ts.to(device, non_blocking=True)
        if base_quality is not None:
            base_quality = base_quality.to(device, non_blocking=True)
        if read_global_features is not None:
            read_global_features = read_global_features.to(device, non_blocking=True)
        if dedup_mask is not None:
            dedup_mask = dedup_mask.to(device, non_blocking=True)
        if reference_match_mask is not None:
            reference_match_mask = reference_match_mask.to(device, non_blocking=True)

        forward_start = time.time()
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                pretrain_out = pretrained_model(
                    base_seq_input,
                    ts_input,
                    percentiles_input=percentiles_input,
                    pad_base_seq=pad_base_seq,
                    pad_ts=pad_ts,
                    embeddings_only=True,
                )
                null_out = model(pretrain_out['fixed_sa_base'], pad_base_seq, base_seq_input=base_seq_input,
                                  rna_structure_features=rna_structure_features)
                raw_scores, _residual = compute_residual_and_score_for_scale(
                    scale_module_for_score,
                    whitening_scorer,
                    pretrain_out,
                    null_out,
                    base_seq_input,
                    valid_mask=pad_base_seq.bool(),
                )
            scores = raw_scores.float()
            scale_inputs = None
            if scale_model is not None:
                scale_module = unwrap_model(scale_model)
                scale_inputs = build_scale_model_inputs(
                    pretrained_model,
                    base_seq_input,
                    pad_base_seq,
                    pad_ts,
                    scale_module.config.kmer_size,
                    base_quality=base_quality,
                    read_global_features=read_global_features,
                    use_base_quality_features=scale_module.config.use_base_quality_features,
                    scale_config=scale_module.config,
                    read_positions=batch[9], read_lengths=batch[10],
                    rna_structure_features=rna_structure_features,
                )
                scale_out = scale_model(
                    pretrain_out['fixed_sa_base'],
                    scale_inputs['kmer_ids'],
                    scale_inputs['token_features'],
                    scale_inputs['global_features'],
                    pad_base_seq=pad_base_seq,
                    base_seq_input=base_seq_input,
                    h_null=null_out['mean_emb'],
                )
                scores = standardize_canonical_scores(
                    raw_scores,
                    scale_out,
                    eps=scale_module.config.eps,
                    score_type=scale_module.config.score_type,
                    hidden_dim=pretrain_config.hidden_dim,
                )
        forward_time += time.time() - forward_start

        if scale_inputs is not None and scale_module.config.kmer_size == calibration_kmer_size:
            kmer_ids = scale_inputs['kmer_ids']
            kmer_valid = scale_inputs['kmer_valid']
        else:
            kmer_ids, kmer_valid = compute_chunk_kmer_ids(base_seq_input, calibration_kmer_size)
        valid_pos = (
            ignore_edges(pad_base_seq.bool(), calibration_base_trim)
            & kmer_valid
            & torch.isfinite(scores)
        )
        if dedup_mask is not None:
            valid_pos = valid_pos & dedup_mask.bool()
        if reference_match_mask is not None:
            valid_pos = valid_pos & reference_match_mask.bool()
        accumulator_start = time.time()
        accumulator.update(kmer_ids[valid_pos], scores[valid_pos])
        accumulator_time += time.time() - accumulator_start

        accepted = int(valid_pos.sum().item())
        total_tokens += accepted
        batch_num += 1
        last_batch_end = time.time()

        if is_main_process and (batch_num == 1 or batch_num % args.log_every == 0):
            covered = int((accumulator.seen_counts > 0).sum().item())
            elapsed = time.time() - t
            print(
                f"{str(datetime.datetime.now())}: calibration rank0_batches={batch_num} "
                f"tokens={total_tokens:,} covered_kmers={covered:,}/{accumulator.num_kmers:,} "
                f"tok/s={total_tokens / max(elapsed, 1e-9):,.0f} "
                f"data_wait={data_wait_time:.2f}s forward={forward_time:.2f}s "
                f"accumulator={accumulator_time:.2f}s time={elapsed:.2f}s",
                flush=True,
            )

    reached_step_limit = int(batch_num >= local_max_steps)
    if world_size > 1:
        accumulator = reduce_histogram_calibration_distributed(
            accumulator,
            rank,
            world_size,
        )

        run_counts = torch.tensor(
            [total_tokens, batch_num, reached_step_limit],
            dtype=torch.long,
            device=device,
        )
        dist.reduce(run_counts, dst=0, op=dist.ReduceOp.SUM)
        if is_main_process:
            total_tokens = int(run_counts[0].item())
            batch_num = int(run_counts[1].item())
            ranks_reached_step_limit = int(run_counts[2].item())
    else:
        ranks_reached_step_limit = reached_step_limit

    if is_main_process:
        accumulated_tokens = int(accumulator.seen_counts.sum().item())
        if accumulated_tokens != total_tokens:
            raise RuntimeError(
                "Calibration token accounting mismatch after aggregation: "
                f"histogram/reservoir saw {accumulated_tokens:,}, loop counted {total_tokens:,}."
            )
        payload = accumulator.finalize(
            metadata={
                'preprocessing': getattr(args, 'preprocessing', None),
                'bam_config': args.bam_config,
                'model_load_path': args.model_load_path,
                'pretrain_load_path': args.pretrain_load_path,
                'num_trim': calibration_base_trim,
                'requested_num_trim': int(args.num_loss_trim[0]),
                'kmer_size': calibration_kmer_size,
                'scale_kmer_size': (
                    int(unwrap_model(scale_model).config.kmer_size)
                    if scale_model is not None
                    else resolved_scale_kmer_size(args)
                ),
                'scale_kmer_embedding_dim': (
                    int(unwrap_model(scale_model).config.kmer_embedding_dim)
                    if scale_model is not None
                    else None
                ),
                'calibration_kmer_size': calibration_kmer_size,
                'reference_match_kmer_size': int(
                    getattr(args, 'reference_match_kmer_size', 0)
                ),
                'whitening_kmer_size': (
                    int(whitening_scorer.whitening_kmer_size)
                    if whitening_scorer is not None
                    else None
                ),
                'total_tokens': total_tokens,
                'batches_seen': batch_num,
                'max_steps_requested': int(args.max_steps),
                'max_steps_scope': 'global',
                'distributed_world_size': int(world_size),
                'ranks_reached_step_limit': ranks_reached_step_limit,
                'score_type': calibration_score_type,
                'raw_score_type': score_type,
                'whitening_artifact': getattr(args, 'whitening_artifact', '') if score_type == RAW_WHITENING_SCORE_TYPE else '',
                'use_base_quality_scale_features': bool(getattr(args, 'use_base_quality_scale_features', False)),
                'scale_feature_schema_version': (
                    int(unwrap_model(scale_model).config.feature_schema_version)
                    if scale_model is not None
                    else None
                ),
                'scale_rna_structure_features': (
                    bool(unwrap_model(scale_model).config.use_rna_structure_features)
                    if scale_model is not None else False
                ),
                'null_rna_structure_features': requires_rna_structure_features(model),
                'null_context_radius': int(getattr(unwrap_model(model).config, 'context_radius', 0)),
                'scale_context_radius': (
                    scale_context_radius(unwrap_model(scale_model))
                    if scale_model is not None
                    else 0
                ),
                'scale_context_radius_affects_trim': False,
                'distribution_type': accumulator.distribution_type,
                'max_scores_per_kmer': args.max_scores_per_kmer,
                'deduplicate_read_positions': True,
                'dedup_min_valid_margin': max(
                    calibration_base_trim,
                    calibration_kmer_size // 2,
                ),
                'finite_bam_pass': ranks_reached_step_limit == 0,
                'hist_min': args.hist_min,
                'hist_max': args.hist_max,
                'hist_bins': args.hist_bins,
                'quantile_body_count': args.quantile_body_count,
                'quantile_tail_count': args.quantile_tail_count,
                'quantile_tail_min_prob': args.quantile_tail_min_prob,
                'pretrain_config': asdict(pretrain_config),
            }
        )
        os.makedirs(os.path.dirname(args.calibration_output) or '.', exist_ok=True)
        torch.save(payload, args.calibration_output)
        if payload.get('distribution_type') == 'histogram':
            stored_kmers = int((payload['stored_counts'] > 0).sum().item())
        elif payload.get('distribution_type') == 'quantile':
            stored_kmers = len(payload['quantile_scores_by_kmer_id'])
        else:
            stored_kmers = len(payload['scores_by_kmer_id'])
        print(
            {
                'calibration_output': args.calibration_output,
                'total_tokens': total_tokens,
                'observed_kmers': int((payload['seen_counts'] > 0).sum().item()),
                'stored_kmers': stored_kmers,
                'distribution_type': payload.get('distribution_type', 'reservoir'),
            },
            flush=True,
        )

def build_training_scheduler(optimizer, args):
    """Keep constant LR independent of checkpoint frequency when requested."""
    if getattr(args, 'lr_schedule', 'legacy') == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return get_warmup_cosine_schedule(
        optimizer, args.checkpoint_batch_step, 50 * args.checkpoint_batch_step,
        args.max_steps, start_factor=0.01, min_factor=0.05,
    )


def run_finite_evaluation(args, model, scale_model, backbone, dataloader,
                          whitening_scorer, device, rank, world_size, *,
                          max_batches, phase, progress_every=0):
    """Evaluate uneven finite shards and reduce target-weighted totals at EOF.

    No DDP forwards or periodic collectives run inside the batch loop. After
    local EOF (or the per-rank batch cap) and loader shutdown, all ranks join
    the store rendezvous before one final reduction. Empty shards contribute
    zero; a globally empty target set raises. Model training modes are restored.
    This is shared by standalone evaluation and held-out checkpoint selection.
    """
    null = unwrap_model(model)
    scale = unwrap_model(scale_model) if scale_model is not None else None
    modes = (null.training, scale.training if scale is not None else None)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    try:
        null.eval()
        if scale is not None:
            scale.eval()
        for batch_index, batch in enumerate(iter_bounded_dataloader(dataloader, max_batches), 1):
            eligible = batch[7] & batch[8]
            if args.mode == 'train-scale':
                *metrics, count = scale_training_step(
                    scale, null, backbone, batch, None, None, device, True,
                    num_base_trim=args.num_loss_trim[0],
                    whitening_scorer=whitening_scorer, loss_mask=eligible,
                )
            else:
                metrics = training_step(null, backbone, batch, None, None, device, True,
                                        num_base_trim=args.num_loss_trim[0], loss_mask=eligible)[:3]
                mask = ignore_edges(batch[3].bool(), args.num_loss_trim[0]) & eligible
                count = int(mask.sum())
            if count:
                # The scaler's denominator comes from its loss, after whitening
                # validity and finite-target filtering, not a surrogate mask.
                totals[:3] += torch.tensor(metrics, device=device, dtype=torch.float64) * count
                totals[3] += count
            if progress_every > 0 and batch_index % progress_every == 0:
                print(f'{phase} rank={rank} local_batches={batch_index} '
                      f'valid_targets={int(totals[3].item())}', flush=True)
        if world_size > 1:
            _wait_for_workflow_ranks(rank, world_size, device, phase=phase)
            dist.all_reduce(totals)
        if totals[3].item() == 0 or not torch.isfinite(totals).all():
            raise ValueError(f"{phase} has no eligible tokens or has non-finite metrics")
        return {"loss": (totals[0] / totals[3]).item(),
                "aux_1": (totals[1] / totals[3]).item(),
                "aux_2": (totals[2] / totals[3]).item(), "tokens": int(totals[3].item())}
    finally:
        null.train(modes[0])
        if scale is not None:
            scale.train(modes[1])


def run_heldout_validation(args, model, scale_model, backbone, config, kmer_table,
                           whitening_scorer, device, rank, world_size):
    """Evaluate a reproducible finite validation subset, preserving all RNGs.

    The shared evaluation path handles uneven shards and uses actual loss
    target counts for checkpoint selection. Calibration remains independent.
    """
    val_args = copy.copy(args)
    val_args.bam_config = args.validation_config
    val_args.batch_size = args.validation_batch_size
    val_args.eval_only = True
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        loader = make_scale_bam_dataloader(
            val_args, config, kmer_table, rank, world_size, drop_last=False,
            use_rna_structure_features=requires_rna_structure_features(
                model, scale_model if args.mode == 'train-scale' else None,
            ),
        )
        return run_finite_evaluation(
            val_args, model, scale_model, backbone, loader, whitening_scorer,
            device, rank, world_size, max_batches=args.validation_steps,
            phase='heldout-validation',
        )
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def train_model(args):
    """Dispatch canonical-null training, scale training, whitening, or calibration.

    The function initializes distributed execution, validates mode-specific
    model/data requirements, owns rank-zero logs/checkpoints, and runs the
    selected workflow until its configured finite pass or maximum step count.
    """
    resolve_cli_preprocessing(args)
    validate_rna_structure_request(args)
    eval_only=args.eval_only
    model_save_path, batch_size = args.model_save_path, args.batch_size
    
    max_steps=args.max_steps
    checkpoint_batch_step, lr = args.checkpoint_batch_step, args.lr
    num_base_trim, num_signal_trim = args.num_loss_trim


    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    from porefm.common.run_outputs import prepare_run_outputs
    output_dirs, output_files = {}, {}
    if args.mode in {'train', 'train-scale'} and not eval_only:
        output_dirs['model_save_path'] = model_save_path
    elif args.mode == 'calibrate':
        output_files['calibration_output'] = args.calibration_output
    elif args.mode == 'fit-whitening':
        output_files['whitening_output'] = args.whitening_output
    prepare_run_outputs(
        args, command=f"zero-shot {args.mode}", directories=output_dirs,
        files=output_files, distributed=dist,
        output_arguments={'model_save_path', 'calibration_output', 'whitening_output'},
    )

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if args.mode == 'calibrate':
        validate_distributed_calibration_mode(
            args.calibration_distribution_type,
            world_size,
        )

    writer = NoOpWriter()
    log_file = None
    log_writer = None

    if args.mode in {'train', 'train-scale'} and not eval_only:
        if not model_save_path:
            raise ValueError(f"--model_save_path is required for stage {args.mode}")
        if is_main_process:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(model_save_path, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'runs'))
            log_file_path = os.path.join(model_save_path, 'training_log.csv')
            log_file = open(log_file_path, 'w', newline='')
            log_writer = csv.writer(log_file)
            log_writer.writerow(['checkpoint', 'task', 'avg_loss', 'aux_1', 'aux_2'])

    pretrained_model, model, scale_model, pretrain_config = load_models_for_args(args, device)
    args.size = min(
        pretrain_config.max_ts_len,
        args.size or pretrain_config.max_ts_len,
    )
    requested_step = (
        None
        if getattr(args, 'chunk_step_source', None) == 'automatic'
        else args.step
    )
    minimum_signal_overlap = (
        required_scale_signal_overlap(pretrain_config, scale_model)
        if scale_model is not None and args.mode in {'train-scale', 'calibrate'}
        else 0
    )
    if requires_rna_structure_features(model):
        minimum_signal_overlap = max(minimum_signal_overlap, required_scale_signal_overlap(pretrain_config, unwrap_model(model)))
    args.step = resolve_downstream_step(
        args.size,
        args.seq_type,
        requested_step,
        minimum_overlap=minimum_signal_overlap,
        context=f"zero-shot {args.mode}",
        announce=(
            is_main_process
            and (
                not hasattr(args, "chunk_step_source")
                or args.step is None
            )
        ),
    )
    if not hasattr(args, "chunk_step_source"):
        args.chunk_step_source = (
            "explicit" if requested_step is not None else "automatic"
        )

    if args.mode == 'train-scale' and args.model_load_path is None:
        raise ValueError("--model_load_path is required for stage train-scale so the null mean predictor is already trained.")

    scale_kmer_size = resolved_scale_kmer_size(args)
    calibration_kmer_size = resolved_calibration_kmer_size(args)
    whitening_kmer_size = resolved_whitening_kmer_size(args)
    args.scale_kmer_size = scale_kmer_size
    args.calibration_kmer_size = calibration_kmer_size
    args.whitening_kmer_size = whitening_kmer_size

    if scale_model is not None and scale_kmer_size != scale_model.config.kmer_size:
        raise ValueError(
            f"Scale model kmer_size={scale_model.config.kmer_size}, but current run requested "
            f"scale_kmer_size={scale_kmer_size}. Use the checkpoint's kmer_size or load a matching scale model."
        )

    if scale_model is not None:
        validate_scale_config_for_backbone(
            unwrap_model(scale_model).config,
            pretrain_config,
            "Scale model for current run",
            expected_use_base_quality_features=(
                bool(getattr(args, 'use_base_quality_scale_features', False))
                if args.mode in {'train-scale', 'calibrate'}
                else None
            ),
            expected_score_type=(
                requested_scale_score_type(args)
                if args.mode in {'train-scale', 'calibrate'}
                else None
            ),
        )
    args.scale_context_radius = (
        scale_context_radius(unwrap_model(scale_model))
        if scale_model is not None and args.mode in {'train-scale', 'calibrate'}
        else 0
    )
    if is_main_process and args.scale_context_radius:
        print(
            {
                'scale_context_radius': args.scale_context_radius,
                'base_eligibility_trim': int(args.num_loss_trim[0]),
                'context_radius_policy': 'overlap_only',
            },
            flush=True,
        )

    whitening_scorer = None
    if args.mode == 'train-scale' and requested_scale_score_type(args) == RAW_WHITENING_SCORE_TYPE:
        whitening_scorer = make_whitening_scorer(args, pretrain_config, device, required=True)

    if not args.disable_compile:
        pretrained_model = torch.compile(pretrained_model, dynamic=True)

    kmer_table=resolve_cli_preprocessing(args).kmer_table

    resume_checkpoint = None
    if args.model_load_path and args.mode in {'train', 'train-scale'} and not eval_only:
        if getattr(args, '_initialized_new_scale_model', False):
            if is_main_process:
                print(
                    "Skipping scale optimizer restore because the loaded checkpoint's scale score type/features "
                    "do not match the requested scale model.",
                    flush=True,
                )
        else:
            resume_checkpoint = torch.load(args.model_load_path, map_location="cpu")

    resume_global_step = 0

    if args.mode == 'train' and not eval_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = build_training_scheduler(optimizer, args)
        model.train()
        if scale_model is not None:
            scale_model.eval()
        resume_global_step = restore_training_state_from_checkpoint(
            args,
            resume_checkpoint,
            optimizer,
            scheduler,
            device,
            is_main_process,
        )
    elif args.mode == 'train-scale' and not eval_only:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        optimizer = torch.optim.Adam(scale_model.parameters(), lr=lr)
        scheduler = build_training_scheduler(optimizer, args)
        scale_model.train()
        resume_global_step = restore_training_state_from_checkpoint(
            args,
            resume_checkpoint,
            optimizer,
            scheduler,
            device,
            is_main_process,
        )
    else:
        optimizer=None
        scheduler=None
        model.eval()
        if scale_model is not None:
            scale_model.eval()

    if not args.disable_compile:
        model = torch.compile(model, dynamic=True)
        if scale_model is not None:
            scale_model = torch.compile(scale_model, dynamic=True)
        
    # Training retains synchronized DDP updates. Finite evaluation instead
    # reduces only after every rank has completed its independent shard.
    if world_size > 1 and not eval_only:
        if args.mode == 'train-scale':
            scale_model = DDP(scale_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        elif args.mode == 'train':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    model_to_print = scale_model if args.mode == 'train-scale' and scale_model is not None else model
    model_to_print = model_to_print.module if isinstance(model_to_print, DDP) else model_to_print
    model_details=str(model_to_print)
    if is_main_process:
        print(model_details, flush=True)
        num_params=sum(p.numel() for p in model_to_print.parameters())
        print('# Parameters=', num_params, flush=True)

    if args.mode == 'fit-whitening':
        run_fit_whitening(
            args,
            model,
            pretrained_model,
            device,
            pretrain_config,
            rank,
            world_size,
            is_main_process,
        )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    use_scale_dataloader = (eval_only or args.mode in {'train-scale', 'calibrate'}
                            or unwrap_model(model).config.use_sequence_context)
    dataloader_builder = (partial(
        make_scale_bam_dataloader,
        use_rna_structure_features=requires_rna_structure_features(
            model, scale_model if args.mode in {'train-scale', 'calibrate'} else None,
        ),
    ) if use_scale_dataloader else make_bam_dataloader)
    dataloader = dataloader_builder(
        args,
        pretrain_config,
        kmer_table,
        rank,
        world_size,
        drop_last=args.mode in {'train', 'train-scale'} and not eval_only,
    )

    if args.mode == 'calibrate':
        run_calibration(
            args,
            model,
            scale_model,
            pretrained_model,
            dataloader,
            device,
            pretrain_config,
            rank,
            world_size,
            is_main_process,
        )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    if eval_only:
        evaluation = run_finite_evaluation(
            args, model, scale_model, pretrained_model, dataloader,
            whitening_scorer, device, rank, world_size,
            max_batches=max_steps, phase='evaluation', progress_every=args.log_every,
        )
        if is_main_process:
            print(f'EVALUATION {evaluation}', flush=True)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return evaluation

    t=time.time()
    sub_epoch_metrics = defaultdict(lambda: {'losses': [], 'cos': [], 'mse': []})
    batch_num = resume_global_step
    checkpoint = batch_num // checkpoint_batch_step if checkpoint_batch_step else 0
    
    
    total_instances_seen = 0
    total_base_tokens_seen = 0
    total_signal_tokens_seen = 0
    best_validation_loss = float('inf')
    if resume_global_step and resume_checkpoint is not None:
        best_validation_loss = float(resume_checkpoint.get('meta', {}).get('args', {}).get(
            'best_validation_loss', float('inf')))
    if getattr(args, 'validation_config', ''):
        if min(args.validation_steps, args.validation_every, args.validation_batch_size) <= 0:
            raise ValueError("Validation interval, steps and batch size must be positive")
        train_paths = {os.path.realpath(p) for p, _ in load_pretraining_bam_list(args.bam_config)}
        val_paths = {os.path.realpath(p) for p, _ in load_pretraining_bam_list(args.validation_config)}
        if train_paths & val_paths:
            raise ValueError("Training and validation BAM configs overlap")
        
    training_batches = dataloader
    if use_scale_dataloader:
        # Scale training uses a cycling BAM stream. Bound it through the same
        # cooperative worker-shutdown path as whitening and calibration so a
        # completed run never tears down workers in the middle of BAM I/O.
        training_batches = iter_bounded_dataloader(
            dataloader,
            max(max_steps - batch_num, 0),
        )

    for batch in training_batches:
        if args.mode == 'train-scale':
            loss, aux_1, aux_2, _valid_target_count = scale_training_step(
                scale_model,
                model,
                pretrained_model,
                batch,
                optimizer,
                scheduler,
                device,
                eval_only,
                num_base_trim=num_base_trim,
                whitening_scorer=whitening_scorer,
            )
            task_name = 'canonical_null_scale'
        else:
            loss, aux_1, aux_2, preds, targets = training_step(
                model,
                pretrained_model,
                batch,
                optimizer,
                scheduler,
                device,
                eval_only,
                num_base_trim=num_base_trim,
                num_signal_trim=num_signal_trim,
            )
            task_name = 'canonical_null'
        
        sub_epoch_metrics[task_name]['losses'].append(loss)
        sub_epoch_metrics[task_name]['cos'].append(aux_1)
        sub_epoch_metrics[task_name]['mse'].append(aux_2)
            
        local_batch_size = len(batch[0]) # counting number of sequences
        local_base_token_count = sum(len(seq) for seq in batch[0]) # counting number of bases in all sequences
        local_signal_token_count = torch.sum(batch[4]).item()
        
        total_instances_seen += local_batch_size
        total_base_tokens_seen += local_base_token_count
        total_signal_tokens_seen += local_signal_token_count
        
        batch_num+=1
        if (getattr(args, 'validation_config', '') and not eval_only
                and (batch_num % args.validation_every == 0 or batch_num == max_steps)):
            validation = run_heldout_validation(
                args, model, scale_model, pretrained_model, pretrain_config, kmer_table,
                whitening_scorer, device, rank, world_size,
            )
            improved = validation['loss'] < best_validation_loss
            if improved:
                best_validation_loss = validation['loss']
            args.best_validation_loss = best_validation_loss
            if is_main_process:
                print(f"VALIDATION step={batch_num} best={improved} {validation}", flush=True)
                with open(os.path.join(model_save_path, 'validation.jsonl'), 'a') as handle:
                    handle.write(json.dumps(dict(step=batch_num, best=improved, **validation)) + '\n')
                if improved:
                    save_canonical_null_checkpoint(
                        model, pretrained_model, optimizer, scheduler, args,
                        os.path.join(model_save_path, 'model.best'), batch_num, scale_model=scale_model,
                    )
        # --- Checkpoint and Logging ---
        if (batch_num)%checkpoint_batch_step==0:
            checkpoint=batch_num//checkpoint_batch_step
            global_instances, global_base_tokens, global_signal_tokens = get_global_counts(total_instances_seen, total_base_tokens_seen, total_signal_tokens_seen, device)
            
            if is_main_process:
                print(f"{str(datetime.datetime.now())}: Global Instances: {global_instances:,} | Global Base Tokens: {global_base_tokens:,} | Global Signal Tokens: {global_signal_tokens:,}", flush=True)
                print(f"{str(datetime.datetime.now())}: Checkpoint:{checkpoint}  Steps:{batch_num}  Time={time.time()-t:.2f}", flush=True)
            
            for task in sub_epoch_metrics:
                task_metrics = sub_epoch_metrics[task]
                if not task_metrics['losses']:
                    continue

                # --- Metric Calculation ---
                total_loss = sum(task_metrics['losses'])
                total_loss, total_count = reduce_loss_stats(total_loss, len(task_metrics['losses']), device, world_size)
                if total_count == 0:
                    continue
                avg_loss = total_loss / total_count
                total_cos, cos_count = reduce_loss_stats(sum(task_metrics['cos']), len(task_metrics['cos']), device, world_size)
                total_mse, mse_count = reduce_loss_stats(sum(task_metrics['mse']), len(task_metrics['mse']), device, world_size)
                avg_cos = total_cos / cos_count if cos_count > 0 else 0.0
                avg_mse = total_mse / mse_count if mse_count > 0 else 0.0

                # Write loss to TensorBoard
                writer.add_scalar(f'Loss/{task}', avg_loss, checkpoint)
                writer.add_scalar(f'Loss/{task}_aux_1', avg_cos, checkpoint)
                writer.add_scalar(f'Loss/{task}_aux_2', avg_mse, checkpoint)


                if is_main_process:
                    if task == 'canonical_null_scale':
                        print(f"  {task.upper():<22} NLL: {avg_loss:.4f} | Z2: {avg_cos:.4f} | log_sigma: {avg_mse:.4f}", flush=True)
                    else:
                        print(f"  {task.upper():<22} Loss: {avg_loss:.4f} | Cos: {avg_cos:.4f} | SmoothL1: {avg_mse:.4f}", flush=True)
                if log_writer:
                    log_writer.writerow([checkpoint, task, avg_loss, avg_cos, avg_mse])

            if is_main_process:
                print('______________________________________________________________', flush=True)

            # Reset metrics for the next sub-epoch
            sub_epoch_metrics = defaultdict(lambda: {'losses': [], 'cos': [], 'mse': []})

            # Save model checkpoint
            if is_main_process:
                model_path = os.path.join(model_save_path, 'model.%d' % (checkpoint))
                save_canonical_null_checkpoint(
                    model,
                    pretrained_model,
                    optimizer,
                    scheduler,
                    args,
                    model_path,
                    batch_num,
                    scale_model=scale_model,
                )

                # Flush CSV writer to save to disk immediately
                if log_file:
                    log_file.flush()
                    
        if not use_scale_dataloader and batch_num >= max_steps:
            break

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return


if __name__=='__main__':
    
    print('\n%s:\nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    args = parse_canonical_null_args()


    print(f"{str(datetime.datetime.now())}: Starting {args.mode}", flush=True)
    
    print('Arguments::')
    for k in vars(args):
        print('{}: {}'.format(k,vars(args)[k]))

    
    train_model(args)
        
    print(f"{str(datetime.datetime.now())}: {args.mode.capitalize()} Finished.", flush=True)
