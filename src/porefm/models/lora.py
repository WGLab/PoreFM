"""Low-rank adaptation layers and PoreFM backbone injection utilities."""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

LORA_TARGET_HELP = {
    "attention": "Self-attention qkv/proj and cross-attention q/kv/out projections.",
    "cross_attention": "FlexCrossAttention q_proj/kv_proj/proj modules.",
    "self_attention": "SelfAttention fused qkv/proj modules, including self-attention pre-mixers.",
    "all_linear": "Every Linear layer in the pretrained backbone except the EMA encoder.",
}


class LoRALinear(nn.Linear):
    """Wrap a linear layer with a trainable low-rank residual update.

    The original weight and bias ``Parameter`` objects are reused, preserving
    loaded values and optimizer identity. The update is
    ``dropout(x) @ A.T @ B.T * (alpha / rank)``; ``B`` starts at zero so
    injection initially leaves outputs unchanged.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__(
            base_linear.in_features,
            base_linear.out_features,
            bias=base_linear.bias is not None,
            device=base_linear.weight.device,
            dtype=base_linear.weight.dtype,
        )
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / float(self.lora_rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(
                self.lora_rank,
                base_linear.in_features,
                device=base_linear.weight.device,
                dtype=base_linear.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                self.lora_rank,
                device=base_linear.weight.device,
                dtype=base_linear.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return the base projection plus its scaled LoRA update."""
        result = F.linear(input, self.weight, self.bias)
        lora_input = self.lora_dropout(input)
        lora_update = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        return result + lora_update * self.lora_scaling


def _is_lora_target(name: str, module: nn.Module, target_groups) -> bool:
    if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
        return False
    if "ema_signal_encoder" in name:
        return False

    target_groups = set(target_groups)
    if "all_linear" in target_groups:
        return True

    is_cross_attention = name.endswith(
        (
            "cross_attention.q_proj",
            "cross_attention.kv_proj",
            "cross_attention.proj",
        )
    )
    is_self_attention = name.endswith(("attn.qkv", "attn.proj"))

    if "attention" in target_groups and (is_cross_attention or is_self_attention):
        return True
    if "cross_attention" in target_groups and is_cross_attention:
        return True
    if "self_attention" in target_groups and is_self_attention:
        return True

    return False


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def add_lora_to_pretrain_backbone(pretrain_model: nn.Module, rank: int, alpha: float, dropout: float, target_groups):
    """Replace selected backbone linear modules with :class:`LoRALinear`.

    Parameters
    ----------
    pretrain_model : torch.nn.Module
        Backbone modified in place. EMA encoder layers are always excluded.
    rank : int
        Positive low-rank dimension.
    alpha : float
        Update scale before division by ``rank``.
    dropout : float
        Dropout probability applied only on the LoRA branch.
    target_groups : iterable of str
        Any of ``attention``, ``cross_attention``, ``self_attention``, or
        ``all_linear`` as described by ``LORA_TARGET_HELP``.

    Returns
    -------
    dict
        Replaced module names, trainable LoRA parameter count, and parameter
        count of affected base layers.

    Raises
    ------
    ValueError
        If ``rank`` is not positive or no module matches the target groups.
    """
    if rank <= 0:
        raise ValueError("--lora_rank must be > 0 when --lora_backbone is set.")

    target_names = [
        name
        for name, module in pretrain_model.named_modules()
        if _is_lora_target(name, module, target_groups)
    ]
    if not target_names:
        raise ValueError(f"No LoRA target modules matched target groups: {target_groups}")

    for name in target_names:
        base_module = pretrain_model.get_submodule(name)
        _replace_module(
            pretrain_model,
            name,
            LoRALinear(base_module, rank=rank, alpha=alpha, dropout=dropout),
        )

    trainable_lora_params = 0
    affected_base_params = 0
    for name in target_names:
        module = pretrain_model.get_submodule(name)
        trainable_lora_params += module.lora_A.numel() + module.lora_B.numel()
        affected_base_params += module.weight.numel()
        if module.bias is not None:
            affected_base_params += module.bias.numel()

    return {
        "target_names": target_names,
        "trainable_lora_params": trainable_lora_params,
        "affected_base_params": affected_base_params,
    }


def set_pretrain_lora_trainability(pretrain_model: nn.Module, train_base_weights: bool):
    """Set base-parameter trainability while always enabling LoRA factors.

    The operation mutates ``requires_grad`` on all parameters in
    ``pretrain_model``.
    """
    for param in pretrain_model.parameters():
        param.requires_grad = bool(train_base_weights)

    for module in pretrain_model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def get_lora_config(args):
    """Serialize LoRA-related argparse attributes into checkpoint metadata."""
    return {
        "enabled": bool(args.lora_backbone),
        "rank": int(args.lora_rank),
        "alpha": float(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": list(args.lora_target_modules),
        "train_base_weights": bool(args.lora_train_base_weights),
    }


def maybe_restore_lora_args_from_checkpoint(args, model_ckpt):
    """Restore LoRA CLI settings when a checkpoint requires adapters.

    ``args`` is mutated only when the checkpoint records enabled LoRA and the
    current invocation did not already request it; explicit CLI settings take
    precedence.
    """
    checkpoint_lora_config = model_ckpt.get("lora_config")
    if not checkpoint_lora_config or not checkpoint_lora_config.get("enabled", False):
        return

    if args.lora_backbone:
        return

    args.lora_backbone = True
    args.lora_rank = int(checkpoint_lora_config.get("rank", args.lora_rank))
    args.lora_alpha = float(checkpoint_lora_config.get("alpha", args.lora_alpha))
    args.lora_dropout = float(checkpoint_lora_config.get("dropout", args.lora_dropout))
    args.lora_target_modules = list(checkpoint_lora_config.get("target_modules", args.lora_target_modules))
    args.lora_train_base_weights = bool(checkpoint_lora_config.get("train_base_weights", args.lora_train_base_weights))
