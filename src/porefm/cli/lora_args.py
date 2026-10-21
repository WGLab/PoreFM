"""Lightweight command-line argument definitions for LoRA options."""

from porefm.models.lora import LORA_TARGET_HELP

def add_lora_arguments(argument_group):
    """Add shared backbone-LoRA options to an argparse group in place.

    Returns ``None``; callers continue using the supplied parser or argument
    group. Target choices are synchronized with the module groups supported by
    ``porefm.models.lora``.
    """
    argument_group.add_argument(
        '--lora_backbone',
        help='Insert LoRA adapters into selected pretrained backbone Linear projections.',
        default=False,
        action='store_true',
    )
    argument_group.add_argument(
        '--lora_rank',
        type=int,
        default=8,
        help='LoRA adapter rank for pretrained backbone projections.',
    )
    argument_group.add_argument(
        '--lora_alpha',
        type=float,
        default=16.0,
        help='LoRA scaling alpha for pretrained backbone projections.',
    )
    argument_group.add_argument(
        '--lora_dropout',
        type=float,
        default=0.0,
        help='Dropout applied before pretrained backbone LoRA adapters.',
    )
    argument_group.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=['attention'],
        choices=sorted(LORA_TARGET_HELP),
        help='Pretrained backbone module groups to receive LoRA adapters.',
    )
    argument_group.add_argument(
        '--lora_train_base_weights',
        help='Also train the original pretrained backbone weights alongside LoRA adapters. By default, only LoRA adapter weights train.',
        default=False,
        action='store_true',
    )
