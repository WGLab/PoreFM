"""Lightweight parser for supervised fine-tuning."""

from porefm.cli.preprocessing_args import add_preprocessing_arguments
from porefm.cli.chunking_args import add_chunking_arguments
import argparse

from porefm.cli._parser import PoreFMArgumentParser
from porefm.common.run_outputs import add_output_safety_argument
from porefm.cli.lora_args import add_lora_arguments
from porefm.models.configuration import EMA_SIGNAL_DROPOUT_MODES


def build_supervised_finetune_parser():
    """Build the parser for labelled supervised Nanopore fine-tuning."""
    parser = PoreFMArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a per-base modification classifier",
    )

    # --- Required Arguments ---
    parser.add_argument(
        '--bam_config',
        type=str,
        nargs='+',
        help='Paths to versioned supervised JSON configs; separate paths with whitespace.',
    )
    parser.add_argument('--group_weights', type=int, nargs='+', help='Weights for each BAM config file.')
    parser.add_argument('--model_save_path', type=str, help='Directory to save trained models.')
    parser.add_argument('--size', type=int, help='Size of the signal chunk window.')
    parser.add_argument('--step', type=int, help='Sliding step size for the signal chunk window.')
    parser.add_argument('--model_load_path', type=str, help='Path to trained classifier head model.')
    parser.add_argument('--pretrain_load_path', type=str, help='Path to pretrained model.')
    parser.add_argument('--load_optimizer', help='Load the optimizer state_dict.',default=False, action='store_true')
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', type=str, help='Path to kmer expected signal file.')

    # --- QC Arguments ---
    parser.add_argument('--min_read_length', type=int, default=100, help='Minimum read length.')
    parser.add_argument('--min_qscore', type=int, default=10, help='Minimum read qscore.')

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--max_steps', type=int, default=1e20, help='Number of classification batch update steps.')
    train_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the classifier optimizer group.')
    train_group.add_argument('--pretrain_lr', type=float, help='Learning rate for trainable pretrained backbone parameters or LoRA adapters. Defaults to --lr when omitted.')
    train_group.add_argument('--classifier_weight_decay', type=float, default=1e-4, help='AdamW weight decay for classifier-head matrix weights only. Biases, normalization weights, gates, and pretrained-backbone parameters are not decayed. Use 0 to restore the previous optimizer behavior.')
    train_group.add_argument('--logit_reg_lambda', type=float, default=0, help='Weight for mean squared raw logits, constraining class differences and their shared offset; computed in float32.')
    train_group.add_argument('--proportion_loss_weight', type=float, default=0.0, help='Weight for tolerant per-config exclusion-sample proportion loss. Zero disables proportion-loss computation.')
    train_group.add_argument('--label_smoothing', type=float, default=0, help='Fallback classification smoothing epsilon in [0, 1). JSON sample label_smoothing overrides config label_smoothing, which overrides this value; explicit zero disables smoothing. Applies once with cross-entropy or focal loss; exclusion is never smoothed.')
    train_group.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    train_group.add_argument('--disable_compile', help='Disable model compile.',default=False, action='store_true')
    train_group.add_argument('--focal_loss', help='Use focal loss for classifier updates only.',default=False, action='store_true')
    train_group.add_argument('--gamma', type=float, default=2.0, help='Nonnegative focusing gamma used with --focal_loss.')
    train_group.add_argument('--alpha', type=float, nargs='+', help='Optional nonnegative focal class weights, one per classifier class.')
    train_group.add_argument('--train_pretrain_tasks', help='When the pretrained backbone is trainable, also run auxiliary pretraining steps during classification fine-tuning.', default=False, action='store_true')
    train_group.add_argument('--pretrain_step_interval', type=int, default=1, help='Run one auxiliary pretraining step after every N classification fine-tuning updates when --train_pretrain_tasks is set. N=1 matches the previous one-per-step behavior.')
    train_group.add_argument('--pretrain_bam_config', type=str, nargs='+', help='Paths to versioned pretraining JSON configs for auxiliary pretraining batches.')
    train_group.add_argument('--pretrain_mask_prob_range', type=float, nargs=2, default=[0.15, 0.4], help='Auxiliary pretraining mask probability range, start and end value after warmup.')
    train_group.add_argument('--pretrain_mask_prob_warmup', type=int, default=10, help='Number of checkpoints used to warm up auxiliary pretraining mask probability.')
    train_group.add_argument('--pretrain_mask_base_span_range', type=int, nargs=2, default=[5, 9], help='Auxiliary pretraining base-token mask span range.')
    train_group.add_argument('--pretrain_mask_signal_span', type=int, default=1, help='Auxiliary pretraining signal-token mask span.')
    train_group.add_argument('--pretrain_task_list', nargs='+', default=['sublm', 'masklm', 'mtr'], choices=['sublm', 'masklm', 'mtr'], help='Auxiliary pretraining tasks to sample.')
    train_group.add_argument('--pretrain_weights', nargs='+', type=int, default=[1, 1, 1], help='Sampling weights corresponding to --pretrain_task_list.')
    train_group.add_argument('--pretrain_num_loss_trim', nargs=2, type=int, default=[16, 32], help='Auxiliary pretraining [BASE_TRIM, SIGNAL_TRIM] loss-trim sizes.')
    signal_dropout_group = train_group.add_mutually_exclusive_group()
    signal_dropout_group.add_argument(
        '--pretrain_signal_dropout_prob',
        type=float,
        default=None,
        help='Explicit signal-modality dropout probability for auxiliary pretraining forwards only.',
    )
    signal_dropout_group.add_argument(
        '--use_loaded_pretrain_signal_dropout_prob',
        default=False,
        action='store_true',
        help='Use signal_dropout_prob recorded in the loaded backbone config for auxiliary pretraining.',
    )
    train_group.add_argument(
        '--pretrain_ema_signal_dropout_mode',
        choices=sorted(EMA_SIGNAL_DROPOUT_MODES),
        default=None,
        help='Override the loaded EMA signal-dropout mode for auxiliary pretraining.',
    )

    # --- Data Processing Arguments ---
    parser.add_argument('--checkpoint_batch_step', type=int, default=10000, help='Number of classification batch updates before saving a model checkpoint and logging.')
    parser.add_argument('--bam_threads', type=int, default=4, help='Number of threads to use for BAM output writing.')
    parser.add_argument('--base', type=str, help='Canonical base saved in checkpoint metadata for inference; inherits from --model_load_path when omitted.')
    parser.add_argument('--mod_code', type=str, nargs='+', help='Modification codes saved in checkpoint metadata, one per class after canonical class 0, in logit order; inherits from --model_load_path when omitted.')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use for data batching.')
    parser.add_argument('--reads_per_chunk', type=int, default=50, help='Number of reads to bundle into a single data chunk.')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='DataLoader worker processes for prefetching/pinning.')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches prefetched per worker (only if workers > 0).')

    # --- Model Architecture Arguments ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--classifier_type', help='Classifier head type for a new --pretrain_load_path model; defaults to cnn when omitted. With --model_load_path, an explicit value must match the checkpoint.', default=None, choices=['cnn', 'transformer_self', 'transformer_cross'])
    model_group.add_argument('--class_labels', nargs='+', type=str, default=[],
                            help='A list of modification class names in the same order as labels')
    model_group.add_argument('--num_classes', type=int, help='Number of classes')
    model_group.add_argument('--fc_dim_ff', type=int, default=32, help='Hidden dimension of classifier MLP')
    model_group.add_argument('--base_emb_only', help='Use only the contextualized base embeddings for classification.', default=False, action='store_true')
    model_group.add_argument('--base_seq_window', type=int, default=0, help='Window size for base sequence self-attn within the classifier head. 0 defaults to pretraining model.')
    model_group.add_argument('--base_cross_window', type=int, default=0, help='Window size for base sequence cross-attn to signal within the classifier head. 0 defaults to pretraining model.')
    model_group.add_argument('--classifier_signal_source', choices=['auto', 'contextual_ts', 'fixed_sa_ts'], default=None,
                             help='Scratch-only signal embedding policy for transformer_cross heads. Auto uses contextual_ts when the pretrained checkpoint supports it and fixed_sa_ts otherwise. With --model_load_path, an explicit value must resolve to the saved source.')
    model_group.add_argument('--backbone_signal_cross_attn', choices=['auto', 'inherit', 'on', 'off'], default=None,
                             help='Scratch signal-to-base tower policy. Auto enables it only when required, inherit preserves checkpoint capability, on requires trained tower weights, and off rejects contextual_ts or MTR. With --model_load_path, an explicit value must match the saved topology.')
    model_group.add_argument('--classifier_residual_control', choices=['none', 'skip', 'gate'], default=None,
                             help='Residual-control mode for newly initialized transformer_cross classifier heads. Omit, or use none, for the previous full-residual behavior. Saved classifier checkpoints keep their saved mode.')
    model_group.add_argument('--classifier_center_source', choices=['head', 'backbone'], default=None,
                             help='Center policy for transformer_cross heads. New heads default to head (learn their own centers); backbone reuses detached final base-to-signal centers without a head center predictor. Continuation preserves the saved policy and rejects changes.')
    model_group.add_argument('--freeze_pretrain', help='Freeze pretrained foundation model.',default=False, action='store_true')
    add_lora_arguments(model_group)
    model_group.add_argument('--num_trim', type=int, default=10, help='Number of bases to trim from either end for loss.')
    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser
