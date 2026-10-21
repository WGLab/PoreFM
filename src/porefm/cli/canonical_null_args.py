"""Lightweight parser for canonical-null training and calibration."""

from porefm.cli.preprocessing_args import add_preprocessing_arguments
from porefm.cli.chunking_args import add_chunking_arguments
import argparse
import sys

from porefm.cli._parser import PoreFMArgumentParser
from porefm.common.run_outputs import add_output_safety_argument


STAGE_DESCRIPTIONS = {
    'train': 'Train a predictor of canonical embeddings from a pretrained backbone.',
    'train-scale': 'Train a location/scale model for canonical deviation scores.',
    'fit-whitening': 'Fit residual covariance transforms on canonical data.',
    'calibrate': 'Estimate empirical score distributions on held-out canonical data.',
}


def parse_canonical_null_args(argv=None):
    """Parse a public fixed stage or the internal distributed launch options.

    Public dispatch prefixes a stage token. Direct module launches may retain
    ``--mode`` for existing torchrun scripts; the fixed-stage parser omits it.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    mode = args.pop(0) if args and args[0] in STAGE_DESCRIPTIONS else None
    return build_canonical_null_parser(mode=mode).parse_args(args or ['--help'])


def build_canonical_null_parser(mode=None):
    """Build reference preparation options, optionally fixing the stage.

    ``mode=None`` retains the internal ``--mode`` interface. A named stage
    sets ``args.mode`` and excludes that switch from the public parser.
    """
    if mode is not None and mode not in STAGE_DESCRIPTIONS:
        raise ValueError(f'Unknown zero-shot stage: {mode}')
    parser = PoreFMArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=STAGE_DESCRIPTIONS.get(mode, "Train canonical references, fit whitening, or calibrate deviation scores."),
    )
    parser.add_argument('--bam_config', type=str, help='Path to a versioned pretraining JSON config containing canonical signal-BAM samples.')
    parser.add_argument('--model_save_path', type=str, help='Directory to save trained models.')
    parser.add_argument('--size', type=int, help='Size of the signal chunk window.')
    parser.add_argument('--step', type=int, help='Sliding step size for the signal chunk window.')
    parser.add_argument('--seq_type', help='Specify DNA or direct RNA sequencing.', choices=['dna','rna'], type=str, required=True)
    parser.add_argument('--model_load_path', type=str, help='Path to existing trained model.')
    parser.add_argument('--pretrain_load_path', type=str, help='Path to pretrained model.')
    add_chunking_arguments(parser)
    parser.add_argument('--kmer_dict_path', type=str, help='Path to kmer expected signal file.')
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Evaluate a finite pass without training/checkpoint writes; aggregate actual target-weighted losses after all ranks finish.')
    if mode is None:
        parser.add_argument('--mode', choices=['train', 'train-scale', 'calibrate', 'fit-whitening'], default='train',
                            help='Train the canonical-null mean predictor, train the scalar scale model, fit whitening covariance, or calibrate score distributions on held-out canonical data.')
    else:
        parser.set_defaults(mode=mode)
    parser.add_argument('--disable_compile', help='Disable torch.compile for backbone/null predictor.', default=False, action='store_true')
    parser.add_argument('--enable_compile', dest='disable_compile', action='store_false',
                        help='Explicitly enable torch.compile.')
    parser.add_argument('--min_read_length', type=int, default=100, help='Minimum read length.')
    parser.add_argument('--min_qscore', type=int, default=10, help='Minimum read qscore.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1e20,
        help=(
            'Maximum batch steps. Distributed calibration treats this as a global cap '
            'split across ranks; other distributed modes apply it per rank.'
        ),
    )
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--lr_schedule', choices=['constant', 'legacy'], default='legacy',
                        help='Constant LR, or the historical checkpoint-based warmup/decay schedule.')
    parser.add_argument('--scale_read_position_features', action='store_true',
                        help='Add full-read position and chunk/read-boundary inputs to new scale checkpoints.')
    parser.add_argument('--scale_rna_structure_features', action='store_true',
                        help='Add 109 RNALfold structure inputs to new RNA scale checkpoints (span/radius 30, noLP); requires rnalocal_ext. Saved checkpoints enable these automatically for calibration/inference.')
    parser.add_argument('--null_sequence_context', action='store_true',
                        help='Add a separate raw-base CNN to a newly initialized null predictor.')
    parser.add_argument('--null_rna_structure_features', action='store_true',
                        help='Concatenate 109 RNALfold channels with base embeddings before the null CNN; requires RNA and --null_sequence_context for a new null model. Saved checkpoints enable extraction automatically.')
    parser.add_argument('--null_context_embedding_dim', type=int, default=32)
    parser.add_argument('--null_context_channels', type=int, default=128)
    parser.add_argument('--null_context_kernel_size', type=int, default=5)
    parser.add_argument('--null_context_dilations', nargs='+', type=int, default=[1, 1, 2, 4, 8, 16])
    parser.add_argument('--null_context_expansion', type=int, default=4)
    parser.add_argument('--null_context_dropout', type=float, default=0.05)
    parser.add_argument('--validation_config', type=str, default='',
                        help='Disjoint pretraining JSON used only for held-out checkpoint selection.')
    parser.add_argument('--validation_steps', type=int, default=128,
                        help='Fixed validation batch cap per rank; includes a reproducible subset each evaluation.')
    parser.add_argument('--validation_batch_size', type=int, default=64)
    parser.add_argument('--validation_every', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--scale_dropout', type=float, default=0.0, help='Dropout used inside the CanonicalNullScaleModel.')
    parser.add_argument('--scale_kmer_embedding_dim', type=int, default=64,
                        help='Learned categorical k-mer embedding width in the scale model.')
    parser.add_argument('--scale_context_embedding_dim', type=int, default=32,
                        help='Raw-base embedding width for the scale-model sequence-context CNN.')
    parser.add_argument('--scale_context_channels', type=int, default=128,
                        help='Hidden channel width of the scale-model sequence-context CNN.')
    parser.add_argument('--scale_context_kernel_size', type=int, default=5,
                        help='Odd kernel width used by every scale-model context block.')
    parser.add_argument('--scale_context_dilations', nargs='+', type=int, default=[1, 1, 2, 4, 8, 16],
                        help='Dilations for successive depthwise-separable scale context blocks.')
    parser.add_argument('--scale_context_expansion', type=int, default=4,
                        help='Pointwise channel expansion factor inside each scale context block.')
    parser.add_argument('--scale_context_dropout', type=float, default=0.05,
                        help='Dropout within the scale-model sequence-context CNN.')
    parser.add_argument('--scale_h_null_feature_dim', type=int, default=128,
                        help='Projected width of detached h_null features supplied to the scale model.')
    parser.add_argument('--disable_scale_sequence_context', default=False, action='store_true',
                        help='Ablate the raw-sequence context CNN when initializing a new scale model.')
    parser.add_argument('--disable_scale_h_null_features', default=False, action='store_true',
                        help='Ablate detached h_null conditioning when initializing a new scale model.')
    parser.add_argument('--enable_scale_sequence_context', dest='disable_scale_sequence_context', action='store_false')
    parser.add_argument('--enable_scale_h_null_features', dest='disable_scale_h_null_features', action='store_false')
    parser.add_argument('--scale_score_type', choices=['cosine', 'whitening'], default='cosine',
                        help='Scalar-standardizer target score family for train-scale/calibrate.')
    parser.add_argument('--use_base_quality_scale_features', default=False, action='store_true',
                        help='Include Dorado base-quality-derived quality_feature and median_q inputs in the CanonicalNullScaleModel.')
    parser.add_argument('--scale_context_quality', action='store_true',
                        help='Also concatenate per-base Q/40 into the scale sequence CNN; requires quality features and sequence context. Stored in new checkpoints.')
    parser.add_argument('--no_base_quality_scale_features', dest='use_base_quality_scale_features', action='store_false',
                        help='Explicitly exclude basecall-quality inputs; does not disable the read Q filter.')
    parser.add_argument('--checkpoint_batch_step', type=int, default=10000, help='Number of batches before saving a model checkpoint and logging.')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use for data batching.')
    parser.add_argument('--reads_per_chunk', type=int, default=500, help='Number of reads to bundle into a single data chunk.')
    parser.add_argument('--pore_type', type=str, default='R10/RNA004', help='Pore type.')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='DataLoader worker processes for prefetching/pinning.')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches prefetched per worker (only if workers > 0).')
    parser.add_argument(
        '--reference_match_kmer_size',
        type=int,
        default=0,
        help=(
            'Require a centered read k-mer of this odd size to match the aligned '
            'reference exactly during whitening, scale fitting, and calibration. '
            'Zero disables the filter; positive values require mapped records with MD+CIGAR.'
        ),
    )
    parser.add_argument('--log_every', type=int, default=100, help='Calibration/evaluation rank-local logging interval in batches.')
    parser.add_argument('--calibration_output', default='', help='Output .pt path for empirical calibration.')
    parser.add_argument(
        '--kmer_size',
        type=int,
        default=9,
        help='Legacy default for scale/calibration context size when their stage-specific options are omitted.',
    )
    parser.add_argument(
        '--scale_kmer_size',
        type=int,
        default=None,
        help='Centered k-mer size used by the learned scalar location/scale model; defaults to --kmer_size.',
    )
    parser.add_argument(
        '--calibration_kmer_size',
        type=int,
        default=None,
        help='Centered k-mer key used by empirical calibration; defaults to --scale_kmer_size.',
    )
    parser.add_argument(
        '--whitening_kmer_size',
        type=int,
        default=1,
        help='Centered k-mer context used to fit/select full-covariance whitening transforms.',
    )
    parser.add_argument('--max_scores_per_kmer', type=int, default=20000, help='GPU reservoir cap per k-mer for reservoir/quantile calibration.')
    parser.add_argument(
        '--calibration_distribution_type',
        choices=['reservoir', 'histogram', 'quantile'],
        default='reservoir',
        help=(
            'How to store per-k-mer canonical score distributions. Only histogram '
            'supports multi-process calibration.'
        ),
    )
    parser.add_argument('--hist_min', type=float, default=-8.0, help='Minimum score edge for histogram calibration storage.')
    parser.add_argument('--hist_max', type=float, default=8.0, help='Maximum score edge for histogram calibration storage.')
    parser.add_argument('--hist_bins', type=int, default=256, help='Number of fixed-width bins for histogram calibration storage.')
    parser.add_argument('--quantile_body_count', type=int, default=201, help='Number of quantile levels between 0 and 0.99 for quantile calibration storage.')
    parser.add_argument('--quantile_tail_count', type=int, default=200, help='Number of log-spaced upper-tail quantile levels for quantile calibration storage.')
    parser.add_argument('--quantile_tail_min_prob', type=float, default=1e-4, help='Smallest upper-tail survival probability for tail-enriched quantile storage.')
    parser.add_argument('--whitening_artifact', default='', help='Base-specific full-space whitening artifact for whitening score training/calibration/inference.')
    parser.add_argument('--allow_whitening_checkpoint_mismatch', default=False, action='store_true', help='Allow whitening artifact checkpoint identifiers to differ from current model metadata.')
    parser.add_argument('--whitening_output', default='', help='Output .pt path for fitted whitening transforms.')
    parser.add_argument('--covariance_mode', choices=['full', 'diagonal', 'isotropic'], default='full', help='Covariance ablation mode for whitening artifact fitting.')
    parser.add_argument('--shrinkage_mode', choices=['fixed', 'oas'], default='fixed', help='Shrinkage coefficient mode for whitening covariance fitting.')
    parser.add_argument('--shrinkage', type=float, default=0.03, help='Fixed scaled-identity covariance shrinkage coefficient.')
    parser.add_argument(
        '--whitening_parent_shrinkage_strength',
        type=float,
        default=0.0,
        help=(
            'Effective-sample strength for shrinking each k-mer mean/covariance toward its central-base parent. '
            'For whitening k>1, non-positive values select 10 * hidden_dim; k=1 disables parent shrinkage.'
        ),
    )
    parser.add_argument(
        '--whitening_covariance_buffer_rows',
        type=int,
        default=128,
        help='Rows buffered per whitening context before a CPU covariance update; reduces tiny matrix products for k-mer fits.',
    )
    parser.add_argument('--relative_eigenvalue_floor', type=float, default=1e-3, help='Eigenvalue floor as a fraction of tau=trace(cov)/hidden_dim.')
    parser.add_argument('--max_condition_number', type=float, default=1e4, help='Maximum condition number enforced through the eigenvalue floor.')
    parser.add_argument('--max_samples_per_base', type=int, default=0, help='Optional residual reservoir cap per central-base group per rank; 0 uses all selected tokens.')
    parser.add_argument('--max_tokens_per_read', type=int, default=0, help='Reserved for read-aware sampling; current BAM chunk loader does not expose stable read IDs to this mode.')
    parser.add_argument('--whitening_token_sample_rate', type=float, default=1.0, help='Independent Bernoulli sampling probability for valid whitening-fit tokens.')
    parser.add_argument('--sampling_seed', type=int, default=0, help='Seed for optional whitening residual/score sampling.')
    parser.add_argument('--robust_fit', dest='robust_fit', default=True, action='store_true', help='Enable pilot-whitening radial downweighting for covariance fitting.')
    parser.add_argument('--disable_robust_fit', dest='robust_fit', action='store_false', help='Disable robust whitening covariance refit.')
    parser.add_argument('--robust_cutoff', type=float, default=0.0, help='Numeric squared-radius cutoff for robust weights; <=0 uses robust_quantile.')
    parser.add_argument('--robust_quantile', type=float, default=0.995, help='Pilot score quantile used as robust squared-radius cutoff when robust_cutoff <= 0.')
    parser.add_argument('--robust_quantile_sample_size', type=int, default=200000, help='Reservoir score sample size per base for estimating robust pilot quantiles.')
    parser.add_argument('--robust_eps', type=float, default=1e-12, help='Small denominator epsilon for robust radial weights.')
    parser.add_argument('--norm_eps', type=float, default=1e-12, help='Embedding normalization epsilon used for cosine and whitening residuals.')
    parser.add_argument('--score_log_eps', type=float, default=1e-8, help='Epsilon used in log scalar-standardizer targets.')
    parser.add_argument('--whitening_score_chunk_size', type=int, default=262144, help='Token chunk size for whitening scorer GEMMs during robust fitting/calibration.')
    parser.add_argument('--include_raw_covariance', default=False, action='store_true', help='Store raw sample covariance matrices in whitening artifacts.')
    parser.add_argument('--require_valid_kmer_for_whitening_fit', dest='require_valid_kmer_for_whitening_fit', default=True, action='store_true', help='Require the existing centered k-mer validity mask during whitening fitting.')
    parser.add_argument('--allow_invalid_kmer_for_whitening_fit', dest='require_valid_kmer_for_whitening_fit', action='store_false', help='Fit whitening on valid canonical central bases even if the centered k-mer has invalid context.')
    task_group = parser.add_argument_group('Training Tasks')
    task_group.add_argument('--num_loss_trim', nargs=2, type=int, default=[16, 32], help='[BASE_TRIM, SIGNAL_TRIM]. How many tokens to trim towards each boundary edge for loss computation.')
    add_output_safety_argument(parser)
    add_preprocessing_arguments(parser)
    return parser
