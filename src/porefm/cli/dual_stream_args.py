"""Lightweight parsers for dual-stream embedding command-line tools."""

from porefm.cli.preprocessing_args import add_preprocessing_arguments
import argparse

from porefm.cli._parser import PoreFMArgumentParser
from porefm.workflows.prototypes.embedding_selection import EMBEDDING_KEY_CHOICES


def add_exclude_bed_argument(parser):
    """Add the shared read-exclusion option with optional strand in column four."""
    parser.add_argument(
        '--exclude-bed', '--exclude_bed', default=None,
        help=(
            'Exclude reads whose full reference span overlaps '
            'a region in a chrom start end [strand] file (zero-based, half-open). '
            'Three columns exclude both strands; column 4, if present, must be + or -.'
        ),
    )
    return parser


def add_anomaly_pvalue_arguments(parser):
    """Add shared chi-square/empirical anomaly p-value options and return ``parser``."""
    parser.add_argument(
        '--pvalue_method',
        choices=['chi2', 'empirical'],
        default='chi2',
        help=(
            'Convert squared Mahalanobis distances with the original chi-square '
            'survival function or a held-out empirical control distribution.'
        ),
    )
    parser.add_argument(
        '--calibration_fraction',
        type=float,
        default=0.2,
        help=(
            'Per-locus control fraction held out when --pvalue_method empirical. '
            'The remaining fit-control coverage must still satisfy '
            '--min_control_cov.'
        ),
    )
    return parser


def add_dual_stream_polishing_arguments(parser):
    """Add independent stream-polishing options and return ``parser``."""
    parser.add_argument(
        '--wt_use_polished_sequence',
        default=False,
        action='store_true',
        help='Use reference-guided polished base sequences for the WT/test stream.',
    )
    parser.add_argument(
        '--ctrl_use_polished_sequence',
        default=False,
        action='store_true',
        help='Use reference-guided polished base sequences for the control stream.',
    )
    parser.add_argument(
        '--polish_indel_limit',
        type=int,
        default=5,
        help='Maximum unmasked insertion/deletion length corrected during polishing.',
    )
    parser.add_argument(
        '--polish_vcf_path',
        default=None,
        help='Optional VCF/BCF whose ALT SNPs and indel loci are preserved during polishing.',
    )
    return parser


def build_pairwise_distance_parser():
    """Build the parser for dual-stream embedding-distance and block output."""
    parser = PoreFMArgumentParser(description="Dual Stream Metrics Computation")
    add_exclude_bed_argument(parser)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embedding_key', choices=EMBEDDING_KEY_CHOICES, default='contextual_base_seq', help='Base-aligned model representation used for pairwise distances and saved embeddings.')
    parser.add_argument('--kmer_dict_path', type=str, required=True)
    parser.add_argument('--seq_type', type=str, choices=['dna','rna'], required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--wt_dir', '--wt_bam_path', dest='wt_input', type=str, required=True, help="WT BAM file or directory")
    parser.add_argument('--ctrl_dir', '--ctrl_bam_path', dest='ctrl_input', type=str, required=True, help="Control BAM file or directory")
    parser.add_argument('--wt_parallel_loader', default=False, action='store_true', help="Process the WT BAM files in parallel and sync via min coordinates")
    parser.add_argument('--ctrl_parallel_loader', default=False, action='store_true', help="Process the control BAM files in parallel and sync via min coordinates")
    parser.add_argument('--chrom_list', type=str, nargs='+', help="Chromosome list, process all by default")
    parser.add_argument('--motif_string', type=str, default='', help="Motif specification supplied as one quoted shell argument, for example 'CG 0;CH 0'.")
    parser.add_argument('--motif_label', type=str, default='', help="Semicolon-delimited class labels corresponding to --motif_string.")
    parser.add_argument('--pos_file', type=str, default='')
    parser.add_argument('--min_control_cov', type=int, default=30)
    parser.add_argument('--min_test_cov', type=int, default=30)
    parser.add_argument('--max_fit_cov', type=int, default=256)
    parser.add_argument('--n_comp', type=int, default=2)
    parser.add_argument('--anomaly_batch_size', type=int, default=2000, help="Batch size for anomaly_kwargs")
    parser.add_argument('--score_batch_size', type=int, default=None, help="Optional chunk size for scoring test reads within each anomaly batch. This does not subsample reads.")
    parser.add_argument('--max_score_cov', type=int, default=None, help="Optional maximum number of test reads scored per position. This subsamples scored reads when set.")
    parser.add_argument('--wt_num_workers', type=int, default=1)
    parser.add_argument('--control_num_workers', type=int, default=1)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=100000)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for get_dual_stream_metrics")
    parser.add_argument('--num_trim', type=int, default=16, help="Number of bases to trim from each read edge before selecting labelled positions.")
    parser.add_argument('--anomaly_workers', type=int, default=1, help="Number of loky subprocesses for block anomaly scoring. Use 1 for CUDA stability; increase only if GPU/host memory allows it.")
    parser.add_argument('--output_mode', type=str, default=None, choices=['pvalues', 'embeddings', 'both'], help="Which block outputs to produce. Default is pvalues, or embeddings when --store_embeddings is used.")
    parser.add_argument('--anomaly_output', type=str, default=None, help="Directory for p-value parquet outputs. Required for --output_mode pvalues or both.")
    parser.add_argument('--embedding_output', type=str, default=None, help="Directory for matched WT/control embedding block outputs. Required for --output_mode embeddings or both.")
    parser.add_argument('--store_embeddings', default=False, action='store_true', help="Legacy alias for --output_mode embeddings when --output_mode is not provided.")
    parser.add_argument('--embedding_write_mode', type=str, default='fast', choices=['fast', 'compressed'], help="fast writes uncompressed .pt blocks; compressed writes smaller .npz blocks with deflate compression.")
    parser.add_argument('--embedding_write_workers', type=int, default=1, help="Number of background writer threads used for embeddings output.")
    parser.add_argument('--max_pending_block_tasks', type=int, default=None, help="Maximum queued anomaly/writer block tasks before applying backpressure. Default is 2x active workers.")
    parser.add_argument('--accumulation_queue_size', type=int, default=128, help="Maximum queued GPU embedding batches awaiting CPU accumulation. Lower values reduce peak GPU memory.")
    add_anomaly_pvalue_arguments(parser)
    add_preprocessing_arguments(parser)
    return add_dual_stream_polishing_arguments(parser)


def build_pairwise_classifier_parser():
    """Build the parser for dual-stream classifier-embedding anomaly scoring."""
    parser = PoreFMArgumentParser(description="Dual Stream Metrics Computation")
    add_exclude_bed_argument(parser)
    parser.add_argument('--embedding_key', choices=EMBEDDING_KEY_CHOICES, default='contextual_base_seq', help='Base-aligned backbone representation used for pairwise distances and saved embeddings.')
    parser.add_argument('--kmer_dict_path', type=str, required=True)
    parser.add_argument('--seq_type', type=str, choices=['dna','rna'], required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="Output dir")
    parser.add_argument('--wt_dir', '--wt_bam_path', dest='wt_input', type=str, required=True, help="WT BAM file or directory")
    parser.add_argument('--ctrl_dir', '--ctrl_bam_path', dest='ctrl_input', type=str, required=True, help="Control BAM file or directory")
    parser.add_argument('--wt_parallel_loader', default=False, action='store_true', help="Process the WT BAM files in parallel and sync via min coordinates")
    parser.add_argument('--ctrl_parallel_loader', default=False, action='store_true', help="Process the control BAM files in parallel and sync via min coordinates")
    parser.add_argument('--chrom_list', type=str, nargs='+', help="Chromosome list, process all by default")
    parser.add_argument('--motif_string', type=str, default='', help="Motif specification supplied as one quoted shell argument, for example 'CG 0;CH 0'.")
    parser.add_argument('--motif_label', type=str, default='', help="Semicolon-delimited class labels corresponding to --motif_string.")
    parser.add_argument('--pos_file', type=str, default='')
    parser.add_argument('--min_control_cov', type=int, default=30)
    parser.add_argument('--min_test_cov', type=int, default=30)
    parser.add_argument('--max_fit_cov', type=int, default=256)
    parser.add_argument('--n_comp', type=int, default=2)
    parser.add_argument('--anomaly_batch_size', type=int, default=2000, help="Batch size for anomaly_kwargs")
    parser.add_argument('--wt_num_workers', type=int, default=1)
    parser.add_argument('--control_num_workers', type=int, default=1)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=100000)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for get_dual_stream_metrics")
    parser.add_argument('--num_trim', type=int, default=16, help="Number of bases to trim from each read edge before selecting labelled positions.")
    parser.add_argument('--anomaly_workers', type=int, default=1, help="Number of spawned subprocesses for block anomaly scoring. Use 1 for CUDA stability; increase only if GPU/host memory allows it.")
    parser.add_argument('--store_embeddings', default=False, action='store_true', help="Write matched accumulated WT/control embeddings to disk and skip p-value calculation.")
    parser.add_argument('--embedding_write_mode', type=str, default='fast', choices=['fast', 'compressed'], help="fast writes uncompressed .pt blocks; compressed writes smaller .npz blocks with deflate compression.")
    parser.add_argument('--embedding_write_workers', type=int, default=1, help="Number of background writer threads used when --store_embeddings is set.")
    parser.add_argument('--max_pending_block_tasks', type=int, default=None, help="Maximum queued anomaly/writer block tasks before applying backpressure. Default is 2x active workers.")
    parser.add_argument('--accumulation_queue_size', type=int, default=128, help="Maximum queued GPU embedding batches awaiting CPU accumulation. Lower values reduce peak GPU memory.")
    parser.add_argument('--model_load_path', type=str, help='Path to trained classifier model.')
    add_anomaly_pvalue_arguments(parser)
    add_preprocessing_arguments(parser)
    return add_dual_stream_polishing_arguments(parser)


def build_dual_stream_metrics_parser():
    """Build the legacy parser for dual-stream matched-position metrics."""
    parser = PoreFMArgumentParser(description="Dual Stream Metrics Computation")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--kmer_dict_path', type=str, required=True)
    parser.add_argument('--seq_type', type=str, choices=['dna','rna'], required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="Output dir")
    parser.add_argument('--wt_dir', '--wt_bam_path', dest='wt_input', type=str, required=True, help="WT BAM file or directory")
    parser.add_argument('--ctrl_dir', '--ctrl_bam_path', dest='ctrl_input', type=str, required=True, help="Control BAM file or directory")
    parser.add_argument('--wt_parallel_loader', default=False, action='store_true', help="Process the WT BAM files in parallel and sync via min coordinates")
    parser.add_argument('--ctrl_parallel_loader', default=False, action='store_true', help="Process the control BAM files in parallel and sync via min coordinates")
    parser.add_argument('--chrom_list', type=str, nargs='+', help="Chromosome list, process all by default")
    parser.add_argument('--motif_string', type=str, default='', help="Motif specification supplied as one quoted shell argument, for example 'CG 0;CH 0'.")
    parser.add_argument('--motif_label', type=str, default='', help="Semicolon-delimited class labels corresponding to --motif_string.")
    parser.add_argument('--pos_file', type=str, default='')
    parser.add_argument('--min_control_cov', type=int, default=30)
    parser.add_argument('--min_test_cov', type=int, default=30)
    parser.add_argument('--max_fit_cov', type=int, default=256)
    parser.add_argument('--n_comp', type=int, default=2)
    parser.add_argument('--anomaly_batch_size', type=int, default=2000, help="Batch size for anomaly_kwargs")
    parser.add_argument('--wt_num_workers', type=int, default=1)
    parser.add_argument('--control_num_workers', type=int, default=1)
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--block_size', type=int, default=100000)
    parser.add_argument('--reads_per_chunk', type=int, default=20)
    parser.add_argument('--min_read_length', type=int, default=100)
    parser.add_argument('--min_qscore', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for get_dual_stream_metrics")
    parser.add_argument('--anomaly_workers', type=int, default=1, help="Number of loky subprocesses for block anomaly scoring. Use 1 for CUDA stability; increase only if GPU/host memory allows it.")
    parser.add_argument('--store_embeddings', default=False, action='store_true', help="Write matched accumulated WT/control embeddings to disk and skip p-value calculation.")
    parser.add_argument('--embedding_write_mode', type=str, default='fast', choices=['fast', 'compressed'], help="fast writes uncompressed .pt blocks; compressed writes smaller .npz blocks with deflate compression.")
    parser.add_argument('--embedding_write_workers', type=int, default=1, help="Number of background writer threads used when --store_embeddings is set.")
    parser.add_argument('--max_pending_block_tasks', type=int, default=None, help="Maximum queued anomaly/writer block tasks before applying backpressure. Default is 2x active workers.")
    parser.add_argument('--accumulation_queue_size', type=int, default=128, help="Maximum queued GPU embedding batches awaiting CPU accumulation. Lower values reduce peak GPU memory.")
    add_anomaly_pvalue_arguments(parser)
    add_preprocessing_arguments(parser)
    return add_dual_stream_polishing_arguments(parser)
