"""Read and summarize PoreFM Parquet outputs without model execution.

Use open_output for shared Arrow/DuckDB access across canonical-null, pairwise,
prototype and supervised runs. Format-specific diagnostics are imported lazily.
"""

from .dataset import OutputDataset, open_output
from .summaries import summarize_loci
from .quantiles import select_quantile_labels
from .thresholds import select_threshold_labels

__all__ = ['OutputDataset', 'open_output', 'summarize_loci', 'select_quantile_labels',
           'select_threshold_labels',
           'load_pairwise_directory', 'summarize_pairwise_loci',
           'iter_pairwise_locus_summaries', 'add_reference_base',
           'make_meth_df_from_predictions']


def __getattr__(name):
    if name in ('load_pairwise_directory', 'summarize_pairwise_loci',
                'iter_pairwise_locus_summaries', 'add_reference_base'):
        from . import pairwise
        return getattr(pairwise, name)
    if name == 'make_meth_df_from_predictions':
        from .prototypes import make_meth_df_from_predictions
        return make_meth_df_from_predictions
    raise AttributeError(name)
