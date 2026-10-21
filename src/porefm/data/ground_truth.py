import pyranges1 as pr
import pandas as pd
import numpy as np

from pyfaidx import Fasta


def check_coordinates(base, fasta_path, df, n=50):
    """Infer whether a strand-aware position table is zero- or one-based.

    Parameters
    ----------
    base : str
        Canonical base expected on the forward strand. Reverse-strand rows are
        checked against its complement.
    fasta_path : str or os.PathLike
        Reference FASTA readable by :class:`pyfaidx.Fasta`.
    df : pandas.DataFrame
        Table containing ``chromosome``, ``position``, and ``strand`` columns.
        Strand values may be ``+``, ``-``, ``0``, or ``1``.
    n : int, default=50
        Maximum number of randomly sampled rows to check.

    Returns
    -------
    tuple
        ``(convention, mismatch_counts)``, where ``convention`` is
        ``"0-based"``, ``"1-based"``, or ``"Uncertain"`` and the mapping
        reports the mismatches under both interpretations.

    Raises
    ------
    ValueError
        If a sampled row has an unsupported strand value.
    """
    fa = Fasta(fasta_path)
    comp = {"A":"T", "C":"G", "G":"C", "T":"A"}
    strand_map = {"+": "+", "0": "+", "-": "-", "1": "-"}
    x = df.sample(min(n, len(df)))

    def mismatches(offset):
        """Return sampled row indices whose reference base disagrees at ``offset``."""
        bad = []
        for i, r in x.iterrows():
            strand = strand_map.get(str(r["strand"]))
            if strand is None:
                raise ValueError(
                    f"Unsupported strand value {r['strand']!r}; expected +, -, 0, or 1"
                )
            expected = base if strand == "+" else comp[base]
            observed = fa[r["chromosome"]][int(r["position"]) - offset].seq.upper()
            if observed != expected:
                bad.append(i)
        return bad
    
    data={
        "0-based_mismatches": len(mismatches(0)),
        "1-based_mismatches": len(mismatches(1)),
    }
    
    
    if data["0-based_mismatches"]==0 and data["1-based_mismatches"]>0:
        flag='0-based'
        
    elif data["0-based_mismatches"]>0 and data["1-based_mismatches"]==0:
        flag='1-based'
        
    else:
        flag='Uncertain'
        
    return flag, data

def split_by_proximity(df_A, df_B, window=10):
    """Partition rows by proximity to a same-strand comparison table.

    Parameters
    ----------
    df_A : pandas.DataFrame
        Rows to partition. Must contain ``chromosome``, ``position``, and
        ``strand`` columns; positions are interpreted as zero-length genomic
        points and converted to one-base half-open intervals.
    df_B : pandas.DataFrame
        Comparison coordinates with the same required columns.
    window : int, default=10
        Maximum distance in bases on either side of a comparison interval.
        Matching also requires equal chromosome and strand values.

    Returns
    -------
    tuple of pandas.DataFrame
        Both retain the same columns, dtypes, index, and row order as df_A.
    """

    a = pr.PyRanges(pd.DataFrame({
        "Chromosome": df_A["chromosome"].to_numpy(),
        "Start": df_A["position"].to_numpy(),
        "End": df_A["position"].to_numpy() + 1,
        "_strand": df_A["strand"].to_numpy(),
        "_row": np.arange(len(df_A), dtype=np.int64),
    }))

    b = pr.PyRanges(pd.DataFrame({
        "Chromosome": df_B["chromosome"].to_numpy(),
        "Start": df_B["position"].to_numpy(),
        "End": df_B["position"].to_numpy() + 1,
        "_strand": df_B["strand"].to_numpy(),
    }))

    hits = a.overlap(
        b,
        slack=window,
        match_by="_strand",
        strand_behavior="ignore",
        multiple=False,
        preserve_input_order=False,
    )

    row_ids = hits["_row"].to_numpy()

    mask = np.zeros(len(df_A), dtype=bool)
    mask[row_ids] = True

    matched = df_A.iloc[mask].copy()
    unmatched = df_A.iloc[~mask].copy()

    return matched, unmatched
