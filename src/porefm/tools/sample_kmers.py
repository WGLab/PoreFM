"""Digitally normalize a shuffled BAM by strand-specific reference k-mers."""

import pysam
import numpy as np
import sys, time
import argparse

from porefm.cli._parser import PoreFMArgumentParser
from porefm.data.bam_headers import add_pg_to_header

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def main():
    """Retain reads whose median previously observed k-mer depth is below target.

    Primary mapped reads are represented by their alignment-derived reference
    sequence in original read orientation. Every ``step``-th k-mer contributes
    to an in-memory count table, which is updated only for retained reads. The
    greedy result therefore depends on input order; callers should randomize the
    BAM when they want an unbiased subsample.
    """
    parser = PoreFMArgumentParser(description="Normalize BAM using reference k-mers (unmapped-style).")
    parser.add_argument("-i", "--input", required=True, help="Input BAM file (must be shuffled/randomized!)")
    parser.add_argument("-o", "--output", required=True, help="Output BAM file")
    parser.add_argument("-t", "--target", type=int, default=50, help="Target coverage (default: 50)")
    parser.add_argument("-k", "--kmer", type=int, default=31, help="K-mer size (default: 31)")
    parser.add_argument("--step", type=int, default=5, help="Sampling step size (default: 5)")
    parser.add_argument("--threads", help='Number of processors to use for merging bam files. Diminishing returns after 4.',type=int, default=1)
    args = parser.parse_args()

    print(f"--- Reference-Based Normalizer ---")
    print(f"Target Depth: {args.target}x")
    print(f"Mode: Strand-Specific Sequence Counting (like raw FASTQ)")
    
    threads=args.threads

    infile = pysam.AlignmentFile(args.input, "rb")
    outfile = pysam.AlignmentFile(args.output, "wb", header=add_pg_to_header(infile.header), threads=threads)

    # A plain dictionary keeps the bacterial-genome use case compact while
    # allowing exact strand-specific sequence keys.
    kmer_counts = {}

    total = 0
    kept = 0
    
    KMER_SIZE=args.kmer
    step=args.step
    
    t=time.time()
    
    for read in infile:
        total += 1
        if total % 5000 == 0:
            elapsed=time.time()-t
            print(f"Processed: {total} | Kept: {kept}|  {elapsed:.2f} ...", end="\r")

        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        # Alignment-derived reference sequence suppresses basecall errors; the
        # reverse complement restores the original read orientation.
        seq = read.get_reference_sequence().upper()
        ref_seq = seq.translate(COMPLEMENT)[::-1] if read.is_reverse else seq

        
        if not seq or len(seq) < args.kmer:
            continue

        # Subsample long reads at a fixed step to bound hashing cost while
        # retaining strand-specific k-mers as distinct keys.
        read_kmers = [hash(ref_seq[i:i+KMER_SIZE]) for i in range(0, len(ref_seq)-KMER_SIZE+1, step)]
        
        if not read_kmers:
            continue

        counts = [kmer_counts.get(k, 0) for k in read_kmers]
        
        # Median abundance tolerates isolated repetitive k-mers while measuring
        # whether most of the read covers already-saturated sequence.
        median_depth = np.median(counts)

        if median_depth < args.target:
            outfile.write(read)
            kept += 1
            
            for k in read_kmers:
                kmer_counts[k] = kmer_counts.get(k, 0) + 1

    print(f"\n\nDone!")
    print(f"Total Reads Processed: {total}")
    print(f"Reads Kept: {kept} ({kept/total:.2%})")
    print(time.time())
    infile.close()
    outfile.close()
    
if __name__=='__main__':
    main()
