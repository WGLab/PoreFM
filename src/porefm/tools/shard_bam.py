"""Split signal BAM streams into approximately move-count-sized shards."""

import random, time, os, sys, argparse, datetime, csv
import math
from collections import defaultdict
from dataclasses import dataclass, asdict

from typing import Optional, Tuple
import pysam
from dataclasses import dataclass, asdict

from porefm.cli._parser import PoreFMArgumentParser
from porefm.config.validation import load_sharding_config
from porefm.data.bam_headers import add_pg_to_header

def get_bam_chunk(output_dir, bam_config, bam_path,
                  chunk_size=1e6,
                  max_reads_per_contigs=1e9,
                  min_read_length=100,
                  min_qscore=10,
                  prefix='features',
                  threads=2):
    """Write quality-filtered BAM records into approximate signal-size shards.

    Input comes either from one ``bam_path`` or a versioned sharding JSON config
    parsed by ``load_sharding_inputs``. Only primary reads meeting ``min_qscore`` and
    ``min_read_length`` are retained. ``chunk_size`` counts move-table entries,
    not reads or bytes; a shard closes after the first read that reaches the
    threshold. With configured contigs, at most ``max_reads_per_contigs`` are
    accepted from each contig.

    Files are named ``<prefix>.<input_basename>.<index>.bam`` in
    ``output_dir``. The final empty shard created exactly at a boundary is
    removed.
    """
    
    if bam_config:
        bam_list, chrom_bam_list = load_sharding_inputs(bam_config)
    else:
        bam_list, chrom_bam_list = [bam_path], [[]]
    
    for chrom_list, bam_path in zip(chrom_bam_list, bam_list):
        
        bam=pysam.AlignmentFile(bam_path,'rb', check_sq=False)
        bam_name=os.path.basename(bam_path)
        bam_header=add_pg_to_header(bam.header)
        current_prefix=f'{prefix}.{bam_name}'
        
        count=0

        if len(chrom_list)==0:
            chrom_list=[None] 
            max_reads_per_contigs=1e12

        file_count=0
        
        file_path=os.path.join(output_dir, f'{current_prefix}.{file_count}.bam')
        out_bam=pysam.AlignmentFile(file_path, "wb", header=bam_header, threads=threads)
        
        current_chunk_size=0
        
        for chrom in chrom_list:
            print(f'{str(datetime.datetime.now())}: File={bam_name}  contig={chrom}', flush=True)
            i=0
            data={'signal':[], 'seq':[]}

            for bam_read in bam.fetch(chrom, until_eof=True):
                
                if bam_read.flag&0x900==0 \
                    and bam_read.get_tag('qs')>=min_qscore and len(bam_read.seq)>=min_read_length:
                    i+=1
                    count+=1

                    move_len=len(bam_read.get_tag('mv',with_value_type=True)[0])-1
                    current_chunk_size+=move_len
                    
                    out_bam.write(bam_read)
                    
                    if current_chunk_size>=chunk_size:
                        out_bam.close()
                        file_count+=1
                        current_chunk_size=0
                        
                        file_path=os.path.join(output_dir, f'{current_prefix}.{file_count}.bam')
                        out_bam=pysam.AlignmentFile(file_path, "wb", header=bam_header, threads=threads)
                        
                    if count%100_000==0:
                        pos_info='' if chrom==None else f'Read Count={count}  Current Position={bam_read.reference_name}:{bam_read.reference_start}'

                        print(f'{str(datetime.datetime.now())}: File={bam_name}| {pos_info}', flush=True)

                if i>max_reads_per_contigs:
                    break

        out_bam.close()
        
        if current_chunk_size==0:
            os.remove(out_bam.filename)
        pos_info='' if chrom==None else f'Read Count={count}  Current Position={bam_read.reference_name}:{bam_read.reference_start}'

        print(f'{str(datetime.datetime.now())}: File={bam_name}| {pos_info}', flush=True)
            
    return

def load_sharding_inputs(bam_config):
    """Load BAM paths and optional contig lists from a sharding JSON config.

    The returned path and contig lists have matching order and length.
    """
    config = load_sharding_config(bam_config)
    return (
        [item.bam_path for item in config.inputs],
        [list(item.contigs) for item in config.inputs],
    )

if __name__=='__main__':
    
    print('\n%s:\nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    parser = PoreFMArgumentParser(description="Split BAM inputs into coordinate shards.")

    # --- Required Arguments ---
    #parser.add_argument('--bam_list', type=str, required=True, help='Path to the file with BAM files, chromosomes and sampling fraction. Format is three columns: SAMPLE_NAME<tab>BAM_PATH<tab>comma separated chromosome list<tab>fraction')
    parser.add_argument('--bam_config', type=str, help='Path to a versioned BAM-sharding JSON config.')
    parser.add_argument('--bam_path', type=str, help='Path to the BAM file.')
    parser.add_argument('--output_dir', type=str, help='Path to the output folder.')
    parser.add_argument('--prefix', type=str, help='Prefix for files.', default='features')

    # --- QC Arguments ---
    parser.add_argument('--min_read_length', type=int, default=100, help='Minimum read length.')
    parser.add_argument('--min_qscore', type=int, default=10, help='Minimum read qscore.')
 
    # --- Data Processing Arguments ---
    parser.add_argument('--chunk_size', type=int, default=1e7, help='Number of instances per chunk (roughly).')
    parser.add_argument('--max_reads_per_contig', type=int, default=int(1e9), help='Maximum number of reads to process per contig.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for writing BAM file.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{str(datetime.datetime.now())}: Starting", flush=True)
    
    get_bam_chunk(output_dir=args.output_dir,
                   bam_config=args.bam_config,
                   bam_path=args.bam_path,
                   chunk_size=args.chunk_size,
                   max_reads_per_contigs=args.max_reads_per_contig,
                   min_read_length=args.min_read_length,
                   min_qscore=args.min_qscore,
                   prefix=args.prefix,
                   threads=args.threads)
        
    print(f"{str(datetime.datetime.now())}: Finished.", flush=True)
