'''
Consensus decoding from a pair of RNN outputs (logits).
Currently breaks the signal into blocks and finds a consensus from each block,
before concatenating. In the future, this segmentation will be informed by the
sequence alignment of invidually basecalled reads.
'''
import numpy as np
from multiprocessing import Pool
import argparse, random, sys, glob, os, re
import decoding
from pair_align_decode import fasta_format, load_logits

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--logits', default='.', help='Paths to both logits', required=True, nargs='+')
    parser.add_argument('--window', type=int, default=200, help='Segment size used for splitting reads')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    args = parser.parse_args()

    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]

    # reverse complement logits of one read, doesn't matter which one
    logits_reshape1 = load_logits(file1)
    logits_reshape2 = load_logits(file2, reverse_complement=True)

    logits1 = np.concatenate(logits_reshape1)
    logits2 = np.concatenate(logits_reshape2)

    U = len(logits1)
    V = len(logits2)

    def basecall_box(u1,u2,v1,v2):
        '''
        Function to be run in parallel.
        '''
        print('\tBasecalling bases {}-{}:{}-{}'.format(u1,u2,v1,v2),file=sys.stderr)
        if (u2-u1)+(v2-v1) < 1:
            return(u1,'')
        else:
            try:
                return((u1, decoding.pair_prefix_search_log(logits1[u1:u2],logits2[v1:v2])[0]))
            except:
                print('WARNING: Error while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))
                return(u1,'')
                
    # calculate ranges on which to split read
    # currently just splitting in boxes that follow the main diagonal
    box_ranges = []
    u_step = args.window
    for u in range(u_step,U,u_step):
        box_ranges.append((u-u_step,u,int(V/U*(u-u_step)),int(V/U*u)))
    box_ranges.append((box_ranges[-1][1],U,box_ranges[-1][3],V)) # add in last box with uneven

    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        basecalls = pool.starmap(basecall_box, box_ranges)

    joined_basecalls = ''.join([b[1] for b in basecalls])
    print(fasta_format('consensus;'+file1+';'+file2,joined_basecalls))
