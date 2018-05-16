'''
Consensus decoding from a pair of RNN outputs (logits).
Currently breaks the signal into blocks and finds a consensus from each block,
before concatenating. In the future, this segmentation will be informed by the
sequence alignment of invidually basecalled reads.
'''
import numpy as np
#import tensorflow as tf
from multiprocessing import Pool
import argparse, random, sys, glob, os, re
import consensus

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

def load_logits(file_path, reverse_complement=False):
    # this is assuming logits are in binary file which does not preserve shape
    # information... for portability will need to change this, but for now
    # assuming we know how it was generated.
    read_raw = np.fromfile(file_path,dtype=np.float32)
    read_reshape = read_raw.reshape(-1,200,5) # assuming alphabet of 5 and window size of 200
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    read_logits = np.concatenate(read_reshape)
    return(read_logits)

def basecall_box(u1,u2,v1,v2):
    '''
    Function to be run in parallel.
    '''
    print(u1,u2,v1,v2)
    return(consensus.pair_prefix_search(logits1[u1:u2],logits2[v1:v2])[0])

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

    # reverse complement logist of one read, doesn't matter which one
    logits1 = load_logits(file1)
    logits2 = load_logits(file2, reverse_complement=True)

    U = len(logits1)
    V = len(logits2)

    # calculate ranges on which to split read
    # currently just splitting in boxes that follow the main diagonal
    # for first test am omitting signals that are not evenly divisible
    box_ranges = []
    u_step = args.window
    for u in range(u_step,U,u_step):
        box_ranges.append((u-u_step,u,int(V/U*(u-u_step)),int(V/U*u)))

    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        basecalls = pool.starmap(basecall_box, box_ranges)

    joined_basecalls = ''.join(basecalls)
    print(fasta_format('consensus;'+file1+';'+file2,joined_basecalls))
