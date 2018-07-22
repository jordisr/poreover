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

def softmax(logits):
    dim = len(logits.shape)
    axis_to_sum = dim-1
    return( (np.exp(logits).T / np.sum(np.exp(logits),axis=axis_to_sum).T).T )

def load_logits(file_path, reverse_complement=False, window=200):
    # this is assuming logits are in binary file which does not preserve shape
    # information... for portability will need to change this, but for now
    # assuming we know how it was generated.
    read_raw = np.fromfile(file_path,dtype=np.float32)
    read_reshape = read_raw.reshape(-1,window,5) # assuming alphabet of 5 and window size of 200
    if np.abs(np.sum(read_reshape[0])) > 1:
        print('WARNING: Logits are not probabilities. Running softmax operation.',file=sys.stderr)
        read_reshape = softmax(read_reshape)
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    read_logits = np.concatenate(read_reshape)
    return(read_logits)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--logits', default='.', help='Paths to both logits', required=True, nargs='+')
    parser.add_argument('--logits_size', type=int, default=200, help='Window width used for basecalling')
    parser.add_argument('--window', type=int, default=200, help='Segment size used for splitting reads')
    parser.add_argument('--band', action='store_true', help='Reduce computation through diagonal band')
    parser.add_argument('--width', type=int, default=0, help='Diagonal band size')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    args = parser.parse_args()

    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]

    # reverse complement logits of one read, doesn't matter which one
    logits1 = load_logits(file1, window=args.logits_size)
    logits2 = load_logits(file2, reverse_complement=True, window=args.logits_size)

    U = len(logits1)
    V = len(logits2)

    def basecall_box(u1,u2,v1,v2):
        '''
        Function to be run in parallel.
        '''
        if args.threads == 1 :
            print(u1,u2,v1,v2,file=sys.stderr)
        return(consensus.pair_prefix_search_vec(logits1[u1:u2],logits2[v1:v2])[0])

    def basecall_box_envelope(u1,u2,v1,v2):
        '''
        Function to be run in parallel.
        '''
        envelope = consensus.diagonal_band_envelope(u2-u1,v2-v1,args.width)
        return(consensus.pair_prefix_search(logits1[u1:u2],logits2[v1:v2], envelope=envelope, forward_algorithm=consensus.pair_forward_sparse)[0])

    # calculate ranges on which to split read
    # currently just splitting in boxes that follow the main diagonal
    box_ranges = []
    u_step = args.window
    for u in range(u_step,U,u_step):
        box_ranges.append((u-u_step,u,int(V/U*(u-u_step)),int(V/U*u)))
    box_ranges.append((box_ranges[-1][1],U,box_ranges[-1][3],V)) # add in last box with uneven

    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        if args.band:
            basecalls = pool.starmap(basecall_box_envelope, box_ranges)
        else:
            basecalls = pool.starmap(basecall_box, box_ranges)

    joined_basecalls = ''.join(basecalls)
    print(fasta_format('consensus;'+file1+';'+file2,joined_basecalls))
