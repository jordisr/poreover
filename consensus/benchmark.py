'''
Synthetic benchmark for consensus decoding. Take logits output from real input,
take highest probability label (through prefix search) as ground truth.
Add Gaussian noise to softmax probabilities and make individual basecalls of
each sequence. Then do consensus basecalling of both sequences and compare
with edit distance to original ground truth sequence.
'''
import numpy as np
from multiprocessing import Pool
import argparse, random, sys, glob, os, re, pickle
import ctc
import consensus

def softmax_(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def softmax(x):
    return(softmax_(x.T).T)

def load_logits(file_path, reverse_complement=False, window=200):
    # this is assuming logits are in binary file which does not preserve shape
    # information... for portability will need to change this, but for now
    # assuming we know how it was generated.
    read_raw = np.fromfile(file_path,dtype=np.float32)
    read_reshape = read_raw.reshape(-1,window,5) # assuming alphabet of 5 and window size of 200
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    read_logits = np.concatenate(read_reshape)
    return(read_logits)

def softmax_with_noise(logits, noise_sd=1):
    noise_tensor1 = np.random.standard_normal(logits.shape)*noise_sd
    noise_tensor2 = np.random.standard_normal(logits.shape)*noise_sd
    noisy_read1 = softmax((noise_tensor1+logits))
    noisy_read2 = softmax((noise_tensor2+logits))
    return((noisy_read1, noisy_read2))

# vectorized numpy implementation from:
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def single_run(logits, basecall, noise_multiplier):
    (noisy_softmax1, noisy_softmax2) = softmax_with_noise(logits,noise_multiplier)
    read1_basecall = ctc.prefix_search(noisy_softmax1)[0]
    read2_basecall = ctc.prefix_search(noisy_softmax2)[0]
    consensus_basecall = consensus.pair_prefix_search(noisy_softmax1,noisy_softmax2)[0]
    print(
    noise_multiplier,
    len(basecall),
    len(read1_basecall),
    len(read2_basecall),
    levenshtein(basecall, read1_basecall),
    levenshtein(basecall, read2_basecall),
    levenshtein(basecall, consensus_basecall)
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--iter', type=int, default=1, help='Interations on each segment')
    parser.add_argument('--logits', type=int, default=1, help='File with logits')
    parser.add_argument('--logits_size', type=int, default=1, help='File with logits')
    parser.add_argument('--segments', type=int, default=1, help='Number of segments in file to use')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    args = parser.parse_args()

    # load logits output by an RNN
    #logits_from_file = pickle.load(open('logits2.p','rb'))
    logits_from_file = load_logits(args.logits,window=args.logits_size)
    #sequence_length = [len(i) for i in logits_from_file]

    def run_segment(i):
        logits = logits_from_file[i]
        softmax_logits = softmax(logits)
        truth_basecall = ctc.prefix_search(softmax_logits)[0]
        # repeat each segment
        for i in range(args.iter):
            # use a variety of error magnitudes
            single_run(logits, truth_basecall, 0.5)
            single_run(logits, truth_basecall, 1)
            single_run(logits, truth_basecall, 1.5)
            single_run(logits, truth_basecall, 2)

    n_segments = min(len(logits_from_file), args.segments)

    with Pool(processes=args.threads) as pool:
        pool.map(run_segment, range(n_segments))
