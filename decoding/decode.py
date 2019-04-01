import numpy as np
import h5py
import sys
import decoding
import network
import os
from scipy.special import logsumexp

# standard CTC model that PoreOver and Chiron use
POREOVER_ALPHABET = np.array(['A','C','G','T',''])

# Flip/flop transitions from Flappie and Guppy
FLIPFLOP_ALPHABET = np.array(['A','C','G','T','a','c','g','t'])
FLIPFLOP_TRANSITION = np.array([
    [1,1,1,1,1,0,0,0],
    [1,1,1,1,0,1,0,0],
    [1,1,1,1,0,0,1,0],
    [1,1,1,1,0,0,0,1],
    [1,1,1,1,1,0,0,0],
    [1,1,1,1,0,1,0,0],
    [1,1,1,1,0,0,1,0],
    [1,1,1,1,0,0,0,1]
])

def softmax(logits):
    dim = len(logits.shape)
    axis_to_sum = dim-1
    return( (np.exp(logits).T / np.sum(np.exp(logits),axis=axis_to_sum).T).T )

def logit_to_log_likelihood(logits):
    # Normalizes logits so they are valid log-likelihoods
    # takes the place of softmax operation in data preprocessing
    dim = len(logits.shape)
    axis_to_sum = dim-1
    return( (logits.T - logsumexp(logits,axis=2).T).T )

def load_logits(file_path, reverse_complement=False):
    #read_raw = np.fromfile(file_path,dtype=np.float32)
    #read_reshape = read_raw.reshape(-1,window,5) # assuming alphabet of 5 and window size of 200
    read_reshape = np.load(file_path)
    if np.isclose(np.sum(read_reshape[0,0]), 1):
        print('WARNING: Logits appear to be probabilities. Taking log.',file=sys.stderr)
        read_reshape = np.log(read_reshape)
    else:
        read_reshape = logit_to_log_likelihood(read_reshape)
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    return(read_reshape)

def remove_repeated(s):
    out = ''
    for i in range(len(s)):
        if (i==0) or (s[i-1] != s[i]):
            out += s[i]
    return(out)

def argmax_decode(trace, alphabet=POREOVER_ALPHABET, return_path=False):
    greedy_path = np.argmax(trace, axis=1)
    if return_path:
        return greedy_path
    else:
        greedy_string = ''.join(np.take(alphabet, greedy_path))
        return(greedy_string)

def trace_from_flappie(p):
    hdf = h5py.File(p, 'r')
    read_id = list(f)[0]
    trace = np.array(hdf[read_id]['trace'])
    # signal = np.array(hdf[read_id]['signal'])
    hdf.close()
    return(trace)

def trace_from_guppy(p):
    hdf = h5py.File(p, 'r')
    trace = np.array(hdf['/Analyses/Basecall_1D_000/BaseCalled_template/Trace'])
    hdf.close()
    return(trace)

def trace_viterbi(log_probs, transition=FLIPFLOP_TRANSITION):
    # initialize variables
    t_max = len(log_probs)
    n_states = len(transition)
    v = np.zeros((t_max, n_states))-np.inf
    ptr = np.zeros_like(v).astype(int)

    # fill out DP matrix
    for t in range(t_max):
        if t==0:
            v[t] = log_probs[0]
        else:
            prev = transition.T + v[t-1]
            ptr[t] = np.argmax(prev, axis=1)
            v[t] = log_probs[t] + np.max(prev, axis=1)

    # traceback
    viterbi_path = np.zeros(t_max, dtype=int)
    viterbi_path[-1] = np.argmax(v[-1])
    for i in reversed(range(0, len(v)-1)):
        viterbi_path[i] = ptr[i][viterbi_path[i+1]]

    return(viterbi_path, v)

def decode(args):
    in_path = getattr(args, 'in')

    # load probabilities from running
    if args.basecaller == 'poreover':
        logits = load_logits(in_path)
        logits = np.concatenate(logits)
        print(logits.shape)
        if args.algorithm == 'viterbi':
            sequence = argmax_decode(logits)

    elif args.basecaller == 'flappie' or args.basecaller == 'guppy':
        if args.basecaller == 'flappie':
            trace = trace_from_flappie(in_path)
        elif args.basecaller == 'guppy':
            trace = trace_from_guppy(in_path)

        eps = 0.0000001
        trace_log = np.log((trace + eps)/(255 + eps))
        if args.algorithm == 'viterbi':
            viterbi_path, _ = trace_viterbi(trace_log)
            sequence = remove_repeated(''.join(np.take(FLIPFLOP_ALPHABET, viterbi_path))).upper()

    # output decoded sequence
    fasta_header = os.path.basename(in_path)
    if args.out is None:
        fasta_file = sys.stdout
    else:
        fasta_file = open(args.out+'.fasta','a')
    print(network.run_model.fasta_format(fasta_header, sequence), file=fasta_file)
