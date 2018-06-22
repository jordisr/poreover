'''
Consensus decoding from a pair of RNN outputs (logits).
Basecall each sequence individually, align, and then use alignment to guide
consensus basecalling on mismatched/gapped regions. Take contiguous matches or
indels above the threshold, and use as anchors. Divide signal in between these
anchors and basecall separately. Finally stitch anchors back with basecalled
sequences.

    indel             match segment
    ______            ****
    TTTTTA-GCA-GACGCAGGAAGAGACGAA
         |            |||| ||| ||
    -----AGCATACCCAG--GAAG-GACAAA
'''
import numpy as np
from multiprocessing import Pool
import argparse, random, sys, glob, os, re

import consensus
import align
import ctc

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

def load_logits(file_path, reverse_complement=False, window=200):
    # this is assuming logits are in binary file which does not preserve shape
    # information... for portability will need to change this, but for now
    # assuming we know how it was generated.
    read_raw = np.fromfile(file_path,dtype=np.float32)
    read_reshape = read_raw.reshape(-1,window,5) # assuming alphabet of 5 and window size of 200
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    return(read_reshape)

def basecall_box(u1,u2,v1,v2):
    '''
    Function to be run in parallel.
    '''
    print('basecalling segment:',u1,u2,v1,v2,file=sys.stderr)
    if (u2-u1)+(v2-v1) < 1:
        return(u1,'')
    else:
        return((u1, consensus.pair_prefix_search_vec(logits1[u1:u2],logits2[v1:v2])[0]))

def basecall_box_envelope(u1,u2,v1,v2):
    '''
    Function to be run in parallel.
    '''
    # set diagonal band of width 10% the mean length of both segments
    width_fraction = 0.1
    width = int(((u2-u1)+(v2-v1))/2*width_fraction)
    #print(u1,u2,v1,v2,width)
    envelope = consensus.diagonal_band_envelope(u2-u1,v2-v1,width)
    return((u1, consensus.pair_prefix_search(logits1[u1:u2],logits2[v1:v2], envelope=envelope, forward_algorithm=consensus.pair_forward_sparse)[0]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--logits', default='.', help='Paths to both logits', required=True, nargs='+')
    parser.add_argument('--logits_size', type=int, default=200, help='Window width used for basecalling')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    parser.add_argument('--matches', type=int, default=4, help='Match size for building anchors')
    parser.add_argument('--indels', type=int, default=4, help='Indel size for building anchors')
    args = parser.parse_args()

    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]

    # reverse complement logist of one read, doesn't matter which one
    logits1_reshape = load_logits(file1, window=args.logits_size)
    logits2_reshape = load_logits(file2, reverse_complement=True,window=args.logits_size)

    # smaller test data for your poor laptop
    #logits1_reshape = logits1_reshape[:30]
    #logits2_reshape = logits2_reshape[:30]

    logits1 = np.concatenate(logits1_reshape)
    logits2 = np.concatenate(logits2_reshape)

    U = len(logits1)
    V = len(logits2)

    s_tot = 0

    read1_prefix = ""
    read2_prefix = ""

    signal_to_sequence1 = []
    signal_to_sequence2 = []

    # Perform 1d basecalling and get signal-sequence mapping by taking
    # argmax of final forward matrix.
    for i,y in enumerate(logits1_reshape):
        (prefix, forward) = ctc.prefix_search(y, return_forward=True)
        s_len = len(prefix)
        forward_indices = np.argmax(forward,axis=0)
        assert(s_len == len(forward_indices))
        signal_to_sequence1.append(forward_indices+200*i)
        read1_prefix += ctc.prefix_search(y)[0]

    for i,y in enumerate(logits2_reshape):
        (prefix, forward) = ctc.prefix_search(y, return_forward=True)
        s_len = len(prefix)
        forward_indices = np.argmax(forward,axis=0)
        assert(s_len == len(forward_indices))
        signal_to_sequence2.append(forward_indices+200*i)
        read2_prefix += ctc.prefix_search(y)[0]

    signal_to_sequence1 = np.concatenate(np.array(signal_to_sequence1))
    signal_to_sequence2 = np.concatenate(np.array(signal_to_sequence2))
    #print(U,len(signal_to_sequence1), len(read1_prefix))

    # alignment should be replaced with a more efficient implementation
    alignment = align.global_align(read1_prefix, read2_prefix)

    # get alignment-sequence mapping
    # no boundary case for first element but it will wrap around to the last (which is zero)
    alignment = np.array(alignment[:2])
    alignment_to_sequence = np.zeros(shape=alignment.shape,dtype=int)
    for i,t in enumerate(alignment.T):
        for j in range(2):
            if t[j] == '-':
                alignment_to_sequence[j,i] = alignment_to_sequence[j,i-1]
            else:
                alignment_to_sequence[j,i] = alignment_to_sequence[j,i-1] + 1

    # find alignment 'anchors' from contiguous stretches of matches
    match_threshold = args.matches
    match = 0
    matches = []
    match_start = 0
    match_end = 0

    for i,(a1,a2) in enumerate(alignment.T):
        if a1 == a2:
            if match > 0:
                match += 1
            else:
                match = 1
                match_start = i
        else:
            if match >= match_threshold:
                match_end = i
                matches.append((match_start,match_end))
            match = 0

    # find alignment 'anchors' from contiguous stretches of matches (or indels)
    state_start = 0
    state_counter = 0
    prev_state = 'START'
    anchor_ranges = []
    anchor_type = []

    for i,(a1,a2) in enumerate(alignment.T):
        # options are match/insertion/deletion/mismatch
        if a1 == a2:
            state = 'mat'
        elif a1 == '-':
            state = 'ins'
        elif a2 == '-':
            state = 'del'
        else:
            state = 'mis'

        if prev_state == state and state != 'mis':
            state_counter += 1
        else:
            if prev_state == 'ins' and state_counter > args.indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'del' and state_counter > args.indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'mat' and state_counter > args.matches:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)

            prev_state = state
            state_counter = 0
            state_start = i

    #for i, r in enumerate(anchor_ranges):
    #    print(anchor_type[i], alignment[0, r[0]:r[1]], alignment[1, r[0]:r[1]])

    basecall_boxes = []
    basecall_anchors = []

    # double check boundary conditions, leaving any out at the beginning/end?
    for i,(curr_start, curr_end) in enumerate(anchor_ranges):

        # get anchor sequences
        if anchor_type[i] == 'mat':
            basecall_anchors.append((signal_to_sequence1[alignment_to_sequence[0,curr_start]],''.join(alignment[0,curr_start:curr_end])))
        elif anchor_type[i] == 'ins':
            basecall_anchors.append((signal_to_sequence1[alignment_to_sequence[1,curr_start]],''.join(alignment[1,curr_start:curr_end])))
        elif anchor_type[i] == 'del':
            basecall_anchors.append((signal_to_sequence1[alignment_to_sequence[0,curr_start]],''.join(alignment[0,curr_start:curr_end])))

        if i > 0:
            basecall_boxes.append((
            signal_to_sequence1[alignment_to_sequence[0,anchor_ranges[i-1][1]]],
            signal_to_sequence1[alignment_to_sequence[0,anchor_ranges[i][0]]],
            signal_to_sequence2[alignment_to_sequence[1,anchor_ranges[i-1][1]]],
            signal_to_sequence2[alignment_to_sequence[1,anchor_ranges[i][0]]]
            ))

    assert(abs(len(basecall_boxes) - len(basecall_anchors))==1)

    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        basecalls = pool.starmap(basecall_box, basecall_boxes)

    #print('ANCHORS', basecall_anchors, file=sys.stderr)
    #print('BASECALLS', basecalls, file=sys.stderr)

    # sort each segment by its first signal index
    joined_basecalls = ''.join([i[1] for i in sorted(basecalls + basecall_anchors)])
    print(fasta_format('consensus_from_alignment;'+file1+';'+file2,joined_basecalls))
