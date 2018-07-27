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
from Bio import pairwise2

import decoding

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
    #print(read_reshape.shape)
    if np.abs(np.sum(read_reshape[0,0])) > 1:
        print('WARNING: Logits are not probabilities. Running softmax operation.',file=sys.stderr)
        read_reshape = softmax(read_reshape)
    if reverse_complement:
        # logit reordering: (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        read_reshape = read_reshape[::-1,::-1,[3,2,1,0,4]]
    return(read_reshape)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--logits', default='.', help='Paths to both logits', required=True, nargs='+')
    parser.add_argument('--logits_size', type=int, default=200, help='Window width used for basecalling')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    parser.add_argument('--matches', type=int, default=8, help='Match size for building anchors')
    parser.add_argument('--indels', type=int, default=10, help='Indel size for building anchors')
    parser.add_argument('--out', default='out',help='Output file name')
    args = parser.parse_args()

    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]

    # reverse complement logist of one read, doesn't matter which one
    logits1_reshape = load_logits(file1, window=args.logits_size)
    logits2_reshape = load_logits(file2, reverse_complement=True,window=args.logits_size)

    # smaller test data for your poor laptop
    #logits1_reshape = logits1_reshape[:10]
    #logits2_reshape = logits2_reshape[:10]

    logits1 = np.concatenate(logits1_reshape)
    logits2 = np.concatenate(logits2_reshape)

    U = len(logits1)
    V = len(logits2)

    read1_prefix = ""
    read2_prefix = ""

    sequence_to_signal1 = []
    sequence_to_signal2 = []

    print('Performing 1D basecalling...',file=sys.stderr)

    def basecall1d(y):
        # Perform 1d basecalling and get signal-sequence mapping by taking
        # argmax of final forward matrix.
        (prefix, forward) = decoding.prefix_search_log(y, return_forward=True)
        s_len = len(prefix)
        print(s_len, forward.shape)
        forward_indices = np.argmax(forward,axis=0)

        assert(s_len == len(forward_indices))
        return((prefix,forward_indices))

    with Pool(processes=args.threads) as pool:
        basecalls1d_1 = pool.map(basecall1d, logits1_reshape)
        for i,out in enumerate(basecalls1d_1):
            read1_prefix += out[0]
            sequence_to_signal1.append(out[1]+args.logits_size*i)

        basecalls1d_2 = pool.map(basecall1d, logits2_reshape)
        for i,out in enumerate(basecalls1d_2):
            read2_prefix += out[0]
            sequence_to_signal2.append(out[1]+args.logits_size*i)

    with open(args.out+'.1d.fasta','a') as f:
        print(fasta_format(file1,read1_prefix),file=f)
        print(fasta_format(file2,read2_prefix),file=f)

    sequence_to_signal1 = np.concatenate(np.array(sequence_to_signal1))
    assert(len(sequence_to_signal1) == len(read1_prefix))

    sequence_to_signal2 = np.concatenate(np.array(sequence_to_signal2))
    assert(len(sequence_to_signal2) == len(read2_prefix))

    print('Aligning basecalled sequences...',file=sys.stderr)
    alignment = pairwise2.align.globalms(read1_prefix, read2_prefix, 2, -1, -.5, -.1)
    alignment = np.array([list(s) for s in alignment[0][:2]])
    print('\tRead sequence identity: {}'.format(np.sum(alignment[0] == alignment[1]) / len(alignment[0])), file=sys.stderr)

    # get alignment_to_sequence mapping
    alignment_to_sequence = np.zeros(shape=alignment.shape,dtype=int)
    for i,col in enumerate(alignment.T):
        # no boundary case for first element but it will wrap around to the last (which is zero)
        for s in range(2):
            if col[s] == '-':
                alignment_to_sequence[s,i] = alignment_to_sequence[s,i-1]
            else:
                alignment_to_sequence[s,i] = alignment_to_sequence[s,i-1] + 1

    #print('LOGITS 1 -- size:{} SEQ_LENGTH:{} sequence_to_signal:{} ALGN_TO_SEQUENCE:{}'.format(U,len(sequence_to_signal1), len(read1_prefix), len(alignment_to_sequence[0])))
    #print('LOGITS 1 -- size:{} SEQ_LENGTH:{} sequence_to_signal:{} ALGN_TO_SEQUENCE:{}'.format(V,len(sequence_to_signal2), len(read2_prefix), len(alignment_to_sequence[0])))

    '''
    # find alignment 'anchors' from contiguous stretches of `matches`
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
    '''

    # find alignment 'anchors' from contiguous stretches of matches (or indels)
    state_start = 0
    state_counter = 1
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
            if prev_state == 'ins' and state_counter >= args.indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'del' and state_counter >= args.indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'mat' and state_counter >= args.matches:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)

            prev_state = state
            state_counter = 1
            state_start = i

    basecall_boxes = []
    basecall_anchors = []

    for i,(curr_start, curr_end) in enumerate(anchor_ranges):

        #print(i,curr_start,curr_end,anchor_type[i])

        # get anchor sequences
        if anchor_type[i] == 'mat':
            basecall_anchors.append((sequence_to_signal1[alignment_to_sequence[0,curr_start]], ''.join(alignment[0,curr_start:curr_end])))
        elif anchor_type[i] == 'ins':
            basecall_anchors.append((sequence_to_signal1[alignment_to_sequence[0,curr_start]], ''.join(alignment[1,curr_start:curr_end])))
        elif anchor_type[i] == 'del':
            basecall_anchors.append((sequence_to_signal1[alignment_to_sequence[0,curr_start]], ''.join(alignment[0,curr_start:curr_end])))

        if i > 0:
            basecall_boxes.append((
            sequence_to_signal1[alignment_to_sequence[0,anchor_ranges[i-1][1]]],
            sequence_to_signal1[alignment_to_sequence[0,anchor_ranges[i][0]]],
            sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[i-1][1]]],
            sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[i][0]]]
            ))
        else:
            basecall_boxes.append((
            0,
            sequence_to_signal1[alignment_to_sequence[0,anchor_ranges[i][0]]],
            0,
            sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[i][0]]]
            ))

    #for i, r in enumerate(anchor_ranges):
    #    print(anchor_type[i], r, alignment[0, r[0]:r[1]], alignment[1, r[0]:r[1]])

    assert len(anchor_ranges) > 0, 'No matches/indels of sufficient length found in alignment. Try decreasing --matches or --indels'

    # add last box on the end
    basecall_boxes.append((
    sequence_to_signal1[alignment_to_sequence[0,anchor_ranges[-1][1]]],
    U,
    sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[-1][1]]],
    V))
    assert(abs(len(basecall_boxes) - len(basecall_anchors))==1)

    print('Starting consensus basecalling...',file=sys.stderr)
    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        basecalls = pool.starmap(basecall_box, basecall_boxes)

    # code for debuggging
    '''
    print('*'*80)
    for i, r in enumerate(anchor_ranges):
        print(anchor_type[i], r, alignment[0, r[0]:r[1]], alignment[1, r[0]:r[1]])
    print('ANCHOR_RANGES',anchor_ranges)
    print('BASECALL_BOXES',basecall_boxes)
    print('ANCHORS', basecall_anchors, file=sys.stderr)
    print('BASECALLS', basecalls, file=sys.stderr)
    print('*'*80)
    '''

    # sort each segment by its first signal index
    joined_basecalls = ''.join([i[1] for i in sorted(basecalls + basecall_anchors)])

    with open(args.out+'.2d.fasta','a') as f:
        print(fasta_format('consensus_from_alignment;'+file1+';'+file2,joined_basecalls), file=f)
