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
from scipy.special import logsumexp
from Bio import pairwise2

import decoding
import cy

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

def get_anchors(alignment, matches, indels):
    # find alignment 'anchors' from contiguous stretches of matches or indels
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
            if prev_state == 'ins' and state_counter >= indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'del' and state_counter >= indels:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)
            if prev_state == 'mat' and state_counter >= matches:
                anchor_ranges.append((state_start,i))
                anchor_type.append(prev_state)

            prev_state = state
            state_counter = 1
            state_start = i

    return(anchor_ranges, anchor_type)

def basecall1d(y):
    # Perform 1d basecalling and get signal-sequence mapping by taking
    # argmax of final forward matrix.
    (prefix, forward) = cy.decoding.prefix_search_log(y, return_forward=True)
    sig_max = forward.shape[0]
    seq_max = forward.shape[1]

    forward_indices = np.zeros(seq_max, dtype=int)
    cumul = 1
    for i in range(1,seq_max):
        forward_indices[i] = np.argmax(forward[cumul:,i])+cumul
        cumul = forward_indices[i]

    '''
    # traceback instead of argmax approach
    forward_indices = np.zeros(seq_max)
    seq_i, sig_i = 1, 0
    while (0 <= seq_i < seq_max-1) and (0 <= sig_i < sig_max-1):
        #print(seq_i, sig_i)
        next_pos = np.argmax([forward[sig_i+1,seq_i], forward[sig_i,seq_i+1], forward[sig_i+1,seq_i+1]])
        if next_pos > 0:
            forward_indices[seq_i] = sig_i
            seq_i += 1
        if (next_pos == 0) or (next_pos == 1):
            sig_i += 1
    forward_indices[-1] = sig_i
    '''

    assert(len(prefix) == len(forward_indices))
    assert(np.all(np.diff(forward_indices) >= 0))
    return((prefix,forward_indices))

def basecall_box(b,b_tot,u1,u2,v1,v2):
    MEM_LIMIT = 1000000000 # 1 GB
    size = (u2-u1+1)*(v2-v1+1)
    '''
    Function to be run in parallel.
    '''
    print('\t {}/{} Basecalling box {}-{}x{}-{} (size: {} elements)...'.format(b,b_tot,u1,u2,v1,v2,size),file=sys.stderr)
    if (u2-u1)+(v2-v1) < 1:
        return(u1,'')
    elif size*8 > MEM_LIMIT:
        print('ERROR: Box too large to basecall {}-{}:{}-{} (size: {} elements)'.format(u1,u2,v1,v2,size))
        return(u1,'')
    else:
        try:
            #return((u1, decoding.pair_prefix_search(np.exp(logits1[u1:u2]),np.exp(logits2[v1:v2]))[0]))
            return((u1, cy.decoding.pair_prefix_search_log(logits1[u1:u2],logits2[v1:v2])[0]))
        except:
            print('WARNING: Error while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))
            return(u1,'')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consensus decoding')
    parser.add_argument('--logits', default='.', help='Paths to both logits', required=True, nargs='+')
    parser.add_argument('--threads', type=int, default=1, help='Processes to use')
    parser.add_argument('--debug', default=False, action='store_true', help='Pickle objects to file for debugging')
    parser.add_argument('--matches', type=int, default=8, help='Match size for building anchors')
    parser.add_argument('--indels', type=int, default=10, help='Indel size for building anchors')
    parser.add_argument('--out', default='out',help='Output file name')
    parser.add_argument('--debug_box', default=False, help='(DEBUGGING) Only bascecall segment in the format u_start-u_end:v_start-v_end. Overrides other options!')

    args = parser.parse_args()

    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]

    # reverse complement logits of one read, doesn't matter which one
    logits1_reshape = load_logits(file1)
    logits2_reshape = load_logits(file2, reverse_complement=True)

    logits1 = np.concatenate(logits1_reshape)
    logits2 = np.concatenate(logits2_reshape)

    U = len(logits1)
    V = len(logits2)

    read1_prefix = ""
    read2_prefix = ""

    sequence_to_signal1 = []
    sequence_to_signal2 = []

    if args.debug_box:
        import re
        re_match = re.match('(\d+)-(\d+):(\d+)-(\d+)',args.debug_box)
        if re_match:
            u1,u2,v1,v2 = int(re_match.group(1)), int(re_match.group(2)), int(re_match.group(3)), int(re_match.group(4))
        else:
            raise "Incorrectly formated box string"

        print(basecall_box(1,1,u1,u2,v1,v2))
        sys.exit()

    print('Read1:{} Read2:{}'.format(file1,file2),file=sys.stderr)
    print('\t Performing 1D basecalling...',file=sys.stderr)

    with Pool(processes=args.threads) as pool:
        basecalls1d_1 = pool.map(basecall1d, logits1_reshape)
        for i,out in enumerate(basecalls1d_1):
            read1_prefix += out[0]
            sequence_to_signal1.append(out[1]+logits1_reshape.shape[1]*i)

        basecalls1d_2 = pool.map(basecall1d, logits2_reshape)
        for i,out in enumerate(basecalls1d_2):
            read2_prefix += out[0]
            sequence_to_signal2.append(out[1]+logits2_reshape.shape[1]*i)

    with open(args.out+'.1d.fasta','a') as f:
        print(fasta_format(file1,read1_prefix),file=f)
        print(fasta_format(file2,read2_prefix),file=f)

    sequence_to_signal1 = np.concatenate(np.array(sequence_to_signal1))
    assert(len(sequence_to_signal1) == len(read1_prefix))

    sequence_to_signal2 = np.concatenate(np.array(sequence_to_signal2))
    assert(len(sequence_to_signal2) == len(read2_prefix))

    print('\t Aligning basecalled sequences...',file=sys.stderr)
    alignment = pairwise2.align.globalms(read1_prefix, read2_prefix, 2, -1, -.5, -.1)
    alignment = np.array([list(s) for s in alignment[0][:2]])
    print('\t Read sequence identity: {}'.format(np.sum(alignment[0] == alignment[1]) / len(alignment[0])), file=sys.stderr)

    # get alignment_to_sequence mapping
    alignment_to_sequence = np.zeros(shape=alignment.shape,dtype=int)
    for i,col in enumerate(alignment.T):
        # no boundary case for first element but it will wrap around to the last (which is zero)
        for s in range(2):
            if col[s] == '-':
                alignment_to_sequence[s,i] = alignment_to_sequence[s,i-1]
            else:
                alignment_to_sequence[s,i] = alignment_to_sequence[s,i-1] + 1

    anchor_ranges, anchor_type = get_anchors(alignment, matches=args.matches, indels=args.indels)

    basecall_boxes = []
    basecall_anchors = []

    for i,(curr_start, curr_end) in enumerate(anchor_ranges):

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

    assert len(anchor_ranges) > 0, 'No matches/indels of sufficient length found in alignment. Try decreasing --matches or --indels'

    # add last box on the end
    basecall_boxes.append((
    sequence_to_signal1[alignment_to_sequence[0,anchor_ranges[-1][1]]],
    U,
    sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[-1][1]]],
    V))
    assert(abs(len(basecall_boxes) - len(basecall_anchors))==1)

    if args.debug:
        with open( "debug.p", "wb" ) as pfile:
            import pickle
            pickle.dump({
            'alignment_to_sequence':alignment_to_sequence,
            'sequence_to_signal1':sequence_to_signal1,
            'sequence_to_signal2':sequence_to_signal2,
            'alignment':alignment,
            'basecall_boxes':basecall_boxes,
            'basecall_anchors':basecall_anchors,
            'anchor_ranges':anchor_ranges
            },pfile)

    print('\t Starting consensus basecalling...',file=sys.stderr)
    starmap_input = []
    for i, b in enumerate(basecall_boxes):
        starmap_input.append((i,len(basecall_boxes)-1,b[0],b[1],b[2],b[3]))

    NUM_THREADS = args.threads
    with Pool(processes=NUM_THREADS) as pool:
        basecalls = pool.starmap(basecall_box, starmap_input)

    # sort each segment by its first signal index
    joined_basecalls = ''.join([i[1] for i in sorted(basecalls + basecall_anchors)])

    with open(args.out+'.2d.fasta','a') as f:
        print(fasta_format('consensus_from_alignment;'+file1+';'+file2,joined_basecalls), file=f)
