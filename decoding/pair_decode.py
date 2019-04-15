'''
Decode a consensus sequence from a pair of RNN outputs.

Due to prohibitive time/memory costs of running DP algorithms on both reads in
their entirety, the 2D search space is broken into segments which are basecalled
individually and the resulting sequences concatenated.

The method of segentation is determined by the --method flag.

--method align
    Basecall each sequence individually, align, and then use alignment to guide
    consensus basecalling on mismatched/gapped regions. Take contiguous matches or
    indels above the threshold, and use as anchors. Divide signal in between these
    anchors and basecall separately. Finally stitch anchors back with basecalled
    sequences. Thresholds are chosen with --matches and --indels.

        indel             match segment
        ______            ****
        TTTTTA-GCA-GACGCAGGAAGAGACGAA
             |            |||| ||| ||
        -----AGCATACCCAG--GAAG-GACAAA

--method split
    Naively splits 2D search space along main diagonal into blocks chosen with
    the --window parameter.
'''
import numpy as np
from multiprocessing import Pool
import argparse, random, sys, glob, os, re
from scipy.special import logsumexp
#from Bio import pairwise2

import decoding
import align

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

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

def argmax_path(forward):
    seq_max = forward.shape[1]
    forward_indices = np.zeros(seq_max, dtype=int)
    cumul = 1
    for i in range(1,seq_max):
        forward_indices[i] = np.argmax(forward[cumul:,i])+cumul
        cumul = forward_indices[i]
    return(forward_indices)

def viterbi_path(forward):
    (sig_max, seq_max) = forward.shape
    forward_indices = np.zeros(seq_max, dtype=int)
    seq_i, sig_i = 1, 0
    while (0 <= seq_i < seq_max-1) and (0 <= sig_i < sig_max-1):
        next_pos = np.argmax([forward[sig_i+1,seq_i], forward[sig_i,seq_i+1], forward[sig_i+1,seq_i+1]])
        if next_pos > 0:
            forward_indices[seq_i] = sig_i
            seq_i += 1
        if (next_pos == 0) or (next_pos == 1):
            sig_i += 1
    forward_indices[seq_i:] = sig_max
    return(forward_indices)

def basecall1d(y):
    # Perform 1d basecalling and get signal-sequence mapping
    (prefix, forward) = decoding.prefix_search_log_cy(y, return_forward=True)
    try:
        forward_indices = viterbi_path(forward)
    except:
        print('WARNING! Best label is blank! y.shape:{} forward.shape:{} prefix:{}'.format(y.shape, forward.shape, prefix))
        return('',[]) # in case of gap being most probable

    assert(len(prefix) == len(forward_indices))
    assert(np.all(np.diff(forward_indices) >= 0))
    return((prefix,forward_indices))

def basecall_box(logits1, logits2, b,b_tot,u1,u2,v1,v2):
    '''
    Helper function for multiprocessing.
    Consensus basecall 2D region defined by u1/u2/v1/v2:

    (u1,v1) -------- (u1,v2)
       |                |
       |                |
    (u2,v1) -------- (u2,v2)
    '''

    MEM_LIMIT = 1000000000 # 1 GB
    size = (u2-u1+1)*(v2-v1+1)
    assert(size > 0)
    print('\t {}/{} Basecalling box {}-{}x{}-{} (size: {} elements)...'.format(b,b_tot,u1,u2,v1,v2,size),file=sys.stderr)

    if size <= 1:
        return(u1,'')
    elif (u2-u1) < 1:
        return((u1, decoding.prefix_search_log_cy(logits2[v1:v2])[0]))
    elif (v2-v1) < 1:
        return((u1, decoding.prefix_search_log_cy(logits1[u1:u2])[0]))
    elif size*8 > MEM_LIMIT:
        print('ERROR: Box too large to basecall {}-{}:{}-{} (size: {} elements)'.format(u1,u2,v1,v2,size))
        return(u1,'')
    else:
        try:
            #return((u1, decoding.pair_prefix_search(np.exp(logits1[u1:u2]),np.exp(logits2[v1:v2]))[0]))
            return((u1, decoding.pair_prefix_search_log_cy(logits1[u1:u2],logits2[v1:v2])[0]))
        except:
            print('WARNING: Error while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))
            return(u1,'')

def pair_decode(args):
    if len(args.logits) != 2:
        raise "Exactly two reads are required"

    file1 = args.logits[0]
    file2 = args.logits[1]
    print('Read1:{} Read2:{}'.format(file1,file2),file=sys.stderr)

    # reverse complement logits of one read, doesn't matter which one
    logits1_reshape = decoding.decode.load_logits(file1)
    logits2_reshape = decoding.decode.load_logits(file2, reverse_complement=True)

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

    elif args.method == 'split':
        # calculate ranges on which to split read
        # currently just splitting in boxes that follow the main diagonal
        box_ranges = []
        u_step = args.window
        for u in range(u_step,U,u_step):
            box_ranges.append((u-u_step,u,int(V/U*(u-u_step)),int(V/U*u)))
        box_ranges.append((box_ranges[-1][1],U,box_ranges[-1][3],V)) # add in last box with uneven

        print('\t Starting consensus basecalling...',file=sys.stderr)
        starmap_input = []
        for i, b in enumerate(box_ranges):
            starmap_input.append((i,len(box_ranges)-1,b[0],b[1],b[2],b[3]))

        with Pool(processes=args.threads) as pool:
            basecalls = pool.starmap(basecall_box, starmap_input)

        joined_basecalls = ''.join([b[1] for b in basecalls])

    elif args.method == 'align':
        print('\t Performing 1D basecalling...',file=sys.stderr)

        with Pool(processes=args.threads) as pool:
            basecalls1d_1 = pool.map(basecall1d, logits1_reshape)
            for i,out in enumerate(basecalls1d_1):
                if out[0] != '':
                    read1_prefix += out[0]
                    sequence_to_signal1.append(out[1]+logits1_reshape.shape[1]*i)

            basecalls1d_2 = pool.map(basecall1d, logits2_reshape)
            for i,out in enumerate(basecalls1d_2):
                if out[0] != '':
                    read2_prefix += out[0]
                    sequence_to_signal2.append(out[1]+logits2_reshape.shape[1]*i)

        with open(args.out+'.1d.fasta','a') as f:
            print(fasta_format(file1,read1_prefix),file=f)
            print(fasta_format(file2,read2_prefix),file=f)

        sequence_to_signal1 = np.concatenate(np.array(sequence_to_signal1))
        assert(len(sequence_to_signal1) == len(read1_prefix))

        sequence_to_signal2 = np.concatenate(np.array(sequence_to_signal2))
        assert(len(sequence_to_signal2) == len(read2_prefix))

        print('\t Aligning basecalled sequences (Read1 is {} bp and Read2 is {} bp)...'.format(len(read1_prefix),len(read2_prefix)),file=sys.stderr)
        #alignment = pairwise2.align.globalms(read1_prefix, read2_prefix, 2, -1, -.5, -.1)
        alignment = align.global_pair(read1_prefix, read2_prefix)
        alignment = np.array([list(s) for s in alignment[:2]])

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
            starmap_input.append((logits1, logits2, i,len(basecall_boxes)-1,b[0],b[1],b[2],b[3]))

        with Pool(processes=args.threads) as pool:
            basecalls = pool.starmap(basecall_box, starmap_input)

        # sort each segment by its first signal index
        joined_basecalls = ''.join([i[1] for i in sorted(basecalls + basecall_anchors)])

    elif args.method == 'envelope':
        print('\t Performing 1D basecalling...',file=sys.stderr)

        with Pool(processes=args.threads) as pool:
            basecalls1d_1 = pool.map(basecall1d, logits1_reshape)
            for i,out in enumerate(basecalls1d_1):
                if out[0] != '':
                    read1_prefix += out[0]
                    sequence_to_signal1.append(out[1]+logits1_reshape.shape[1]*i)

            basecalls1d_2 = pool.map(basecall1d, logits2_reshape)
            for i,out in enumerate(basecalls1d_2):
                if out[0] != '':
                    read2_prefix += out[0]
                    sequence_to_signal2.append(out[1]+logits2_reshape.shape[1]*i)

        with open(args.out+'.1d.fasta','a') as f:
            print(fasta_format(file1,read1_prefix),file=f)
            print(fasta_format(file2,read2_prefix),file=f)

        sequence_to_signal1 = np.concatenate(np.array(sequence_to_signal1))
        assert(len(sequence_to_signal1) == len(read1_prefix))

        sequence_to_signal2 = np.concatenate(np.array(sequence_to_signal2))
        assert(len(sequence_to_signal2) == len(read2_prefix))

        print('\t Aligning basecalled sequences (Read1 is {} bp and Read2 is {} bp)...'.format(len(read1_prefix),len(read2_prefix)),file=sys.stderr)
        #alignment = pairwise2.align.globalms(read1_prefix, read2_prefix, 2, -1, -.5, -.1)
        alignment = align.global_pair(read1_prefix, read2_prefix)
        alignment = np.array([list(s) for s in alignment[:2]])

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

        if args.debug:
            with open( "debug.p", "wb" ) as pfile:
                import pickle
                pickle.dump({
                'alignment_to_sequence':alignment_to_sequence,
                'sequence_to_signal1':sequence_to_signal1,
                'sequence_to_signal2':sequence_to_signal2,
                'alignment':alignment
                },pfile)

        # prepare data for passing to C++
        y1 = logits1.astype(np.float64)
        y2 = logits2.astype(np.float64)

        # Build envelope
        alignment_col = decoding.pair_envelope_decode.get_alignment_columns(alignment)
        full_envelope = decoding.pair_envelope_decode.build_envelope(y1,y2,alignment_col, sequence_to_signal1, sequence_to_signal2, padding=args.padding)

        # split envelope into subsets
        number_subsets = args.segments
        window = int(len(y1)/number_subsets)
        subsets = np.zeros(shape=(number_subsets, 4)).astype(int)
        s = 0
        u = 0
        while s < number_subsets:
            start = u
            end = u+window
            subsets[s,0] = start
            subsets[s,1] = end
            subsets[s,2] = np.min(full_envelope[start:end,0])
            subsets[s,3] = np.max(full_envelope[start:end,1])
            s += 1
            u = end

        def basecall_subset(subset):
            y1_subset = y1[subset[0]:subset[1]]
            y2_subset = y2[subset[2]:subset[3]]
            subset_envelope = decoding.pair_envelope_decode.offset_envelope(full_envelope, subset)
            subset_envelope = decoding.pair_envelope_decode.pad_envelope(subset_envelope,len(y1_subset), len(y2_subset))
            return(decoding.decoding_cpp.cpp_pair_prefix_search_log(
            y1_subset,
            y2_subset,
            subset_envelope.tolist(),
            "ACGT"))

        print('\t Starting consensus basecalling...',file=sys.stderr)
        with Pool(processes=args.threads) as pool:
            basecalls = pool.map(basecall_subset, subsets)
        joined_basecalls = ''.join([i.decode() for i in basecalls])

    # output final basecalled sequence
    with open(args.out+'.2d.fasta','a') as f:
        print(fasta_format('consensus_{};{};{}'.format(args.method,file1,file2),joined_basecalls), file=f)
