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

def get_sequence_mapping(path, kind):
    signal_to_sequence = []
    sequence_to_signal = []
    label_len = 0
    if kind is 'poreover':
        for i, p in enumerate(path):
            if p < 4:
                sequence_to_signal.append(i)
                signal_to_sequence.append(label_len)
                label_len += 1
    elif kind is 'flipflop':
        for i, p in enumerate(path):
            if i == 0:
                sequence_to_signal.append(i)
                signal_to_sequence.append(label_len)
            else:
                if path[i] != path[i-1]:
                    label_len += 1
                    sequence_to_signal.append(i)
                signal_to_sequence.append(label_len)
    return(sequence_to_signal, signal_to_sequence)

def _beam_search_2d(logits1, logits2, b, b_tot, u1, u2, v1, v2):
    size = (u2-u1+1)*(v2-v1+1)
    print('\t {}/{} Basecalling box {}-{}x{}-{} (size: {} elements)...'.format(b,b_tot,u1,u2,v1,v2,size),file=sys.stderr)
    seq = decoding.decoding_cpp.cpp_beam_search_2d_by_row(
    logits1[u1:u2],
    logits2[v1:v2])
    return((u1, seq))

def _beam_search_2d_envelope(y1_subset, y2_subset, subset_envelope):
    return(decoding.decoding_cpp.cpp_beam_search_2d_by_row(
    y1_subset,
    y2_subset,
    subset_envelope.tolist()))

def _prefix_search_1d(y):
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

def _prefix_search_2d(logits1, logits2, b, b_tot, u1, u2, v1, v2):
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
            return((u1, decoding.pair_prefix_search_log_cy(logits1[u1:u2],logits2[v1:v2])[0]))
        except:
            print('WARNING: Error while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))
            return(u1,'')

def _prefix_search_2d_envelope(y1_subset, y2_subset, subset_envelope):
    return(decoding.decoding_cpp.cpp_pair_prefix_search_log(
    y1_subset,
    y2_subset,
    subset_envelope.tolist(),
    "ACGT"))

def pair_decode(args):
    in_path = getattr(args, 'in')
    if len(in_path) != 2:
        raise "Exactly two reads are required"

    print('Read1:{} Read2:{}'.format(in_path[0], in_path[1]),file=sys.stderr)

    model1 = decoding.decode.model_from_trace(in_path[0])
    model2 = decoding.decode.model_from_trace(in_path[1])
    model2.reverse_complement()

    assert(model1.kind == model2.kind)

    if args.method == 'split':
        # calculate ranges on which to split read
        # currently just splitting in boxes that follow the main diagonal
        U = model1.t_max
        V = model2.t_max
        box_ranges = []
        u_step = args.window
        for u in range(u_step,U,u_step):
            box_ranges.append((u-u_step,u,int(V/U*(u-u_step)),int(V/U*u)))
        box_ranges.append((box_ranges[-1][1],U,box_ranges[-1][3],V)) # add in last box with uneven

        print('\t Starting consensus basecalling...',file=sys.stderr)
        starmap_input = []
        for i, b in enumerate(box_ranges):
            starmap_input.append((model1, model2, i,len(box_ranges)-1,b[0],b[1],b[2],b[3]))

        assert(model1.kind == 'poreover')
        with Pool(processes=args.threads) as pool:
            if args.algorithm == 'beam':
                parallel_fn = _beam_search_2d
            elif args.algorithm == 'prefix':
                parallel_fn = _prefix_search_2d
            basecalls = pool.starmap(parallel_fn, starmap_input)

        joined_basecalls = ''.join([b[1] for b in basecalls])

    else:
        print('\t Performing 1D basecalling...',file=sys.stderr)

        basecall1, _viterbi_path = model1.viterbi_decode(return_path=True)
        sequence_to_signal1, _ = get_sequence_mapping(_viterbi_path, model1.kind)
        assert(len(sequence_to_signal1) == len(basecall1))

        basecall2, _viterbi_path = model2.viterbi_decode(return_path=True)
        sequence_to_signal2, _ = get_sequence_mapping(_viterbi_path, model2.kind)
        assert(len(sequence_to_signal2) == len(basecall2))

        with open(args.out+'.1d.fasta','a') as f:
            print(fasta_format(in_path[0], basecall1),file=f)
            print(fasta_format(in_path[1], basecall2),file=f)

        print('\t Aligning basecalled sequences (Read1 is {} bp and Read2 is {} bp)...'.format(len(basecall1),len(basecall2)),file=sys.stderr)
        #alignment = pairwise2.align.globalms(, , 2, -1, -.5, -.1)
        alignment = align.global_pair(basecall1, basecall2)
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

    if args.method == 'align':

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
        model1.t_max,
        sequence_to_signal2[alignment_to_sequence[1,anchor_ranges[-1][1]]],
        model2.t_max))
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
            starmap_input.append((model1, model2, i,len(basecall_boxes)-1,b[0],b[1],b[2],b[3]))

        assert(model1.kind == 'poreover')
        with Pool(processes=args.threads) as pool:
            if args.algorithm == 'beam':
                parallel_fn = _beam_search_2d
            elif args.algorithm == 'prefix':
                parallel_fn = _prefix_search_2d
            basecalls = pool.starmap(parallel_fn, starmap_input)

        # sort each segment by its first signal index
        joined_basecalls = ''.join([i[1] for i in sorted(basecalls + basecall_anchors)])

    elif args.method == 'envelope':

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
        y1 = model1.log_prob
        y2 = model2.log_prob

        # Build envelope
        alignment_col = decoding.envelope.get_alignment_columns(alignment)
        full_envelope = decoding.envelope.build_envelope(y1,y2,alignment_col, sequence_to_signal1, sequence_to_signal2, padding=args.padding)

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

        starmap_input = []
        for subset in subsets:
            y1_subset = y1[subset[0]:subset[1]]
            y2_subset = y2[subset[2]:subset[3]]
            subset_envelope = decoding.envelope.offset_envelope(full_envelope, subset)
            subset_envelope = decoding.envelope.pad_envelope(subset_envelope, len(y1_subset), len(y2_subset))
            starmap_input.append( (y1_subset, y2_subset, subset_envelope) )

        assert(model1.kind == 'poreover')
        print('\t Starting consensus basecalling...',file=sys.stderr)
        with Pool(processes=args.threads) as pool:
            if args.algorithm == 'beam':
                parallel_fn = _beam_search_2d_envelope
            elif args.algorithm == 'prefix':
                parallel_fn = _prefix_search_2d_envelope
            basecalls = pool.starmap(parallel_fn, starmap_input)
        joined_basecalls = ''.join(basecalls)

    # output final basecalled sequence
    with open(args.out+'.2d.fasta','a') as f:
        print(fasta_format('consensus_{};{};{}'.format(args.method,in_path[0],in_path[1]), joined_basecalls), file=f)
