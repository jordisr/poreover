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
from multiprocessing import Pool, get_logger
import argparse, random, sys, glob, os, re
from scipy.special import logsumexp
#from Bio import pairwise2
import logging
import copy
import progressbar
from itertools import starmap

from . import decode
from . import decoding_cpp
from . import envelope
import poreover.align as align

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
    elif kind is 'bonito':
        for i, p in enumerate(path):
            if p == 4 or path[i] == path[i-1]:
                pass
            else:
                sequence_to_signal.append(i)
                signal_to_sequence.append(label_len)
                label_len += 1
    return(sequence_to_signal, signal_to_sequence)

class parallel_decoder:
    def __init__(self, args, kind):
        self.args = args
        self.kind = {'poreover':'ctc', 'guppy':'ctc_flipflop', 'flappie':'ctc_flipflop', 'bonito':'ctc_merge_repeats'}[self.args.basecaller]

    def _beam_search_2d(self, logits1, logits2, b, b_tot, u1, u2, v1, v2):
        size = (u2-u1+1)*(v2-v1+1)
        print('\t {}/{} Basecalling box {}-{}x{}-{} (size: {} elements)...'.format(b,b_tot,u1,u2,v1,v2,size),file=sys.stderr)
        if size <= 1:
            return(u1,'')
        elif (u2-u1) < 1:
            return((u1, decoding.prefix_search_log_cy(logits2[v1:v2])[0]))
        elif (v2-v1) < 1:
            return((u1, decoding.prefix_search_log_cy(logits1[u1:u2])[0]))
        else:
            seq = decoding_cpp.cpp_beam_search_2d(
            logits1[u1:u2],
            logits2[v1:v2],
            beam_width_=self.args.beam_width,
            model_=self.kind)
            return((u1, seq))

    def _beam_search_2d_envelope(self, y1_subset, y2_subset, subset_envelope):
        return(decoding_cpp.cpp_beam_search_2d(
        y1_subset,
        y2_subset,
        subset_envelope.tolist(),
        beam_width_=self.args.beam_width,
        model_=self.kind))

    def _prefix_search_1d(self, y):
        # Perform 1d basecalling and get signal-sequence mapping
        (prefix, forward) = decoding.prefix_search_log_cy(y, return_forward=True)
        try:
            forward_indices = viterbi_path(forward)
        except:
            logger.warning('WARNING! Best label is blank! y.shape:{} forward.shape:{} prefix:{}'.format(y.shape, forward.shape, prefix))
            return('',[]) # in case of gap being most probable

        assert(len(prefix) == len(forward_indices))
        assert(np.all(np.diff(forward_indices) >= 0))
        return((prefix,forward_indices))

    def _prefix_search_2d(self, logits1, logits2, b, b_tot, u1, u2, v1, v2):
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
            logger.error('ERROR: Box too large to basecall {}-{}:{}-{} (size: {} elements)'.format(u1,u2,v1,v2,size))
            return(u1,'')
        else:
            try:
                return((u1, decoding.pair_prefix_search_log_cy(logits1[u1:u2],logits2[v1:v2])[0]))
            except:
                logger.warning('WARNING: Error while basecalling box {}-{}:{}-{}'.format(u1,u2,v1,v2))
                return(u1,'')

    def _prefix_search_2d_envelope(self, y1_subset, y2_subset, subset_envelope):
        return(decoding_cpp.cpp_pair_prefix_search_log(
        y1_subset,
        y2_subset,
        subset_envelope.tolist(),
        "ACGT"))

    def get_function(self):
        if self.args.algorithm == 'beam':
            if self.args.method == 'envelope':
                return(self._beam_search_2d_envelope)
            else:
                return(self._beam_search_2d)
        elif self.args.algorithm == 'prefix':
            assert(self.kind == "poreover")
            if self.args.method == 'envelope':
                return(self._prefix_search_2d_envelope)
            else:
                return(self._prefix_search_2d)

def pair_decode(args):

    # set up logger - should make it global
    progressbar.streams.wrap_stderr()
    #logging.basicConfig()
    logger = get_logger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # print software message, should incorporate to other subroutines as well
    coffee_emoji = u'\U00002615'
    dna_emoji = u'\U0001F9EC'
    logger.info('{0:2}{1:3}{0:2} {2:^30} {0:2}{1:3}{0:2}'.format(coffee_emoji, dna_emoji,'PoreOver pair-decode (version 0.0)'))
    #logger.info(('{0:2}{1:3}'*9+'{0:2}').format(coffee_emoji, dna_emoji))

    in_path = getattr(args, 'in')
    if len(in_path) == 1:
        args_list = []
        with open(in_path[0], 'r') as read_pairs:
            for n, line in enumerate(read_pairs):
                args_copy = copy.deepcopy(args)
                setattr(args_copy, 'in', line.split())
                #args_copy.out = "pair{}".format(n)
                args_list.append(args_copy)

        # set up progressbar and manage output
        class callback_helper:
            def __init__(self):
                self.counter = 0
                self.pbar = progressbar.ProgressBar(max_value=len(args_list))
                self.out_1d_f = open(args.out+'.1d.fasta','w')
                self.out_2d_f = open(args.out+'.2d.fasta','w')
                self.log_f = open(args.out+'.log','w',1)
                print('# PoreOver pair-decode', file=self.log_f)
                print('# '+str(vars(args)), file=self.log_f)
                print('# '+'\t'.join(map(str,["read1", "read2", "length1", "length2", "sequence_identity"])), file=self.log_f)
            def callback(self, x):
                self.counter += 1
                self.pbar.update(self.counter)
                if len(x) == 3:
                    print(x[0], file=self.out_1d_f)
                    print(x[1], file=self.out_2d_f)
                    print('\t'.join(map(str,[x[2][k] for k in ["read1", "read2", "length1", "length2", "sequence_identity"]])), file=self.log_f)
        callback_helper_ = callback_helper()

        bullet_point = u'\u25B8'+" "
        logger.info(bullet_point + "found {} read pairs in {}".format(len(args_list), in_path[0]))
        logger.info(bullet_point + "writing sequences to {0}.1d.fasta and {0}.2d.fasta".format(args.out))
        logger.info(bullet_point + "pair alignment statistics saved to {}.log".format(args.out))
        logger.info(bullet_point + "starting {} decoding processes...".format(args.threads))

        with Pool(processes=args.threads) as pool:
            #basecalls = pool.map(pair_decode_helper, args_list) #works but no logging
            for i, arg in enumerate(args_list):
                pool.apply_async(pair_decode_helper, (args_list[i],), callback=callback_helper_.callback)
            pool.close()
            pool.join()

    else:
        seqs_1d, seq_2d, summary = pair_decode_helper(args)
        print(summary, file=sys.stderr)
        with open(args.out+'.fasta', 'w') as out_fasta:
            print(seq_2d, file=out_fasta)

def pair_decode_helper(args):
    #logger = getattr(args, 'logger') # should set it globally but just testing for now
    logger = get_logger() # get multiprocessing logger
    in_path = getattr(args, 'in')
    if len(in_path) != 2:
        logger.error("ERROR: Exactly two reads are required")

    logger.debug('Read1:{} Read2:{}'.format(in_path[0], in_path[1]))
    model1 = decode.model_from_trace(os.path.join(args.dir, in_path[0]), args.basecaller)
    model2 = decode.model_from_trace(os.path.join(args.dir, in_path[1]), args.basecaller)
    if args.reverse_complement:
        model2.reverse_complement()

    assert(model1.kind == model2.kind)

    # get appropriate helper function for multiprocessing
    decoding_fn = parallel_decoder(args, model1.kind).get_function()
    pair_decode_summary = dict()

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

        logger.debug('\t Starting consensus basecalling...')
        starmap_input = []
        for i, b in enumerate(box_ranges):
            starmap_input.append((model1, model2, i,len(box_ranges)-1,b[0],b[1],b[2],b[3]))

        #with Pool(processes=args.threads) as pool:
        #    basecalls = pool.starmap(decoding_fn, starmap_input)
        basecalls = starmap(decoding_fn, starmap_input)

        joined_basecalls = ''.join([b[1] for b in basecalls])

    else:
        logger.debug('\t Performing 1D basecalling...')

        if args.single == 'viterbi':
            basecall1, viterbi_path1 = model1.viterbi_decode(return_path=True)
            basecall2, viterbi_path2 = model2.viterbi_decode(return_path=True)
        elif args.single == 'beam':
            print("Basecalling 1")
            basecall1 = decoding_cpp.cpp_beam_search(model1.log_prob)
            print("Resquiggling 1")
            viterbi_path1 = decoding_cpp.cpp_viterbi_acceptor(model1.log_prob, basecall1, band_size=1000)
            print("Basecalling 2")
            basecall2 = decoding_cpp.cpp_beam_search(model2.log_prob)
            viterbi_path2 = decoding_cpp.cpp_viterbi_acceptor(model2.log_prob, basecall2, band_size=1000)

        sequence_to_signal1, _ = get_sequence_mapping(viterbi_path1, model1.kind)
        assert(len(sequence_to_signal1) == len(basecall1))

        sequence_to_signal2, _ = get_sequence_mapping(viterbi_path2, model2.kind)
        assert(len(sequence_to_signal2) == len(basecall2))

        #if not getattr(args, 'unittest', False):
        #    with open(args.out+'.1d.fasta','a') as f:
        #        print(fasta_format(in_path[0], basecall1),file=f)
        #        print(fasta_format(in_path[1], basecall2),file=f)

        logger.debug('\t Aligning basecalled sequences (Read1 is {} bp and Read2 is {} bp)...'.format(len(basecall1),len(basecall2)))
        #alignment = pairwise2.align.globalms(, , 2, -1, -.5, -.1)
        alignment = align.global_pair(basecall1, basecall2)
        alignment = np.array([list(s) for s in alignment[:2]])
        sequence_identity = np.sum(alignment[0] == alignment[1]) / len(alignment[0])
        logger.debug('\t Read sequence identity: {}'.format(sequence_identity))

        pair_decode_summary = {'read1':in_path[0], 'read2':in_path[1], 'length1':len(basecall1), 'length2':len(basecall2), 'sequence_identity':sequence_identity}

        if sequence_identity < 0.5:
            logger.error("ERROR: Pairwise sequence identity between reads is below 50%. Did you mean to take the --reverse-complement of one of the reads?")
            return ()

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

        logger.debug('\t Starting consensus basecalling...')
        starmap_input = []
        for i, b in enumerate(basecall_boxes):
            starmap_input.append((model1, model2, i,len(basecall_boxes)-1,b[0],b[1],b[2],b[3]))

        #with Pool(processes=args.threads) as pool:
        #    basecalls = pool.starmap(decoding_fn, starmap_input)
        basecalls = starmap(decoding_fn, starmap_input)

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
        alignment_col = envelope.get_alignment_columns(alignment)
        alignment_envelope = envelope.build_envelope(y1,y2,alignment_col, sequence_to_signal1, sequence_to_signal2, padding=args.padding)

        logger.debug('\t Starting consensus basecalling...')
        joined_basecalls = decoding_fn(y1, y2, alignment_envelope)

    # output final basecalled sequence
    #if not getattr(args, 'unittest', False):
    #    with open(args.out+'.2d.fasta','a') as f:
    #        print(fasta_format('consensus_{};{};{}'.format(args.method,in_path[0],in_path[1]), joined_basecalls), file=f)

    # return formatted strings but do output in main pair_decode function
    return (fasta_format(in_path[0], basecall1)+fasta_format(in_path[1], basecall2), fasta_format('consensus_{};{};{}'.format(args.method,in_path[0],in_path[1]), joined_basecalls), pair_decode_summary)
    #return((basecall1, basecall2), joined_basecalls)
