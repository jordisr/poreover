import os
import sys
import time
import glob
import copy
import pickle
import argparse
import logging
import progressbar
import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from itertools import starmap
from collections import defaultdict
from multiprocessing import Pool, get_logger

from . import decoding_cpp
from . import decode
from . import pair_decode

def revcomp(s):
    return ''.join([{'A':'T','C':'G','G':'C','T':'A'}[x] for x in s[::-1]])

def ref_envelope_from_cs(cs):
    string_counter = [0,0]
    coord_r = 0
    coord_q = 0
    q2r = []
    r2q = []
    align_coord = []
    operation = ""
    operation_field = ""
    envelope = []

    for i, c in enumerate(cs):
        if c in [':','+','-','*',"~", "="] or (i == len(cs)-1):
            if (i == len(cs)-1):
                operation_field += c
            if operation != "":
                # at the start of next field do something for previous
                if operation == ":":
                    match_length = int(operation_field)
                    string_counter[0] += match_length
                    string_counter[1] += match_length
                    align_coord.append(string_counter[:])
                    for j in range(match_length):
                        coord_r += 1
                        coord_q += 1
                        r2q.append(coord_q)
                        q2r.append(coord_r)

                if operation == "=":
                    match_length = len(operation_field)
                    string_counter[0] += match_length
                    string_counter[1] += match_length
                    align_coord.append(string_counter[:])
                    for j in range(match_length):
                        coord_r += 1
                        coord_q += 1
                        r2q.append(coord_q)
                        q2r.append(coord_r)

                elif operation == "+":
                    # insertion with respect to the reference
                    string_counter[1] += len(operation_field)
                    align_coord.append(string_counter[:])

                    for j in range(len(operation_field)):
                        q2r.append(coord_r)

                    coord_q += len(operation_field)

                elif operation == "-":
                    # deletion with respect to the reference
                    string_counter[0] += len(operation_field)
                    align_coord.append(string_counter[:])

                    for j in range(len(operation_field)):
                        r2q.append(coord_q)

                    coord_r += len(operation_field)

                elif operation == "*":
                    # substitution
                    assert(len(operation_field) == 2)
                    string_counter[0] += 1
                    string_counter[1] += 1
                    align_coord.append(string_counter[:])

                    r2q.append(coord_q)
                    q2r.append(coord_r)
                    coord_r += 1
                    coord_q += 1

                elif operation == "~":
                    pass
            operation = c
            operation_field = ""
        else:
            operation_field += c

    return np.array(r2q), np.array(q2r)

def envelope_from_coord(c, s, w=10):
    envelope = np.zeros(shape=(len(c),2))
    for i,r in enumerate(c):
        if i > 0:
            envelope[i][0] = max(0,c[i-1]-w)
        else:
            envelope[i][0] = max(0,r-w)
        envelope[i][1] = min(s,r+w)

    return envelope.astype(int)


def consensus_helper(in_path, fastq_, logit_paths, args):
    # consensus from single overlap PAF file

    basecalls_1d = ""
    reads = []

    # path to overlap file
    paf_columns = ['query_name','query_length','query_start','query_end','strand','target_name', 'target_length', 'target_start','target_end','number_matches','alignment_length','quality','NM','ms','AS','nn','tp','cm','s1','s2','de', 'rl', 'cs']
    paf = pd.read_csv(in_path, delimiter='\t', names=paf_columns, header=None, usecols=range(len(paf_columns)), index_col=0)

    if args.strand != "both":
        #paf = paf.loc[[x[-2] != "_" for x in paf.index] & (paf["strand"] == args.strand)]
        paf = paf.loc[paf["strand"] == args.strand]
    else:
        #paf = paf.loc[[x[-2] != "_" for x in paf.index]]
        pass

    if (args.max > 0) and (len(paf) > args.max):
        paf = paf.sample(args.max, random_state=args.seed)
    assert(len(paf) > 0)

    # need to standardize input
    if (args.fastq is not None) and (args.bins is not None):
        fastq = read_fastq(os.path.join(os.path.split(in_path)[0], args.fastq))
    else:
        fastq = fastq_

    #print("Total of {} reads in overlap file".format(len(paf)))

    # this could be an option like full overlap only
    # find part of reference present in all reads
    ref_start, ref_end = (max(paf["target_start"]), min(paf["target_end"]))

    all_logits = []
    all_envelopes = []

    for r in paf.itertuples():

        ref2query, query2ref = ref_envelope_from_cs(r.cs[5:])

        assert(len(ref2query) == r.target_end-r.target_start)
        assert(len(query2ref) == r.query_end-r.query_start)

        # split reads labeled e.g. read_1
        read_split = r.Index.split("_")
        read_base = read_split[0]

        # load logits and generate sequence/signal
        if read_base in logit_paths:
            logits = load_logits(logit_paths[read_base])
        else:
            logits = load_logits("{}.npy".format(read_base), base_dir=args.logits)

        if r.strand == "-":
            logits.reverse_complement()
            query_start = r.query_length - r.query_end
            query_end = r.query_length - r.query_start
        else:
            query_start = int(r.query_start)
            query_end = int(r.query_end)

        viterbi_sequence, viterbi_path = logits.viterbi_decode(return_path=True)
        basecalls_1d += decode.fasta_format(r.Index, viterbi_sequence)
        query2signal, _ = pair_decode.get_sequence_mapping(viterbi_path, "bonito")
        assert(len(query2signal) == len(viterbi_sequence))

        basecall_offset = 0
        if len(read_split) > 1:
            if r.strand == "-":
                basecall_bounds = split_read(full_seq=revcomp(viterbi_sequence), sub_seq=fastq[r.Index])
            else:
                basecall_bounds = split_read(full_seq=viterbi_sequence, sub_seq=fastq[r.Index])

            if basecall_bounds is not None:
                if r.strand == "-":
                    basecall_offset = len(viterbi_sequence) - basecall_bounds[1]
                else:
                    basecall_offset = basecall_bounds[0]
            else:
                print("Can't resolve split read with read:{}".format(r.Index))
                continue

        #                 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR full reference
        #                       ^target_start   target_end^
        #                       |-------------------------|     aligned region
        #                          query2ref/ref2query          from cs string
        #                 QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ   full query
        #                       ^query_start     query_end^
        # BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB Viterbi basecall
        #                 ^basecall_offset
        # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL logits

        reads.append(r.Index)

        # only want subset of reference alignment present in all reads
        rel_ref_start = ref_start - int(r.target_start)
        rel_ref_end = ref_end - int(r.target_start)

        raw_query_indices = ref2query[rel_ref_start:rel_ref_end] + query_start + basecall_offset
        raw_signal_indices = np.array([query2signal[min(x, len(query2signal)-1)] for x in raw_query_indices])

        #print(query_signal_start == raw_signal_indices[0], query_signal_end, raw_signal_indices[-1])

        #print(len(raw_query_indices), ref_end-ref_start)

        logits_trim = logits.log_prob[raw_signal_indices[0]:raw_signal_indices[-1]]

        #print(raw_signal_indices[0] >= 0, raw_signal_indices[-1] <= len(logits.log_prob))

        envelope = envelope_from_coord(raw_signal_indices-raw_signal_indices[0], logits_trim.shape[0])

        #print(raw_signal_indices-raw_signal_indices[0])

        all_logits.append(logits_trim)
        all_envelopes.append(envelope)

    if args.debug:
        pickle.dump({"reads":reads, "paf":in_path, "mapping":paf, "reference":r.target_name, "logits":all_logits, "envelopes":all_envelopes, "ref_length":(ref_end-ref_start)}, open("{}.pickle".format(Path(in_path).parent.name), "wb") )

    # run polishing
    fasta_header = '{};{}'.format(Path(in_path).parent.name, len(all_logits))
    sequence = decoding_cpp.cpp_beam_polish(all_logits,"A"*(ref_end-ref_start), all_envelopes, beam_width_=args.beam_width, verbose_=False, length_norm_=args.length_norm)

    if args.bins is not None:
        with open(os.path.join(os.path.split(in_path)[0], "{}.fasta".format(args.out)), 'w') as out_fasta:
            print(decode.fasta_format(fasta_header, sequence), file=out_fasta)

    return (decode.fasta_format(fasta_header, sequence), basecalls_1d)

def load_logits(f, base_dir='.'):
    return decode.model_from_trace(os.path.join(base_dir, f), basecaller="bonito")

def split_read(full_seq, sub_seq):
    # naive way of doing it, not sure how to pull that information
    len_full = len(full_seq)
    len_sub = len(sub_seq)
    sub_start =  full_seq.find(sub_seq)
    rel_start = 0
    rel_end = 0
    if sub_start >= 0:
        rel_start = sub_start
        rel_end = sub_start+len_sub
    if (rel_end > 0) and (rel_end <= len_full):
        return rel_start, rel_end
    else:
        return None

def split_read_mappy(full_seq, sub_seq):
    a = mp.Aligner(seq=full_seq)
    for hit in a.map(sub_seq):
        print("{}\t{}\t{}\t{}".format(hit.ctg, hit.r_st, hit.r_en, hit.cigar_str))

def read_fastq(f):
    fastq = {}
    with open(f, "r") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            fastq[record.id] = str(record.seq)
    return fastq

def consensus(args):
    # set up logger - should make it global
    progressbar.streams.wrap_stderr()
    #logging.basicConfig()
    logger = get_logger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # print software message
    coffee_emoji = u'\U00002615'
    dna_emoji = u'\U0001F9EC'
    logger.info('{0:2}{1:3}{0:2} {2:^30} {0:2}{1:3}{0:2}'.format(coffee_emoji, dna_emoji,'PoreOver consensus'))

    # load FASTQ
    if (args.fastq is not None) and (args.bins is None):
        fastq = read_fastq(args.fastq)
    else:
        fastq = {}

    # look for logits one level deeper if none found in args.logits
    if os.path.isdir(args.logits):
        logit_paths = {}
        if not next(glob.iglob("{}/*.{}".format(args.logits, "npy")), False):
            for f in glob.iglob("{}/*/*.{}".format(args.logits, "npy")):
                logit_paths[os.path.splitext(os.path.split(f)[1])[0]] = f
            if len(logit_paths) < 1:
                sys.exit("No files in --logits")
    else:
        sys.exit("--logits is not a directory")

    # get list of read bin directories
    in_files = []
    if args.bins is not None and os.path.isdir(args.bins):
        in_files = glob.glob("{}/*/{}".format(args.bins, args.paf))
    else:
        in_files = [args.paf]

    if len(in_files) > 1:
        # set up progressbar and manage output
        class callback_helper:
            def __init__(self):
                self.counter = 0
                self.pbar = progressbar.ProgressBar(max_value=len(in_files))
                self.out_f1 = open(args.out+'.fasta','w')
                self.out_f2 = open(args.out+'_1d.fasta','w')
            def callback(self, x):
                self.counter += 1
                self.pbar.update(self.counter)
                print(x[0], file=self.out_f1)
                print(x[1], file=self.out_f2)
        callback_helper_ = callback_helper()

        bullet_point = u'\u25B8'+" "
        logger.info(bullet_point + "found {} read bins for consensus".format(len(in_files)))
        logger.info(bullet_point + "writing sequences to {0}.fasta".format(args.out))
        logger.info(bullet_point + "starting {} decoding processes...".format(args.threads))

        with Pool(processes=args.threads) as pool: # maxtasksperchild=1?
            for p in in_files:
                pool.apply_async(consensus_helper, (p, fastq, logit_paths, args,), callback=callback_helper_.callback)
            pool.close()
            pool.join()

    elif len(in_files) == 1:
        seqs = consensus_helper(in_files[0], fastq, logit_paths, args)
        print(seqs[0])
        #with open(args.out+'.fasta', 'w') as out_fasta:
        #    print(seqs, file=out_fasta)

    else:
        sys.exit("Could not find any reads for consensus!")
