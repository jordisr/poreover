import sys, os, argparse, pickle
import numpy as np
import mappy as mp
import pandas as pd
from Bio import SeqIO

def fasta_format(name, seq, width=60):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

def get_top_hit(aligner, seq):
    alignment = aligner.map(seq, cs=True)
    for hit in alignment:
        return hit
    return None

def reverse_complement(seq):
    complement = {'A':'T','C':'G','G':'C','T':'A'}
    return ''.join([complement[x] for x in seq][::-1])

def get_homopolymers(seq, k=2):
    homopolymers = []
    homopolymer_length = 0
    homopolymer_base = ""
    homopolymer_pos = 0
    for i, base in enumerate(seq):
        if base == homopolymer_base:
            homopolymer_length += 1
        else:
            if homopolymer_base != "" and homopolymer_length >= k:
                homopolymers.append([homopolymer_pos, homopolymer_base, homopolymer_length])
            homopolymer_base = base
            homopolymer_length = 1
            homopolymer_pos = i
    return homopolymers

def get_homopolymers_alignment(ref, query, k=2):
    homopolymers = []
    homopolymer_length = 0
    homopolymer_base = ""
    homopolymer_start = 0
    homopolymer_end = 0
    for i, base in enumerate(ref):
        if base == "-":
            continue
        if base == homopolymer_base:
            homopolymer_length += 1
        else:
            if homopolymer_base != "" and homopolymer_length >= k:
                homopolymer_end = i
                ref_seq = ref[homopolymer_start:homopolymer_end]
                query_seq = str(query[homopolymer_start:homopolymer_end])
                homopolymers.append([homopolymer_base, homopolymer_length, ref_seq.replace('-',''), query_seq.replace('-','')])
            homopolymer_base = base
            homopolymer_length = 1
            homopolymer_start = i

    homopolymer_summary = {'match':0, 'insertion':0, 'deletion':0, 'mismatch':0, 'bases_inserted':0, 'bases_deleted':0, 'total':0,'ref_bases':0}
    for h in homopolymers:
        r_bases = h[2]
        q_bases = h[3]
        homopolymer_summary['total'] += 1
        homopolymer_summary['ref_bases'] += h[1]
        if r_bases == q_bases:
            homopolymer_summary['match'] += 1
        elif len(r_bases) < len(q_bases):
            homopolymer_summary['insertion'] += 1
            homopolymer_summary['bases_inserted'] += len(q_bases) - len(r_bases)
        elif len(r_bases) > len(q_bases):
            homopolymer_summary['deletion'] += 1
            homopolymer_summary['bases_deleted'] += len(r_bases) - len(q_bases)
        else:
            # if both indel and substitution (e.g. AAA > AC) will get counted as indel
            homopolymer_summary['mismatch'] += 1

    #print(homopolymer_summary)
    return homopolymer_summary

def parse_cigar(hit, q_seq, r_seq):
    cigar = hit.cigar
    summary = {'insertion':0,'deletion':0,'mismatch':0,'match':0}
    for i, c in enumerate(cigar):
        if c[1] == 0:
            summary['match'] += c[0]
        elif c[1] == 1:
            summary['insertion'] += c[0]
        elif c[1] == 2:
            summary['deletion'] += c[0]
    return summary

def parse_cs(hit, q_seq, r_seq):
    cs = hit.cs
    summary = {'insertion':0,'deletion':0,'mismatch':0,'match':0}
    summary_string = ["","", ""]
    string_counter = [0,0]
    seq_to_align = [[],[]]
    operation = ""
    operation_field = ""

    q2r = []
    r2q = []
    coord_r = 0
    coord_q = 0

    error_context = {'insertion':[], 'deletion':[], 'mismatch':[]}

    for i, c in enumerate(cs):
        if c in [':','+','-','*',"~"] or (i == len(cs)-1):
            if (i == len(cs)-1):
                operation_field += c
            if operation != "":
                # at the start of next field do something for previous
                if operation == ":":
                    match_length = int(operation_field)
                    summary_string[0] += r_seq[string_counter[0]:string_counter[0]+match_length].upper()
                    summary_string[1] += q_seq[string_counter[1]:string_counter[1]+match_length].upper()
                    string_counter[0] += match_length
                    string_counter[1] += match_length
                    summary_string[2] += ("|"*match_length)
                    summary['match'] += match_length

                    for j in range(match_length):
                        coord_r += 1
                        coord_q += 1
                        r2q.append(coord_q)
                        q2r.append(coord_r)

                elif operation == "+":
                    # insertion with respect to the reference
                    error_context['insertion'].append([*string_counter, len(operation_field)])
                    summary_string[0] += "-"*len(operation_field)
                    summary_string[1] += operation_field.upper()
                    string_counter[1] += len(operation_field)
                    summary_string[2] += " "*len(operation_field)
                    summary['insertion'] += len(operation_field)

                    for j in range(len(operation_field)):
                        q2r.append(coord_r)
                    coord_q += len(operation_field)

                elif operation == "-":
                    # deletion with respect to the reference
                    error_context['deletion'].append([*string_counter, len(operation_field)])
                    summary_string[0] += operation_field.upper()
                    summary_string[1] += "-"*len(operation_field)
                    string_counter[0] += len(operation_field)
                    summary_string[2] += " "*len(operation_field)
                    summary['deletion'] += len(operation_field)

                    for j in range(len(operation_field)):
                        r2q.append(coord_q)
                    coord_r += len(operation_field)

                elif operation == "*":
                    # substitution
                    assert(len(operation_field) == 2)
                    summary_string[0] += operation_field[0].upper()
                    #summary_string[1] += ":"
                    summary_string[1] += operation_field[1].upper()
                    error_context['mismatch'].append([*string_counter, 1])
                    string_counter[0] += 1
                    string_counter[1] += 1
                    summary_string[2] += ":"
                    summary['mismatch'] += 1

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

    # print alignments for debugging
    #print('r:',summary_string[0][:100])
    #print('  ',summary_string[2][:100])
    #print('q:',summary_string[1][:100])

    summary['alignment_length'] = summary['match']+summary['mismatch']+summary['deletion']+summary['insertion']
    summary['identity'] = summary['match']/summary['alignment_length']

    return summary, summary_string, [np.array(r2q), np.array(q2r)], error_context

def alignment_identity(hit):
    if hit is not None:
        return hit.mlen/hit.blen
    else:
        return 0

def benchmark_sequence_file(in_file, format_flag, aligner, full=False):
    records = list(SeqIO.parse(in_file, format_flag))
    results = pd.DataFrame()
    results_kmers = pd.DataFrame()
    out_fasta = open(os.path.splitext(in_file)[0]+'.benchmark.ref.fasta', 'w')

    # full options
    homopolymer_summary = {}
    error_positions = {'insertion':np.zeros(200), 'deletion':np.zeros(200), 'mismatch':np.zeros(200)}

    for r in records:
        hit = get_top_hit(aligner, str(r.seq))
        results_row = {"name":r.id}
        try:
            if hit is not None:
                results_row.update({"blen":hit.blen, "strand":hit.strand, "mlen":hit.mlen, "primary":hit.is_primary, 'ref_start':hit.r_st, 'ref_end':hit.r_en})
                q_seq = r.seq[hit.q_st:hit.q_en]
                r_seq = aligner.seq(hit.ctg, start=hit.r_st, end=hit.r_en)
                if hit.strand == -1:
                    q_seq = reverse_complement(q_seq)
                print(r.id, sep=',', file=sys.stderr)
                print(fasta_format(r.id, r_seq), file=out_fasta)

                # parsing CS since cigar doesn't differentiate between match and mismatch
                summary, alignment, seq_index, error_context = parse_cs(hit, q_seq=q_seq, r_seq=r_seq)
                results_row.update(summary)

                if full:
                    results_kmers_row = {"name":r.id}
                    # statistics on homopolymers (are deletions concentrated in homopolymer regions?)
                    #homopolymers = get_homopolymers(r_seq, 3)
                    results_kmers_row.update(get_homopolymers_alignment(alignment[0], alignment[1], 3))
                    results_kmers = results_kmers.append(results_kmers_row, ignore_index=True)

                    # error positions (are errors uniformly distributed or more towards the ends?)
                    ref_length = len(r_seq)
                    for error_type in ['mismatch', 'deletion', 'insertion']:
                        for e in error_context[error_type]:
                            rel_pos = int(200*e[0]/ref_length)
                            error_positions[error_type][rel_pos] += 1

                    #for c in error_context['deletion']:
                    #    print(c, aligner.seq(hit.ctg, start=c[0]-10, end=c[0]), "-", aligner.seq(hit.ctg, start=c[0], end=c[0]+10))
                    #for c in error_context['insertion']:
                    #    print(c, aligner.seq(hit.ctg, start=c[0]-10, end=c[0]), r.seq[c[1]:c[1]+c[2]], aligner.seq(hit.ctg, start=c[0], end=c[0]+10))

                # compare with cigar string
                #cigar_parse = parse_cigar(hit, q_seq=q_seq, r_seq=r_seq)
                #print(hit.mlen == results_row["match"], results_row["insertion"] == cigar_parse["insertion"], results_row["deletion"] == cigar_parse["deletion"], results_row["match"]+results_row["mismatch"] == cigar_parse["match"])

                alignment_match = hit.blen - (results_row["match"]+results_row["mismatch"]+results_row["deletion"]+results_row["insertion"])
                if alignment_match != 0:
                    print("WARNING: Alignment parsing. {} alignment character(s) unaccounted for".format(alignment_match))
            results = results.append(results_row, ignore_index=True)
        except:
            pass

    results.to_csv(os.path.splitext(in_file)[0]+'.benchmark.csv')
    if full:
        results_kmers.to_csv(os.path.splitext(in_file)[0]+'.benchmark_kmers.csv')

    if full:
        with open(os.path.splitext(in_file)[0]+'.benchmark.pickle', 'wb') as p:
            pickle.dump({"homopolymers":homopolymer_summary, "error_positions":error_positions}, p)

def benchmark(args):
    aligner = mp.Aligner(args.reference, preset='map-ont')

    if args.fasta is not None:
        format_flag = "fasta"
        in_file = args.fasta
    elif args.fastq is not None:
        format_flag = "fastq"
        in_file = args.fastq
    elif args.fasta_pair is None:
        sys.exit("Must specify FASTA or FASTQ sequence file!")

    if args.fasta_pair is not None:
        benchmark_sequence_file(args.fasta_pair+'.1d.fasta', "fasta", aligner=aligner, full=args.full)
        benchmark_sequence_file(args.fasta_pair+'.2d.fasta', "fasta", aligner=aligner, full=args.full)
    else:
        benchmark_sequence_file(in_file, format_flag, full=args.full, aligner=aligner)
