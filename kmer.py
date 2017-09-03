import numpy as np
import itertools
import sys

# enumerate all possible 6-mers
ALPHABET = 'ACGT'
KMER_SIZE = 6

# dictionary storing kmer encodings
KMER_TO_LABEL = dict()
LABEL_TO_KMER = dict()
kmer_count = 1 # start at 1 to save 0 for padding

for kmer in itertools.product(ALPHABET, repeat=KMER_SIZE):
    kmer_string = ''.join(kmer)
    KMER_TO_LABEL[kmer_string] = kmer_count
    LABEL_TO_KMER[kmer_count] = kmer_string
    #print(kmer_count, kmer_string)
    kmer_count += 1
NUM_KMER = kmer_count

def kmer2label(kmer):
    return(KMER_TO_LABEL[kmer])

def label2kmer(kmer):
    return(LABEL_TO_KMER[kmer])

def stitch_kmers(list_of_kmers,direction='+'):
    '''
    Stitch together kmers, maximizing overlap. If two kmers do not overlap
    they are concatenated.
    '''
    if len(list_of_kmers) < 1: return ''
    sequence = list_of_kmers[0]
    previous_kmer = list_of_kmers[0]
    ksize = len(previous_kmer) # all kmers should be the same size
    for kmer in list_of_kmers[1:]:
        if kmer == previous_kmer:
            continue
        else:
            overlap = 0
            for overlap_size in reversed(range(1,ksize)):
                if (kmer[:overlap_size] == previous_kmer[ksize-overlap_size:]) and overlap == 0:
                    overlap = overlap_size
                    sequence = sequence + kmer[overlap:]
            if overlap == 0:
                #sys.stderr.write("WARNING: Stiching k-mers without any overlap.\n")
                sequence += kmer
        previous_kmer = kmer
    return sequence

def fasta_format(name, seq, width=80):
    fasta_string = ''
    # need to write...
    return(fasta_string)
