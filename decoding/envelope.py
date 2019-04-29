import pickle, sys, time
import numpy as np
from scipy.special import logsumexp

import decoding

def add_block(b,envelope):
    '''
    Add single block to row-based envelope
    '''
    (sx,sy,ex,ey) = b
    for i in range(sx,ex):
        this_min = sy
        this_max = ey
        if i < len(envelope):
            if this_min < envelope[i,0] or envelope[i,0] < 0:
                envelope[i,0] = this_min
            if this_max > envelope[i,1] or envelope[i,1] < 0:
                envelope[i,1] = this_max

def check_envelope(envelope,U,V):
    check_greater = all(envelope[:,1]>envelope[:,0])
    check_overlap = all(envelope[:-1,1]-envelope[1:,0])
    check_length = len(envelope) == U+2
    check_range = all(envelope[:,1]<=V)
    return(check_greater and check_overlap and check_length and check_range)

def get_alignment_columns(alignment):
    # extract column type and read sequence indices
    x_index=-1
    y_index=-1
    alignment_col = []
    for (x,y) in alignment.T:
        if (x != '-'):
            x_index += 1
        if (y != '-'):
            y_index += 1

        if (x == '-'):
            label = 'i'
        elif (y == '-'):
            label = 'd'
        else:
            label = 'm'
        alignment_col.append((label, x_index, y_index))
    return(alignment_col)

def build_envelope(y1, y2, alignment_col, sequence_to_signal1, sequence_to_signal2, padding=150):
    U = len(y1)
    V = len(y2)

    # get [sequence] to [signal range] mapping
    sequence_to_signal_range1 = []
    for i,v in enumerate(sequence_to_signal1[:-1]):
        sequence_to_signal_range1.append([sequence_to_signal1[i],sequence_to_signal1[i+1]])
    sequence_to_signal_range1.append([sequence_to_signal1[-1],U])

    sequence_to_signal_range2 = []
    for i,v in enumerate(sequence_to_signal2[:-1]):
        sequence_to_signal_range2.append([sequence_to_signal2[i],sequence_to_signal2[i+1]])
    sequence_to_signal_range2.append([sequence_to_signal2[-1],V])

    # build alignment envelope
    alignment_envelope = np.zeros(shape=(U,2),dtype=int)-1

    for (i,tup) in enumerate(alignment_col):
        (label, seq1, seq2) = tup
        block = (int(sequence_to_signal_range1[seq1][0]), int(sequence_to_signal_range2[seq2][0]),
                 int(sequence_to_signal_range1[seq1][1]), int(sequence_to_signal_range2[seq2][1])
                )
        add_block(block, alignment_envelope)

    # add a little padding to ensure some overlap
    for i in range(len(alignment_envelope)):
        alignment_envelope[i,0] = max(0,alignment_envelope[i,0]-padding)
        alignment_envelope[i,1] = min(V,alignment_envelope[i,1]+padding)

    # try and fix any problems
    for i in range(len(alignment_envelope)):
        if alignment_envelope[i,0] > alignment_envelope[i,1]:
            alignment_envelope[i,0] = 0

    return(alignment_envelope)

def offset_envelope(full_envelope, subset):
    (u1,u2,v1,v2) = subset
    subset_envelope = np.copy(full_envelope[u1:u2])
    subset_envelope[:,0] = subset_envelope[:,0] - v1
    subset_envelope[:,1] = subset_envelope[:,1] - v1
    return(subset_envelope)

def pad_envelope(envelope, U, V):
    new_envelope = np.concatenate((envelope, [envelope[-1], envelope[-1]]))
    for i,_ in enumerate(new_envelope):
        if new_envelope[i,1] == V-1:
            new_envelope[i,1] = V
    new_envelope[U] = new_envelope[U-1]
    new_envelope[U+1] = new_envelope[U-1]
    return(new_envelope)
