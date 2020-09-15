# cython: language_level=3
cimport cython
cimport numpy as np
from numpy cimport ndarray
import numpy as np

DTYPE = np.intc

# default parameter values
cdef int MATCH_DEFAULT = 2
cdef int MISMATCH_DEFAULT = -1
cdef int GAP_DEFAULT = -1
cdef int BAND_DEFAULT = 500

cdef int scoring_function(str a, str b, int match=MATCH_DEFAULT, int mismatch=MISMATCH_DEFAULT):
    if a == b:
        return match
    else:
        return mismatch

# needed for banded alignment
cdef extern from "SparseMatrix.h":
    cdef cppclass SparseMatrix[int]:
        int length;
        void push_row(int, int);
        void set(int,int,int);
        int get(int,int);

def global_pair(str seq1, str seq2, int match=MATCH_DEFAULT, int mismatch=MISMATCH_DEFAULT, int gap_cost=GAP_DEFAULT):
    '''
    Needleman-Wunsch algorithm. Uses constant gap penalty.
    '''
    cdef ssize_t l1 = len(seq1)
    cdef ssize_t l2 = len(seq2)
    cdef int i, j

    # initialize DP matrix memoryview
    dpMatrix_np = np.zeros(shape=(l1+1,l2+1), dtype=DTYPE)
    cdef int [:,:] dpMatrix = dpMatrix_np

    for i in range(l1+1):
        dpMatrix[i,0] = gap_cost*i

    for j in range(l2+1):
        dpMatrix[0,j] = gap_cost*j

    # fill in DP matrix
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            dpMatrix[i,j] = max(
                dpMatrix[i-1,j-1] + scoring_function(seq1[i-1],seq2[j-1], match, mismatch),
                dpMatrix[i-1,j] + gap_cost,
                dpMatrix[i,j-1] + gap_cost
            )

    # traceback and build alignment
    i = l1+1
    j = l2+1
    i = i-1
    j = j-1
    align1 = list()
    align2 = list()
    while i>0 and j>0:
        # recalculating max, slower but saves memory
        neighbor_cells = [
            dpMatrix[i-1,j-1] + scoring_function(seq1[i-1],seq2[j-1]),
            dpMatrix[i-1,j] + gap_cost,
            dpMatrix[i,j-1] + gap_cost]
        max_val = max(neighbor_cells)
        for index,val in enumerate(neighbor_cells):
            if val == max_val:
                if index == 0:
                    i = i - 1
                    j = j - 1
                    align1.extend(seq1[i])
                    align2.extend(seq2[j])
                elif index == 1:
                    i = i - 1
                    align1.extend(seq1[i])
                    align2.extend('-')
                elif index == 2:
                    j = j - 1
                    align1.extend('-')
                    align2.extend(seq2[j])
    while i>0 or j>0:
        if i>0:
            i = i - 1
            align1.extend(seq1[i])
            align2.extend('-')
        elif j>0:
            j = j - 1
            align1.extend('-')
            align2.extend(seq2[j])

    align1.reverse()
    align2.reverse()

    return(align1,align2,dpMatrix)

def global_pair_banded(str seq1, str seq2, int band_width=BAND_DEFAULT, int match=MATCH_DEFAULT, int mismatch=MISMATCH_DEFAULT, int gap_cost=GAP_DEFAULT):
    '''
    Banded Needleman-Wunsch algorithm. Uses constant gap penalty.
    '''
    cdef ssize_t l1 = len(seq1)
    cdef ssize_t l2 = len(seq2)
    cdef int i, j
    cdef int center, start, end

    # initialize sparse DP matrix
    dpMatrix = new SparseMatrix()

    for i in range(l1+1):
        dpMatrix.set(i,0,gap_cost*i)

    for j in range(l2+1):
        dpMatrix.set(0,j,gap_cost*j)

    # fill in DP matrix
    for i in range(l1):

      # bounds of diagonal band
      center = int(np.round(l2/l1*i))
      start = max(center-band_width,0)
      end = min(center+band_width,l2-1)
      dpMatrix.push_row(start,end)

      for j in range(start, end):
          dpMatrix.set(i,j, max(
              dpMatrix.get(i-1,j-1) + scoring_function(seq1[i-1],seq2[j-1], match, mismatch),
              dpMatrix.get(i-1,j) + gap_cost,
              dpMatrix.get(i,j-1) + gap_cost
          ))

    # traceback and build alignment
    i = l1+1
    j = l2+1
    i = i-1
    j = j-1
    align1 = list()
    align2 = list()
    while i>0 and j>0:
        # recalculating max, slower but saves memory
        neighbor_cells = [
            dpMatrix.get(i-1,j-1) + scoring_function(seq1[i-1],seq2[j-1]),
            dpMatrix.get(i-1,j) + gap_cost,
            dpMatrix.get(i,j-1) + gap_cost]
        max_val = max(neighbor_cells)
        for index,val in enumerate(neighbor_cells):
            if val == max_val:
                if index == 0:
                    i = i - 1
                    j = j - 1
                    align1.extend(seq1[i])
                    align2.extend(seq2[j])
                elif index == 1:
                    i = i - 1
                    align1.extend(seq1[i])
                    align2.extend('-')
                elif index == 2:
                    j = j - 1
                    align1.extend('-')
                    align2.extend(seq2[j])
    while i>0 or j>0:
        if i>0:
            i = i - 1
            align1.extend(seq1[i])
            align2.extend('-')
        elif j>0:
            j = j - 1
            align1.extend('-')
            align2.extend(seq2[j])

    align1.reverse()
    align2.reverse()

    del dpMatrix

    return(align1,align2)
