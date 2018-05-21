import numpy as np

# default parameter values
MATCH_DEFAULT = 5
MISMATCH_DEFAULT = -4
GAP_DEFAULT = -5

def global_align(seq1,seq2, match=MATCH_DEFAULT, mismatch=MISMATCH_DEFAULT, gap_cost=GAP_DEFAULT):
    '''
    Needleman-Wunsch algorithm. Uses constant gap penalty.
    '''
    def scoring_function(a,b):
        if a==b:
            return match
        else:
            return mismatch

    # initialize DP matrix
    dpMatrix = np.zeros((len(seq1)+1,len(seq2)+1))
    dpMatrix[:,0] = [gap_cost*i for i in range(len(seq1)+1)]
    dpMatrix[0,:] = [gap_cost*j for j in range(len(seq2)+1)]

    # fill in DP matrix
    for i in range(1,len(seq1)+1):
        for j in range(1,len(seq2)+1):
            dpMatrix[i,j] = max(
                dpMatrix[i-1,j-1] + scoring_function(seq1[i-1],seq2[j-1]),
                dpMatrix[i-1,j] + gap_cost,
                dpMatrix[i,j-1] + gap_cost
            )

    # traceback --update this
    (i,j) = dpMatrix.shape
    i = i-1
    j = j-1
    align1 = []
    align2 = []
    while i>0 and j>0:
        # there is probably a better way of doing this
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
