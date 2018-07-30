'''
Optimizing prefix search with an alignment envelope. Idea is instead of
summing over all UxV pairs for calculating prefix probabilities, consider a
reduced subset. This still requires calculation of the full gamma matrix, which
can be quite large for real-world read sizes. Currently focusing on approaches
that segment reads and calculate consensus separately, so moving some code here.
'''

import numpy as np
import consensus

class alignment_envelope_dense:
    '''
    Store envelope in dense numpy array and keep list of tuples of entries.
    '''
    def __init__(self,U,V):
        self.U = U
        self.V = V
        self.envelope = np.zeros(shape=(U,V),dtype=int)
        self.keys_ = []
    def __contains__(self,t):
        if self.envelope[t[0],t[1]] == 1:
            return True
        else:
            return False
    def add(self,u,v):
        self.envelope[u,v] = 1
        self.keys_.append((u,v))
    def keys(self):
        return(sorted(self.keys_))
    def toarray(self):
        return self.envelope

def diagonal_band_envelope(U,V,width):
    envelope = alignment_envelope_dense(U,V)
    for u in range(U):
        # just steps across main diagonal line. Width is size above and below
        # the main diagonal
        center = int(np.round(V/U*u))
        for v in range(center-width, center+width+1):
            if 0 <= v < V:
                envelope.add(u,v)
    return(envelope)

def sample_gamma(y1,y2,n=1):
    U = len(y1)
    V = len(y2)

    # get gamma matrix
    gamma_ = pair_gamma(y1,y2)
    gamma_[:,-1] = gamma_[-1,:] = 0

    # initialize alignment envelope
    envelope = alignment_envelope_dense(U,V)
    envelope.add(0,0)

    for i in range(n):
        (u,v) = (0,0)
        while (u < U-1) or (v < V-1):
            samples = np.array([gamma_[u+1,v],gamma_[u,v+1],gamma_[u+1,v+1]])
            (u,v) = np.array([u,v]) + [(1,0),(0,1),(1,1)][np.random.choice([0,1,2], p=samples/np.sum(samples))]
            if (u,v) not in envelope:
                envelope.add(u,v)

    return(gamma_, envelope)

def pair_forward_sparse(l, y1, y2, mask, previous=None):
    '''
    Requires alignment envelope. Iteration is only done over entries in the envelope.
    '''
    U = len(y1)
    V = len(y2)
    S = len(l)
    shift = 2

    alpha = np.zeros(shape=(S+shift,U+shift,V+shift))
    alpha_ast_ast = np.zeros(shape=(S+shift,U+shift,V+shift))
    alpha_ast = np.zeros(shape=(S+shift,U+shift,V+shift))

    if previous is not None:
        alpha[:-1] = previous[0]
        alpha_ast[:-1] = previous[1]
        alpha_ast_ast[:-1] = previous[2]
        s_range = [S + shift - 1]
    else:
        s_range = range(1,S+shift)

    alpha_ast_ast[1][1,1] = 1
    alpha_ast[1][1,1] = 1
    alpha[1][1,1] = 1

    # list of (u,v) tuples in alignment envelope
    sorted_keys = sorted(mask.keys())
    envelope_size = len(sorted_keys)

    s=1
    for u in range(2,U+shift):
        v = 1
        alpha_eps = y1[u-shift,-1]*alpha[s][u-1,v]
        alpha_ast_eps = y2[v-shift,-1]*alpha_ast[s][u,v-1]
        alpha_ast_ast[s][u,v] = y1[u-shift,l[s-shift]]*y2[v-shift,l[s-shift]]*alpha[s-1][u-1,v-1]
        alpha_ast[s][u,v] = alpha_ast_ast[s][u,v] + alpha_ast_eps
        alpha[s][u,v] = alpha_eps + alpha_ast[s][u,v]

    for v in range(2,V+shift):
        u = 1
        alpha_eps = y1[u-shift,-1]*alpha[s][u-1,v]
        alpha_ast_eps = y2[v-shift,-1]*alpha_ast[s][u,v-1]
        alpha_ast_ast[s][u,v] = y1[u-shift,l[s-shift]]*y2[v-shift,l[s-shift]]*alpha[s-1][u-1,v-1]
        alpha_ast[s][u,v] = alpha_ast_ast[s][u,v] + alpha_ast_eps
        alpha[s][u,v] = alpha_eps + alpha_ast[s][u,v]

    for s in s_range:
        for i,k in enumerate(sorted_keys):
            (u_env, v_env) = k
            u = u_env + shift
            v = v_env + shift

            alpha_eps = y1[u-shift,-1]*alpha[s][u-1,v]
            alpha_ast_eps = y2[v-shift,-1]*alpha_ast[s][u,v-1]
            alpha_ast_ast[s][u,v] = y1[u-shift,l[s-shift]]*y2[v-shift,l[s-shift]]*alpha[s-1][u-1,v-1]

            alpha_ast[s][u,v] = alpha_ast_ast[s][u,v] + alpha_ast_eps
            alpha[s][u,v] = alpha_eps + alpha_ast[s][u,v]

    return(alpha, alpha_ast_ast, alpha_ast)

def basecall_box_envelope(u1,u2,v1,v2):
    '''
    Function to be run in parallel.
    '''
    # set diagonal band of width 10% the mean length of both segments
    width_fraction = 0.1
    width = int(((u2-u1)+(v2-v1))/2*width_fraction)
    #print(u1,u2,v1,v2,width)
    envelope = consensus.diagonal_band_envelope(u2-u1,v2-v1,width)
    return((u1, consensus.pair_prefix_search(logits1[u1:u2],logits2[v1:v2], envelope=envelope, forward_algorithm=consensus.pair_forward_sparse)[0]))

if __name__ == '__main__':

    import testing

    print('--- Testing forward algorithm with full alignment envelope ---')
    y1 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.1,0.1,0.8]])
    y2 = np.array([[0.7,0.2,0.1],[0.2,0.3,0.5],[0.7,0.2,0.1],[0.05,0.05,0.9]])
    U = len(y1)
    V = len(y2)
    examples = ['AAAA','ABBA','ABA','AA','BB','A','B']
    full_envelope = alignment_envelope_dense(U,V)
    for u in range(U):
        for v in range(V):
            full_envelope.add(u,v)
    testing.test_pair_forward(y1,y2,examples=examples,envelope=full_envelope)

    print('alternative implementation (should be the same)')
    testing.test_pair_forward(y1,y2,examples=examples,envelope=full_envelope,forward_algorithm=pair_forward_sparse)

    print('--- Testing banded alignment envelope ---')
    (U, V) = (5,5)
    print('width=0\n',diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',diagonal_band_envelope(U,V,3).toarray())

    (U, V) = (5,4)
    print('width=0\n',diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',diagonal_band_envelope(U,V,3).toarray())

    (U, V) = (2,10)
    print('width=0\n',diagonal_band_envelope(U,V,0).toarray())
    print('width=1\n',diagonal_band_envelope(U,V,1).toarray())
    print('width=2\n',diagonal_band_envelope(U,V,2).toarray())
    print('width=3\n',diagonal_band_envelope(U,V,3).toarray())

    print('--- Single diagonal band doesn\'t match ---')
    y1 = y2 = np.array([[0.8,0.1,0.1],[0.1,0.3,0.6],[0.7,0.2,0.1],[0.8,0.1,0.1]])
    (U, V) = (len(y1),len(y2))
    band_envelope = diagonal_band_envelope(U,V,0)
    testing.test_pair_forward(y1,y2,examples=examples,envelope=band_envelope)
    testing.test_prefix_search(y1,y2,envelope=band_envelope)

    print('except when most of the probability passes through it')
    y1 = y2 = np.array([[1,0,0],[0,1,0]])
    (U, V) = (len(y1),len(y2))
    band_envelope = diagonal_band_envelope(U,V,0)
    testing.test_prefix_search(y1,y2,envelope=band_envelope)
