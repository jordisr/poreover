import numpy as np

def remove_repeated(s):
    out = ''
    for i in range(len(s)):
        if (i==0) or (s[i-1] != s[i]):
            out += s[i]
    return(out)

class transducer:
    '''
    Class for building CTC-style automata from table of log-probabilities
    '''
    def __init__(self, log_prob, kind, alphabet):
        self.log_prob = log_prob.astype(np.float64)
        self.t_max = len(log_prob)
        self.alphabet = alphabet
        self.num_states = len(alphabet)
        self.kind = kind
        assert(self.num_states == len(self.log_prob[0]))
        self.transition = np.ones((self.t_max, self.num_states))

    def __getitem__(self, i):
        return(self.log_prob.__getitem__(i))

    def argmax_decode(self, return_path=False):
        greedy_path = np.argmax(self.log_prob, axis=1)
        greedy_string = ''.join(np.take(self.alphabet, greedy_path))
        if return_path:
            return(greedy_string, greedy_path)
        else:
            return(greedy_string)

    def viterbi_decode(self, return_path=False):
        v = np.zeros((self.t_max, self.num_states))-np.inf
        ptr = np.zeros_like(v).astype(int)

        # fill out DP matrix
        for t in range(self.t_max):
            if t==0:
                v[t] = self.log_prob[0]
            else:
                prev = self.transition.T + v[t-1]
                ptr[t] = np.argmax(prev, axis=1)
                v[t] = self.log_prob[t] + np.max(prev, axis=1)

        # path traceback
        viterbi_path = np.zeros(self.t_max, dtype=int)
        viterbi_path[-1] = np.argmax(v[-1])
        for i in reversed(range(0, len(v)-1)):
            viterbi_path[i] = ptr[i][viterbi_path[i+1]]

        # sequence
        sequence = remove_repeated(''.join(np.take(self.alphabet, viterbi_path))).upper()
        if return_path:
            return(sequence, viterbi_path)
        else:
            return(sequence)

    def __repr__(self):
        return('transducer(kind=%s, alphabet=%s, t_max=%s)' % (self.kind, self.alphabet, self.t_max))

class poreover(transducer):
    def __init__(self, log_prob):
        super().__init__(log_prob, 'poreover', np.array(['A','C','G','T','']))

    def reverse_complement(self):
        # (A,C,G,T,-)/(0,1,2,3,4) => (T,G,C,A,-)/(3,2,1,0,4)
        self.log_prob = self.log_prob[::-1,[3,2,1,0,4]]

    def viterbi_decode(self, return_path=False):
        return(self.argmax_decode(return_path))

class flipflop(transducer):
    def __init__(self, log_prob):
        super().__init__(log_prob, 'flipflop', np.array(['A','C','G','T','a','c','g','t']))
        self.transition = np.array([
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,0,1,0,0],
            [1,1,1,1,0,0,1,0],
            [1,1,1,1,0,0,0,1],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,0,1,0,0],
            [1,1,1,1,0,0,1,0],
            [1,1,1,1,0,0,0,1]
        ])
    def reverse_complement(self):
        # (A,C,G,T,a,c,g,t)/(0,1,2,3,4,5,6,7) => (T,G,C,A,t,g,c,a)/(3,2,1,0,4)
        self.log_prob = self.log_prob[::-1,[3,2,1,0,7,6,5,4]]

if __name__ == '__main__':
    y = np.random.random((20,5))
    y = (y.T / np.sum(y, axis=1)).T
    y = np.log(y)
    machine = poreover(y)
    print(machine.log_prob)
    print(machine[3:5,:3])
    print(machine.t_max)
    print(machine.argmax_decode())
    machine.reverse_complement()
    print(machine.argmax_decode())
    print(machine.viterbi_decode())
