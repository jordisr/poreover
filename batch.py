import numpy as np

BLANK_INDEX = 4
BLANK_VALUE = ''

def label2base(l):
    if l == 0:
        return 'A'
    elif l == 1:
        return 'C'
    elif l == 2:
        return 'G'
    elif l == 3:
        return 'T'
    elif l == 4:
        return ''

def decode_list(l):
    return(''.join(list(map(label2base,l))))

def base2label(b):
    if b == 'A':
        return 0
    elif b == 'C':
        return 1
    elif b == 'G':
        return 2
    elif b == 'T':
        return 3

def format_string(l):
    return(' '.join(list(map(str, l)))+'\n')

# returns (padded_data, sizes)
def pad(data):
    sizes = np.array([len(i) for i in data])
    DATA_SIZE = len(data)
    padded_data = np.zeros((DATA_SIZE,max(sizes)))
    for i, length in enumerate(sizes):
        assert(len(padded_data[i,:length]) == len(np.array(data[i])))
        padded_data[i,:length] = np.array(data[i])
    return(padded_data)

def load_data(path, dim=1):
    raw_events = []
    raw_bases = []
    with open(path+'.signal','r') as ef, open(path+'.bases','r') as bf:
        for eline, bline in zip(ef,bf):
            events = eline.split()
            bases = bline.split()
            #if (len(events) == (len(bases)*dim)): # output length doesnt need to equal input lenght
            raw_events.append(np.array(list(map(lambda x: float(x),events))))
            raw_bases.append(np.array(list(map(base2label,bases))))

    # pad data and labels
    padded_events = pad(raw_events)
    (DATA_SIZE, MAX_SEQ) = padded_events.shape
    # dim is dimension of flattened input and is used for resizing to vector
    padded_events = np.reshape(padded_events, (DATA_SIZE, -1, dim))

    #padded_bases = pad(raw_bases)
    #padded_bases = padded_bases.astype(int)
    #return(padded_events, padded_bases)

    # pad signal but not bases
    return(padded_events, raw_bases)

class data_helper:
    '''
    Simple class to hold data/labels and return consecutive minibatches
    small_batch=True returns the last batch even if it is smaller
    return_length=True returns a list with the length of each sequence
    '''
    def __init__(self, X, y, small_batch=True, return_length=False):
        # core data structure
        self.X = np.array(X)
        self.y = np.array(y)

        # pointer to current minibatch
        self.batch_i = 0
        self.epoch = 0

        # fixed options
        self.LENGTH = len(self.X)
        self.SMALL_BATCH = small_batch
        self.RETURN_LENGTH = return_length

        # if True return list of sequence lengths
        #if self.RETURN_LENGTH:
        #    self.sequence_length = [len(i) for i in self.X]

    # reset counter for testing purposes
    def reset(self):
        self.batch_i = 0

    # shuffle order of data
    def shuffle(self):
        indices = np.arange(self.LENGTH)
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]

    # get next minibatch
    def next_batch(self,batch_size):
        if self.batch_i+batch_size < self.LENGTH:
            batch_start = self.batch_i
            batch_end = batch_start + batch_size
            self.batch_i += batch_size
            NEW_EPOCH = False
        elif self.SMALL_BATCH:
            batch_start = self.batch_i
            batch_end = self.LENGTH
            self.batch_i = 0
            NEW_EPOCH = True
        else:
            self.batch_i = 0
            batch_start = self.batch_i
            batch_end = batch_start + batch_size
            NEW_EPOCH = True

        batch_X = self.X[batch_start:batch_end]
        batch_y = self.y[batch_start:batch_end]
        #batch_length = self.sequence_length[batch_start:batch_end]

        if NEW_EPOCH:
            self.epoch += 1
            self.shuffle()

        if self.RETURN_LENGTH:
            batch_length = [len(i) for i in batch_X]
            return(batch_X, batch_y, batch_length)
        else:
            return(batch_X, batch_y)

if __name__ == '__main__':

    print("Testing package with real data!")

    (train_events, train_bases) = load_data('/home/jordi/work/nanopore/poreover/data/toy')

    assert(len(train_events) == len(train_bases))
    print('NUMBER OF SEQUENCES:',len(train_events))

    print('PADDED RAW SIGNAL')
    print(train_events[:5])
    print('PADDED BASES')
    print(train_bases[:5])

    def sparse_tuple_from(sequences, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n]*len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

        return (indices, values, shape)

    sparse_tuple = sparse_tuple_from(train_bases)
    print('INDICES', sparse_tuple[0][:200])
    print('VALUES', sparse_tuple[1][:200])
    print('SHAPE',sparse_tuple[2])

    def one_hot(n, depth=4):
        v = np.zeros(depth)
        if n > 0:
            v[int(n)-1] = 1
        return(v)

    def random_data(length):
        return [np.random.randint(low=1,high=5, size=np.random.randint(1,11)) for i in range(length) ]

    '''

    print("Testing package with random data!")

    # generate some sample data to play with
    data = random_data(100)
    #labels = np.random.randint(low=0,high=4,size=100)
    labels = data
    print(labels)
    rows = len(data)

    # Pad smaller sequences with 0
    padded_data = pad(data)
    print(padded_data)

    #encode data (pad with zero vector)
    encoded_padded_data = np.array([[one_hot(elem) for elem in row] for row in padded_data])

    print("size should be [length, max_batch_size, alphabet_length]")
    print("size is ", encoded_padded_data.shape)

    print("Testing minibatch iteration")
    dataset = data_helper(encoded_padded_data, labels, small_batch=False)

    # testing minibatch iteration
    BATCH_SIZE = 16
    for i in range(100):
        (X, y) = dataset.next_batch(BATCH_SIZE)
        print(y)

        #print(X)
        #print("iteration",i+1,"epoch",dataset.epoch, "batch is shape", X.shape, len(y))

    '''
