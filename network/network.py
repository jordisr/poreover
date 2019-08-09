import numpy as np
import tensorflow as tf
import sys, os
from pathlib import Path

# some custom helper functions
import batch
from helpers import sparse_tuple_from

class rnn(tf.keras.Model):
    def __init__(self, num_neurons=128, num_labels=4, input_size=1000):
        super(rnn, self).__init__()
        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True), input_shape=(input_size,1))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_labels+1, activation=None)

    def call(self, x):
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.rnn3(x)
        return self.dense(x)

class cnn_rnn(tf.keras.Model):
    def __init__(self, num_neurons=128, num_labels=4, input_size=1000):
        super(cnn_rnn, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(kernel_size=3, filters=256, strides=1, input_shape=(input_size,1), activation="relu", padding="same")
        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True))
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_labels+1, activation=None)

    def call(self, x):
        x = self.cnn1(x)
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.rnn3(x)
        return self.dense(x)

def train_ctc_model(model, dataset, training_steps=10, optimizer=tf.keras.optimizers.Adam(), save_frequency=10, log_frequency=10, log_file=sys.stderr, ctc_merge_repeated=False):
    avg_loss = []
    checkpoint = 0
    checkpoint_dir = 'saved'
    for t in range(training_steps):

        X, y, sequence_length = dataset.next_batch()
        y_true = tf.sparse.SparseTensor(*sparse_tuple_from(y))

        with tf.GradientTape() as tape:
            y_pred = model(X.astype(np.float32))
            loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                            labels=y_true,
                                            sequence_length=sequence_length,
                                            time_major=False,
                                            preprocess_collapse_repeated=False,
                                            ctc_merge_repeated=ctc_merge_repeated))
            avg_loss.append(loss)

            if t % save_frequency == 0:
                model.save_weights(os.path.join(checkpoint_dir,"checkpoint-{}".format(checkpoint)))
                checkpoint += 1

            if t % log_frequency == 0:
                print(t,loss.numpy(),file=log_file)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    model.save_weights(os.path.join(checkpoint_dir, "final"))

    return avg_loss

def train(args):
    log_file = open(args.save_dir+'/'+args.name+'.out','w')
    print('Command-line arguments:',file=log_file)
    for k,v in args.__dict__.items():
        print(k,'=',v, file=log_file)

    (train_events, train_bases) = batch.load_data(args.data, INPUT_DIM)
    data = batch.data_helper(train_events, train_bases, batch_size=64, small_batch=False, return_length=True)

    model = rnn()
    train_ctc_model(model, data, training_steps=args.training_steps, save_frequency=args.save_every, log_frequency=args.loss_every, log_file=log_file)

def call(args):
    WINDOW_SIZE = args.window

    if args.fast5:
        if os.path.isdir(args.fast5):
            fast5_files =  glob.glob(args.fast5+'/*.fast5')
            print("Found",len(fast5_files),"files to basecall")
            for fast5 in fast5_files:
                args.fast5 = fast5
                if args.logits:
                    # use different filename for each saved logit
                    args.out = os.path.basename(fast5)
                call(args)
            sys.exit()
        else:
            hdf = h5py.File(args.fast5,'r')

            # basic parameters
            read_string = list(hdf['/Raw/Reads'].keys())[0]
            read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
            read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
            read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

            # raw events and signals
            raw_signal_path = '/Raw/Reads/'+read_string+'/Signal'
            raw_signal = hdf[raw_signal_path]
            assert(len(raw_signal) == read_duration)

            # for converting raw signal to current (pA)
            alpha = hdf['UniqueGlobalKey']['channel_id'].attrs['digitisation'] / hdf['UniqueGlobalKey']['channel_id'].attrs['range']
            offset = hdf['UniqueGlobalKey']['channel_id'].attrs['offset']
            sampling_rate = hdf['UniqueGlobalKey']['channel_id'].attrs['sampling_rate']

            raw_signal = [s for s in raw_signal if 200 < s < 800] # very rough heuristic for abasic region

            # rescale signal (should be same as option selected in make_labeled_data.py)
            if args.scaling == 'standard':
                # standardize
                signal = (raw_signal - np.mean(raw_signal))/np.std(raw_signal)
            elif args.scaling == 'current':
                # convert to current
                signal = (raw_signal+offset)/alpha # convert to pA
            elif args.scaling == 'median':
                # divide by median
                signal = raw_signal / np.median(raw_signal)
            elif args.scaling == 'rescale':
                signal = (raw_signal - np.mean(raw_signal))/(np.max(raw_signal) - np.min(raw_signal))

    elif args.signal:
        raw_events = []
        with open(args.signal, 'r') as f:
            for line in f:
                if len(line.split()) > 1:
                    raw_events.append(np.array(list(map(lambda x: float(x),line.split()))))
        # pad input sequences
        #sizes = [len(i) for i in raw_events]
        #padded_X = np.reshape(batch.pad(raw_events), (len(raw_events), -1, INPUT_DIM))
        signal = raw_events[0] # just take first line, don't normalize

    else:
        sys.exit("An input file must be specified with --signal or --fast5!")

    if not args.no_stack:
        # split signal into blocks to allow for faster basecalling with the GPU
        rounded = int(len(signal)/WINDOW_SIZE)*WINDOW_SIZE
        stacked = np.reshape(signal[:rounded], (-1, WINDOW_SIZE, INPUT_DIM))
        sizes = [len(i) for i in stacked]

        if rounded < len(signal):
            last_row = np.zeros(WINDOW_SIZE)
            last_row[:len(signal)-rounded] = signal[rounded:]
            last_row = np.expand_dims(last_row,1)
            sizes.append(len(signal)-rounded)
            stacked = np.vstack((stacked,np.expand_dims(last_row,0)))
    else:
        stacked = np.reshape(np.expand_dims(signal,axis=1), (1, len(signal), INPUT_DIM))
        sizes = [len(i) for i in stacked]

    # if model argument is a directory load the latest model in it
    if os.path.isdir(args.model):
        model_file = tf.train.latest_checkpoint(args.model)
    else:
        model_file = args.model

    if args.model == 'rnn':
        model = rnn()
    elif args.model == 'cnn_rnn':
        model = cnn_rnn()

    model.load_weights(model_file)

    logits_ = model.call(stacked)
    softmax = tf.nn.softmax(logits_)

    # now just saving logits, can use decode submodule for decoding probabilities
    logits_concatenate = np.concatenate(softmax)[:-1*max(sizes[0]-sizes[-1], 1)]
    if args.logits == 'csv':
        np.savetxt(args.out+'.csv', logits_concatenate, delimiter=',', header=','.join(['A','C','G','T','']), comments='', )
    else:
        np.save(args.out, logits_concatenate)
