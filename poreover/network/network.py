import numpy as np
import tensorflow as tf
import sys, os
import h5py
import glob
import datetime
import progressbar
import pathlib
from pathlib import Path
from pkg_resources import get_distribution

INPUT_DIM=1

class build_model:
    def __init__(self, args):
        self.num_neurons = getattr(args, "num_neurons", 128)
        self.kernel_size = getattr(args, "kernel_size", 9)
        self.filters = getattr(args, "filters", 256)

    # bigru3: 3 bidirectional GRU layers
    def bigru3(self, num_labels=4, input_size=1000):
        return tf.keras.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True), input_shape=(input_size,1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Dense(num_labels+1, activation=None)])

    # conv1_bigru3: 1 Conv1D layers + 3 bidirectional GRU layers
    def conv1_bigru3(self, num_labels=4, input_size=1000, strides=1):
        return tf.keras.Sequential([tf.keras.layers.Conv1D(kernel_size=self.kernel_size, filters=self.filters, strides=strides, input_shape=(input_size,1), activation="relu", padding="same"),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Dense(num_labels+1, activation=None)])

    # conv2_bigru3: 2 Conv1D layers + 3 Bidirectional GRU
    def conv2_bigru3(self, num_labels=4, input_size=1000, strides=1):
        return tf.keras.Sequential([
        tf.keras.layers.Conv1D(kernel_size=self.kernel_size, filters=self.filters, strides=strides, input_shape=(input_size,1), activation="relu", padding="same"),
        tf.keras.layers.Conv1D(kernel_size=self.kernel_size, filters=self.filters, strides=strides, activation="relu", padding="same"),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.num_neurons, return_sequences=True)),
        tf.keras.layers.Dense(num_labels+1, activation=None)])

    # conv1_gru5: 1 Conv1D layers + 5 alternating GRU layers
    def conv1_gru5(self, num_labels=4, input_size=1000, strides=1):
        return tf.keras.Sequential([
        tf.keras.layers.Conv1D(kernel_size=self.kernel_size, filters=self.filters, strides=strides, input_shape=(input_size,1), activation="relu", padding="same"),
        tf.keras.layers.GRU(self.num_neurons, return_sequences=True, go_backwards=False),
        tf.keras.layers.GRU(self.num_neurons, return_sequences=True, go_backwards=True),
        tf.keras.layers.GRU(self.num_neurons, return_sequences=True, go_backwards=False),
        tf.keras.layers.GRU(self.num_neurons, return_sequences=True, go_backwards=True),
        tf.keras.layers.GRU(self.num_neurons, return_sequences=True, go_backwards=False),
        tf.keras.layers.Dense(num_labels+1, activation=None)])

    def taiyaki_like(self, input_size=1000, num_labels=4):
        return tf.keras.Sequential([
        tf.keras.layers.Conv1D(kernel_size=19, filters=256, strides=2, input_shape=(input_size,1), activation="relu", padding="same"),
        tf.keras.layers.GRU(256, return_sequences=True, go_backwards=False),
        tf.keras.layers.GRU(256, return_sequences=True, go_backwards=True),
        tf.keras.layers.GRU(256, return_sequences=True, go_backwards=False),
        tf.keras.layers.GRU(256, return_sequences=True, go_backwards=True),
        tf.keras.layers.GRU(256, return_sequences=True, go_backwards=False),
        tf.keras.layers.Dense(num_labels+1, activation=None)])

def ragged_from_list_of_lists(l):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(l), np.array([len(i) for i in l]))

def validation_error(model, dataset):
    edit_distance = []
    for X,y in dataset:
        tmp1 = np.argmax(tf.nn.softmax(model(X)).numpy(), axis=2).astype(np.int32)
        tmp2 = [x[np.where(x < 4)] for x in tmp1]
        tmp3 = ragged_from_list_of_lists(tmp2).to_sparse()
        edit_distance.append(tf.reduce_mean(tf.edit_distance(hypothesis=tmp3, truth=y, normalize=True)).numpy())
    return(np.mean(edit_distance))

def train_ctc_model(model, dataset, optimizer=tf.keras.optimizers.Adam(), out_dir="save", save_frequency=10, log_frequency=10, ctc_merge_repeated=False, validation_size=0, early_stopping=True):
    avg_loss = []
    checkpoint = 0
    t = 0

    training_dataset = dataset.skip(validation_size)
    test_dataset = dataset.take(validation_size)

    json_config = model.to_json()
    with open(out_dir+'/model.json', 'w') as json_file:
        json_file.write(json_config)

    writer = tf.compat.v2.summary.create_file_writer(out_dir)
    with writer.as_default():
        for X,y in training_dataset:

            sequence_length = tf.ones(X.shape[0], dtype=np.int32)*1000
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                                labels=y,
                                                sequence_length=sequence_length,
                                                time_major=False,
                                                preprocess_collapse_repeated=False,
                                                ctc_merge_repeated=ctc_merge_repeated))
                avg_loss.append(loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if t % save_frequency == 0:
                model.save_weights(os.path.join(out_dir,"checkpoint-{}".format(checkpoint)))
                checkpoint += 1

            if t % log_frequency == 0:
                print("Iteration:{}\tLoss:{}".format(t,loss.numpy()), file=sys.stderr)

            tf.compat.v2.summary.scalar("loss", loss, step=t)
            if t % log_frequency:
                writer.flush()

            if t % save_frequency == 0 and validation_size > 0:
                edit_distance = validation_error(model, test_dataset)
                print("Iteration:{}\tEdit distance (test):{}".format(t,edit_distance), file=sys.stderr)
                tf.compat.v2.summary.scalar("test_edit_distance",edit_distance, step=t)
                writer.flush()

            t += 1

    model.save_weights(os.path.join(out_dir, "final"))

    return avg_loss

def train(args):

    # print software message, should incorporate to other subroutines as well
    coffee_emoji = u'\U00002615'
    dna_emoji = u'\U0001F9EC'
    print('{0:2}{1:3}{0:2} {2:^30} {0:2}{1:3}{0:2}'.format(coffee_emoji, dna_emoji,'PoreOver train'), file=sys.stderr)

    # directory for model checkpoints and logging
    out_dir = "{}_{}_{}".format(args.model, args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_file = open(out_dir+'/train.log','w')
    print('Command-line arguments:',file=log_file)
    for k,v in args.__dict__.items():
        print(k,'=',v, file=log_file)

    # set random seed
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    # load npz training data
    training = np.load(args.data)
    signal = np.expand_dims(training['signal'],axis=2)
    labels = tf.RaggedTensor.from_row_lengths(training['labels'].astype(np.int32),training['row_lengths'].astype(np.int32)).to_sparse()
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(signal), tf.data.Dataset.from_tensor_slices(labels)))
    dataset.shuffle(buffer_size=2000000)

    #if args.model == 'rnn':
    #    model = rnn(num_neurons=args.num_neurons)
    #elif args.model == 'cnn_rnn':
    #    model = cnn_rnn(num_neurons=args.num_neurons, kernel_size=args.kernel_size)

    # get the neural network architecture
    model = getattr(build_model(args), args.model)()

    # restart training from weights in checkpoint
    if args.restart:
        if os.path.isdir(args.restart):
            model_file = tf.train.latest_checkpoint(args.restart)
        else:
            model_file = args.restart
        model.load_weights(model_file)

    validation_size = int(int(len(list(dataset))/args.batch_size)*args.holdout)
    print("Setting aside {}% of data for validation: {} batches".format(args.holdout*100, validation_size), file=log_file)
    log_file.close()
    train_ctc_model(model, dataset.shuffle(buffer_size=2000000).repeat(args.epochs).batch(args.batch_size, drop_remainder=True), out_dir=out_dir, save_frequency=args.save_every, log_frequency=args.loss_every, validation_size=validation_size)

def call(args):

    if args.model is None:
        # if no model architecture is specified, use default architecture
        model = build_model(args).conv1_bigru3()
    else:
        # otherwise, load architecture from JSON file
        # for some reason, this is much slower than specifying model explicitly
        # possibly (?) related to https://github.com/tensorflow/tensorflow/issues/31243
        json_config_path = args.model
        with open(json_config_path) as json_file:
            json_config = json_file.read()
            model = tf.keras.models.model_from_json(json_config)

    # load trained model weights
    if args.weights is None:
        model_file = str(Path(__file__).parent.parent.parent.joinpath("data").joinpath("model").joinpath("checkpoint-124"))
    elif os.path.isdir(args.weights):
        model_file = tf.train.latest_checkpoint(args.weights)
    else:
        model_file = args.weights
    model.load_weights(model_file)

    if os.path.isdir(getattr(args,"in")):
        fast5_files =  glob.glob(os.path.join(getattr(args,"in"), "*.fast5"))
        for i in progressbar.progressbar(range(len(fast5_files))):
            fast5 = fast5_files[i]
            setattr(args, 'in', fast5)
            call_helper(args, model)
    else:
        call_helper(args, model)

def parse_fast5(f, scaling='standard'):

    hdf = h5py.File(f,'r')

    # basic parameters
    read_string = list(hdf['/Raw/Reads'].keys())[0]
    read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
    read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
    read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

    # raw events and signals
    raw_signal_path = '/Raw/Reads/'+read_string+'/Signal'
    raw_signal = np.array(hdf[raw_signal_path])
    assert(len(raw_signal) == read_duration)

    # for converting raw signal to current (pA)
    alpha = hdf['UniqueGlobalKey']['channel_id'].attrs['digitisation'] / hdf['UniqueGlobalKey']['channel_id'].attrs['range']
    offset = hdf['UniqueGlobalKey']['channel_id'].attrs['offset']
    sampling_rate = hdf['UniqueGlobalKey']['channel_id'].attrs['sampling_rate']

    # very rough heuristic for abasic region (still needed?)
    raw_signal = raw_signal[np.logical_and(raw_signal > 200, raw_signal < 800)]

    # rescale signal (should be same as option used in training)
    if scaling == 'standard':
        # standardize
        signal = (raw_signal - np.mean(raw_signal))/np.std(raw_signal)
    elif scaling == 'current':
        # convert to current (pA)
        signal = (raw_signal+offset)/alpha
    elif scaling == 'median':
        # divide by median
        signal = raw_signal / np.median(raw_signal)
    elif scaling == 'rescale':
        signal = (raw_signal - np.mean(raw_signal))/(np.max(raw_signal) - np.min(raw_signal))
    elif scaling == 'raw':
        signal = raw_signal

    return read_id, signal

def batch_input(signal, window_size, batch_size=128):

    num_padded_batches, last_batch_index = divmod(len(signal),window_size*batch_size)
    if last_batch_index > 0:
        num_padded_batches += 1

    padded_signal = np.zeros(window_size*batch_size*num_padded_batches)
    padded_signal[:len(signal)] = signal
    padded_batches = padded_signal.reshape((num_padded_batches, batch_size, window_size, INPUT_DIM))

    return padded_batches, last_batch_index

def call_helper(args, model):

    # load scaled signal from FAST5 file
    fast5_file = getattr(args, 'in')
    read_id, signal = parse_fast5(fast5_file, scaling=args.scaling)

    # split signal into blocks to allow for faster basecalling with the GPU
    padded_batches, last_batch_index = batch_input(signal, window_size=args.window)
    output = []

    # run forward pass
    for batch in padded_batches:
        if args.model is None:
            softmax_batch = tf.nn.softmax(model(batch))
        else:
             # ~4x slower than model(batch) if using model_from_json
            softmax_batch = tf.nn.softmax(model.predict_on_batch(batch))
        output.append(np.concatenate(softmax_batch))

    if last_batch_index > 0:
        output[-1] = output[-1][:last_batch_index]

    # now just saving logits, can use decode submodule for decoding probabilities
    logits_concatenate = np.concatenate(output)

    if args.use_id:
        out_prefix = os.path.join(args.dir, read_id.decode('utf-8'))
    else:
        out_prefix = os.path.join(args.dir, Path(fast5_file).stem)

    if args.format == 'csv':
        np.savetxt(out_prefix+'.csv', logits_concatenate, delimiter=',', header=','.join(['A','C','G','T','']), comments='', )
    else:
        np.save(out_prefix, logits_concatenate)
