import numpy as np
import tensorflow as tf
import sys, os
import h5py
import glob
import datetime
from pathlib import Path

INPUT_DIM=1

def rnn(num_neurons=128, num_labels=4, input_size=1000):
    return tf.keras.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True), input_shape=(input_size,1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True)),
    tf.keras.layers.Dense(num_labels+1, activation=None)])

def cnn_rnn(num_neurons=128, num_labels=4, input_size=1000, kernel_size=3, filters=256, strides=1):
    return tf.keras.Sequential([tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=filters, strides=strides, input_shape=(input_size,1), activation="relu", padding="same"),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_neurons, return_sequences=True)),
    tf.keras.layers.Dense(num_labels+1, activation=None)])

def taiyaki_like(input_size=1000, num_labels=4):
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

def train_ctc_model(model, dataset, optimizer=tf.keras.optimizers.Adam(), checkpoint_dir="run", save_frequency=10, log_frequency=10, log_file=sys.stderr, ctc_merge_repeated=False, validation_size=0, early_stopping=True):
    avg_loss = []
    checkpoint = 0
    t = 0

    out_dir = checkpoint_dir+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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
    log_file = sys.stderr
    print('Command-line arguments:',file=log_file)
    for k,v in args.__dict__.items():
        print(k,'=',v, file=log_file)

    # load npz training data
    training = np.load(args.data)
    signal = np.expand_dims(training['signal'],axis=2)
    labels = tf.RaggedTensor.from_row_lengths(training['labels'].astype(np.int32),training['row_lengths'].astype(np.int32)).to_sparse()
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(signal), tf.data.Dataset.from_tensor_slices(labels)))
    dataset.shuffle(buffer_size=2000000)

    if args.model == 'rnn':
        model = rnn(num_neurons=args.num_neurons)
    elif args.model == 'cnn_rnn':
        model = cnn_rnn(num_neurons=args.num_neurons, kernel_size=args.kernel_size)

    # restart training from weights in checkpoint
    if args.restart:
        if os.path.isdir(args.restart):
            model_file = tf.train.latest_checkpoint(args.restart)
        else:
            model_file = args.restart
        model.load_weights(model_file)

    train_ctc_model(model, dataset.shuffle(buffer_size=5000).repeat(args.epochs).batch(args.batch_size, drop_remainder=True), checkpoint_dir=args.name, save_frequency=args.save_every, log_frequency=args.loss_every, log_file=log_file)

def call(args):

    if os.path.isdir(args.weights):
        model_file = tf.train.latest_checkpoint(args.weights)
        if args.model is None:
            json_config_path = args.weights+'/model.json'
    else:
        model_file = args.weights
        json_config_path = args.model

    with open(json_config_path) as json_file:
        json_config = json_file.read()

    model = tf.keras.models.model_from_json(json_config)
    model.load_weights(model_file)

    if args.fast5:
        if os.path.isdir(args.fast5):
            fast5_files =  glob.glob(args.fast5+'/*.fast5')
            print("Found",len(fast5_files),"files to basecall")
            for fast5 in fast5_files:
                args.fast5 = fast5
                args.out = os.path.basename(fast5)
                call_helper(args, model)
        else:
            call_helper(args, model)
    else:
        sys.exit("An input file must be specified with --fast5!")

def call_helper(args, model):

    # read raw signal from FAST5 file
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

    if not args.no_stack:
        # split signal into blocks to allow for faster basecalling with the GPU
        rounded = int(len(signal)/args.window)*args.window
        stacked = np.reshape(signal[:rounded], (-1, args.window, INPUT_DIM))
        sizes = [len(i) for i in stacked]

        if rounded < len(signal):
            last_row = np.zeros(args.window)
            last_row[:len(signal)-rounded] = signal[rounded:]
            last_row = np.expand_dims(last_row,1)
            sizes.append(len(signal)-rounded)
            stacked = np.vstack((stacked,np.expand_dims(last_row,0)))
    else:
        stacked = np.reshape(np.expand_dims(signal,axis=1), (1, len(signal), INPUT_DIM))
        sizes = [len(i) for i in stacked]

    # run forward pass
    logits_ = model(stacked)
    softmax = tf.nn.softmax(logits_)

    # now just saving logits, can use decode submodule for decoding probabilities
    logits_concatenate = np.concatenate(softmax)[:-1*max(sizes[0]-sizes[-1], 1)]
    if args.format == 'csv':
        np.savetxt(args.out+'.csv', logits_concatenate, delimiter=',', header=','.join(['A','C','G','T','']), comments='', )
    else:
        np.save(args.out, logits_concatenate)
