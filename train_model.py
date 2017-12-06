'''
Build multilayer RNN and train it for basecalling
'''
import numpy as np
import tensorflow as tf
import argparse

# some custom helper functions
import batch
from helpers import sparse_tuple_from

def build_rnn(X,y):
    # model parameters
    NUM_NEURONS = 150 # how many neurons
    NUM_LABELS = 4 # A,C,G,T
    NUM_OUTPUTS = NUM_LABELS+1 # + blank
    NUM_LAYERS = 3

    # use MultiRNNCell for multiple RNN layers
    cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS,state_is_tuple=True) for _ in range(NUM_LAYERS)])
    cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=NUM_NEURONS,state_is_tuple=True) for _ in range(NUM_LAYERS)])

    # use dynamic RNN to allow for flexibility in input size
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X, dtype=tf.float32, time_major=False, sequence_length=sequence_length)
    outputs = tf.concat(outputs, 2)

    # dense layer connecting to output
    #logits = tf.contrib.layers.linear(outputs, NUM_OUTPUTS)
    logits = tf.add(tf.layers.dense(outputs, NUM_OUTPUTS, activation=None),0,name="logits")

    # decoding and log probabilities
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(logits, (1, 0, 2)), top_paths=1, beam_width=100, sequence_length=sequence_length, merge_repeated=False)

    # tensor of top prediction
    prediction = tf.sparse_tensor_to_dense(decoded[0], name='prediction', default_value=4)

    # edit distance
    edit_distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y), name='edit_distance')

    # set up minimization of loss
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y, inputs=logits, sequence_length=sequence_length, time_major=False, preprocess_collapse_repeated=False, ctc_merge_repeated=bool(args.ctc_merge_repeated)), name='loss')
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return(train_op, logits, loss, decoded[0], prediction, log_prob, edit_distance)

# parse command line arguments
parser = argparse.ArgumentParser(description='Train the basecaller')
parser.add_argument('--data', help='Location of training data', required=True)
parser.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
parser.add_argument('--name', default='run', help='Name of run')
parser.add_argument('--training_steps', type=int, default=1000, help='Number of iterations to run training (default: 1000)')
parser.add_argument('--save_every', type=int, default=10000, help='Frequency with which to save checkpoint files (default: 10000)')
parser.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output minibatch loss')
parser.add_argument('--ctc_merge_repeated', type=int, default=1, help='boolean option for tf.nn.ctc_loss, 0:False/1:True')
args = parser.parse_args()

print("CTC_MERGE_REPEATED:",bool(args.ctc_merge_repeated))

# user options
TRAINING_STEPS = args.training_steps #number of iterations of SGD
CHECKPOINT_ITER = args.save_every
LOSS_ITER = args.loss_every # how often to output loss
BATCH_SIZE = 64 # number of read fragments to use at a time

# load training data into memory (small files so this is OK for now)
INPUT_DIM = 1 # Raw signal has only one dimension
(train_events, train_bases) = batch.load_data(args.data, INPUT_DIM)
EPOCH_SIZE = len(train_events)

# log file for training statuts
log_file = open(args.save_dir+'/'+args.name+'.out','w')

# pass data to batch iterator class
dataset = batch.data_helper(train_events, train_bases, small_batch=False, return_length=True)

# Set up the model
# data X are [BATCH_SIZE, MAX_INPUT_SIZE, 1]
X = tf.placeholder(shape=[None, None, INPUT_DIM], dtype=tf.float32, name='X')

# labels y are [BATCH_SIZE, MAX_INPUT_SIZE]
y = tf.sparse_placeholder(dtype=tf.int32,name='y')

# sequence length is [BATCH_SIZE]
sequence_length = tf.placeholder(shape=[None],dtype=tf.int32,name='sequence_length')

(train_op, logits, loss, decoded, prediction, log_prob, edit_distance) = build_rnn(X,y)

# Start training network
saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_counter = 0

    # main training loop
    for iteration in range(TRAINING_STEPS):

        # get current minibatch and run minimization
        (X_batch, y_batch, sequence_length_batch) = dataset.next_batch(BATCH_SIZE)

        # For checking sparse representation of targets
        #print(sess.run(labels.indices))
        #print(labels.indices, labels.dense_shape, labels.values)
        #print(X_batch, labels, sequence_length_batch)

        sparse_tuple = sparse_tuple_from(y_batch)
        sess.run(train_op, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch})

        # periodically output the minibatch loss
        if (iteration+1) % LOSS_ITER == 0:
            print(len(X_batch), len(sequence_length_batch))
            print(iteration)
            #print(sess.run(log_prob, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch}))
            #decoded_out = sess.run(decoded, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch})
            #print('Values decoded:',len(decoded_out.values),'Decoding values:',decoded_out.values)
            #pred_out = sess.run(prediction, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch})
            #print('Prediction shape:',pred_out.shape)
            #print('Edit distance:',sess.run(edit_distance, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch}))
            #print(list(map(batch.decode_list,pred_out)))
            log_file.write(batch.format_string(('iteration:',iteration+1,'epoch:',dataset.epoch,'minibatch_loss:',sess.run(loss, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch}),'edit_distance:',sess.run(edit_distance, feed_dict={X:X_batch, y:sparse_tuple, sequence_length:sequence_length_batch}))))

        # periodically save the current model parameters
        if (iteration+1) % CHECKPOINT_ITER == 0:
            saver.save(sess, args.save_dir+'/'+args.name, global_step=checkpoint_counter)
            log_file.write(batch.format_string(('iteration:',iteration+1,'epoch:',dataset.epoch, 'model:',checkpoint_counter)))
            checkpoint_counter += 1

    # save extra model at the end of training unless you just saved one
    if (iteration+1) % CHECKPOINT_ITER != 0:
        saver.save(sess, args.save_dir+'/'+args.name, global_step=checkpoint_counter)
        log_file.write(batch.format_string(('iteration:',iteration+1,'epoch:',dataset.epoch, 'model:',checkpoint_counter)))

    #print("NAMED TENSORS")
    #[print(tensor.name) for tensor in tf.get_default_graph().as_graph_def().node]
