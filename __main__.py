'''
PoreOver
'''
import argparse, sys, glob, os
from run_model import call
from train_model import train
#from pair_decode import pair

script_dir = os.path.dirname(__file__)

# Set up argument parser
parser = argparse.ArgumentParser(description='PoreOver: Consensus Basecalling for Nanopore Sequencing')
subparsers = parser.add_subparsers(dest="command")
subparsers.required=True

# Train
parser_train = subparsers.add_parser('train', help='Train a neural network base calling model')
parser_train.set_defaults(func=train)
parser_train.add_argument('--data', help='Location of training data', required=True)
parser_train.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
parser_train.add_argument('--name', default='run', help='Name of run')
parser_train.add_argument('--training_steps', type=int, default=1000, help='Number of iterations to run training (default: 1000)')
parser_train.add_argument('--save_every', type=int, default=10000, help='Frequency with which to save checkpoint files (default: 10000)')
parser_train.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output minibatch loss')
parser_train.add_argument('--ctc_merge_repeated', type=int, default=1, help='boolean option for tf.nn.ctc_loss, 0:False/1:True')

# Call
parser_call = subparsers.add_parser('call', help='Base call one or multiple reads using neural network')
parser_call.set_defaults(func=call)
parser_call.add_argument('--model', default=os.path.join(script_dir,'models/r9.5'), help='Trained model to load (if directory, loads latest from checkpoint file)')
parser_call.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale'], help='Type of preprocessing (should be same as training)')
parser_call.add_argument('--signal', help='File with space-delimited signal for testing')
parser_call.add_argument('--fast5', default=False, help='FAST5 file to basecall (directories not currently supported)')
parser_call.add_argument('--out', default='out', help='Prefix for sequence output')
parser_call.add_argument('--window', type=int, default=400, help='Call read using chunks of this size')
parser_call.add_argument('--logits', choices=['csv', 'npy'], default=False, help='Save softmax probabilities to CSV file or logits to binarized NumPy format')
parser_call.add_argument('--decoding', default='beam', choices=['beam', 'prefix', 'none'], help='Choice of CTC decoding algorithm to use. Beam uses TensorFlow\'s built-in beam search. Prefix uses CTC prefix search decoding (but does not collapse repeated characters). None skips decoding and just runs neural network (output can be saved with --logits)')
parser_call.add_argument('--ctc_threads', type=int, default=1, help='Number of threads to use for prefix decoding')
parser_call.add_argument('--no_stack', default=False, action='store_true', help='Basecall [1xSIGNAL_LENGTH] tensor instead of splitting it into windows (slower)')

# Pair
#parser_pair = subparsers.add_parser('pair', help='Pair decoding of probabilities from neural network')
#parser_pair.set_defaults(func=pair)

# Parse arguments and call corresponding command
args = parser.parse_args()
args.func(args)

print(args)
