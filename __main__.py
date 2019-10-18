'''
PoreOver
'''
import argparse, sys, glob, os

script_dir = os.path.dirname(__file__)
sys.path.insert(1, script_dir+'/network')

from network.network import call, train
from decoding.decode import decode
from decoding.pair_decode import pair_decode

# Set up argument parser
parser = argparse.ArgumentParser(description='PoreOver: Consensus Basecalling for Nanopore Sequencing')
subparsers = parser.add_subparsers(dest="command")
subparsers.required=True

# Train
parser_train = subparsers.add_parser('train', help='Train a neural network base calling model')
parser_train.set_defaults(func=train)
parser_train.add_argument('--data', help='Location of training data in compressed npz format', required=True)
parser_train.add_argument('--save_dir', default='.',help='Directory to save checkpoints')
parser_train.add_argument('--name', default='run', help='Name of run')
parser_train.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on (default: 1)')
parser_train.add_argument('--save_every', type=int, default=1000, help='Frequency with which to save checkpoint files (default: 1000)')
parser_train.add_argument('--holdout', default=0.05, type=float, help='Fraction of training data to hold out for calculating test error')
parser_train.add_argument('--loss_every', type=int, default=100, help='Frequency with which to output minibatch loss')
parser_train.add_argument('--ctc_merge_repeated', action='store_true', default=False, help='boolean option for tf.compat.v1.nn.ctc_loss')
parser_train.add_argument('--model', default='rnn', choices=['rnn', 'cnn_rnn'], help='Neural network architecture to use')
parser_train.add_argument('--restart', default=False, help='Trained model to load (if directory, loads latest from checkpoint file)')
parser_train.add_argument('--batch_size', default=64, type=int, help='Minibatch size for training')
parser_train.add_argument('--num_neurons', type=int, default=128, help='Number of neurons in RNN layers')
parser_train.add_argument('--kernel_size', type=int, default=9, help='Kernel size in Conv1D layer')

# Call
parser_call = subparsers.add_parser('call', help='Base call one or multiple reads using neural network')
parser_call.set_defaults(func=call)
parser_call.add_argument('--weights', help='Trained weights to load into model (if directory, loads latest from checkpoint file)', required=True)
parser_call.add_argument('--model', help='Model config JSON file', default=None)
parser_call.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale'], help='Type of preprocessing (should be same as training)')
parser_call.add_argument('--fast5', default=False, help='Single FAST5 file or directory of FAST5 files', required=True)
parser_call.add_argument('--out', default='out', help='Prefix for sequence output')
parser_call.add_argument('--window', type=int, default=1000, help='Call read using chunks of this size')
parser_call.add_argument('--format', choices=['csv', 'npy'], default='npy', help='Save softmax probabilities to CSV file or logits to binarized NumPy format')
parser_call.add_argument('--no_stack', default=False, action='store_true', help='Basecall [1xSIGNAL_LENGTH] tensor instead of splitting it into windows (slower)')

# Decode
parser_decode = subparsers.add_parser('decode', help='Decode probabilities from another basecaller')
parser_decode.set_defaults(func=decode)
parser_decode.add_argument('in', help='Probabilities to decode (either .npy from PoreOver of HDF5/FAST5 from Flappie or Guppy)')
parser_decode.add_argument('--out', help='Save FASTA sequence to file (default: stdout)')
parser_decode.add_argument('--basecaller', choices=['poreover', 'flappie', 'guppy'], help='Basecaller used to generate probabilitiess')
parser_decode.add_argument('--algorithm', default='viterbi', choices=['viterbi' ,'beam', 'prefix'], help='')
parser_decode.add_argument('--window', type=int, default=400, help='Use chunks of this size for prefix search')
parser_decode.add_argument('--beam_width', type=int, default=25, help='Width for beam search')

# Pair decode
parser_pair= subparsers.add_parser('pair-decode', help='1d2 consensus decoding of two output probabilities')
parser_pair.set_defaults(func=pair_decode)
# general options
parser_pair.add_argument('in', nargs='+', help='Probabilities to decode (either .npy from PoreOver of HDF5/FAST5 from Flappie or Guppy)')
parser_pair.add_argument('--basecaller', choices=['poreover', 'flappie', 'guppy'], help='Basecaller used to generate probabilitiess')
parser_pair.add_argument('--reverse_complement', default=False, action='store_true', help='Whether to reverse complement the second sequence (default: False)')
parser_pair.add_argument('--out', default='out',help='Save FASTA sequence to file (default: stdout)')
parser_pair.add_argument('--threads', type=int, default=1, help='Processes to use')
parser_pair.add_argument('--method', choices=['align', 'split', 'envelope'],default='align',help='Method for dividing up search space (see code)')
parser_pair.add_argument('--single', choices=['beam', 'viterbi'], default='viterbi', help='')
parser_pair.add_argument('--debug', default=False, action='store_true', help='Pickle objects to file for debugging')
parser_pair.add_argument('--algorithm', default='beam', choices=['prefix' ,'beam'], help='')
parser_pair.add_argument('--beam_width', type=int, default=25, help='Width for beam search')
# --method envelope
parser_pair.add_argument('--padding', type=int, default=150, help='Padding for building alignment envelope')
parser_pair.add_argument('--segments', type=int, default=8, help='Split full alignment envelope into N segments')
# --method split
parser_pair.add_argument('--window', type=int, default=200, help='Segment size used for splitting reads')
# --method align
parser_pair.add_argument('--matches', type=int, default=8, help='Match size for building anchors')
parser_pair.add_argument('--indels', type=int, default=10, help='Indel size for building anchors')

# Parse arguments and call corresponding command
args = parser.parse_args()
args.func(args)

print(args, file=sys.stderr)
