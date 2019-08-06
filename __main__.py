'''
PoreOver
'''
import argparse, sys, glob, os

script_dir = os.path.dirname(__file__)
sys.path.insert(1, script_dir+'/network')

from network.run_model import call
from network.train_model import train
from decoding.decode import decode
from decoding.pair_decode import pair_decode

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
parser_train.add_argument('--ctc_merge_repeated', action='store_true', default=False, help='boolean option for tf.compat.v1.nn.ctc_loss')
parser_call.add_argument('--model', default='rnn', choices=['rnn'], help='Neural network architecture to use')

# Call
parser_call = subparsers.add_parser('call', help='Base call one or multiple reads using neural network')
parser_call.set_defaults(func=call)
parser_call.add_argument('--model', default=os.path.join(script_dir,'models/r9.5'), help='Trained model to load (if directory, loads latest from checkpoint file)')
parser_call.add_argument('--scaling', default='standard', choices=['standard', 'current', 'median', 'rescale'], help='Type of preprocessing (should be same as training)')
parser_call.add_argument('--signal', help='File with space-delimited signal for testing')
parser_call.add_argument('--fast5', default=False, help='Single FAST5 file or directory of FAST5 files')
parser_call.add_argument('--out', default='out', help='Prefix for sequence output')
parser_call.add_argument('--window', type=int, default=400, help='Call read using chunks of this size')
parser_call.add_argument('--format', choices=['csv', 'npy'], default='npy', help='Save softmax probabilities to CSV file or logits to binarized NumPy format')
parser_call.add_argument('--decoding', default='greedy', choices=['greedy','beam', 'prefix', 'none'], help='Choice of CTC decoding algorithm to use. Greedy takes best path. Beam uses TensorFlow\'s built-in beam search. Prefix uses CTC prefix search decoding (but does not collapse repeated characters). None skips decoding and just runs neural network (output can be saved with --logits)')
parser_call.add_argument('--threads', type=int, default=1, help='Number of threads to use for prefix decoding')
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
