![Logo](logo.png)
# PoreOver: Nanopore basecalling and consensus decoding
## Introduction
PoreOver is a basecalling tool for the Oxford Nanopore sequencing platform and is under active development.
It can serve as a standalone neural network basecaller or decode the output probabilities of CTC-style models from other basecallers including [Bonito](https://github.com/nanoporetech/bonito), [Flappie](https://github.com/nanoporetech/flappie), and Guppy. PoreOver additionally implements pair decoding of basecaller probabilities for [higher accuracy 1D<sup>2</sup> sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1).

PoreOver is intended as a platform on which to explore new algorithms and architectures for basecalling and decoding.
Its current basecaller implements a bidirectional RNN with GRU cells and CTC loss to call bases from raw signal, an architecture inspired by other community basecallers such as [DeepNano](https://bitbucket.org/vboza/deepnano/src/master/) (Boža et al. 2017) and [Chiron](https://github.com/haotianteng/Chiron) (Teng et al. 2018).

### Requirements
* Python 3
* TensorFlow (needed for basecalling)

### Installation

~~~
git clone https://github.com/jordisr/poreover
cd poreover
pip install -r requirements.txt
pip install -e .
~~~

Now the software can be run with:

`poreover --help`

## Usage examples

### Pair decoding of 1D<sup>2</sup> reads

`poreover pair-decode examples/1d2/read1.npy examples/1d2/read2.npy --basecaller poreover `

### Training a new basecalling model

We can run a few iterations of gradient descent to train a model on a small dataset.

`poreover train --data poreover/examples/training/toy --training_steps 100`

Once there is a model to load, we can make a basecall on a sample read (of course, after only a little training on a toy dataset we would not expect it to be very accurate).

`poreover call --fast5 poreover/examples/read.fast5 --model run-0 --out test.fasta`

### Decoding a flip-flop trace

The newer versions of the ONT basecallers implement a CTC-style "flip-flop" model for basecalling and can optionally output the probabilities of each nucleotide. This is done either with `--fast5-out` in Guppy or the `--trace` option in [Flappie](https://github.com/nanoporetech/flappie).

PoreOver can read and decode these probabilities, yielding a basecalled sequence.

`poreover decode poreover/examples/flappie_trace.hdf5`

`poreover decode poreover/examples/guppy_flipflop.fast5`
