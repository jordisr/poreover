![Logo](logo.png)
# PoreOver: Nanopore basecalling in TensorFlow
## Introduction
PoreOver is a basecalling tool for the Oxford Nanopore sequencing platform and is under active development.
It can serve as a standalone neural network basecaller or decode the output probabilities of CTC-style models from other basecallers (i.e. [Flappie](https://github.com/nanoporetech/flappie) and Guppy).
It is intended as a platform on which to explore new algorithms and architectures for basecalling and decoding.
The current version implements a bidirectional RNN with LSTM cells and CTC loss to call bases from raw signal, an architecture inspired by other community basecallers such as DeepNano (Boža et al. 2017) and Chiron (Teng et al. 2018).

### Requirements
* Python 3
* TensorFlow (GPU installation recommended)

### Installation

To install, clone the repository and compile the Cython/C++ extensions:

~~~
git clone https://github.com/jordisr/poreover
cd poreover
pip install -r requirements.txt
pip install -e .
~~~

Now the software can be run with:

`poreover --help`

## Usage examples

### Training a basecalling model

We can run a few iterations of gradient descent to train a model on a small dataset.

`poreover train --data poreover/examples/training/toy --training_steps 100`

Once there is a model to load, we can make a basecall on a sample read (of course, after only a little training on a toy dataset we would not expect it to be very accurate).

`poreover call --fast5 poreover/examples/read.fast5 --model run-0 --out test.fasta`

### Basecalling a read with a trained model

We can compare this to the output of a model that has seen more training on a larger dataset. In the `models/` directory there are models for pore versions R9 (trained on E.coli) and R9.5 (trained on human).

`poreover call --fast5 poreover/examples/read.fast5 --model poreover/models/r9.5 --out read.fasta`

### Decoding a flip-flop trace

The newer versions of the ONT basecallers implement a CTC-style "flip-flop" model for basecalling and can optionally output the probabilities of each nucleotide. This is done either with `--fast5-out` in Guppy or the `--trace` option in [Flappie](https://github.com/nanoporetech/flappie).

PoreOver can read and decode these probabilities, yielding a basecalled sequence.

`poreover decode poreover/examples/flappie_trace.hdf5`

`poreover decode poreover/examples/guppy_flipflop.fast5`

### Pair decoding of 1D<sup>2</sup> reads

Pair decoding of probability profiles (as described [here](https://link.springer.com/chapter/10.1007/978-3-319-91938-6_11)) is under development with the goal of higher accuracy 1D<sup>2</sup> sequencing.
