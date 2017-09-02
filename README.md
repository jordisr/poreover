# PoreOver: Nanopore basecalling in TensorFlow
### Introduction
PoreOver is an RNN basecaller for the Oxford Nanopore sequencing platform and is under active development. The current version uses a bidirectional RNN with LSTM cells to predict 6-mers for each event. 

### Requirements
* TensorFlow 1.2
* Python 3
* sense of adventure

### Example
Full options for `train_model.py` and `run_model.py` available with the `--help` flag.

We can first train the model for a few iterations on a toy training set.

`python train_model.py --data data/train`

Once there is a model to load, we can make a basecall on a sample "read".

`python run_model.py --events data/read.events`

![Logo](logo.png)
