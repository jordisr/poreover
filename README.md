![Logo](logo.png)
# PoreOver: Nanopore basecalling in TensorFlow
### Introduction
PoreOver is an RNN basecaller for the Oxford Nanopore sequencing platform and is under active development. The current version uses a bidirectional RNN with LSTM cells to call bases from raw signal (without using events).

### Requirements
* TensorFlow 1.2
* Python 3
* sense of adventure

### Example
Full options for `train_model.py` and `run_model.py` available with the `--help` flag.

We can first train the model for a few iterations on a toy training set.

`python train_model.py --data data/tiny.train`

Once there is a model to load, we can make a basecall on a sample "read".

`python run_model.py --signal data/test.signal`

Finally the model can be evaluated in terms of accuracy on a test set.

`python eval_model.py --test data/tiny.test --train data/tiny.train`
