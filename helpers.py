import tensorflow as tf
import numpy as np

# from TensorFlow unit test test_ctc_loss_op.py
def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).

  Args:
    x: a list of lists of type int

  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  #return tf.SparseTensor(x_ix, x_val, x_shape)
  return ([x_ix, x_val, x_shape])

# from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return (indices, values, shape)
