import functools

import tensorflow as tf

from composite_layers.utils import wrap


@functools.wraps(tf.SparseTensor)
def SparseTensor(indices, values, dense_shape):
    return wrap(tf.SparseTensor, indices, values, dense_shape)
