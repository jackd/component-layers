import functools

import tensorflow as tf

Lambda = tf.keras.layers.Lambda


def _unpacked_args(args, fn, **kwargs):
    return fn(*args, **kwargs)


def _unpacked(fn):
    return functools.partial(_unpacked_args, fn=fn)


def wrap(fn, *args, **kwargs):
    return Lambda(_unpacked(fn), arguments=kwargs)(args)
