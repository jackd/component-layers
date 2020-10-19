import functools
from typing import Optional

import tensorflow as tf

from composite_layers.types import is_ragged
from composite_layers.utils import wrap

Lambda = tf.keras.layers.Lambda


@functools.wraps(tf.RaggedTensor.from_row_splits)
def from_row_splits(
    values, row_splits, name: Optional[str] = None, validate: bool = True
):
    return wrap(
        tf.RaggedTensor.from_row_splits,
        values,
        row_splits,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_nested_row_splits)
def from_nested_row_splits(flat_values, nested_row_splits, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_nested_row_splits,
        flat_values,
        nested_row_splits,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_value_rowids)
def from_value_rowids(values, value_rowids, nrows=None, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_value_rowids,
        values,
        value_rowids,
        nrows,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_nested_value_rowids)
def from_nested_value_rowids(
    flat_values, nested_value_rowids, nested_nrows=None, name=None, validate=True
):
    return wrap(
        tf.RaggedTensor.from_nested_value_rowids,
        flat_values,
        nested_value_rowids,
        nested_nrows,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_row_lengths)
def from_row_lengths(values, row_lengths, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_row_lengths,
        values,
        row_lengths,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_nested_row_lengths)
def from_nested_row_lengths(flat_values, nested_row_lengths, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_nested_row_lengths,
        flat_values,
        nested_row_lengths,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_uniform_row_length)
def from_uniform_row_length(
    values, uniform_row_length, nrows=None, validate=True, name=None
):
    return wrap(
        tf.RaggedTensor.from_uniform_row_length,
        values,
        uniform_row_length,
        nrows,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_row_limits)
def from_row_limits(values, row_limits, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_row_limits,
        values,
        row_limits,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_row_starts)
def from_row_starts(values, row_starts, name=None, validate=True):
    return wrap(
        tf.RaggedTensor.from_row_starts,
        values,
        row_starts,
        name=name,
        validate=validate,
    )


@functools.wraps(tf.RaggedTensor.from_sparse)
def from_sparse(st_input, name=None, row_splits_dtype=tf.int64):
    return wrap(
        tf.RaggedTensor.from_sparse,
        st_input,
        name=name,
        row_splits_dtype=row_splits_dtype,
    )


@functools.wraps(tf.RaggedTensor.from_tensor)
def from_tensor(
    tensor,
    lengths=None,
    padding=None,
    ragged_rank=1,
    name=None,
    row_splits_dtype=tf.int64,
):
    return wrap(
        tf.RaggedTensor.from_tensor,
        tensor,
        lengths,
        padding,
        ragged_rank=1,
        name=None,
        row_splits_dtype=row_splits_dtype,
    )


# components


@functools.wraps(tf.RaggedTensor.values)
def values(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.values))(rt)


@functools.wraps(tf.RaggedTensor.row_splits)
def row_splits(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.row_splits))(rt)


@functools.wraps(tf.RaggedTensor.flat_values)
def flat_values(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.flat_values))(rt)


@functools.wraps(tf.RaggedTensor.nested_row_splits)
def nested_row_splits(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: [tf.identity(s) for s in rt.nested_row_splits])(rt)


# other derived quantities


@functools.wraps(tf.RaggedTensor.row_lengths)
def row_lengths(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.row_lengths()))(rt)


@functools.wraps(tf.RaggedTensor.row_starts)
def row_starts(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.row_starts()))(rt)


@functools.wraps(tf.RaggedTensor.value_rowids)
def value_rowids(rt):
    assert is_ragged(rt)
    return Lambda(lambda rt: tf.identity(rt.value_rowids()))(rt)


@functools.wraps(tf.RaggedTensor.to_tensor)
def to_tensor(rt, default_value=None, name=None, shape=None):
    """See `tf.RaggedTensor.to_tensor`."""
    assert is_ragged(rt)
    return Lambda(lambda args: args[0].to_tensor(args[1], name=name, shape=shape))(
        [rt, default_value]
    )


@functools.wraps(tf.RaggedTensor.ragged_rank)
def ragged_rank(rt) -> int:
    """ragged_rank from `RaggedTensor` or `KerasTensor` equivalent."""
    if isinstance(rt, tf.RaggedTensor):
        return rt.ragged_rank
    else:
        assert tf.keras.backend.is_keras_tensor(rt)
        type_spec = rt.type_spec
        assert isinstance(type_spec, tf.RaggedTensorSpec)
        return type_spec.ragged_rank
