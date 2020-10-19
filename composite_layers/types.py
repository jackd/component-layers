import tensorflow as tf


def is_ragged(x) -> bool:
    """Indicate if x is a `tf.RaggedTensor` equivalent `KerasTensor`."""
    return (
        isinstance(x, tf.RaggedTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.RaggedTensorSpec)
    )


def is_sparse(x) -> bool:
    """Indicate if x is a `tf.SparseTensor` or equivalent `KerasTensor`."""
    return (
        isinstance(x, tf.SparseTensor)
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, tf.SparseTensorSpec)
    )


def is_dense(x) -> bool:
    """Indicate if x is a `tf.Tensor`, `tf.Variable` or KerasTensor representing one."""
    return (
        isinstance(x, (tf.Tensor, tf.Variable))
        or tf.is_tensor(x)
        and tf.keras.backend.is_keras_tensor(x)
        and isinstance(x.type_spec, x.Tensorspec)
    )
