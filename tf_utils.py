import tensorflow as tf


def dense_layer(
    inputs,
    output_units,
    bias=True,
    activation=None,
    batch_norm=None,
    dropout=None,
    scope='dense-layer',
    reuse=False,
):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        in_dim = shape(inputs, -1)
        W = tf.compat.v1.get_variable(
            name='weights',
            initializer=tf.compat.v1.variance_scaling_initializer(),
            shape=[in_dim, output_units],
        )
        z = tf.matmul(inputs, W)

        if bias:
            b = tf.compat.v1.get_variable(
                name='biases',
                initializer=tf.compat.v1.constant_initializer(),
                shape=[output_units],
            )
            z = z + b

        if batch_norm is not None:
            # 그래프 모드 + v1 API 사용
            z = tf.compat.v1.layers.batch_normalization(
                z,
                training=batch_norm,
                reuse=reuse,
            )

        if activation is not None:
            z = activation(z)

        if dropout is not None:
            # keep_prob 방식(v1) 그대로 사용
            z = tf.compat.v1.nn.dropout(z, keep_prob=dropout)

        return z


def time_distributed_dense_layer(
    inputs,
    output_units,
    bias=True,
    activation=None,
    batch_norm=None,
    dropout=None,
    scope='time-distributed-dense-layer',
    reuse=False,
):
    """
    Applies a shared dense layer to each timestep of a tensor of shape
    [batch_size, max_seq_len, input_units] to produce a tensor of shape
    [batch_size, max_seq_len, output_units].
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        in_dim = shape(inputs, -1)  # 마지막 차원 (input_units)
        W = tf.compat.v1.get_variable(
            name='weights',
            initializer=tf.compat.v1.variance_scaling_initializer(),
            shape=[in_dim, output_units],
        )
        # [B, T, D] x [D, O] -> [B, T, O]
        z = tf.einsum('ijk,kl->ijl', inputs, W)

        if bias:
            b = tf.compat.v1.get_variable(
                name='biases',
                initializer=tf.compat.v1.constant_initializer(),
                shape=[output_units],
            )
            z = z + b

        if batch_norm is not None:
            # time dimension 그대로 두고 feature 차원만 BN
            z = tf.compat.v1.layers.batch_normalization(
                z,
                training=batch_norm,
                reuse=reuse,
            )

        if activation is not None:
            z = activation(z)

        if dropout is not None:
            z = tf.compat.v1.nn.dropout(z, keep_prob=dropout)

        return z


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int (static shape only)."""
    s = tensor.shape.as_list()
    if dim is None:
        return s
    return s[dim]


def rank(tensor):
    """Get tensor rank as python int."""
    return len(tensor.shape.as_list())
