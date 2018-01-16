import tensorflow as tf


def _weight_variable(shape):
    return tf.get_variable("weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))


def _bias_variable(shape):
    return tf.get_variable("biases", shape=shape, initializer=tf.constant_initializer(0.05))


def _conv_layer(x, kernel_shape):
    kernel = _weight_variable(kernel_shape)
    pre_activation = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
    bias = _bias_variable([kernel_shape[3]])
    activated = tf.nn.relu(pre_activation + bias)
    pooled = tf.nn.max_pool(activated, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    return pooled


def _fully_connected_layer(x, shape):
    weights = _weight_variable(shape)
    bias = _bias_variable([shape[1]])
    pre_activation = tf.matmul(x, weights) + bias
    return pre_activation


def _fully_connected_with_relu(x, shape):
    return tf.nn.relu(_fully_connected_layer(x, shape))


def get_network(x):
    with tf.variable_scope("conv1"):
        conv_1 = _conv_layer(x, [5, 5, 1, 32])
    with tf.variable_scope("conv2"):
        conv_2 = _conv_layer(conv_1, [5, 5, 32, 64])
    flattened_size = 7 * 7 * 64  # this depends on image size
    with tf.variable_scope("flatten"):
        reshaped = tf.reshape(conv_2, [-1, flattened_size])
        flat = _fully_connected_with_relu(reshaped, [flattened_size, 1024])
    with tf.variable_scope("fc1"):
        y_hat = _fully_connected_layer(flat, [1024, 10])

    return y_hat
