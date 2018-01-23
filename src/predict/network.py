import tensorflow as tf


def _weight_variable(shape):
    return tf.get_variable("weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))


def _weight_variable_with_dropconnect(shape, p):
    return tf.nn.dropout(_weight_variable(shape), keep_prob=p) * p


def _bias_variable(shape):
    return tf.get_variable("biases", shape=shape, initializer=tf.constant_initializer(0.05))


def _conv_layer_with_pooling(x, kernel_shape, dropconnect_prob=None):
    if dropconnect_prob:
        kernel = _weight_variable_with_dropconnect(kernel_shape, dropconnect_prob)
    else:
        kernel = _weight_variable(kernel_shape)
    pre_activation = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
    bias = _bias_variable([kernel_shape[3]])
    activated = tf.nn.relu(pre_activation + bias)
    pooled = tf.nn.max_pool(activated, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    return pooled


def _fully_connected_layer(x, shape, dropconnect_prob=None):
    if dropconnect_prob:
        weights = _weight_variable_with_dropconnect(shape, dropconnect_prob)
    else:
        weights = _weight_variable(shape)
    bias = _bias_variable([shape[1]])
    pre_activation = tf.matmul(x, weights) + bias
    return pre_activation


def _fully_connected_with_relu(x, shape, dropconnect_prob=None):
    return tf.nn.relu(_fully_connected_layer(x, shape, dropconnect_prob=dropconnect_prob))


def get_network(x):
    with tf.variable_scope("conv1"):
        conv_1 = _conv_layer_with_pooling(x, [5, 5, 1, 32], dropconnect_prob=0.25)
    with tf.variable_scope("conv2"):
        conv_2 = _conv_layer_with_pooling(conv_1, [5, 5, 32, 64], dropconnect_prob=0.25)
    flattened_size = 7 * 7 * 64  # this depends on image size
    with tf.variable_scope("flatten"):
        reshaped = tf.reshape(conv_2, [-1, flattened_size])
        flat = _fully_connected_with_relu(reshaped, [flattened_size, 1024], dropconnect_prob=0.5)
    with tf.variable_scope("fc1"):
        y_hat = _fully_connected_layer(flat, [1024, 10])

    return y_hat
