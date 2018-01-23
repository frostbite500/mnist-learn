import tensorflow as tf
import logging
import util.log as log
import predict.network as network
import predict.reader as reader
import numpy as np
import time


def _loss_fn(y_hat, y):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)


def _accuracy_fn(y_hat, y):
    return tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))


def train(train_dataset, validation_dataset, epochs):

    train_dataset = train_dataset.shuffle(100000)
    train_batch_size = 500
    train_dataset = train_dataset.batch(train_batch_size)
    train_iterator = train_dataset.make_initializable_iterator()

    validation_dataset = validation_dataset.batch(train_batch_size)
    validation_iterator = validation_dataset.make_initializable_iterator()

    x, y = train_iterator.get_next()
    x_val, y_val = validation_iterator.get_next()
    with tf.variable_scope("network") as scope:
        y_hat = network.get_network(x)
        scope.reuse_variables()
        y_hat_val = network.get_network(x_val)

    cross_entropy = _loss_fn(y_hat, y)
    cross_entropy_validation = _loss_fn(y_hat_val, y_val)
    minimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
    training_accuracy = _accuracy_fn(y_hat, y)
    validation_accuracy = _accuracy_fn(y_hat_val, y_val)

    # Todo: Monitoring, checkpoints
    batch_counter = 0
    total_training_time = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):

            # Reset iterator, run training, evaluate accuracy
            sess.run(train_iterator.initializer)
            sess.run(validation_iterator.initializer)
            start = time.time()
            while True:
                try:
                    [loss, _, acc] = sess.run([cross_entropy, minimize, training_accuracy])
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
            end = time.time()
            total_training_time += (end-start)

            acc = []
            while True:
                try:
                    acc.extend(sess.run(validation_accuracy))
                except tf.errors.OutOfRangeError:
                    break

            training_rate = (batch_counter * train_batch_size) / total_training_time
            logger.info("Epoch {} - Training rate: {} samples/sec, Validation accuracy {:0.3f}%".format(i, int(training_rate), np.mean(acc)))


def main():
    logger.info("Starting training")
    train_datafile = '../data/training/training.tfrecord'
    validation_datafile = '../data/test/test.tfrecord'
    train_dataset = reader.decode(train_datafile)
    validation_dataset = reader.decode(validation_datafile)
    train(train_dataset, validation_dataset, 100)


if __name__ == '__main__':
    global logger
    log.setup_log()
    logger = logging.getLogger("minst-train")
    main()
