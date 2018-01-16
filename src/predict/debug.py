import tensorflow as tf
import predict.reader as reader
import logging
import util.log as log
import os


def read_and_print(next_image, next_label):
    with tf.Session() as session:
        for i in range(10):
            image, label = session.run([next_image, next_label])
            logger.info("Image has shape {} and label {}".format(image.shape, label))


def main():
    logger.info("Current directory: {}".format(os.getcwd()))
    datafile = '../../data/test/test.tfrecord'
    image, label = reader.decode(datafile).make_one_shot_iterator().get_next()
    read_and_print(image, label)

if __name__ == '__main__':
    global logger
    log.setup_log()
    logger = logging.getLogger("debug")
    main()
