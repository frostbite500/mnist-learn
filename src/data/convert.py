import struct
import numpy as np
import tensorflow as tf
import util.log as log
import logging

IMAGE_LABEL_FEATURE = 'image/label'
IMAGE_DATA_FEATURE = 'image/data'
IMAGE_WIDTH_FEATURE = 'image/width'
IMAGE_HEIGHT_FEATURE = 'image/height'


def create_dataset(data_path, label_path, dataset_path):
    records = convert_to_image(data_path, label_path)
    convert_to_tfrecord(records, dataset_path)


def convert_to_image(data_path, label_path):
    with open(data_path, 'rb') as f:
        # First sixteen bytes contain magic number, number of images, image row dimension and image column dimension
        magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(size, rows, cols)

    with open(label_path, 'rb') as f:
        # First eight bytes contain magic number and number of items in file
        magic, size = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.int8)

    return list(zip(images, labels))


def convert_to_tfrecord(records, record_filename):
    def _bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    with tf.python_io.TFRecordWriter(record_filename) as writer:
        for record in records:
            image = record[0]
            height, width = image.shape
            label = record[1]
            feature = {IMAGE_HEIGHT_FEATURE: _int64_feature(height),
                       IMAGE_WIDTH_FEATURE: _int64_feature(width),
                       IMAGE_DATA_FEATURE: _bytes_feature(image.tobytes()),
                       IMAGE_LABEL_FEATURE: _int64_feature(label)}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def main():
    log.setup_log()
    logger = logging.getLogger("convert")
    training_data_path = '../data/training/orig/train-images-idx3-ubyte'
    training_label_path = '../data/training/orig/train-labels-idx1-ubyte'
    test_data_path = '../data/test/orig/t10k-images-idx3-ubyte'
    test_label_path = '../data/test/orig/t10k-labels-idx1-ubyte'
    training_dataset_path = '../data/training/training.tfrecord'
    test_dataset_path = '../data/test/test.tfrecord'
    logger.info("Creating data sets")
    create_dataset(training_data_path, training_label_path, training_dataset_path)
    logger.info("Training done")
    create_dataset(test_data_path, test_label_path, test_dataset_path)
    logger.info("Test done")


if __name__ == '__main__':
    main()
