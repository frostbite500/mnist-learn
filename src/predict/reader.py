import tensorflow as tf
import logging
import util.log as log

from data.convert import IMAGE_HEIGHT_FEATURE, IMAGE_WIDTH_FEATURE, IMAGE_DATA_FEATURE, IMAGE_LABEL_FEATURE


def decode(filename):
    """Set up pipeline to decode contents in tfrecord files"""

    def _parse_data(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                IMAGE_HEIGHT_FEATURE: tf.FixedLenFeature([], tf.int64),
                IMAGE_WIDTH_FEATURE: tf.FixedLenFeature([], tf.int64),
                IMAGE_LABEL_FEATURE: tf.FixedLenFeature([], tf.int64),
                IMAGE_DATA_FEATURE: tf.FixedLenFeature([], tf.string)
            })

        height = tf.cast(features[IMAGE_HEIGHT_FEATURE], tf.int32)
        width = tf.cast(features[IMAGE_WIDTH_FEATURE], tf.int32)
        image = features[IMAGE_DATA_FEATURE]
        label = features[IMAGE_LABEL_FEATURE]

        image = tf.decode_raw(image, tf.uint8)
        image = tf.reshape(image, shape=[width, height, 1])

        # Transform to float from uint8
        image = tf.cast(image, tf.float32) * (1.0 / 255) - 0.5
        label = tf.one_hot(tf.cast(label, tf.int32), depth=10)

        return image, label

    dataset = tf.data.TFRecordDataset(filename)
    return dataset.map(_parse_data, num_parallel_calls=5)
