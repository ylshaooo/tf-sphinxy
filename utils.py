import tensorflow as tf


def conv_layer(inputs, filters, ksize=1, strides=1, name='conv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, ksize, (strides, strides), 'same')


def max_pool(inputs, ksize, strides, name='pool'):
    return tf.layers.max_pooling2d(inputs, ksize, strides, 'same')


def batch_norm(inputs, training=True):
    inputs = tf.layers.batch_normalization(inputs, momentum=0.997, epsilon=1e-5, training=training, fused=True)
    return tf.nn.relu(inputs)


def dropout(inputs, rate, training=True, name='dropout'):
    with tf.name_scope(name):
        return tf.layers.dropout(inputs, rate, training=training)
