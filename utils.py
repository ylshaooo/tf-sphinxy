from collections import namedtuple

import tensorflow as tf


# ---------------------------- ResNet -------------------------------

block = namedtuple('Bottleneck', ['name', 'unit_fn', 'args'])

RESNET_50_UNIT = [3, 4, 6, 3]
RESNET_101_UNIT = [3, 4, 23, 3]
RESNET_152_UNIT = [3, 8, 36, 3]
RESNET_200_UNIT = [3, 24, 36, 3]


def bottleneck(inputs, depth, depth_neck=None, stride=1, training=True, name='bottleneck'):
    with tf.name_scope(name):
        if depth_neck is None:
            depth_neck = depth / 4

        net = batch_norm(inputs, training)
        depth_in = inputs.get_shape().as_list()[3]
        if depth_in == depth:
            shortcut = inputs
        else:
            shortcut = conv_layer(inputs, depth, 1, 1)

        net = conv_layer(net, depth_neck, 1, 1, 'conv1')
        net = batch_norm(net, training)
        net = conv_layer(net, depth_neck, 3, stride, 'conv2')
        net = batch_norm(net, training)
        net = conv_layer(net, depth, 1, 1, 'conv3')
        output = net + shortcut
        return output


def stack_block_dense(inputs, blocks, training):
    net = inputs
    for b in blocks:
        with tf.name_scope(b.name):
            for i, unit in enumerate(b.args):
                with tf.name_scope('unit_%d' % (i + 1)):
                    depth, depth_neck, stride = unit
                    net = b.unit_fn(net, depth, depth_neck, stride, training)
    return net


# ---------------------------- Hourglass ----------------------------

def hourglass(inputs, units, depth, name='hourglass'):
    """
    Hourglass Module
    :param inputs: Input tensor
    :param units: Number of down-sampling step
    :param depth: Number of output features (channels)
    :param name: Name of the block
    :return: Output tensor
    """
    with tf.name_scope(name):
        # Upper Branch
        up_1 = bottleneck(inputs, depth, name='up_1')
        # Lower Branch
        net = max_pool(inputs, 2, 2)
        net = bottleneck(net, depth, name='low_1')

        if units > 0:
            net = hourglass(net, units - 1, depth, name='low_2')
        else:
            net = bottleneck(net, depth, name='low_2')

        net = bottleneck(net, depth, name='low_3')
        up_2 = tf.image.resize_bilinear(net, tf.shape(net)[1:3] * 2, name='upsampling')
        return tf.add_n([up_2, up_1], name='out_hg')


# ---------------------- Basic Network Module -----------------------

def fc_layer(inputs, num_out, activation=None, name='fc_layer'):
    with tf.name_scope(name):
        return tf.layers.dense(inputs, num_out, activation)


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


# ---------------------------- Other Utils --------------------------
