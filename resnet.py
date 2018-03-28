import tensorflow as tf
from collections import namedtuple

block = namedtuple('Bottleneck', ['name', 'unit_fn', 'args'])

RESNET_50_UNIT = [3, 4, 6, 3]
RESNET_101_UNIT = [3, 4, 23, 3]
RESNET_152_UNIT = [3, 8, 36, 3]
RESNET_200_UNIT = [3, 24, 36, 3]


def resnet(image, num_classes, model='resnet_50', training=True):
    units = RESNET_50_UNIT
    if model is 'resnet_101':
        units = RESNET_101_UNIT
    if model is 'resnet_152':
        units = RESNET_152_UNIT
    if model is 'resnet_200':
        units = RESNET_200_UNIT
    blocks = [
        block('block1', bottleneck, [(256, 64, 1)] * (units[0] - 1) + [(256, 64, 2)]),
        block('block2', bottleneck, [(512, 128, 1)] * (units[1] - 1) + [(512, 128, 2)]),
        block('block3', bottleneck, [(1024, 256, 1)] * (units[2] - 1) + [(1024, 256, 2)]),
        block('block4', bottleneck, [(2048, 512, 1)] * units[3])
    ]

    net = conv_layer(image, 64, ksize=7, strides=2)
    net = max_pool(net, ksize=3, strides=2)
    net = stack_block_dense(net, blocks, training)
    feature = net
    # global average pooling
    with tf.name_scope('global_avg_pool'):
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='net_flat')
    net = fc_layer(net, num_classes, name='fc')
    prediction = tf.nn.softmax(net)
    if training:
        return feature, prediction
    else:
        return feature


def bottleneck(inputs, depth, depth_neck, stride, training):
    net = batch_norm(inputs, training)
    depth_in = inputs.get_shape().as_list()[3]
    if depth_in == depth:
        shortcut = inputs
    else:
        shortcut = conv_layer(inputs, depth, ksize=1, strides=1)

    net = conv_layer(net, depth_neck, ksize=1, strides=1)
    net = batch_norm(net, training)
    net = conv_layer(net, depth_neck, ksize=3, strides=stride)
    net = batch_norm(net, training)
    net = conv_layer(net, depth, ksize=1, strides=1)
    output = net + shortcut
    return output


def batch_norm(inputs, training=True):
    inputs = tf.layers.batch_normalization(inputs, momentum=0.997, epsilon=1e-5, training=training, fused=True)
    return tf.nn.relu(inputs)


def stack_block_dense(inputs, blocks, training):
    net = inputs
    for b in blocks:
        with tf.name_scope(b.name):
            for i, unit in enumerate(b.args):
                with tf.name_scope('unit_%d' % (i + 1)):
                    depth, depth_neck, stride = unit
                    net = b.unit_fn(net, depth, depth_neck, stride, training)
    return net


def conv_layer(inputs, filters, ksize=1, strides=1, name='conv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, ksize, (strides, strides), 'same')


def fc_layer(inputs, num_out, activation=None, name='fc_layer'):
    with tf.name_scope(name):
        return tf.layers.dense(inputs, num_out, activation)


def max_pool(inputs, ksize, strides):
    return tf.layers.max_pooling2d(inputs, ksize, strides, 'same')
