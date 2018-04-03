import configparser
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
            shortcut = sub_sample(inputs, stride)
        else:
            shortcut = conv_layer(inputs, depth, 1, stride)

        net = conv_layer(net, depth_neck, 1, 1, name='conv1')
        net = batch_norm(net, training)
        net = conv_layer(net, depth_neck, 3, stride, name='conv2')
        net = batch_norm(net, training)
        net = conv_layer(net, depth, 1, 1, name='conv3')
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


def conv_layer(inputs, filters, ksize=1, stride=1, activation=None, name='conv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, ksize, (stride, stride), 'same', activation=activation)


def deconv_layer(inputs, filters, ksize, stride, activation=None, name='deconv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d_transpose(inputs, filters, ksize, (stride, stride), 'same', activation=activation)


def max_pool(inputs, ksize, stride, name='pool'):
    return tf.layers.max_pooling2d(inputs, ksize, stride, 'same', name=name)


def sub_sample(inputs, stride, name='subsample'):
    with tf.name_scope(name):
        if stride == 1:
            return inputs
        else:
            return max_pool(inputs, 1, stride)


def batch_norm(inputs, training=True):
    inputs = tf.layers.batch_normalization(inputs, momentum=0.997, epsilon=1e-5, training=training, fused=True)
    return tf.nn.relu(inputs)


def group_norm(inputs, group=16, esp=1e-5, name='group_norm'):
    with tf.name_scope(name):
        N, H, W, C = inputs.get_shape().as_list()
        G = min(C, group)
        x = tf.transpose(inputs, [0, 3, 1, 2])
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        gamma = tf.Variable(tf.ones([1, C, 1, 1]), name='gamma')
        beta = tf.Variable(tf.zeros([1, C, 1, 1]), name='beta')
        x = tf.reshape(x, [N, C, H, W]) * gamma + beta
        return tf.transpose(x, [0, 2, 3, 1])


def dropout(inputs, rate, training=True, name='dropout'):
    return tf.layers.dropout(inputs, rate, training=training, name=name)


# ---------------------------- Other Utils --------------------------

VALID_POINTS = {
    'blouse': [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'dress': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    'outwear': [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'skirt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'trousers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
}


def process_config(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSet':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params
