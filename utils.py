import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


# ---------------------------- Hourglass ----------------------------

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

        net = conv_layer_bn(net, depth_neck, 1, 1, training, name='conv1')
        net = conv_layer_bn(net, depth_neck, 3, stride, training, name='conv2')
        net = conv_layer(net, depth, 1, 1, name='conv3')
        output = net + shortcut
        return output


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
        up_2 = tf.image.resize_nearest_neighbor(net, tf.shape(net)[1:3] * 2, name='upsampling')
        return tf.add_n([up_2, up_1], name='out_hg')


# ---------------------- Basic Network Module -----------------------

def fc_layer(inputs, num_out, name='fc_layer'):
    with tf.name_scope(name):
        return tf.layers.dense(inputs, num_out)


def conv_layer(inputs, filters, ksize=1, stride=1, name='conv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d(inputs, filters, ksize, (stride, stride), 'same',
                                kernel_initializer=xavier_initializer())


def conv_layer_bn(inputs, filters, ksize=1, stride=1, training=True, name='conv_layer'):
    net = conv_layer(inputs, filters, ksize, stride, name=name)
    net = batch_norm(net, training)
    return net


def deconv_layer(inputs, filters, ksize, stride, name='deconv_layer'):
    with tf.name_scope(name):
        return tf.layers.conv2d_transpose(inputs, filters, ksize, (stride, stride), 'same',
                                          kernel_initializer=xavier_initializer())


def max_pool(inputs, ksize, stride, name='pool'):
    return tf.layers.max_pooling2d(inputs, ksize, stride, 'same', name=name)


def sub_sample(inputs, stride, name='subsample'):
    with tf.name_scope(name):
        if stride == 1:
            return inputs
        else:
            return max_pool(inputs, 1, stride)


def batch_norm(inputs, training=True):
    inputs = tf.layers.batch_normalization(inputs, momentum=0.9, epsilon=1e-5, training=training, fused=True)
    return tf.nn.relu(inputs)


def dropout(inputs, rate, training=True, name='dropout'):
    return tf.layers.dropout(inputs, rate, training=training, name=name)


# ---------------------------- Other Utils --------------------------

VALID_POSITION = {
    'blouse': [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'dress': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    'outwear': [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'skirt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'trousers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
}
