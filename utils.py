import numpy as np
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

def weighted_loss(model_weight, nStacks, hm_size, out_size, logits, gt_hm, step):
    with tf.name_scope('weighted_loss_' + str(step)):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gt_hm)
        weight = tf.cast(tf.equal(model_weight, 1), tf.float32)
        weight = tf.expand_dims(weight, 1)
        weight = tf.expand_dims(weight, 1)
        if step == 0:
            weight = tf.expand_dims(weight, 1)
            weight = tf.tile(weight, [1, nStacks, hm_size, hm_size, 1])
        elif step == 1:
            weight = tf.tile(weight, [1, hm_size * 2, hm_size * 2, 1])
        elif step == 2:
            weight = tf.tile(weight, [1, hm_size * 4, hm_size * 4, 1])
        elif step == 3:
            weight = tf.tile(weight, [1, out_size, out_size, 1])
        else:
            raise ValueError('Wrong up step of output.')
        loss = tf.multiply(loss, weight, name='weighted_out')
        return loss

# ---------------------- Evaluation Utils -----------------------

def error(num_points, pred, gt_map, weight, category):
    """
    Compute point error for each image and store in self.batch_point_error.
    :param pred: Heat map of shape (hm_size, hm_size, num_points)
    :param gt_map: Ground truth heat map
    :param weight: Point weight
    """
    total_dist = 0.0
    for i in range(num_points):
        if weight[i] == 1:
            pred_idx = np.array(np.where(pred[:, :, i] == np.max(pred[:, :, i])))
            gt_idx = np.array(np.where(gt_map[:, :, i] == np.max(gt_map[:, :, i])))
            total_dist += np.linalg.norm(pred_idx - gt_idx)
    # select the normalization points
    if category in ['blouse', 'outwear', 'dress']:
        norm_idx1 = np.array(np.where(gt_map[:, :, 5] == np.max(gt_map[:, :, 5])))
        norm_idx2 = np.array(np.where(gt_map[:, :, 6] == np.max(gt_map[:, :, 6])))
    else:
        norm_idx1 = np.array(np.where(gt_map[:, :, 0] == np.max(gt_map[:, :, 0])))
        norm_idx2 = np.array(np.where(gt_map[:, :, 1] == np.max(gt_map[:, :, 1])))
    norm_dist = np.linalg.norm(norm_idx2 - norm_idx1)
    return total_dist / norm_dist

def error_computation(batch_size, output, gt_map, weight, num_points, category):
    # point distances for every image in batch
    batch_point_error = []
    for i in range(batch_size):
        batch_point_error.append(
            error(
                num_points,
                output[i, :, :, :],
                gt_map[i, :, :, :],
                weight[i],
                category
            )
        )
    return batch_point_error


# ---------------------------- Other Utils --------------------------

VALID_POSITION = {
    'blouse': np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'dress': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
    'outwear': np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'skirt': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    'trousers': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]),
}