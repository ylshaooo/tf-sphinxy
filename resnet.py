import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections


def resnet(inputs, blocks, num_classes=None, is_training=True, scope=None):
    with tf.name_scope(scope):
        with slim.arg_scope([slim.conv2d, residual_block, stack_blocks_dense]):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # We do not include batch normalization or activation functions in
                    # conv1 because the first ResNet unit will perform these. Cf.
                    # Appendix of [2].
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_blocks_dense(net, blocks, output_stride)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], scope='logits')
                    net = tf.squeeze(net, [1, 2])
                    prediction = slim.softmax(net, scope='predictions')
                    return net, prediction
                return net


def resnet_block(depth_out, num_units, stride, scope):
    pass


@slim.add_arg_scope
def residual_block(inputs, depth, depth_out, stride, scope='bottleneck'):
    with tf.name_scope(scope):
        depth_in = inputs.get_shape().as_list()[3]
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth_out == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth_out, [1, 1], stride=stride, scope='shortcut')

        residual = slim.conv2d(preact, depth, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth_out, [1, 1], stride=1, scope='conv3')
        output = shortcut + residual
        return output


@slim.add_arg_scope
def stack_blocks_dense():
    return 0


def subsample(inputs, factor, scope=None):
    """
    Subsamples the input along the spatial dimensions.
    :param inputs: Input tensor
    :param factor: Subsampling stride
    :param scope: Optional variable scope
    :return: Output tensor
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, out_dim, kernel_size, stride, rate=1, scope=None):
    return 0
