import tensorflow as tf

from resnet import bottleneck
from utils import conv_layer, batch_norm, max_pool, dropout


def graph_hourglass(inputs, nFeats, nStacks, nLow, out_dim, dropout_rate, training):
    with tf.name_scope('hourglass'):
        net = conv_layer(inputs, 64, ksize=6, strides=2, name='conv1')
        net = batch_norm(net, training=training)
        net = bottleneck(net, 128, 32, 1, training=training, name='res1')
        net = max_pool(net, 2, 2, name='max_pool')
        net = bottleneck(net, int(nFeats / 2), 1, training=training, name='res2')
        net = bottleneck(net, nFeats, 1, training=training, name='res3')

        final_out = []
        with tf.name_scope('stacks'):
            with tf.name_scope('stage_0'):
                hg = _hourglass(net, nLow, nFeats, 'hourglass')
                drop = dropout(hg, dropout_rate, training, name='dropout')
                ll = conv_layer(drop, nFeats, 1, 1)
                ll = batch_norm(ll, training)
                ll_ = conv_layer(ll, nFeats, 1, 1, name='ll')
                out = conv_layer(ll, out_dim, 1, 1, name='out')
                out_ = conv_layer(out, nFeats, 1, 1, name='out_')
                sum_ = tf.add_n([out_, net, ll_], name='merge')
                final_out.append(out)
            for i in range(1, nStacks - 1):
                with tf.name_scope('stage_' + str(i)):
                    hg = _hourglass(sum_, nLow, nFeats, 'hourglass')
                    drop = dropout(hg, dropout_rate, training, name='dropout')
                    ll = conv_layer(drop, nFeats, 1, 1, name='conv')
                    ll = batch_norm(ll, training)
                    ll_ = conv_layer(ll, nFeats, 1, 1, 'll')
                    out = conv_layer(ll, out_dim, 1, 1, 'out')
                    out_[i] = conv_layer(out, nFeats, 1, 1, 'out_')
                    sum_ = tf.add_n([out_, sum_, ll_], name='merge')
                    final_out.append(out)
            with tf.name_scope('stage_' + str(nStacks - 1)):
                hg = _hourglass(sum_, nLow, nFeats, 'hourglass')
                drop = tf.layers.dropout(hg, dropout_rate, training, name='dropout')
                ll = conv_layer(drop[nStacks - 1], nFeats, 1, 1, 'conv')
                ll = batch_norm(ll, training)
                out = conv_layer(ll, out_dim, 1, 1, 'out')
                final_out.append(out)
        return tf.stack(final_out, axis=1, name='output')


def _hourglass(self, inputs, n, depth, name='hourglass'):
    """
    Hourglass Module
    :param inputs: Input tensor
    :param n: Number of down-sampling step
    :param out_dim: Number of output features (channels)
    :param name: Name of the block
    :return: Output tensor
    """
    with tf.name_scope(name):
        # Upper Branch
        up_1 = self._residual(inputs, depth, name='up_1')
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2])
        low_1 = self._residual(low_, depth, name='low_1')

        if n > 0:
            low_2 = self._hourglass(low_1, n - 1, depth, name='low_2')
        else:
            low_2 = self._residual(low_1, depth, name='low_2')

        low_3 = self._residual(low_2, depth, name='low_3')
        up_2 = tf.image.resize_bilinear(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
        return tf.add_n([up_2, up_1], name='out_hg')