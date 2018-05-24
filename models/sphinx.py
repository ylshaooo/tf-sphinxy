import numpy as np
import tensorflow as tf

import utils as ut
from config import Config


class SphinxModel:
    def __init__(self, cfg: Config):
        self.nStacks = cfg.nStacks
        self.nFeats = cfg.nFeats
        self.nLow = cfg.nLow
        
        self.category = cfg.category
        self.points = []
        for i in range(len(cfg.points_list)):
            if ut.VALID_POSITION[self.category][i] == 1:
                self.points.append(cfg.points_list[i])
        self.num_points = len(self.points)
        self.dropout_rate = cfg.dropout_rate

    def graph(self, input, is_training):
        with tf.name_scope('model'):
            net = ut.conv_layer(input, 64, 7, 2, name='conv1')
            net = ut.bottleneck(net, 128, stride=1, training=is_training, name='res1')
            net = ut.max_pool(net, 2, 2, 'max_pool')
            net = ut.bottleneck(net, int(self.nFeats / 2), stride=1, training=is_training, name='res2')
            net = ut.bottleneck(net, self.nFeats, stride=1, training=is_training, name='res3')

            with tf.name_scope('stacks'):
                stack_out = []
                with tf.name_scope('stage_0'):
                    hg = ut.hourglass(net, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, is_training, 'dropout')
                    ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, is_training)
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                    out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                    sum_ = tf.add(net, out_, name='merge')
                    stack_out.append(out)
                for i in range(1, self.nStacks):
                    with tf.name_scope('stage_' + str(i)):
                        hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                        drop = ut.dropout(hg, self.dropout_rate, is_training, 'dropout')
                        ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, is_training)
                        out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                        out_ = ut.conv_layer(ll, self.nFeats, 1, 1, name='out_')
                        sum_ = tf.add(sum_, out_, name='merge')
                        stack_out.append(out)
            with tf.name_scope('upsampling'):
                net = ut.batch_norm(sum_, is_training)
                net = ut.conv_layer_bn(net, self.nFeats, 3, 1, is_training)
                up1 = ut.deconv_layer(net, self.num_points, 1, 2, name='up_1')
                net = ut.conv_layer_bn(up1, self.nFeats, 3, 1, is_training)
                up2 = ut.deconv_layer(net, self.num_points, 1, 2, name='up_2')
            return tf.stack(stack_out, axis=1, name='stack_out'), up1, up2
