import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass


def residual_block(base_depth, num_units, stride, scope):
    pass

