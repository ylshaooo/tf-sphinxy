import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class SphinxModel(object):
    default_points = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                      'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                      'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                      'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                      'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    def __init__(self, nFeats=512, nStacks=4, nLow=4, out_dim=24, batch_size=16, drop_rate=0.2,
                 learning_rate=1e-3, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train=None, logdir_test=None, w_loss=False, points=default_points, name='sphinx'):

        self.nStacks = nStacks
        self.nFeats = nFeats
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.training = training
        self.dropout_rate = drop_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.points = points
        self.w_summary = w_summary
        self.w_loss = w_loss
        self.resume = {}

    def generate_model(self):
        start_time = time.time()

        print('CREATE MODEL:')
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                self.img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input')
                if self.w_loss:
                    self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.out_dim))
                self.gt_maps = tf.placeholder(dtype=tf.float32, shape=(None, self.nStacks, 64, 64, self.out_dim))
            print('---Inputs : Done.')
            self.output = self._graph_sphinx()
            print('---Graph : Done.')
            with tf.name_scope('loss'):
                if self.w_loss:
                    self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt_maps),
                        name='cross_entropy_loss'
                    )
            print('---Loss : Done.')

        with tf.device(self.cpu):
            with tf.name_scope('error'):
                self._error_computation()
            print('---Error : Done.')
            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.name_scope('lr'):
                lr = tf.train.exponential_decay(
                    self.learning_rate, self.train_step, self.decay_step, self.decay,
                    staircase=True, name='learning_rate'
                )
            print('---Learning Rate : Done.')

        with tf.device(self.gpu):
            with tf.name_scope('rmsprop'):
                rmsprop = tf.train.RMSPropOptimizer(learning_rate=lr)
            print('---Optimizer : Done.')
            with tf.name_scope('minimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_rmsprop = rmsprop.minimize(self.loss, self.train_step)
            print('---Minimizer : Done.')
        self.init = tf.global_variables_initializer()
        print('---Init : Done.')

        with tf.device(self.cpu):
            with tf.name_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train'])
                tf.summary.scalar('learning_rate', lr, collections=['train'])
            with tf.name_scope('summary'):
                for i in range(len(self.points)):
                    tf.summary.scalar(self.points[i], self.point_error[i], collections=['train', 'test'])

        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')

        end_time = time.time()
        print('Model created (' + str(int(abs(end_time - start_time))) + ' sec.)')
        del start_time, end_time

    def _train(self, nEpochs=10, epoch_size=1000, save_step=500, valid_iter=10):
        with tf.name_scope('Train'):
            self.generator = self.dataset.aux_generator(self.batch_size, self.nStacks, normalize=True,
                                                        sample_set='train')
            self.valid_gen = self.dataset.aux_generator(self.batch_size, self.nStacks, normalize=True,
                                                        sample_set='valid')
            start_time = time.time()
            self.resume['loss'] = []
            self.resume['error'] = []
            for epoch in range(nEpochs):
                epoch_start_time = time.time()
                avg_cost = 0.
                cost = 0.
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')

                # Training Set
                for i in range(epoch_size):
                    percent = ((i + 1) / epoch_size) * 100
                    num = np.int(20 * percent / 100)
                    time2end = int((time.time() - epoch_start_time) * (100 - percent) / percent)
                    sys.stdout.write(
                        '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' +
                        str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' +
                        str(avg_cost)[:5] + ' -timeToEnd: ' + str(time2end) + ' sec.'
                    )
                    sys.stdout.flush()

                    img_train, gt_train, weight_train = next(self.generator)
                    if i % save_step == 0:
                        if self.w_loss:
                            _, c, summary = self.Session.run(
                                [self.train_rmsprop, self.loss, self.train_op],
                                {self.img: img_train, self.gt_maps: gt_train, self.weights: weight_train}
                            )
                        else:
                            _, c, summary = self.Session.run(
                                [self.train_rmsprop, self.loss, self.train_op],
                                {self.img: img_train, self.gt_maps: gt_train}
                            )
                        # Save summary (Loss + Error)
                        self.train_summary.add_summary(summary, epoch * epoch_size + i)
                        self.train_summary.flush()
                    else:
                        if self.w_loss:
                            _, c, = self.Session.run(
                                [self.train_rmsprop, self.loss],
                                {self.img: img_train, self.gt_maps: gt_train, self.weights: weight_train}
                            )
                        else:
                            _, c, = self.Session.run(
                                [self.train_rmsprop, self.loss],
                                {self.img: img_train, self.gt_maps: gt_train}
                            )

                    cost += c
                    avg_cost += c / epoch_size
                epoch_finish_time = time.time()

                if self.w_loss:
                    weight_summary = self.Session.run(
                        self.weight_op,
                        {self.img: img_train, self.gt_maps: gt_train, self.weights: weight_train}
                    )
                else:
                    weight_summary = self.Session.run(
                        self.weight_op,
                        {self.img: img_train, self.gt_maps: gt_train}
                    )

                self.train_summary.add_summary(weight_summary, epoch)
                self.train_summary.flush()

                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                    int(epoch_finish_time - epoch_start_time)) + ' sec.' + ' -avg_time/batch: ' + str(
                    ((epoch_finish_time - epoch_start_time) / epoch_size))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session,
                                    os.path.join('checkpoints/', str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)

                # Validation Set
                error_array = np.array([0.0] * len(self.point_error))
                for i in range(valid_iter):
                    img_valid, gt_valid, w_valid = next(self.valid_gen)
                    error_pred = self.Session.run(self.point_error,
                                                  feed_dict={self.img: img_valid, self.gt_maps: gt_valid})
                    error_array += np.array(error_pred, dtype=np.float32) / valid_iter
                print('--Avg. Error =', str((np.sum(error_array) / len(error_array)) * 100)[:6], '%')
                self.resume['error'].append(np.sum(error_array) / len(error_array))
                valid_summary = self.Session.run(self.test_op, feed_dict={self.img: img_valid, self.gt_maps: gt_valid})
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()

            print('Training Done')
            print('Resume:')
            print('  Epochs: ' + str(nEpochs))
            print('  n. Images: ' + str(nEpochs * epoch_size * self.batch_size))
            print('  Final Loss: ' + str(cost))
            print('  Relative Loss: ' + str(100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
            print('  Relative Improvement: ' + str((self.resume['error'][0] - self.resume['error'][-1]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - start_time)))

    def training_init(self, nEpochs=10, epoch_size=1000, save_step=500, load=None):
        """
        Initialize the training
        :param nEpochs: Number of epochs to train
        :param epoch_size: Size of one epoch
        :param save_step: Step to save 'train' summary (has to be lower than epoch size)
        :param load: Model to load (None if training from scratch)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)
                self._train(nEpochs, epoch_size, save_step, valid_iter=10)

    def weighted_bce_loss(self):
        self.bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt_maps),
                                       name='cross_entropy_loss')
        e1 = tf.expand_dims(self.weights, axis=1, name='exp_dim1')
        e2 = tf.expand_dims(e1, axis=1, name='exp_dim2')
        e3 = tf.expand_dims(e2, axis=1, name='exp_dim3')
        return tf.multiply(e3, self.bce_loss, name='lossW')

    def _error_computation(self):
        self.point_error = []
        for i in range(len(self.points)):
            self.point_error.append(
                self._error(
                    self.output[:, self.nStacks - 1, :, :, i],
                    self.gt_maps[:, self.nStacks - 1, :, :, i],
                    self.batch_size
                )
            )

    def _define_saver_summary(self, summary=True):
        if self.logdir_train is None or self.logdir_test is None:
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver(max_to_keep=10)
            if summary:
                with tf.device(self.gpu):
                    self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                    self.test_summary = tf.summary.FileWriter(self.logdir_test)

    def _init_session(self):
        print('Session initialization')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.Session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _graph_sphinx(self):
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                pad = tf.pad(self.img, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad')
                conv1 = self._conv_bn_relu(pad, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                r1 = self._residual(conv1, out_dim=128, name='r1')
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2])
                r2 = self._residual(pool1, out_dim=int(self.nFeats / 2), name='r2')
                r3 = self._residual(r2, out_dim=self.nFeats, name='r3')
            # Storage Table
            hg = [None] * self.nStacks
            ll = [None] * self.nStacks
            ll_ = [None] * self.nStacks
            drop = [None] * self.nStacks
            out = [None] * self.nStacks
            out_ = [None] * self.nStacks
            sum_ = [None] * self.nStacks

            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = self._hourglass(r3, self.nLow, self.nFeats, 'hourglass')
                    drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,
                                                name='dropout')
                    ll[0] = self._conv_bn_relu(drop[0], self.nFeats, 1, 1, name='conv')
                    ll_[0] = self._conv(ll[0], self.nFeats, 1, 1, 'll')
                    out[0] = self._conv(ll[0], self.out_dim, 1, 1, 'out')
                    out_[0] = self._conv(out[0], self.nFeats, 1, 1, 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, self.nStacks - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeats, 'hourglass')
                        drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                                                    name='dropout')
                        ll[i] = self._conv_bn_relu(drop[i], self.nFeats, 1, 1, name='conv')
                        ll_[i] = self._conv(ll[i], self.nFeats, 1, 1, 'll')
                        out[i] = self._conv(ll[i], self.out_dim, 1, 1, 'out')
                        out_[i] = self._conv(out[i], self.nFeats, 1, 1, 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.name_scope('stage_' + str(self.nStacks - 1)):
                    hg[self.nStacks - 1] = self._hourglass(sum_[self.nStacks - 2], self.nLow, self.nFeats, 'hourglass')
                    drop[self.nStacks - 1] = tf.layers.dropout(hg[self.nStacks - 1], rate=self.dropout_rate,
                                                               training=self.training, name='dropout')
                    ll[self.nStacks - 1] = self._conv_bn_relu(drop[self.nStacks - 1], self.nFeats, 1, 1, 'conv')
                    out[self.nStacks - 1] = self._conv(ll[self.nStacks - 1], self.out_dim, 1, 1, 'out')
            return tf.stack(out, axis=1, name='final_output')

    def _conv(self, inputs, filters, kernel_size=1, strides=1, name='conv'):
        """
        Spatial Convolution (CONV2D)
        :param inputs: Input tensor (data type: NHWC)
        :param filters: Number of filters (channels)
        :param kernel_size: Size of kernel
        :param strides: Stride
        :param name: Name of the block
        :return: Output tensor (convolved input)
        """
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='SAME', data_format='NHWC')
            if self.w_summary:
                with tf.device(self.cpu):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, name='conv_bn_relu'):
        """
        Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        :param inputs: Input tensor (data type: NHWC)
        :param filters: Number of filters (channels)
        :param kernel_size: Size of kernel
        :param strides: Stride
        :param name: Name of the block
        :return: Output tensor
        """
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights'
            )
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='SAME', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            if self.w_summary:
                with tf.device(self.cpu):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return norm

    def _conv_block(self, inputs, out_dim, name='conv_block'):
        """
        Convolutional Block
        :param inputs: Input tensor
        :param out_dim: Desired output number of channel
        :param name: Name of the block
        :return: Output tensor
        """
        with tf.name_scope(name):
            with tf.name_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.training)
                conv_1 = self._conv(norm_1, int(out_dim / 2), 1, 1, name='conv')
            with tf.name_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, int(out_dim / 2), 3, 1, name='conv')
            with tf.name_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=self.training)
                conv_3 = self._conv(norm_3, int(out_dim), 1, 1, name='conv')
            return conv_3

    def _skip_layer(self, inputs, out_dim, name='skip_layer'):
        """
        :param inputs: Input Tensor
        :param out_dim: Desired output number of channel
        :param name: Name of the block
        :return: Tensor of shape (None, inputs.height, inputs.width, out_dim)
        """
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == out_dim:
                return inputs
            else:
                conv = self._conv(inputs, out_dim, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, out_dim, name='residual_block'):
        """
        Residual Unit
        :param inputs: Input tensor
        :param out_dim: Number of output features (channels)
        :param name: Name of the block
        :return: Output tensor
        """
        with tf.name_scope(name):
            conv_block = self._conv_block(inputs, out_dim)
            skip_layer = self._skip_layer(inputs, out_dim)
            return tf.add_n([conv_block, skip_layer], name='residual_block')

    def _hourglass(self, inputs, n, out_dim, name='hourglass'):
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
            up_1 = self._residual(inputs, out_dim, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2])
            low_1 = self._residual(low_, out_dim, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, out_dim, name='low_2')
            else:
                low_2 = self._residual(low_1, out_dim, name='low_2')

            low_3 = self._residual(low_2, out_dim, name='low_3')
            up_2 = tf.image.resize_bilinear(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            return tf.add_n([up_2, up_1], name='out_hg')

    def _argmax(self, tensor):
        """
        ArgMax
        :param tensor: 2D - Tensor (height x width)
        :return: Tuple of max position
        """
        reshape = tf.reshape(tensor, [-1])
        arg_max = tf.argmax(reshape, 0)
        return arg_max // tensor.get_shape().as_list()[0], arg_max % tensor.get_shape().as_list()[0]

    def _error(self, pred, gt_maps, num_image):
        """
        Given a Prediction batch and a Ground Truth batch, returns normalized error distance.
        :param pred: Prediction batch (shape = num_image x 64 x 64)
        :param gt_maps: Ground truth batch (shape = num_image x 64 x 64)
        :param num_image: (int) Number of images in batch
        :return: (float)
        """
        for i in range(num_image):
            pass

    def _resnet(self, num_classes, name):
        pass

    def _residual_block(self, depth, units, stride, scope):
        pass
