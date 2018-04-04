import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf

import utils as ut


class SphinxModel:
    key_points = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                  'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                  'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                  'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                  'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    def __init__(self, nFeats=512, nStacks=4, nLow=4, out_dim=24, img_size=256, hm_size=64, points=key_points,
                 batch_size=16, num_classes=5, drop_rate=0.5, learning_rate=1e-3, decay=0.96, decay_step=2000,
                 dataset=None, training=True, w_loss=False, num_validation=1000, logdir_train=None,
                 logdir_test=None, name='sphinx'):
        self.nStacks = nStacks
        self.nFeats = nFeats
        self.out_dim = out_dim
        self.img_size = img_size
        self.hm_size = hm_size
        self.points = points
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.training = training
        self.dropout_rate = drop_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset = dataset
        self.valid_iter = num_validation // batch_size
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.w_loss = w_loss
        self.name = name
        self.resume = {}

    def generate_model(self):
        start_time = time.time()

        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, 3))
            self.weight = tf.placeholder(tf.int32, (None, self.out_dim))
            self.gt_label = tf.placeholder(tf.float32, (None, self.num_classes))
            self.gt_map = tf.placeholder(tf.float32,
                                         (None, self.nStacks, self.hm_size, self.hm_size, self.out_dim))
        print('---Inputs : Done.')
        self.output = self._graph_sphinx()
        print('---Graph : Done.')
        with tf.name_scope('loss'):
            self.cls_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output[0], labels=self.gt_label),
                name='cls_loss'
            )
            if self.w_loss:
                self.hm_loss = tf.reduce_mean(self.weighted_bce_loss(), name='hm_loss')
            else:
                self.hm_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.output[1], labels=self.gt_map),
                    name='hm_loss'
                )
            self.loss = tf.add(self.cls_loss, self.hm_loss, name='loss')
        print('---Loss : Done.')

        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('lr'):
            self.lr = tf.train.exponential_decay(
                self.learning_rate, self.train_step, self.decay_step, self.decay,
                staircase=True, name='learning_rate'
            )
        print('---Learning Rate : Done.')

        with tf.name_scope('rmsprop'):
            rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        print('---Optimizer : Done.')
        with tf.name_scope('minimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_rmsprop = rmsprop.minimize(self.loss, self.train_step)
        print('---Minimizer : Done.')
        self.init = tf.global_variables_initializer()
        print('---Init : Done.')

        with tf.name_scope('training'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        with tf.name_scope('summary'):
            self.point_error = tf.placeholder(tf.float32)
            self.label_error = tf.placeholder(tf.float32)
            tf.summary.scalar('point_error', self.point_error, collections=['train', 'test'])
            tf.summary.scalar('label_error', self.label_error, collections=['train', 'test'])

        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')

        end_time = time.time()
        print('Model created (' + str(int(abs(end_time - start_time))) + ' sec.)')
        del start_time, end_time

    def _train(self, nEpochs=10, epoch_size=1000, save_step=500, valid_iter=10):
        with tf.name_scope('Train'):
            self.train_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size,
                                                    self.num_classes, self.nStacks, True, 'train')
            self.valid_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size,
                                                    self.num_classes, self.nStacks, True, 'valid')
            start_time = time.time()
            self.resume['loss'] = []
            self.resume['point_error'] = []
            self.resume['label_error'] = []
            cost = 0.
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

                    img_train, lb_train, gt_train, w_train = next(self.train_gen)
                    if i % save_step == 0:
                        if self.w_loss:
                            _, c, summary = self.Session.run(
                                [self.train_rmsprop, self.loss, self.train_op],
                                {
                                    self.img: img_train,
                                    self.gt_label: lb_train,
                                    self.gt_map: gt_train,
                                    self.weight: w_train
                                }
                            )
                        else:
                            _, c, summary = self.Session.run(
                                [self.train_rmsprop, self.loss, self.train_op],
                                {self.img: img_train, self.gt_label: lb_train, self.gt_map: gt_train}
                            )
                        # Save summary (Loss + Error)
                        self.train_summary.add_summary(summary, epoch * epoch_size + i)
                        self.train_summary.flush()
                    else:
                        if self.w_loss:
                            _, c, = self.Session.run(
                                [self.train_rmsprop, self.loss],
                                {
                                    self.img: img_train,
                                    self.gt_label: lb_train,
                                    self.gt_map: gt_train,
                                    self.weight: w_train
                                }
                            )
                        else:
                            _, c, = self.Session.run(
                                [self.train_rmsprop, self.loss],
                                {self.img: img_train, self.gt_label: lb_train, self.gt_map: gt_train}
                            )

                    cost += c
                    avg_cost += c / epoch_size
                epoch_finish_time = time.time()

                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                    int(epoch_finish_time - epoch_start_time)) + ' sec.' + ' -avg_time/batch: ' + str(
                    ((epoch_finish_time - epoch_start_time) / epoch_size))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session,
                                    os.path.join('checkpoints/', str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)

                # Validation Set
                point_error, label_error = self._valid(valid_iter)
                self.resume['point_error'].append(point_error)
                self.resume['label_error'].append(label_error)
                valid_summary = self.Session.run(
                    self.test_op,
                    {self.point_error: point_error, self.label_error: label_error}
                )
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()

            print('Training Done')
            print('Resume:')
            print('  Epochs: ' + str(nEpochs))
            print('  n. Images: ' + str(nEpochs * epoch_size * self.batch_size))
            print('  Final Loss: ' + str(cost))
            print('  Relative Loss: ' + str(100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
            print('  Relative Improvement - Point: ' + str(
                (self.resume['point_error'][0] - self.resume['point_error'][-1]) * 100) + '%')
            print('  Relative Improvement - Label: ' + str(
                (self.resume['label_error'][0] - self.resume['label_error'][-1]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - start_time)))

    def _valid(self, valid_iter):
        if not hasattr(self, 'valid_gen'):
            self.valid_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size,
                                                    self.num_classes, self.nStacks, True, 'valid')
        point_error = 0
        num_points = 0
        correct_label = 0
        for it in range(valid_iter):
            img_valid, lb_valid, gt_valid, w_valid = next(self.valid_gen)
            num_points += np.sum(w_valid)

            out = self.Session.run(
                self.output,
                feed_dict={
                    self.img: img_valid,
                    self.gt_map: gt_valid,
                    self.gt_label: lb_valid,
                    self.weight: w_valid
                }
            )

            batch_point_error, batch_correct_label = self._error_computation(out, lb_valid, gt_valid, w_valid)
            point_error += sum(batch_point_error)
            correct_label += batch_correct_label
        point_error = point_error / num_points
        label_error = 1 - correct_label / (valid_iter * self.batch_size)
        print('--Avg. Point Error = %.2f%%', point_error * 100)
        print('--Avg. Label Error = %.2f%%', label_error * 100)
        return point_error, label_error

    def predict(self, images):
        prediction = self.Session.run(self.output[1], feed_dict={self.img: images})
        return prediction

    def training_init(self, nEpochs=10, epoch_size=1000, save_step=500, load=None):
        """
        Initialize the training.
        :param nEpochs: Number of epochs to train
        :param epoch_size: Size of one epoch
        :param save_step: Step to save 'train' summary (has to be lower than epoch size)
        :param load: Model to load (None if training from scratch)
        """
        with tf.name_scope('Session'):
            self._init_session()
            self._define_saver_summary()
            if load is not None:
                self.saver.restore(self.Session, load)
            self._train(nEpochs, epoch_size, save_step, self.valid_iter)

    def inference_init(self, load):
        """
        Initialize model for inference only.
        :param load: Model to load
        """
        with tf.name_scope('Session'):
            self._init_session()
            self.saver.restore(self.Session, load)

    def weighted_bce_loss(self):
        bce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.output[1], labels=self.gt_map),
            name='cross_entropy_loss'
        )
        e1 = tf.expand_dims(self.weight, axis=1, name='exp_dim1')
        e2 = tf.expand_dims(e1, axis=1, name='exp_dim2')
        e3 = tf.expand_dims(e2, axis=1, name='exp_dim3')
        return tf.multiply(e3, bce_loss, name='lossW')

    def _error_computation(self, output, gt_label, gt_map, weight):
        # point distances for every image in batch
        batch_point_error = []
        for i in range(self.batch_size):
            batch_point_error.append(
                self._error(
                    output[1][i, self.nStacks - 1, :, :, :],
                    gt_label[i],
                    gt_map[i, self.nStacks - 1, :, :, :],
                    weight[i],
                )
            )

        # label correct count for this batch
        pred_label = np.argmax(output[0], axis=1)
        gt_label = np.argmax(gt_label, axis=1)
        correct_label = np.count_nonzero(pred_label == gt_label)
        return batch_point_error, correct_label

    def _error(self, pred, gt_label, gt_map, weight):
        """
        Compute point error for each image and store in self.batch_point_error.
        :param pred: Heat map of shape (hm_size, hm_size, num_points)
        :param gt_label: One hot label of shape: (num_classes,)
        :param gt_map: Ground truth heat map
        :param weight: Point weight
        """
        total_dist = 0.0
        for i in range(len(self.points)):
            if weight[i] != 0:
                pred_idx = np.array(np.where(pred[:, :, i] == 1))
                gt_idx = np.array(np.where(gt_map[:, :, i] == 1))
                total_dist += np.linalg.norm(pred_idx - gt_idx)

        # select the normalization points
        # label {0 1 2}: [5, 6]
        # label {3 4}: [15, 16]
        gt_label = tf.argmax(gt_label)
        if gt_label <= 2:
            norm_idx = np.where(gt_map[:, :, 5:7] == 1)[:2]
        else:
            norm_idx = np.where(gt_map[:, :, 15:17] == 1)[:2]
        norm_dist = np.linalg.norm(norm_idx[:, 1] - norm_idx[:, 0])
        return total_dist / norm_dist

    def _define_saver_summary(self, summary=True):
        if self.logdir_train is None or self.logdir_test is None:
            raise ValueError('Train/Test directory not assigned')
        else:
            self.saver = tf.train.Saver(max_to_keep=10)
            if summary:
                self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                self.test_summary = tf.summary.FileWriter(self.logdir_test)

    def _init_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.Session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('Session initialization')

        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _graph_sphinx(self):
        with tf.name_scope('model'):
            feat, cls_pred = self._graph_resnet('resnet_50')
            feat = ut.deconv_layer(feat, self.nFeats, 1, 8, name='upsample')
            hm_pred = self._graph_hourglass(feat)
            return cls_pred, hm_pred

    def _graph_hourglass(self, feature):
        with tf.name_scope('hourglass'):
            net = ut.conv_layer(self.img, 64, 6, 2, name='conv1')
            net = ut.batch_norm(net, self.training)
            net = ut.bottleneck(net, 128, 32, 1, self.training, name='res1')
            net = ut.max_pool(net, 2, 2, 'max_pool')
            net = ut.bottleneck(net, int(self.nFeats / 2), stride=1, training=self.training, name='res2')
            net = ut.bottleneck(net, self.nFeats, stride=1, training=self.training, name='res3')
            net = tf.add(net, feature, name='feat_merge')

            final_out = []
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg = ut.hourglass(net, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.training, 'dropout')
                    ll = ut.conv_layer(drop, self.nFeats, 1, 1)
                    ll = ut.batch_norm(ll, self.training)
                    ll_ = ut.conv_layer(ll, self.nFeats, 1, 1, name='ll')
                    out = ut.conv_layer(ll, self.out_dim, 1, 1, name='out')
                    out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                    sum_ = tf.add_n([out_, net, ll_], name='merge')
                    final_out.append(out)
                for i in range(1, self.nStacks - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                        drop = ut.dropout(hg, self.dropout_rate, self.training, 'dropout')
                        ll = ut.conv_layer(drop, self.nFeats, 1, 1, name='conv')
                        ll = ut.batch_norm(ll, self.training)
                        ll_ = ut.conv_layer(ll, self.nFeats, 1, 1, name='ll')
                        out = ut.conv_layer(ll, self.out_dim, 1, 1, name='out')
                        out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                        sum_ = tf.add_n([out_, sum_, ll_], name='merge')
                        final_out.append(out)
                with tf.name_scope('stage_' + str(self.nStacks - 1)):
                    hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.training, 'dropout')
                    ll = ut.conv_layer(drop, self.nFeats, 1, 1, name='conv')
                    ll = ut.batch_norm(ll, self.training)
                    out = ut.conv_layer(ll, self.out_dim, 1, 1, name='out')
                    final_out.append(out)
            return tf.stack(final_out, axis=1, name='output')

    def _graph_resnet(self, model='resnet_50'):
        units = ut.RESNET_50_UNIT
        if model == 'resnet_101':
            units = ut.RESNET_101_UNIT
        if model == 'resnet_152':
            units = ut.RESNET_152_UNIT
        if model == 'resnet_200':
            units = ut.RESNET_200_UNIT
        blocks = [
            ut.block('block1', ut.bottleneck, [(256, 64, 1)] * (units[0] - 1) + [(256, 64, 2)]),
            ut.block('block2', ut.bottleneck, [(512, 128, 1)] * (units[1] - 1) + [(512, 128, 2)]),
            ut.block('block3', ut.bottleneck, [(1024, 256, 1)] * (units[2] - 1) + [(1024, 256, 2)]),
            ut.block('block4', ut.bottleneck, [(2048, 512, 1)] * units[3])
        ]

        net = ut.conv_layer(self.img, 64, 7, 2, name='conv')
        net = ut.max_pool(net, 3, 2)
        net = ut.stack_block_dense(net, blocks, self.training)
        feature = net
        # global average pooling
        with tf.name_scope('global_avg_pool'):
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='net_flat')
        prediction = ut.fc_layer(net, self.num_classes, name='fc')
        return feature, prediction
