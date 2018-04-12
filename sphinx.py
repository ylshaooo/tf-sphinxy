import csv
import datetime
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import utils as ut
from config import Config
from datagen import DataGenerator


class SphinxModel:
    def __init__(self, cfg: Config, dataset: DataGenerator):
        self.img_size = cfg.img_size
        self.hm_size = cfg.hm_size
        self.nStacks = cfg.nStacks
        self.nFeats = cfg.nFeats
        self.nLow = cfg.nLow
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes
        self.points_list = cfg.points_list
        self.num_points = len(self.points_list)
        self.dropout_rate = cfg.dropout_rate
        self.learning_rate = cfg.learning_rate
        self.decay = cfg.learning_rate_decay
        self.decay_step = cfg.decay_step
        self.nEpochs = cfg.nEpochs
        self.epoch_size = cfg.epoch_size
        self.valid_iter = cfg.valid_iter
        self.logdir = cfg.logdir
        self.save_step = cfg.save_step
        self.saver_dir = cfg.saver_dir
        self.test_txt_file = cfg.test_data_file
        self.test_output_file = cfg.test_output_file
        self.name = cfg.name
        self.load = cfg.load
        if self.load is None:
            self.start_epoch = 0
        else:
            self.start_epoch = int(self.load.split('_')[1])

        self.dataset = dataset
        self.is_training = True
        self.resume = {}

    def generate_model(self, train):
        start_time = time.time()

        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, 3))
            if train:
                self.gt_label = tf.placeholder(tf.float32, (None, self.num_classes))
                self.gt_map = tf.placeholder(tf.float32,
                                             (None, self.nStacks, self.hm_size, self.hm_size, self.num_points))
                self.weight = tf.placeholder(tf.float32, (None, self.num_points))
        print('---Inputs : Done.')
        self.output = self._graph_hourglass()
        print('---Graph : Done.')

        end_time = time.time()
        print('Model created in ' + str(int(end_time - start_time)) + ' sec.')

    def _generate_training_graph(self):
        start_time = time.time()

        print('CREATE GRAPH:')
        with tf.name_scope('loss'):
            self.hm_loss = tf.reduce_mean(self._weighted_loss(), name='hm_loss')
        print('---Loss : Done.')

        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('lr'):
            self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                 staircase=True, name='learning_rate')
        print('---Learning Rate : Done.')

        with tf.name_scope('rmsprop'):
            rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        print('---Optimizer : Done.')
        with tf.name_scope('minimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_rmsprop = rmsprop.minimize(self.hm_loss, self.train_step)
        print('---Minimizer : Done.')
        self.init = tf.global_variables_initializer()
        print('---Init : Done.')

        with tf.name_scope('training'):
            tf.summary.scalar('hm_loss', self.hm_loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        with tf.name_scope('summary'):
            self.point_error = tf.placeholder(tf.float32)
            tf.summary.scalar('point_error', self.point_error, collections=['valid'])

        self.train_op = tf.summary.merge_all('train')
        self.valid_op = tf.summary.merge_all('valid')
        self._log_summary()
        print('---Summary : Done.')

        end_time = time.time()
        print('Graph created in ' + str(int(end_time - start_time)) + ' sec.')

    def _weighted_loss(self):
        with tf.name_scope('weighted_loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gt_map)
            weight = tf.cast(tf.equal(self.weight, 1), tf.float32)
            weight = tf.expand_dims(weight, 1)
            weight = tf.expand_dims(weight, 1)
            weight = tf.expand_dims(weight, 1)
            weight = tf.tile(weight, [1, self.nStacks, self.hm_size, self.hm_size, 1])
            loss = tf.multiply(loss, weight, name='weighted_out')
            return loss

    def _error_computation(self, output, gt_label, gt_map, weight):
        # point distances for every image in batch
        batch_point_error = []
        for i in range(self.batch_size):
            batch_point_error.append(
                self._error(
                    output[i, self.nStacks - 1, :, :, :],
                    gt_label[i],
                    gt_map[i, self.nStacks - 1, :, :, :],
                    weight[i],
                )
            )

        return batch_point_error

    def _error(self, pred, gt_label, gt_map, weight):
        """
        Compute point error for each image and store in self.batch_point_error.
        :param pred: Heat map of shape (hm_size, hm_size, num_points)
        :param gt_label: One hot label of shape: (num_classes,)
        :param gt_map: Ground truth heat map
        :param weight: Point weight
        """
        total_dist = 0.0
        for i in range(len(self.points_list)):
            if weight[i] == 1:
                pred_idx = np.array(np.where(pred[:, :, i] == np.max(pred[:, :, i])))
                gt_idx = np.array(np.where(gt_map[:, :, i] == np.max(gt_map[:, :, i])))
                total_dist += np.linalg.norm(pred_idx - gt_idx)

        # select the normalization points
        # label {0 1 2}: [5, 6]
        # label {3 4}: [15, 16]
        gt_label = np.argmax(gt_label)
        if gt_label <= 2:
            norm_idx1 = np.array(np.where(gt_map[:, :, 5] == np.max(gt_map[:, :, 5])))
            norm_idx2 = np.array(np.where(gt_map[:, :, 6] == np.max(gt_map[:, :, 6])))
        else:
            norm_idx1 = np.array(np.where(gt_map[:, :, 15] == np.max(gt_map[:, :, 15])))
            norm_idx2 = np.array(np.where(gt_map[:, :, 16] == np.max(gt_map[:, :, 16])))
        norm_dist = np.linalg.norm(norm_idx2 - norm_idx1)
        return total_dist / norm_dist

    def training(self):
        self._generate_training_graph()
        with tf.name_scope('Session'):
            with tf.Session() as self.Session:
                self._init_variable()
                self._train()

    def inference(self):
        with tf.name_scope('Session'):
            with tf.Session() as self.Session:
                self._init_variable()
                self._test()

    def _log_summary(self):
        if self.logdir is None:
            raise ValueError('Train log directory not assigned')
        else:
            self.train_summary = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
            self.valid_summary = tf.summary.FileWriter(self.logdir)

    def _init_variable(self):
        t_start = time.time()
        print('Variable initialization')
        self.saver = tf.train.Saver(max_to_keep=10)
        if self.load is None:
            self.Session.run(self.init)
        else:
            self.saver.restore(self.Session, os.path.join(self.saver_dir, self.load))
        t_end = time.time()
        print('Variable initialized in ' + str(int(t_end - t_start)) + ' sec.')

    def _train(self):
        start_time = time.time()
        train_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size,
                                           self.num_classes, self.nStacks, 'train')
        valid_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size,
                                           self.num_classes, self.nStacks, 'valid')
        self.resume['loss'] = []
        self.resume['point_error'] = []
        # self.resume['label_error'] = []
        cost = 0.
        for epoch in range(self.start_epoch, self.nEpochs):
            self.is_training = True
            epoch_start_time = time.time()
            avg_cost = 0.
            cost = 0.
            self.dataset.randomize()
            print('Epoch :' + str(epoch + 1) + '/' + str(self.nEpochs) + '\n')

            # Training Set
            for i in range(self.epoch_size):
                percent = ((i + 1) / self.epoch_size) * 100
                num = np.int(20 * percent / 100)
                sys.stdout.write(
                    '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' +
                    ' -loss: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] +
                    ' -completion: ' + str(i + 1) + '/' + str(self.epoch_size)
                )
                sys.stdout.flush()

                img_train, _, hm_train, w_train = next(train_gen)
                if i % self.save_step == 0:
                    _, c, summary = self.Session.run(
                        [self.train_rmsprop, self.hm_loss, self.train_op],
                        {self.img: img_train, self.gt_map: hm_train, self.weight: w_train}
                    )
                    # Save summary (Loss + Error)
                    self.train_summary.add_summary(summary, epoch * self.epoch_size + i)
                    self.train_summary.flush()
                else:
                    _, c, = self.Session.run(
                        [self.train_rmsprop, self.hm_loss],
                        {self.img: img_train, self.gt_map: hm_train, self.weight: w_train}
                    )
                cost += c
                avg_cost = cost / (i + 1)
            epoch_finish_time = time.time()

            print('\nEpoch done in ' + str(int(epoch_finish_time - epoch_start_time)) +
                  ' sec.' + ' -avg_time/batch: ' +
                  str(((epoch_finish_time - epoch_start_time) / self.epoch_size))[:4] + ' sec.')
            with tf.name_scope('save'):
                self.saver.save(self.Session,
                                os.path.join(self.saver_dir, str(self.name + '_' + str(epoch + 1))))
            self.resume['loss'].append(cost)

            # Validation Set
            point_error = self._valid(valid_gen)
            self.resume['point_error'].append(point_error)
            valid_summary = self.Session.run(
                self.valid_op,
                {self.point_error: point_error}
            )
            self.valid_summary.add_summary(valid_summary, epoch)
            self.valid_summary.flush()

            # reset train and validation index
            self.dataset.reset_batch_index()

        print('Training Done')
        print('Resume:')
        print('  Epochs:', self.nEpochs)
        print('  n. Images: ', self.nEpochs * self.epoch_size * self.batch_size)
        print('  Final Loss: %.3f' % cost)
        print('  Relative Loss: %.2f%%' % 100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1))
        print('  Relative Improvement - Point: %.2f%%' %
            (self.resume['point_error'][0] - self.resume['point_error'][-1]) * 100)
        print('  Training Time: ' + str(datetime.timedelta(seconds=int(time.time() - start_time))))

    def _valid(self, data_gen):
        self.is_training = False
        point_error = 0
        num_points = 0
        # correct_label = 0
        self.dataset.randomize('valid')
        for it in range(self.valid_iter):
            img_valid, lb_valid, gt_valid, w_valid = next(data_gen)
            num_points += np.sum(w_valid == 1)

            out = self.Session.run(
                self.output,
                {self.img: img_valid}
            )

            batch_point_error = self._error_computation(out, lb_valid, gt_valid, w_valid)
            point_error += sum(batch_point_error)
        point_error = point_error / num_points
        print('--Avg. Point Error = %.2f%%' % (point_error * 100))
        return point_error

    def _test(self):
        self.is_training = False
        test_gen = self.dataset.test_generator(self.img_size, self.batch_size)

        with open(self.test_output_file, 'w') as outfile:
            spam_writer = csv.writer(outfile)
            # Traversal the test set
            for _ in tqdm(range(len(self.dataset.test_set) // self.batch_size + 1)):
                images, categories, offsets, names, sizes = next(test_gen)
                prediction = self.Session.run(self.output, feed_dict={self.img: images})
                hms = prediction[:, self.nStacks - 1, :, :, :]

                # Formatting to lines
                for i in range(hms.shape[0]):
                    hm = hms[i]
                    offset = offsets[i]
                    category = categories[i]
                    name = names[i]
                    size = sizes[i]

                    write_line = [name, category]
                    for j in range(self.num_points):
                        if ut.VALID_POSITION[category][j] is 1:
                            # Calculate predictions from heat map
                            index = np.unravel_index(hm[:, :, j].argmax(), (self.hm_size, self.hm_size))
                            point = np.array(index) / self.hm_size * size
                            point -= offset
                            write_line.append(str(int(round(point[1]))) + '_' + str(int(round(point[0]))) + '_1')
                        else:
                            write_line.append('-1_-1_-1')
                    spam_writer.writerow(write_line)

    def _graph_hourglass(self):
        with tf.name_scope('model'):
            net = ut.conv_layer_bn(self.img, 64, 6, 2, self.is_training, name='conv1')
            net = ut.bottleneck(net, 128, stride=1, training=self.is_training, name='res1')
            net = ut.max_pool(net, 2, 2, 'max_pool')
            net = ut.bottleneck(net, int(self.nFeats / 2), stride=1, training=self.is_training, name='res2')
            net = ut.bottleneck(net, self.nFeats, stride=1, training=self.is_training, name='res3')
            # net = tf.add(net, feature, name='feat_merge')

            final_out = []
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg = ut.hourglass(net, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.is_training, 'dropout')
                    ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, self.is_training)
                    ll = ut.conv_layer(ll, self.nFeats, 1, 1, name='ll')
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                    out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                    sum_ = tf.add_n([out_, net, ll], name='merge')
                    final_out.append(out)
                for i in range(1, self.nStacks - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                        drop = ut.dropout(hg, self.dropout_rate, self.is_training, 'dropout')
                        ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, self.is_training)
                        ll = ut.conv_layer(ll, self.nFeats, 1, 1, name='ll')
                        out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                        out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                        sum_ = tf.add_n([out_, sum_, ll], name='merge')
                        final_out.append(out)
                with tf.name_scope('stage_' + str(self.nStacks - 1)):
                    hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.is_training, 'dropout')
                    ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, self.is_training)
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                    final_out.append(out)
            return tf.stack(final_out, axis=1, name='output')
