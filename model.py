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
        self.out_size = cfg.hm_size * cfg.out_rate
        self.nStacks = cfg.nStacks
        self.nFeats = cfg.nFeats
        self.nLow = cfg.nLow
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes
        self.is_top = cfg.is_top
        if self.is_top:
            self.points = cfg.top_points
        else:
            self.points = cfg.bottom_points
        self.num_points = len(self.points)
        self.total_points = len(cfg.points_list)
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
                self.gt_hm0 = tf.placeholder(tf.float32,
                                             (None, self.nStacks, self.hm_size, self.hm_size, self.num_points))
                self.gt_hm1 = tf.placeholder(tf.float32, (None, self.hm_size * 2, self.hm_size * 2, self.num_points))
                self.gt_hm2 = tf.placeholder(tf.float32, (None, self.hm_size * 4, self.hm_size * 4, self.num_points))
                self.gt_hm3 = tf.placeholder(tf.float32, (None, self.out_size, self.out_size, self.num_points))
                self.weight = tf.placeholder(tf.float32, (None, self.num_points))
        print('---Inputs : Done.')
        self.output = self._graph_sphinx()
        print('---Graph : Done.')

        end_time = time.time()
        print('Model created in ' + str(int(end_time - start_time)) + ' sec.')

    def _generate_training_graph(self):
        start_time = time.time()

        print('CREATE GRAPH:')
        with tf.name_scope('loss'):
            self.loss0 = tf.reduce_mean(self._weighted_loss(self.output[0], self.gt_hm0, 0), name='stack_loss')
            self.loss1 = tf.reduce_mean(self._weighted_loss(self.output[1], self.gt_hm1, 1), name='up1_loss')
            self.loss2 = tf.reduce_mean(self._weighted_loss(self.output[2], self.gt_hm2, 2), name='up2_loss')
            self.loss3 = tf.reduce_mean(self._weighted_loss(self.output[3], self.gt_hm3, 3), name='up3_loss')
            self.loss = tf.add_n([self.loss0, self.loss1, self.loss2, self.loss3], name='total_loss')
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
                self.train_rmsprop = rmsprop.minimize(self.loss, self.train_step)
        print('---Minimizer : Done.')
        self.init = tf.global_variables_initializer()
        print('---Init : Done.')

        with tf.name_scope('training'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
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

    def _weighted_loss(self, logits, gt_hm, step):
        with tf.name_scope('weighted_loss_' + str(step)):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gt_hm)
            weight = tf.cast(tf.equal(self.weight, 1), tf.float32)
            weight = tf.expand_dims(weight, 1)
            weight = tf.expand_dims(weight, 1)
            if step == 0:
                weight = tf.expand_dims(weight, 1)
                weight = tf.tile(weight, [1, self.nStacks, self.hm_size, self.hm_size, 1])
            elif step == 1:
                weight = tf.tile(weight, [1, self.hm_size * 2, self.hm_size * 2, 1])
            elif step == 2:
                weight = tf.tile(weight, [1, self.hm_size * 4, self.hm_size * 4, 1])
            elif step == 3:
                weight = tf.tile(weight, [1, self.out_size, self.out_size, 1])
            else:
                raise ValueError('Wrong up step of output.')
            loss = tf.multiply(loss, weight, name='weighted_out')
            return loss

    def _error_computation(self, output, gt_map, weight):
        # point distances for every image in batch
        batch_point_error = []
        for i in range(self.batch_size):
            batch_point_error.append(
                self._error(
                    output[i, :, :, :],
                    gt_map[i, :, :, :],
                    weight[i],
                )
            )

        return batch_point_error

    def _error(self, pred, gt_map, weight):
        """
        Compute point error for each image and store in self.batch_point_error.
        :param pred: Heat map of shape (hm_size, hm_size, num_points)
        :param gt_map: Ground truth heat map
        :param weight: Point weight
        """
        total_dist = 0.0
        for i in range(len(self.points)):
            if weight[i] == 1:
                pred_idx = np.array(np.where(pred[:, :, i] == np.max(pred[:, :, i])))
                gt_idx = np.array(np.where(gt_map[:, :, i] == np.max(gt_map[:, :, i])))
                total_dist += np.linalg.norm(pred_idx - gt_idx)

        # select the normalization points
        if self.is_top:
            norm_idx1 = np.array(np.where(gt_map[:, :, 5] == np.max(gt_map[:, :, 5])))
            norm_idx2 = np.array(np.where(gt_map[:, :, 6] == np.max(gt_map[:, :, 6])))
        else:
            norm_idx1 = np.array(np.where(gt_map[:, :, 0] == np.max(gt_map[:, :, 0])))
            norm_idx2 = np.array(np.where(gt_map[:, :, 1] == np.max(gt_map[:, :, 1])))
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
        print('Variable initialized in %d sec.' % int(t_end - t_start))

    def _train(self):
        start_time = time.time()
        train_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size, self.nStacks, 'train')
        valid_gen = self.dataset.generator(self.img_size, self.hm_size, self.batch_size, self.nStacks, 'valid')
        self.resume['loss'] = []
        self.resume['point_error'] = []
        cost = 0.
        for epoch in range(self.start_epoch, self.nEpochs):
            self.is_training = True
            epoch_start_time = time.time()
            avg_cost = 0.
            cost = 0.
            self.dataset.randomize()
            print('\nEpoch : %d/%d' % (epoch + 1, self.nEpochs))

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

                img_train, hm0_train, hm1_train, hm2_train, hm3_train, w_train = next(train_gen)
                if i % self.save_step == 0:
                    _, c, summary = self.Session.run(
                        [self.train_rmsprop, self.loss, self.train_op],
                        {
                            self.img: img_train,
                            self.gt_hm0: hm0_train,
                            self.gt_hm1: hm1_train,
                            self.gt_hm2: hm2_train,
                            self.gt_hm3: hm3_train,
                            self.weight: w_train
                         }
                    )
                    # Save summary (Loss + Error)
                    self.train_summary.add_summary(summary, epoch * self.epoch_size + i)
                    self.train_summary.flush()
                else:
                    _, c, = self.Session.run(
                        [self.train_rmsprop, self.loss],
                        {
                            self.img: img_train,
                            self.gt_hm0: hm0_train,
                            self.gt_hm1: hm1_train,
                            self.gt_hm2: hm2_train,
                            self.gt_hm3: hm3_train,
                            self.weight: w_train
                        }
                    )
                cost += c
                avg_cost = cost / (i + 1)
            epoch_finish_time = time.time()
            duration = epoch_finish_time - epoch_start_time

            print('\nEpoch done in %d sec. -avg_time/batch: %.2f sec.' % (int(duration), duration / self.epoch_size))
            with tf.name_scope('save'):
                self.saver.save(self.Session, os.path.join(self.saver_dir, str(self.name + '_' + str(epoch + 1))))
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
        self.dataset.randomize('valid')
        for it in range(self.valid_iter):
            img_valid, _, _, _, hm3_valid, w_valid = next(data_gen)
            num_points += np.sum(w_valid == 1)

            out = self.Session.run(
                self.output[3],
                {self.img: img_valid}
            )

            batch_point_error = self._error_computation(out, hm3_valid, w_valid)
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
                prediction = self.Session.run(self.output[3], feed_dict={self.img: images})
                hms = prediction

                # Formatting to lines
                for i in range(hms.shape[0]):
                    hm = hms[i]
                    offset = offsets[i]
                    category = categories[i]
                    name = names[i]
                    size = sizes[i]

                    write_line = [name, category]
                    cnt = 0
                    for j in range(self.total_points):
                        if ut.VALID_POINTS[self.is_top][j]:
                            if ut.VALID_POSITION[category][j] == 1:
                                index = np.unravel_index(hm[:, :, cnt].argmax(), (self.out_size, self.out_size))
                                index = (index[1], index[0])
                                point = np.array(index) / self.out_size * size
                                point -= offset
                                write_line.append(str(int(round(point[0]))) + '_' + str(int(round(point[1]))) + '_1')
                            else:
                                write_line.append('-1_-1_-1')
                            cnt += 1
                        else:
                            write_line.append('-1_-1_-1')
                    spam_writer.writerow(write_line)

    def _graph_sphinx(self):
        with tf.name_scope('model'):
            net = ut.conv_layer(self.img, 64, 7, 2, name='conv1')
            net = ut.bottleneck(net, 128, stride=1, training=self.is_training, name='res1')
            net = ut.max_pool(net, 2, 2, 'max_pool')
            net = ut.bottleneck(net, int(self.nFeats / 2), stride=1, training=self.is_training, name='res2')
            net = ut.bottleneck(net, self.nFeats, stride=1, training=self.is_training, name='res3')

            with tf.name_scope('stacks'):
                stack_out = []
                with tf.name_scope('stage_0'):
                    hg = ut.hourglass(net, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.is_training, 'dropout')
                    ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, self.is_training)
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                    out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                    sum_ = tf.add(net, out_, name='merge')
                    stack_out.append(out)
                for i in range(1, self.nStacks):
                    with tf.name_scope('stage_' + str(i)):
                        hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                        drop = ut.dropout(hg, self.dropout_rate, self.is_training, 'dropout')
                        ll = ut.conv_layer_bn(drop, self.nFeats, 1, 1, self.is_training)
                        out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                        out_ = ut.conv_layer(ll, self.nFeats, 1, 1, name='out_')
                        sum_ = tf.add(sum_, out_, name='merge')
                        stack_out.append(out)
            with tf.name_scope('upsampling'):
                net = ut.batch_norm(sum_, self.is_training)
                net = ut.conv_layer_bn(net, self.nFeats, 3, 1, self.is_training)
                up1 = ut.deconv_layer(net, self.num_points, 1, 2, name='up_1')
                net = ut.conv_layer_bn(up1, self.nFeats, 3, 1, self.is_training)
                up2 = ut.deconv_layer(net, self.num_points, 1, 2, name='up_2')
                net = ut.conv_layer_bn(up2, self.nFeats, 3, 1, self.is_training)
                up3 = ut.deconv_layer(net, self.num_points, 1, 2, name='up3')
            return tf.stack(stack_out, axis=1, name='stack_out'), up1, up2, up3
