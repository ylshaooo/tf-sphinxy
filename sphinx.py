import csv
import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf

import utils as ut
from config import Config
from datagen import DataGenerator


class SphinxModel:
    def __init__(self, cfg: Config, dataset: DataGenerator, training=True):
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
        self.logdir_train = cfg.logdir_train
        self.logdir_valid = cfg.logdir_valid
        self.saver_step = cfg.saver_step
        self.saver_dir = cfg.saver_dir
        self.load = cfg.load
        self.test_txt_file = cfg.test_txt_file
        self.test_output_file = cfg.test_output_file
        self.name = cfg.name
        self.dataset = dataset
        self.training = training
        self.resume = {}

    def generate_model(self):
        start_time = time.time()

        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, 3))
            self.gt_label = tf.placeholder(tf.float32, (None, self.num_classes))
            self.gt_map = tf.placeholder(tf.float32,
                                         (None, self.nStacks, self.hm_size, self.hm_size, self.num_points))
        print('---Inputs : Done.')
        self.output = self._graph_sphinx()
        print('---Graph : Done.')
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.gt_label),
                name='loss'
            )
            # self.hm_loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(logits=self.output[1], labels=self.gt_map),
            #     name='hm_loss'
            # )
            # self.loss = tf.add(self.cls_loss, self.hm_loss, name='loss')
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
            # tf.summary.scalar('hm_loss', self.hm_loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        with tf.name_scope('summary'):
            self.point_error = tf.placeholder(tf.float32)
            self.label_error = tf.placeholder(tf.float32)
            # tf.summary.scalar('point_error', self.point_error, collections=['valid'])
            tf.summary.scalar('label_error', self.label_error, collections=['valid'])

        self.train_op = tf.summary.merge_all('train')
        self.valid_op = tf.summary.merge_all('valid')

        end_time = time.time()
        print('Model created (' + str(int(abs(end_time - start_time))) + ' sec.)')
        del start_time, end_time

    def _train(self):
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
            for epoch in range(self.nEpochs):
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
                        ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] +
                        ' -completion: ' + str(i + 1) + '/' + str(self.epoch_size)
                    )
                    sys.stdout.flush()

                    img_train, lb_train, gt_train, w_train = next(self.train_gen)
                    if i % self.saver_step == 0:
                        _, c, summary = self.Session.run(
                            [self.train_rmsprop, self.loss, self.train_op],
                            {self.img: img_train, self.gt_label: lb_train, self.gt_map: gt_train}
                        )
                        # Save summary (Loss + Error)
                        self.train_summary.add_summary(summary, epoch * self.epoch_size + i)
                        self.train_summary.flush()
                    else:
                        _, c, = self.Session.run(
                            [self.train_rmsprop, self.loss],
                            {self.img: img_train, self.gt_label: lb_train, self.gt_map: gt_train}
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
                label_error = self._valid()
                # self.resume['point_error'].append(point_error)
                self.resume['label_error'].append(label_error)
                valid_summary = self.Session.run(
                    self.valid_op,
                    {self.label_error: label_error}
                )
                self.valid_summary.add_summary(valid_summary, epoch)
                self.valid_summary.flush()

                # reset train and validation index
                self.dataset.reset_batch_index()

            print('Training Done')
            print('Resume:')
            print('  Epochs: ' + str(self.nEpochs))
            print('  n. Images: ' + str(self.nEpochs * self.epoch_size * self.batch_size))
            print('  Final Loss: ' + str(cost))
            print('  Relative Loss: ' + str(100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
            print('  Relative Improvement - Point: ' + str(
                (self.resume['point_error'][0] - self.resume['point_error'][-1]) * 100) + '%')
            print('  Relative Improvement - Label: ' + str(
                (self.resume['label_error'][0] - self.resume['label_error'][-1]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - start_time)))

    def _valid(self):
        # point_error = 0
        num_points = 0
        correct_label = 0
        self.dataset.randomize('valid')
        for it in range(self.valid_iter):
            img_valid, lb_valid, gt_valid, w_valid = next(self.valid_gen)
            num_points += np.sum(w_valid == 1)

            out = self.Session.run(
                self.output,
                feed_dict={
                    self.img: img_valid,
                    # self.gt_map: gt_valid,
                    # self.gt_label: lb_valid,
                }
            )

            batch_correct_label = self._error_computation(out, lb_valid, gt_valid, w_valid)
            # point_error += sum(batch_point_error)
            # print('point error:', point_error)
            correct_label += batch_correct_label
            print('correct label:', correct_label)
        # point_error = point_error / num_points
        label_error = 1 - correct_label / (self.valid_iter * self.batch_size)
        # print('--Avg. Point Error = %.2f%%' % (point_error * 100))
        print('--Avg. Label Error = %.2f%%' % (label_error * 100))
        return label_error

    def _test(self):
        test_gen = self.dataset.test_generator(self.img_size, self.batch_size, True)

        with open(self.test_output_file, 'w', newline='') as outfile:
            spam_writer = csv.writer(outfile)
            # Traversal the test set
            for images, categories, offsets, names in test_gen:
                hms = self.predict(images)

                # Formatting to lines
                for i in range(hms.shape[0]):
                    hm = hms[i]
                    offset = offsets[i]
                    category = categories[i]
                    name = names[i]

                    write_line = [name, category]
                    for i in range(self.num_points):
                        if ut.VALID_POSITION[category][i] is 1:
                            # Calculate predictions from heat map
                            index = np.unravel_index(hm[:, :, i].argmax(), (self.hm_size, self.hm_size))
                            point = np.array(index) / self.hm_size * self.img_size
                            point -= offset
                            write_line.append(str(point[0]) + '_' + str(point[1]) + '_1')
                        else:
                            write_line.append('-1_-1_-1')
                    spam_writer.writerow(write_line)

    def predict(self, images):
        prediction = self.Session.run(self.output[1], feed_dict={self.img: images})
        return prediction

    def training_init(self):
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
            if self.load is not None:
                self.saver.restore(self.Session, self.load)
            self._train()

    def inference_init(self, load, test=True):
        """
        Initialize model for inference only.
        :param load: Model to load
        :param test: True to run test graph, False to run validation graph
        """
        with tf.name_scope('Session'):
            self._init_session()
            self.saver.restore(self.Session, load)
            if test:
                self._test()
            else:
                self._valid()

    def _error_computation(self, output, gt_label, gt_map, weight):
        # point distances for every image in batch
        # batch_point_error = []
        # for i in range(self.batch_size):
        #     batch_point_error.append(
        #         self._error(
        #             output[1][i, self.nStacks - 1, :, :, :],
        #             gt_label[i],
        #             gt_map[i, self.nStacks - 1, :, :, :],
        #             weight[i],
        #         )
        #     )

        # label correct count for this batch
        pred_label = np.argmax(output, axis=1)
        gt_label = np.argmax(gt_label, axis=1)
        correct_label = np.count_nonzero(pred_label == gt_label)
        print('pred_label:', pred_label)
        print('correct_label:', correct_label)
        return correct_label

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
                gt_idx = np.array(np.where(gt_map[:, :, i] == 1))
                total_dist += np.linalg.norm(pred_idx - gt_idx)

        # select the normalization points
        # label {0 1 2}: [5, 6]
        # label {3 4}: [15, 16]
        gt_label = np.argmax(gt_label)
        if gt_label <= 2:
            norm_idx1 = np.array(np.where(gt_map[:, :, 5] == 1))
            norm_idx2 = np.array(np.where(gt_map[:, :, 6] == 1))
        else:
            norm_idx1 = np.array(np.where(gt_map[:, :, 15] == 1))
            norm_idx2 = np.array(np.where(gt_map[:, :, 16] == 1))
        norm_dist = np.linalg.norm(norm_idx2 - norm_idx1)
        return total_dist / norm_dist

    def _define_saver_summary(self, summary=True):
        if self.logdir_train is None or self.logdir_valid is None:
            raise ValueError('Train/Valid directory not assigned')
        else:
            self.saver = tf.train.Saver(max_to_keep=10)
            if summary:
                self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                self.valid_summary = tf.summary.FileWriter(self.logdir_valid)

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
            # hm_pred = self._graph_hourglass(feat)
            return cls_pred

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
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
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
                        out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                        out_ = ut.conv_layer(out, self.nFeats, 1, 1, name='out_')
                        sum_ = tf.add_n([out_, sum_, ll_], name='merge')
                        final_out.append(out)
                with tf.name_scope('stage_' + str(self.nStacks - 1)):
                    hg = ut.hourglass(sum_, self.nLow, self.nFeats, 'hourglass')
                    drop = ut.dropout(hg, self.dropout_rate, self.training, 'dropout')
                    ll = ut.conv_layer(drop, self.nFeats, 1, 1, name='conv')
                    ll = ut.batch_norm(ll, self.training)
                    out = ut.conv_layer(ll, self.num_points, 1, 1, name='out')
                    final_out.append(out)
            return tf.stack(final_out, axis=1, name='output')

    def _graph_resnet(self, model='resnet_50'):
        with tf.name_scope('resnet'):
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
                net = tf.reduce_mean(net, [1, 2], keep_dims=False, name='net_flat')
            prediction = ut.fc_layer(net, self.num_classes, name='fc')
            return feature, prediction
