from models.sphinx import SphinxModel
from config import Config
from preprocess.datagen import DataGenerator

import os
import sys
import time
import datetime

import tensorflow as tf
import numpy as np
import utils as ut


class Trainer:
    def __init__(self, model: SphinxModel, cfg: Config, dataset: DataGenerator):
        self.model = model
        self.dataset = dataset
        self.name = cfg.name
        self.logdir = cfg.logdir
        self.save_step = cfg.save_step
        self.saver_dir = cfg.saver_dir
        self.load = cfg.load
        if self.load is None:
            self.start_epoch = 0
        else:
            self.start_epoch = int(self.load.split('_')[1])

        self.img_size = cfg.img_size
        self.hm_size = cfg.hm_size
        self.out_size = cfg.hm_size * cfg.out_rate
        self.nStacks = cfg.nStacks
        self.num_classes = cfg.num_classes

        self.dataset = dataset
        self.category = cfg.category
        self.points = []
        for i in range(len(cfg.points_list)):
            if ut.VALID_POSITION[self.category][i] == 1:
                self.points.append(cfg.points_list[i])
        self.num_points = len(self.points)

        self.batch_size = cfg.batch_size
        self.nEpochs = cfg.nEpochs
        self.epoch_size = cfg.epoch_size
        self.learning_rate = cfg.learning_rate
        self.valid_iter = cfg.valid_iter
        self.decay = cfg.learning_rate_decay
        self.decay_step = cfg.decay_step
        self.is_training = True
        self.resume = {}

    def training(self):
        self._generate_model()
        self._generate_training_graph()
        with tf.name_scope('Session'):
            with tf.Session() as self.Session:
                self.saver = tf.train.Saver(max_to_keep=10)
                self._init_variable()
                self._train()
                
    def _init_variable(self):
        t_start = time.time()
        print('Variable initialization')
        if self.load is None:
            self.Session.run(self.init)
        else:
            self.saver.restore(self.Session, os.path.join(self.saver_dir, self.load))
        t_end = time.time()
        print('Variable initialized in %d sec.' % int(t_end - t_start))

    def _generate_model(self):
        self.is_training = True

        start_time = time.time()

        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, 3))

            self.gt_label = tf.placeholder(tf.float32, (None, self.num_classes))
            self.gt_hm0 = tf.placeholder(tf.float32,
                                         (None, self.nStacks, self.hm_size, self.hm_size, self.num_points))
            self.gt_hm1 = tf.placeholder(tf.float32, (None, self.hm_size * 2, self.hm_size * 2, self.num_points))
            self.gt_hm2 = tf.placeholder(tf.float32, (None, self.hm_size * 4, self.hm_size * 4, self.num_points))
            self.weight = tf.placeholder(tf.float32, (None, self.num_points))
        print('---Inputs : Done.')
        self.output = self.model.graph(self.img, self.is_training)
        print('---Graph : Done.')

        end_time = time.time()
        print('Model created in ' + str(int(end_time - start_time)) + ' sec.')

    def _generate_training_graph(self):
        start_time = time.time()

        print('CREATE GRAPH:')
        with tf.name_scope('loss'):
            self.loss0 = tf.reduce_mean(ut.weighted_loss(
                self.weight, self.nStacks, self.hm_size, self.out_size,
                self.output[0], self.gt_hm0, 0), name='stack_loss')
            self.loss1 = tf.reduce_mean(ut.weighted_loss(
                self.weight, self.nStacks, self.hm_size, self.out_size,
                self.output[1], self.gt_hm1, 1), name='up1_loss')
            self.loss2 = tf.reduce_mean(ut.weighted_loss(
                self.weight, self.nStacks, self.hm_size, self.out_size,
                self.output[2], self.gt_hm2, 2), name='up2_loss')
            self.loss = tf.add_n(
                [self.loss0, self.loss1, self.loss2], name='total_loss')
        print('---Loss : Done.')

        with tf.name_scope('steps'):
            self.train_step = tf.Variable(
                0, name='global_step', trainable=False)
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
                self.train_rmsprop = rmsprop.minimize(
                    self.loss, self.train_step)
        print('---Minimizer : Done.')
        self.init = tf.global_variables_initializer()
        print('---Init : Done.')

        with tf.name_scope('training'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        with tf.name_scope('summary'):
            self.point_error = tf.placeholder(tf.float32)
            tf.summary.scalar('point_error', self.point_error,
                              collections=['valid'])

        self.train_op = tf.summary.merge_all('train')
        self.valid_op = tf.summary.merge_all('valid')
        self._log_summary()
        print('---Summary : Done.')

        end_time = time.time()
        print('Graph created in ' + str(int(end_time - start_time)) + ' sec.')

    def _log_summary(self):
        if self.logdir is None:
            raise ValueError('Train log directory not assigned')
        else:
            self.train_summary = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
            self.valid_summary = tf.summary.FileWriter(self.logdir)

    def _train(self):
        start_time = time.time()
        train_gen = self.dataset.generator(
            self.img_size, self.hm_size, self.batch_size, self.nStacks, 'train')
        valid_gen = self.dataset.generator(
            self.img_size, self.hm_size, self.batch_size, self.nStacks, 'valid')
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
                img_train, hm0_train, hm1_train, hm2_train, w_train = next(
                    train_gen)
                if i % self.save_step == 0:
                    _, c, summary = self.Session.run(
                        [self.train_rmsprop, self.loss, self.train_op],
                        {
                            self.img: img_train,
                            self.gt_hm0: hm0_train,
                            self.gt_hm1: hm1_train,
                            self.gt_hm2: hm2_train,
                            self.weight: w_train
                        }
                    )
                    # Save summary (Loss + Error)
                    self.train_summary.add_summary(
                        summary, epoch * self.epoch_size + i)
                    self.train_summary.flush()
                else:
                    _, c, = self.Session.run(
                        [self.train_rmsprop, self.loss],
                        {
                            self.img: img_train,
                            self.gt_hm0: hm0_train,
                            self.gt_hm1: hm1_train,
                            self.gt_hm2: hm2_train,
                            self.weight: w_train
                        }
                    )
                cost += c
                avg_cost = cost / (i + 1)
            epoch_finish_time = time.time()
            duration = epoch_finish_time - epoch_start_time
            print('\nEpoch done in %d sec. -avg_time/batch: %.2f sec.' %
                (int(duration), duration / self.epoch_size))
            with tf.name_scope('save'):
                self.saver.save(self.Session, os.path.join(
                    self.saver_dir, str(self.name + '_' + str(epoch + 1))))
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
        print('  n. Images: ', self.nEpochs *
            self.epoch_size * self.batch_size)
        print('  Final Loss: %.3f' % cost)
        print('  Relative Loss: %.2f%%' %
            (100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)))
        print('  Relative Improvement - Point: %.2f%%' %
            ((self.resume['point_error'][0] - self.resume['point_error'][-1]) * 100))
        print('  Training Time: ' +
            str(datetime.timedelta(seconds=int(time.time() - start_time))))

    def _valid(self, data_gen):
        self.is_training = False
        point_error = 0
        num_points = 0
        self.dataset.randomize('valid')
        for it in range(self.valid_iter):
            img_valid, _, _, hm2_valid, w_valid = next(data_gen)
            num_points += np.sum(w_valid == 1)
            out = self.Session.run(
                self.output[-1],
                {self.img: img_valid}
            )
            batch_point_error = ut.error_computation(
                self.batch_size, out, hm2_valid, w_valid, self.num_points, self.category)
            point_error += sum(batch_point_error)
        point_error = point_error / num_points
        print('--Avg. Point Error = %.2f%%' % (point_error * 100))
        return point_error
