from models.sphinx import SphinxModel
from config import Config
from preprocess.datagen import DataGenerator

import os
import csv
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import utils as ut

class Tester:
    def __init__(self, model: SphinxModel, cfg: Config, dataset: DataGenerator):
        self.model = model
        self.dataset = dataset

        self.category = cfg.category
        self.img_size = cfg.img_size
        self.hm_size = cfg.hm_size
        self.out_size = cfg.hm_size * cfg.out_rate
        self.nStacks = cfg.nStacks
        self.batch_size = cfg.batch_size
        self.total_points = len(cfg.points_list)
        self.load = cfg.load
        self.saver_dir = cfg.saver_dir
        self.test_txt_file = cfg.test_data_file
        self.test_output_file = cfg.test_output_file

    def inference(self):
        self._generate_model()
        with tf.name_scope('Session'):
            with tf.Session() as self.Session:
                self.saver = tf.train.Saver(max_to_keep=10)
                self._init_variable()
                self._test()
                
    def _init_variable(self):
        assert (self.load is not None), 'Config "Load" must be specified.'
        
        t_start = time.time()
        print('Variable initialization')
        self.saver.restore(self.Session, os.path.join(self.saver_dir, self.load))
        t_end = time.time()
        print('Variable initialized in %d sec.' % int(t_end - t_start))

    def _generate_model(self):
        self.is_training = False

        start_time = time.time()

        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(tf.float32, (None, self.img_size, self.img_size, 3))
        print('---Inputs : Done.')
        self.output = self.model.graph(self.img, self.is_training)
        print('---Graph : Done.')

        end_time = time.time()
        print('Model created in ' + str(int(end_time - start_time)) + ' sec.')

    def _test(self):
        test_gen = self.dataset.test_generator(self.img_size, self.batch_size)
        with open(self.test_output_file, 'w') as outfile:
            spam_writer = csv.writer(outfile)
            # Traversal the test set
            for _ in tqdm(range(len(self.dataset.test_set) // self.batch_size + 1)):
                images, offsets, names, sizes = next(test_gen)
                prediction = self.Session.run(self.output[-1], feed_dict={self.img: images})
                hms = prediction
                # Formatting to lines
                for i in range(hms.shape[0]):
                    hm = hms[i]
                    offset = offsets[i]
                    name = names[i]
                    size = sizes[i]
                    write_line = [name, self.category]
                    cnt = 0
                    for j in range(self.total_points):
                        if ut.VALID_POSITION[self.category][j] == 1:
                            index = np.unravel_index(hm[:, :, cnt].argmax(), (self.out_size, self.out_size))
                            index = (index[1], index[0])
                            point = np.array(index) / self.out_size * size
                            point -= offset
                            write_line.append(str(int(round(point[0]))) + '_' + str(int(round(point[1]))) + '_1')
                            cnt += 1
                        else:
                            write_line.append('-1_-1_-1')
                    spam_writer.writerow(write_line)
