import math
import os
import random

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

from config import Config


# -------------------- Image Processing Utils --------------------

def _pad_img(image):
    # pad the image using white pixels
    h, w, _ = image.shape
    if h == w:
        return image
    elif h > w:
        offset = math.floor((h - w) / 2)
        background = np.ones([h, h, 3], dtype=np.uint8) * 255
        background[:, offset:offset + w, :] = image
        return background
    else:
        offset = math.floor((w - h) / 2)
        background = np.ones([w, w, 3], dtype=np.uint8) * 255
        background[offset:offset + h, :, :] = image
        return background


def _relative_points(points, shape):
    new_p = np.copy(points)
    h, w, _ = shape
    for i in range(len(new_p)):
        if (new_p[i] == [-1, -1]).any():
            continue
        if h > w:
            offset = math.floor((h - w) / 2)
            new_p[i, 0] += offset
        else:
            offset = math.floor((w - h) / 2)
            new_p[i, 1] += offset
    return new_p


def _padding_offset(shape):
    h, w, _ = shape
    if h > w:
        offset = (math.floor((h - w) / 2), 0)
    else:
        offset = (0, math.floor((w - h) / 2))
    return offset


def _augment(img, hm, max_rotation=30):
    # use rotation to do data augmentation
    if random.choice([0, 1]):
        r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        img = transform.rotate(img, r_angle, cval=255, preserve_range=True)
        hm = transform.rotate(hm, r_angle)
    return img, hm


def _make_gaussian(hm_size, center, sigma=3):
    """
    Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which can be thought of as an effective radius.
    """
    x = np.arange(0, hm_size, 1, np.float32)
    y = np.arange(0, hm_size, 1, np.float32)[:, np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def _make_one_hot(center, shape):
    one_hot = np.zeros(shape)
    one_hot[center] = 1
    return one_hot


class DataGenerator:
    def __init__(self, cfg: Config):
        """
        Initializer
        :param points_list: List of points considered
        :param img_dir: Directory of images
        :param train_data_file: Text file with training set data
        """
        if cfg.points_list is None:
            self.points_list = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                                'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                                'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                                'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                                'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        else:
            self.points_list = cfg.points_list
        self.label = {'blouse': 0, 'dress': 1, 'outwear': 2, 'skirt': 3, 'trousers': 4}

        self.train_img_dir = cfg.train_img_dir
        self.test_img_dir = cfg.test_img_dir
        self.train_data_file = cfg.train_data_file
        self.test_data_file = cfg.test_data_file
        self.current_train_index = 0
        self.current_valid_index = 0

    # -------------------- Generator Initialization Methods --------------------

    def reset_batch_index(self):
        self.current_train_index = 0
        self.current_valid_index = 0

    def _create_train_table(self):
        # Create Table of samples from TEXT file
        self.train_table = []
        self.no_intel = []
        self.data_dict = {}
        with open(self.train_data_file, 'r') as input_file:
            print('READING TRAIN DATA')
            for line in input_file:
                line = line.strip()
                line = line.split(' ')
                name = line[0]
                category = line[1]
                label = self.label[category]
                points = list(map(int, line[2:]))
                if points == [-1] * len(points):
                    self.no_intel.append(name)
                else:
                    points = np.reshape(points, (-1, 3))
                    weight = points[:, 2]
                    points = points[:, :2]
                    self.data_dict[name] = {'points': points, 'label': label, 'weight': weight}
                    self.train_table.append(name)

    def _create_test_set(self):
        self.test_set = []
        self.test_data_dict = {}
        with open(self.test_data_file, 'r') as input_file:
            print('READING TEST DATA')
            spam_reader = csv.reader(input_file)
            head = True
            for row in spam_reader:
                if head:
                    head = False
                    continue

                name = row[0]
                self.test_set.append(name)
                self.test_data_dict[name] = {'category': row[1]}
        print('--Test set :', len(self.test_set), ' samples.')

    def randomize(self, dataset='train'):
        # randomize the set
        if dataset == 'train':
            random.shuffle(self.train_set)
        if dataset == 'valid':
            random.shuffle(self.valid_set)

    def _create_sets(self):
        # Select Elements to feed training and validation set
        num_images = len(self.train_table)
        self.train_set = []
        self.valid_set = []
        valid_index = [i for i in range(0, num_images, 10)]
        for i in range(num_images):
            if i in valid_index:
                self.valid_set.append(self.train_table[i])
            else:
                self.train_set.append(self.train_table[i])
        print('SET CREATED')
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generate_set(self, train=True):
        """
        Generate the dataset
        :param train: (bool) True to generate training table
        """
        if train:
            self._create_train_table()
            self._create_sets()
        else:
            self._create_test_set()

    # ---------------------------- Generating Methods --------------------------

    @staticmethod
    def _generate_hm(orig_size, hm_size, points, weight, keep_invisible):
        """
        Generate a full Heap Map for every points in an array
        :param orig_size: Size for the padded image
        :param hm_size: Size for the heat map
        :param points: Array of points
        :param keep_invisible: True to keep invisible coordinates
        """
        num_points = points.shape[0]
        hm = np.zeros((hm_size, hm_size, num_points), dtype=np.float32)
        for i in range(num_points):
            if weight[i] == 1 or (weight[i] == 0 and keep_invisible):
                new_p = (points[i] * hm_size / orig_size).astype(np.int32)
                hm[:, :, i] = _make_gaussian(hm_size, (new_p[1], new_p[0]))
        return hm

    def generator(self, img_size=256, hm_size=64, batch_size=16, num_classes=5, stacks=4, sample_set='train'):
        """
        Batch Generator
        :param num_classes: Number of classes
        :param hm_size: Size of heat map
        :param img_size: Size of image
        :param batch_size: Number of images per batch
        :param stacks: Number of stacks/module in the network
        :param sample_set: train/valid Default: 'train'
        """
        while True:
            images = np.zeros((batch_size, img_size, img_size, 3), np.float32)
            gt_labels = np.zeros((batch_size, num_classes), np.float32)
            gt_maps = np.zeros((batch_size, stacks, hm_size, hm_size, len(self.points_list)), np.float32)
            weights = np.zeros((batch_size, len(self.points_list)), np.int32)

            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = self.train_set[self.current_train_index]
                    self.current_train_index += 1
                    if self.current_train_index == len(self.train_set):
                        self.current_train_index = 0
                    keep_invisible = False
                if sample_set == 'valid':
                    name = self.valid_set[self.current_valid_index]
                    self.current_valid_index += 1
                    if self.current_valid_index == len(self.valid_set):
                        self.current_valid_index = 0
                    keep_invisible = True

                point = self.data_dict[name]['points']
                label = self.data_dict[name]['label']
                gt_labels[i] = _make_one_hot(label, num_classes)
                weight = np.asarray(self.data_dict[name]['weight'])
                weights[i] = weight
                img = self.open_img(self.train_img_dir, name)
                new_p = _relative_points(point, img.shape)
                orig_size = max(img.shape)
                hm = self._generate_hm(orig_size, hm_size, new_p, weight, keep_invisible)
                img = _pad_img(img)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                if sample_set == 'train':
                    img, hm = _augment(img, hm)
                images[i] = img.astype(np.float32) / 255

                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                gt_maps[i] = hm
                i = i + 1
            yield images, gt_labels, gt_maps, weights

    def test_generator(self, img_size=256, batch_size=16):
        images = np.zeros((batch_size, img_size, img_size, 3), np.float32)

        num_test = len(self.test_set)
        idx = 0
        while idx < num_test:
            if num_test - idx >= batch_size:
                next_size = batch_size
            else:
                next_size = num_test - idx
                images = np.zeros((next_size, img_size, img_size, 3), np.float32)

            categories = []
            offsets = []
            names = []
            sizes = []
            for i in range(next_size):
                name = self.test_set[idx]
                img = self.open_img(self.test_img_dir, name)
                img = _pad_img(img)

                categories.append(self.test_data_dict[name]['category'])
                offsets.append(_padding_offset(img.shape))
                names.append(name)
                sizes.append(max(img.shape))
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                images[i] = img.astype(np.float32) / 255
                idx += 1
            yield images, categories, offsets, names, sizes

    # ---------------------------- Image Reader --------------------------------

    @staticmethod
    def open_img(img_dir, name):
        """
        Open an image
        :param img_dir: Directory of images
        :param name: Name of the sample
        """
        img = cv2.imread(os.path.join(img_dir, name))
        return img

    def plot_img(self, name, plot='cv2'):
        """
        Plot an image
        :param name: Name of the Sample
        :param plot: Library to use (cv2: OpenCV, plt: matplotlib)
        :return:
        """
        img = self.open_img(name)
        if plot == 'cv2':
            cv2.imshow('Image', img)
        elif plot == 'plt':
            plt.imshow(img)
            plt.show()
