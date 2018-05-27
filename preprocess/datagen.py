import math
import os
import random

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from PIL import Image

from config import Config
import utils as ut

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


def _augment(img, hms, max_rotation=30):
    # random data augmentation choices
    if random.choice([0, 1]):
        r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        img = transform.rotate(img, r_angle, cval=255, preserve_range=True)
        hms[0] = transform.rotate(hms[0], r_angle)
        hms[1] = transform.rotate(hms[1], r_angle)
        hms[2] = transform.rotate(hms[2], r_angle)
    return img, hms


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

def _clamp_bound(path, bound):
    im = Image.open(path)
    orig_width, orig_height = im.size
    
    bound[0] = max(0, bound[0])
    bound[1] = max(0, bound[1])
    bound[2] = min(orig_width, bound[2])
    bound[3] = min(orig_height, bound[3])    


class DataGenerator:
    def __init__(self, cfg: Config):
        """
        Initializer
        :param points_list: List of points considered
        :param img_dir: Directory of images
        :param train_data_file: Text file with training set data
        """

        self.category = cfg.category
        self.points_list = []
        for i in range(len(cfg.points_list)):
            if ut.VALID_POSITION[self.category][i] == 1:
                self.points_list.append(cfg.points_list[i])

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
        # Create Table of samples from csv(txt) file
        self.train_table = []
        self.no_intel = []
        self.data_dict = {}
        
        with open(self.train_data_file, newline='') as infile:
            print('READING TRAIN DATA')
            spamreader = csv.reader(infile, delimiter=' ')

            for row in spamreader:
                name = row[0]
                bound = list(map(int, row[1:5]))
                points = list(map(int, row[5:]))

                points = np.reshape(points, (-1, 3))
                weight = points[:, 2]
                points = points[:, :2]
                
                _clamp_bound(os.path.join(self.train_img_dir, name), bound)

                points[np.not_equal(weight, -1)] -= np.array(bound[0:2])
                
                self.data_dict[name] = {'points': points, 'bound': bound, 'weight': weight}
                self.train_table.append(name)

    def _create_test_set(self):
        self.test_set = []
        self.test_data_dict = {}
        with open(self.test_data_file, newline='') as infile:
            print('READING TEST DATA')
            spam_reader = csv.reader(infile, delimiter=' ')
            for row in spam_reader:
                name = row[0]
                bound = list(map(int, row[1:5]))
                
                _clamp_bound(os.path.join(self.test_img_dir, name), bound)
                
                self.test_data_dict[name] = {'bound': bound }
                self.test_set.append(name)
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
    def _generate_hm(orig_size, hm_size, points, weight, keep_invisible, sigma=3):
        """
        Generate a full Heap Map for every points in an array
        :param orig_size: Size for the padded image
        :param hm_size: Size for the heat map
        :param points: Array of points
        :param keep_invisible: True to keep invisible coordinates
        :param sigma: Variance of Gaussian distribution
        """
        num_points = points.shape[0]
        hm = np.zeros((hm_size, hm_size, num_points), dtype=np.float32)
        for i in range(num_points):
            if weight[i] == 1 or (weight[i] == 0 and keep_invisible):
                new_p = (points[i] * hm_size / orig_size).astype(np.int32)
                hm[:, :, i] = _make_gaussian(hm_size, (new_p[0], new_p[1]), sigma)
        return hm

    def generator(self, img_size=256, hm_size=64, batch_size=16, stacks=4, sample_set='train'):
        """
        Batch Generator
        :param hm_size: Size of heat map
        :param img_size: Size of image
        :param batch_size: Number of images per batch
        :param stacks: Number of stacks/module in the network
        :param sample_set: train/valid Default: 'train'
        """
        while True:
            images = np.zeros((batch_size, img_size, img_size, 3), np.float32)
            weights = np.zeros((batch_size, len(self.points_list)), np.int32)

            # generate hm of different size
            gt_hms0 = np.zeros((batch_size, stacks, hm_size, hm_size, len(self.points_list)), np.float32)
            gt_hms1 = np.zeros((batch_size, hm_size * 2, hm_size * 2, len(self.points_list)), np.float32)
            gt_hms2 = np.zeros((batch_size, hm_size * 4, hm_size * 4, len(self.points_list)), np.float32)

            i = 0
            keep_invisible = False
            while i < batch_size:
                # cycling indexing
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

                # fetch data
                data = self.data_dict[name]
                point = data['points']
                weight = np.asarray(data['weight'])
                bound = data['bound']
                
                # get cropped image
                img = self.open_img(self.train_img_dir, name, bound)
                new_p = _relative_points(point, img.shape)
                
                # generate hm
                orig_size = max(img.shape)
                hm0 = self._generate_hm(orig_size, hm_size, new_p, weight, keep_invisible, sigma=3)
                hm1 = self._generate_hm(orig_size, hm_size * 2, new_p, weight, keep_invisible, sigma=6)
                hm2 = self._generate_hm(orig_size, hm_size * 4, new_p, weight, keep_invisible, sigma=12)
                
                # generate image
                img = _pad_img(img)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                if sample_set == 'train':
                    img, hms = _augment(img, [hm0, hm1, hm2])
                    hm0, hm1, hm2 = hms

                # batching output
                images[i] = img.astype(np.float32) / 255
                hm0 = np.expand_dims(hm0, axis=0)
                hm0 = np.repeat(hm0, stacks, axis=0)
                gt_hms0[i] = hm0
                gt_hms1[i] = hm1
                gt_hms2[i] = hm2
                weights[i] = weight
                
                i = i + 1
            yield images, gt_hms0, gt_hms1, gt_hms2, weights

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
                # fetch data
                name = self.test_set[idx]
                bound = self.test_data_dict[name]['bound']
                img = self.open_img(self.test_img_dir, name, bound)
                
                # batching output
                offsets.append(np.array(_padding_offset(img.shape)) - bound[0:2])
                names.append(name)
                sizes.append(max(img.shape))

                img = _pad_img(img)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                images[i] = img.astype(np.float32) / 255
                idx += 1
            yield images, offsets, names, sizes

    # ---------------------------- Image Reader --------------------------------

    @staticmethod
    def open_img(img_dir, name, bound=None):
        """
        Open an image
        :param img_dir: Directory of images
        :param name: Name of the sample
        :param bound: The interested bound
        """
        img = cv2.imread(os.path.join(img_dir, name))
        if bound is not None:
            img = img[bound[1]:bound[3], bound[0]:bound[2]]
        return img

    def plot_img(self, img_dir, name, plot='cv2'):
        """
        Plot an image
        :param name: Name of the Sample
        :param plot: Library to use (cv2: OpenCV, plt: matplotlib)
        :return:
        """
        img = self.open_img(img_dir, name)
        if plot == 'cv2':
            cv2.imshow('Image', img)
        elif plot == 'plt':
            plt.imshow(img)
            plt.show()
