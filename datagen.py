import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform


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
    points = np.copy(points)
    h, w, _ = shape
    if h > w:
        offset = math.floor((h - w) / 2)
        points[:, 0] += offset
    else:
        offset = math.floor((w - h) / 2)
        points[:, 1] += offset
    return points


def _augment(img, hm, max_rotation=30):
    # use rotation to do data augmentation
    if random.choice([0, 1]):
        r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        img = transform.rotate(img, r_angle, preserve_range=True)
        hm = transform.rotate(hm, r_angle)
    return img, hm


def _make_gaussian(height, width, center, sigma=3):
    """
    Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, np.float32)
    y = np.arange(0, height, 1, np.float32)[:, np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def _make_one_hot(center, shape):
    one_hot = np.zeros(shape)
    one_hot[center] = 1
    return one_hot


class DataGenerator:
    def __init__(self, points_list=None, num_validation=None,
                 img_dir=None, train_data_file=None, keep_invalid=False):
        """
        Initializer
        :param points_list: List of points considered
        :param num_validation: Number of image to use in validation
        :param img_dir: Directory of images
        :param train_data_file: Text file with training set data
        :param keep_invalid: True to keep nonexistent points
        """
        if points_list is None:
            self.points_list = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                                'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                                'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                                'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                                'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        else:
            self.points_list = points_list
        self.label = {'blouse': 0, 'dress': 1, 'outwear': 2, 'skirt': 3, 'trousers': 4}

        self.num_validation = num_validation
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.images = os.listdir(img_dir)
        self.keep_invalid = keep_invalid
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
        input_file = open(self.train_data_file, 'r')
        print('READING TRAIN DATA')
        for line in input_file:
            line = line.strip()
            line = line.split(' ')
            name = line[0]
            category = line[1]
            label = self.label[category]
            points = list(map(int, line[6:]))
            if points == [-1] * len(points):
                self.no_intel.append(name)
            else:
                points = np.reshape(points, (-1, 3))
                w = points[:, 2]
                points = points[:, :2]
                if not self.keep_invalid:
                    points[np.equal(w, 0)] = [-1, -1]
                self.data_dict[name] = {'points': points, 'label': label, 'weight': w}
                self.train_table.append(name)
        input_file.close()

    def _randomize(self):
        # randomize the set
        random.shuffle(self.train_set)

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
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generate_set(self, rand=False):
        """
        Generate the training and validation set
        :param rand: (bool) True to shuffle the set
        """
        self._create_train_table()
        self._create_sets()
        if rand:
            self._randomize()

    # ---------------------------- Generating Methods --------------------------

    @staticmethod
    def _generate_hm(img_size, hm_size, points, weight):
        """
        Generate a full Heap Map for every points in an array
        :param img_size: Size for input image
        :param hm_size: Size for the heat map
        :param points: Array of points
        """
        num_points = points.shape[0]
        hm = np.zeros((hm_size, hm_size, num_points), dtype=np.float32)
        new_p = np.round(points * hm_size / img_size)
        new_p = new_p.astype(np.int32)
        for i in range(num_points):
            if not np.array_equal(new_p[i], [-1, -1]) and weight[i] == 1:
                hm[:, :, i] = _make_one_hot((new_p[i, 1], new_p[i, 0]), (hm_size, hm_size))
            else:
                hm[:, :, i] = np.zeros((hm_size, hm_size))
        return hm

    def generator(self, img_size=256, hm_size=64, batch_size=16, num_classes=5, stacks=4,
                  normalize=True, sample_set='train'):
        """
        Batch Generator
        :param num_classes: Number of classes
        :param hm_size: Size of heat map
        :param img_size: Size of image
        :param batch_size: Number of images per batch
        :param stacks: Number of stacks/module in the network
        :param normalize: True to return Image Value between 0 and 1
        :param sample_set: 'train'/'valid' Default: 'train'
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
                if sample_set == 'valid':
                    name = self.valid_set[self.current_valid_index]
                    self.current_valid_index += 1
                    if self.current_valid_index == len(self.valid_set):
                        self.current_valid_index = 0

                point = self.data_dict[name]['points']
                label = self.data_dict[name]['label']
                gt_labels[i] = _make_one_hot(label, num_classes)
                weight = np.asarray(self.data_dict[name]['weight'])
                weights[i] = weight
                img = self.open_img(name)
                new_p = _relative_points(point, img.shape)
                hm = self._generate_hm(img_size, hm_size, new_p, weight)
                img = _pad_img(img)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                if sample_set == 'train':
                    images, hm = _augment(images, hm)
                if normalize:
                    images[i] = img.astype(np.float32) / 255
                else:
                    images[i] = img.astype(np.float32)

                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                gt_maps[i] = hm
                i = i + 1
            yield images, gt_labels, gt_maps, weights

    # ---------------------------- Image Reader --------------------------------

    def open_img(self, name, color='RGB'):
        """
        Open an image
        :param name: Name of the sample
        :param color: Color Mode (RGB/BGR)
        """
        assert color in ['RGB', 'BGR'], 'Color mode supported: RGB/BGR'
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        if color == 'BGR':
            return img

    def plot_img(self, name, plot='cv2'):
        """
        Plot an image
        :param name: Name of the Sample
        :param plot: Library to use (cv2: OpenCV, plt: matplotlib)
        :return:
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()
