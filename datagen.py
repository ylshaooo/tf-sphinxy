import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as scm
from skimage import transform


class DataGenerator:
    def __init__(self, points_list=None, img_dir=None, train_data_file=None, keep_invalid=False):
        """
        Initializer
        :param points_list: List of points considered
        :param img_dir: Directory of images
        :param train_data_file: Text file with training set data
        :param keep_invalid: Whether to use invalid data
        """
        if points_list is None:
            self.points_list = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                                'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                                'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                                'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                                'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        else:
            self.points_list = points_list

        self.letter = ['A']
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.images = os.listdir(img_dir)
        self.keep_invalid = keep_invalid
        self.current_train_index = 0
        self.current_valid_index = 0

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
            points = list(map(int, line[6:]))
            if points == [-1] * len(points):
                self.no_intel.append(name)
            else:
                points = np.reshape(points, (-1, 3))
                w = points[:, 2]
                points = points[:, :2]
                if not self.keep_invalid:
                    points[np.equal(w, 0)] = [-1, -1]
                self.data_dict[name] = {'points': points, 'weights': w, 'category': category}
                self.train_table.append(name)
        input_file.close()

    def _randomize(self):
        # randomize the set
        random.shuffle(self.train_set)

    def _complete_sample(self, name):
        """
        Check if a sample has no missing value
        :param name: Name of the sample
        """
        for i in range(self.data_dict[name]['points'].shape[0]):
            if np.array_equal(self.data_dict[name]['points'][i], [-1, -1]):
                return False
        return True

    def _give_batch_name(self, batch_size=16, set_type='train'):
        """
        :param batch_size: Number of sample
        :param set_type: Set to use (valid/train)
        :return: a List of Samples
        """
        list_file = []
        for i in range(batch_size):
            if set_type == 'train':
                list_file.append(random.choice(self.train_set))
            elif set_type == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

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
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        self._create_sets()
        if rand:
            self._randomize()

    # ---------------------------- Generating Methods --------------------------

    def _make_gaussian(self, height, width, sigma=3, center=None):
        """
        Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, points, max_length, weight):
        """ Generate a full Heap Map for every points in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            points			: Array of points
            max_length		: Length of the Bounding Box
        """
        num_points = points.shape[0]
        hm = np.zeros((height, width, num_points), dtype=np.float32)
        new_j = points.astype(np.int32)
        for i in range(num_points):
            if not (np.array_equal(new_j[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
                hm[:, :, i] = self._make_gaussian(height, width, sigma=s, center=(new_j[i, 0], new_j[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _crop_data(self, height, width, boxp=0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of points.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            points		: Array of points
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        crop_box = [0, 0, width - 1, height - 1]
        # print(crop_box)
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[1][1] = abs(width - bounds[1])
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
            if bounds[0] < 0:
                padding[0][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[0][1] = abs(height - bounds[1])
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        return padding, crop_box

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        hm = np.pad(hm, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        hm = hm[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
             crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img, hm

    def _relative_points(self, box, padding, points, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(points)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j

    def _augment(self, img, hm, max_rotation=30):
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm

    # ----------------------- Batch Generator ----------------------------------

    def _generator(self, batch_size=16, stacks=4, set_type='train', normalize=True, debug=False):
        """ Create Generator for Training
        Args:
            batch_size	: Number of images per batch
            stacks			: Number of stacks/module in the network
            set				: Training/Testing/Validation set # TODO: Not implemented yet
            normalize		: True to return Image Value between 0 and 1
            _debug			: Boolean to test the computation time (/!\ Keep False)
        # Done : Optimize Computation time
            16 Images --> 1.3 sec (on i7 6700hq)
        """
        while True:
            if debug:
                t = time.time()
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gt = np.zeros((batch_size, stacks, 64, 64, len(self.points_list)), np.float32)
            files = self._give_batch_name(batch_size=batch_size, set_type=set_type)
            for i, name in enumerate(files):
                if name[:-1] in self.images:
                    try:
                        img = self.open_img(name)
                        points = self.data_dict[name]['points']
                        box = self.data_dict[name]['box']
                        weight = self.data_dict[name]['weights']
                        if debug:
                            print(box)
                        pad, cbox = self._crop_data(img.shape[0], img.shape[1], box, points, boxp=0.2)
                        if debug:
                            print(cbox)
                            print('maxl :', max(cbox[2], cbox[3]))
                        new_j = self._relative_points(cbox, pad, points, to_size=64)
                        hm = self._generate_hm(64, 64, new_j, 64, weight)
                        img = self._crop_img(img, pad, cbox)
                        img = img.astype(np.uint8)
                        # On 16 image per batch
                        # Avg Time -OpenCV : 1.0 s -skimage: 1.25 s -scipy.misc.imresize: 1.05s
                        img = scm.imresize(img, (256, 256))
                        # Less efficient that OpenCV resize method
                        # img = transform.resize(img, (256,256), preserve_range = True, mode = 'constant')
                        # May Cause trouble, bug in OpenCV imgwrap.cpp:3229
                        # error: (-215) ssize.area() > 0 in function cv::resize
                        # img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
                        img, hm = self._augment(img, hm)
                        hm = np.expand_dims(hm, axis=0)
                        hm = np.repeat(hm, stacks, axis=0)
                        if normalize:
                            train_img[i] = img.astype(np.float32) / 255
                        else:
                            train_img[i] = img.astype(np.float32)
                        train_gt[i] = hm
                    except:
                        i = i - 1
                else:
                    i = i - 1
            if debug:
                print('Batch : ', time.time() - t, ' sec.')
            yield train_img, train_gt

    def aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        # Auxiliary Generator
        global name
        while True:
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.points_list)), np.float32)
            train_weights = np.zeros((batch_size, len(self.points_list)), np.float32)
            i = 0
            while i < batch_size:
                try:
                    if sample_set == 'train':
                        name = self.train_set[self.current_train_index]
                        self.current_train_index += 1
                        if self.current_train_index == len(self.train_set):
                            self.current_train_index = 0
                    elif sample_set == 'valid':
                        name = self.valid_set[self.current_valid_index]
                        self.current_valid_index += 1
                        if self.current_valid_index == len(self.valid_set):
                            self.current_valid_index = 0
                    box = self.data_dict[name]['box']
                    weight = np.asarray(self.data_dict[name]['weights'])
                    train_weights[i] = weight
                    img = self.open_img(name)
                    # print("Image Opened")
                    padd, cbox = self._crop_data(img.shape[0], img.shape[1], boxp=0)
                    # print([padd, cbox])
                    new_j = self._relative_points(cbox, padd, points, to_size=64)
                    # print(new_j)
                    hm = self._generate_hm(64, 64, new_j, 64, weight)
                    img = self._crop_img(img, padd, cbox)
                    img = img.astype(np.uint8)
                    img = scm.imresize(img, (256, 256))

                    if sample_set == 'train':
                        img, hm = self._augment(img, hm)
                    hm = np.expand_dims(hm, axis=0)
                    hm = np.repeat(hm, stacks, axis=0)
                    if normalize:
                        train_img[i] = img.astype(np.float32) / 255
                    else:
                        train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = hm
                    i = i + 1
                except:
                    # print('error file: ', name)
                    return
            yield train_img, train_gtmap, train_weights

    def generator(self, batchSize=16, stacks=4, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self.aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted points
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self.create_train_table()
        self.create_sets()
        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            pad, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'],
                                        self.data_dict[self.train_set[i]]['points'], boxp=0.0)
            new_j = self._relative_points(box, pad, self.data_dict[self.train_set[i]]['points'], to_size=256)
            rhm = self._generate_hm(256, 256, new_j, 256, w)
            rimg = self._crop_img(img, pad, box)
            # See Error in self._generator
            # rimg = cv2.resize(rimg, (256,256))
            rimg = scm.imresize(rimg, (256, 256))
            # rhm = np.zeros((256,256,16))
            # for i in range(16):
            # rhm[:,:,i] = cv2.resize(rHM[:,:,i], (256,256))
            grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2))
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

    # ------------------------------- PCK METHODS-------------------------------
    def pck_ready(self, idlh=3, idrs=12, test_set=None):
        # Creates a list with all PCK ready samples
        # (PCK: Percentage of Correct Keypoints)
        id_lhip = idlh
        id_rsho = idrs
        self.total_points = 0
        self.pck_samples = []
        for s in self.data_dict.keys():
            if test_set is None:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_points += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    if s in test_set:
                        self.pck_samples.append(s)
                        wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                        self.total_points += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.points', self.total_points)

    def get_sample(self, sample=None):
        """ Returns information of a sample
        Args:
            sample : (str) Name of the sample
        Returns:
            img: RGB Image
            new_j: Resized points
            w: Weights of points
            joint_full: Raw points
            max_l: Maximum Size of Input Image
        """
        if sample is not None:
            try:
                points = self.data_dict[sample]['points']
                box = self.data_dict[sample]['box']
                w = self.data_dict[sample]['weights']
                img = self.open_img(sample)
                padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, points, boxp=0.2)
                new_j = self._relative_points(cbox, padd, points, to_size=256)
                joint_full = np.copy(points)
                max_l = max(cbox[2], cbox[3])
                joint_full = joint_full + [padd[1][0], padd[0][0]]
                joint_full = joint_full - [cbox[0] - max_l // 2, cbox[1] - max_l // 2]
                img = self._crop_img(img, padd, cbox)
                img = img.astype(np.uint8)
                img = scm.imresize(img, (256, 256))
                return img, new_j, w, joint_full, max_l
            except:
                return False
        else:
            print('Specify a sample name')
