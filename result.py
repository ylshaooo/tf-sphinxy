import csv

import numpy as np

from datagen import DataGenerator
from sphinx import SphinxModel
from utils import VALID_POINTS, read_config


def test_all(params, model, dataset):
    img_size = params['img_size']
    batch_size = params['batch_size']
    num_points = params['num_points']
    hm_size = params['hm_size']

    test_gen = dataset.test_generator(img_size, batch_size, True)

    with open(params['test_output_file'], 'w', newline='') as outfile:
        spam_writer = csv.writer(outfile)
        # Traversal the test set
        for images, categories, offsets, names in test_gen:
            hms = model.predict(images)

            # Formatting to lines
            for i in range(hms.shape[0]):
                hm = hms[i]
                offset = offsets[i]
                category = categories[i]
                name = names[i]

                write_line = [name, category]
                for i in range(num_points):
                    if VALID_POINTS[category][i] is 1:
                        # Calculate predictions from heat map
                        index = np.unravel_index(hm[:, :, i].argmax(), (hm_size, hm_size))
                        point = np.array(index) / hm_size * img_size
                        point -= offset
                        write_line.append(str(point[0]) + '_' + str(point[1]) + '_1')
                    else:
                        write_line.append('-1_-1_-1')
                spam_writer.writerow(write_line)


if __name__ == '__main__':
    print('--Parsing Config File')
    params = read_config('config.cfg')

    print('--Creating Dataset')
    dataset = DataGenerator(params['points_list'], params['test_img_directory'], test_data_file=params['test_txt_file'])

    model = SphinxModel(
        nFeats=params['nfeats'], nStacks=params['nstacks'], nLow=params['nlow'],
        out_dim=params['num_points'], img_size=params['img_size'], hm_size=params['hm_size'],
        points=params['point_list'], batch_size=params['batch_size'], num_classes=params['num_classes'],
        drop_rate=params['dropout_rate'], learning_rate=params['learning_rate'],
        decay=params['learning_rate_decay'], decay_step=params['decay_step'], training=True,
        dataset=dataset, num_validation=params['num_validation'], logdir_train=params['logdir_train'],
        logdir_test=params['logdir_test'], w_loss=params['weighted_loss'], name=params['name']
    )
    model.generate_model()
    model.inference_init(params['load'])

    test_all(params, model, dataset)
