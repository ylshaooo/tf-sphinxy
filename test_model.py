from datagen import DataGenerator
from sphinx import SphinxModel

if __name__ == '__main__':
    print('--Parsing Config File')
    params = ('config.cfg')

    print('--Creating Dataset')
    dataset = DataGenerator(params['points_list'], params['test_img_directory'], test_data_file=params['test_txt_file'])

    model = SphinxModel(
        nFeats=params['nfeats'], nStacks=params['nstacks'], nLow=params['nlow'],
        num_points=params['num_points'], img_size=params['img_size'], hm_size=params['hm_size'],
        points=params['point_list'], batch_size=params['batch_size'], num_classes=params['num_classes'],
        drop_rate=params['dropout_rate'], learning_rate=params['learning_rate'],
        decay=params['learning_rate_decay'], decay_step=params['decay_step'], training=True,
        dataset=dataset, logdir_train=params['logdir_train'], logdir_valid=params['logdir_valid'],
        test_txt_file=params['test_txt_file'], test_output_file=params['test_output_file'], name=params['name']
    )

    model.generate_model()
    model.inference_init(params['load'], True)
