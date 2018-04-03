from datagen import DataGenerator
from sphinx import SphinxModel
from utils import process_config

if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('config.cfg')

    print('--Creating Dataset')
    dataset = DataGenerator(params['points_list'], params['num_validation'],
                            params['img_directory'], params['training_txt_file'])
    dataset.generate_set(rand=True)

    model = SphinxModel(
        nFeats=params['nfeats'], nStacks=params['nstacks'], nLow=params['nlow'],
        out_dim=params['num_points'], img_size=params['img_size'], hm_size=params['hm_size'],
        points=params['points_list'], batch_size=params['batch_size'], num_classes=params['num_classes'],
        drop_rate=params['dropout_rate'], learning_rate=params['learning_rate'],
        decay=params['learning_rate_decay'], decay_step=params['decay_step'], training=True,
        dataset=dataset, num_validation=params['num_validation'],logdir_train=params['logdir_train'],
        logdir_test=params['logdir_test'], w_loss=params['weighted_loss'], name=params['name']
    )
    model.generate_model()
    model.training_init(nEpochs=params['nEpochs'], epoch_size=params['epoch_size'],
                        save_step=params['saver_step'], load=params['load'])
