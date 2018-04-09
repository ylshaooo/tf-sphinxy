import argparse
import os

from config import Config
from datagen import DataGenerator
from sphinx import SphinxModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode',
        default='train',
        help='mode of task: train/valid/test'
    )
    args = parser.parse_args()
    mode = args.mode
    assert mode in ['train', 'test'], 'invalid mode'

    # load config
    cfg = Config()
    print('--Creating Dataset')
    dataset = DataGenerator(cfg)

    if mode == 'train':
        dataset.generate_set()
        model = SphinxModel(cfg, dataset)
        model.generate_model()
        model.training()
    else:
        dataset.generate_set(train=False)
        model = SphinxModel(cfg, dataset)
        model.generate_model()
        model.inference()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
