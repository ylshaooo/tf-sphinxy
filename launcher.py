import argparse
import os

from config import Config
from datagen import DataGenerator
from model import SphinxModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode',
        default='train',
        help='mode of task: train/test'
    )
    args = parser.parse_args()
    mode = args.mode
    assert mode in ['train', 'test'], 'invalid mode'

    # load config
    cfg = Config()
    print('--Creating Dataset')
    dataset = DataGenerator(cfg)

    dataset.generate_set(train=True if mode == 'train' else False)
    model = SphinxModel(cfg, dataset)
    if mode == 'train':
        model.generate_model(True)
        model.training()
    else:
        model.generate_model(False)
        model.inference()


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
