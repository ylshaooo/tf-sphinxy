import argparse
import os

from config import Config
from preprocess.datagen import DataGenerator
from models.sphinx import SphinxModel
from scripts.test import Tester
from scripts.train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode',
        default='train',
        help='mode of task: train/test'
    )
    parser.add_argument(
        '-d', '--devices',
        default='-1',
        help='gpus to use: "0", "1", or "0,1"'
    )
    args = parser.parse_args()
    mode = args.mode
    assert mode in ['train', 'test'], 'invalid mode'

    devices = args.devices
    if devices != '-1':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = devices

    # load config
    cfg = Config()
    print('--Creating Dataset')
    dataset = DataGenerator(cfg)

    dataset.generate_set(train=True if mode == 'train' else False)
    model = SphinxModel(cfg)

    if mode == 'train':
        trainer = Trainer(model, cfg, dataset)
        trainer.training()
    else:
        tester = Tester(model, cfg, dataset)
        tester.inference()


if __name__ == '__main__':
    main()
