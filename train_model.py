from config import Config
from datagen import DataGenerator
from sphinx import SphinxModel

if __name__ == '__main__':
    cfg = Config()

    print('--Creating Dataset')
    dataset = DataGenerator(cfg.points_list, cfg.img_dir, cfg.training_txt_file)
    dataset.generate_set()

    model = SphinxModel(cfg, dataset, True)
    model.generate_model()
    model.training_init()
