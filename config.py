class Config:
    # Dataset
    train_img_dir = 'data/train/Images'
    test_img_dir = 'data/test'
    train_data_file = 'data/train/dataset.txt'
    test_data_file = 'data/test/test.csv'
    test_output_file = 'result.csv'

    # Network
    name = 'sphinx'
    img_size = 256
    hm_size = 64
    num_classes = 5
    nFeats = 256
    nStacks = 8
    nLow = 4
    points_list = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                   'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                   'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                   'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                   'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    # Train
    batch_size = 8
    nEpochs = 30
    epoch_size = 4000
    learning_rate = 0.001
    learning_rate_decay = 0.93
    decay_step = 2000
    dropout_rate = 0.3

    # Validation
    valid_iter = 360

    # Saver
    logdir = 'logs'
    save_step = 100
    saver_dir = 'checkpoints'
    load = 'checkpoints/sphinx_28'
