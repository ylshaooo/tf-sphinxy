class Config:
    # Dataset
    category = 'trousers'
    train_img_dir = 'data/train/trousers/Images'
    test_img_dir = 'data/test/trousers/Images'
    train_data_file = 'data/train/trousers/points.txt'
    test_data_file = 'data/test/test_bottom.csv'
    test_output_file = 'result_bottom.csv'

    # Network
    name = 'trousers'
    img_size = 256
    hm_size = 64
    out_rate = 4
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
    nEpochs = 1
    epoch_size = 1250
    learning_rate = 0.01
    learning_rate_decay = 0.96
    decay_step = 600
    dropout_rate = 0.3

    # Validation
    valid_iter = 200

    # Saver
    logdir = 'logs/trousers'
    save_step = 100
    saver_dir = 'checkpoints/trousers'
    load = None
