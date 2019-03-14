class Config:
    # Dataset
    category = 'outwear'
    train_img_dir = 'data/train/' + category + '/Images'
    test_img_dir = 'data/test_a/' + category + '/Images'
    train_data_file = 'data/train/' + category + '/points.txt'
    test_data_file = 'data/test_a/' + category + '/bbox.txt'
    test_output_file = 'results/result_' + category + '.csv'

    # Network
    name = category
    img_size = 256
    hm_size = 64

    out_rate = 8
    nFeats = 256
    nStacks = 8
    nLow = 4

    top_points = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                  'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                  'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                  'hemline_left', 'hemline_right']

    bottom_points = ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                     'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    points_list = top_points + bottom_points

    # Train
    batch_size = 8
    nEpochs = 10
    epoch_size = 1250
    learning_rate = 0.003
    learning_rate_decay = 0.96
    decay_step = 600
    dropout_rate = 0.3

    # Validation
    valid_iter = 200

    # Saver
    logdir = 'logs/' + category
    save_step = 100
    saver_dir = 'checkpoints/' + category
    load = category + '_10'
