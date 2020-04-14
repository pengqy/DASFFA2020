# -*- coding: utf-8 -*-
import numpy as np


class DefaultConfig:

    model = 'NRSG'  # prior gru double attention network
    dataset = 'Office_Products_data'

    norm_emb = False  # whether norm word embedding or not
    drop_out = 0.5

    # --------------optimizer---------------------#
    optimizer = 'Adam'
    weight_decay = 1e-3  # optimizer rameteri
    lr = 2e-3
    eps = 1e-8
    update_method = 'mse'

    # -------------main.py-----------------------#
    seed = 2019
    gpu_id = 1
    multi_gpu = False
    gpu_ids = []
    use_gpu = True   # user GPU or not
    num_epochs = 30  # the number of epochs for training
    num_workers = 0  # how many workers for loading data

    load_ckp = False
    ckp_path = ""
    fine_tune = True
    num_aspect = 5

    # ----------for confirmint the data -------------#
    use_word_embedding = True
    doc_len = 300

    #  ----------id_embedding------------------------#
    att_id_emb_size = 32
    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 100

    # ----------------self att ----------------------#
    self_att = True

    # --------------------CNN------------------------#
    doc_len = 300
    r_filters_num = 100
    kernel_size = 3
    attention_size = 32
    att_method = 'matrix'
    review_weight = 'softmax'

    # ----------------------GCN----------------------#
    gcn_hidden_num = 100

    # ---------------multi head----------------------#
    multi_size = 5
    word_att = True
    review_att = True
    user_att = 'use'
    item_att = 'use'
    gate = True
    # -----------------gru/cnn-----------------------#

    r_id_merge = 'add'
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    use_mask = False
    print_opt = 'def'

    fine_step = False
    use_word_drop = True

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train/npy'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

        self.edge_path = f'{prefix}/edge_index.npy'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.user_list = np.load(self.user_list_path, encoding='bytes')
        self.item_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_dict = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_dict = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print("{} => {}".format("model_name", self.model))
        print('*************************************************')


class Office_Products_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Office_Products_data')

    vocab_size = 42301
    word_dim = 300
    r_max_len = 248  # review max length

    train_data_size = 42611
    test_data_size = 5323

    user_num = 4905 + 2
    item_num = 2420 + 2

    u_max_r = 14
    i_max_r = 35

    user_mlp = [500, 80]
    item_mlp = [500, 80]

    batch_size = 128
    print_step = 500


class Gourmet_Food_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Gourmet_Food_data')

    vocab_size = 74572
    word_dim = 300
    r_max_len = 168  # review max length

    u_max_r = 15
    i_max_r = 22

    train_data_size = 121003
    test_data_size = 15125
    user_num = 14681 + 2
    item_num = 8713 + 2
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 64
    print_step = 1000


class Video_Games_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Video_Games_data')

    vocab_size = 169398
    word_dim = 300
    r_max_len = 394  # review max length

    train_data_size = 185439
    test_data_size = 23170
    user_num = 24303 + 2
    item_num = 10672 + 2
    u_max_r = 10
    i_max_r = 27
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 128
    print_step = 1000


class Toys_and_Games_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Toys_and_Games_data')

    vocab_size = 69213
    word_dim = 300

    r_max_len = 178  # review max length

    train_data_size = 134104
    test_data_size = 16755
    user_num = 19412 + 2
    item_num = 11924 + 2
    u_max_r = 9
    i_max_r = 18
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000


class Musical_Instruments_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Musical_Instruments_data')

    vocab_size = 17226
    word_dim = 300

    r_max_len = 153

    u_max_r = 8
    i_max_r = 13

    train_data_size = 8218
    test_data_size = 1021

    user_num = 1429 + 2
    item_num = 900 + 2

    batch_size = 32
    print_step = 11


class Digital_Music_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Digital_Music_data')

    vocab_size = 96287
    word_dim = 300

    r_max_len = 366

    u_max_r = 13
    i_max_r = 24

    train_data_size = 51764
    test_data_size = 6471

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 64


class yelp2016_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('yelp2016_data')

    vocab_size = 264045
    word_dim = 300

    r_max_len = 189

    u_max_r = 9
    i_max_r = 16

    train_data_size = 1024505
    test_data_size = 121317

    user_num = 164179 + 2
    item_num = 100125 + 2

    batch_size = 1024
    print_step = 200


class Tools_Improvement_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Tools_Improvement_data')

    vocab_size = 66612
    word_dim = 300

    r_max_len = 192

    u_max_r = 9
    i_max_r = 16

    train_data_size = 107595
    test_data_size = 13440

    user_num = 16638 + 2
    item_num = 10217 + 2

    batch_size = 128
    print_step = 100


class Automotive_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Automotive_data')

    vocab_size = 22757
    word_dim = 300

    r_max_len = 145

    u_max_r = 7
    i_max_r = 13

    train_data_size = 16383
    test_data_size = 2045

    user_num = 2928 + 2
    item_num = 1835 + 2

    batch_size = 32


class Kindle_Store_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Kindle_Store_data')

    vocab_size = 278914
    word_dim = 300

    r_max_len = 211  # review max length

    train_data_size = 786159
    test_data_size = 98230
    user_num = 68223 + 2
    item_num = 61934 + 2
    u_max_r = 20
    i_max_r = 24
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 4
    print_step = 1000


class Movies_and_TV_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Movies_and_TV_data')

    vocab_size = 764339
    word_dim = 300

    r_max_len = 326  # review max length

    train_data_size = 1358101
    test_data_size = 169716
    user_num = 123960
    item_num = 50052
    u_max_r = 16
    i_max_r = 49
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 16
    print_step = 5000


class Clothing_Shoes_and_Jewelry_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Movies_and_TV_data')

    vocab_size = 67812
    word_dim = 300
    r_max_len = 97  # review max length
    s_max_len = 31  # summary max length

    train_data_size = 222984
    test_data_size = 55693
    user_num = 39387
    item_num = 23033
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class Sports_and_Outdoors_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Sports_and_Outdoors_data')

    vocab_size = 100129
    word_dim = 300
    r_max_len = 146  # review max length
    s_max_len = 29  # summary max length

    train_data_size = 237095
    test_data_size = 59242
    user_num = 35598
    item_num = 18357
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class yelp2013_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('yelp2013_data')

    vocab_size = 65751
    word_dim = 300

    r_max_len = 271

    u_max_r = 57
    i_max_r = 59

    train_data_size = 63172
    test_data_size = 7897

    user_num = 1631 + 2
    item_num = 1633 + 2

    batch_size = 32
    print_step = 1000


class yelp2014_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('yelp2014_data')

    vocab_size = 113166
    word_dim = 300

    r_max_len = 284

    u_max_r = 58
    i_max_r = 68

    train_data_size = 184930
    test_data_size = 23116

    user_num = 4818 + 2
    item_num = 4194 + 2

    batch_size = 32
    print_step = 1000


