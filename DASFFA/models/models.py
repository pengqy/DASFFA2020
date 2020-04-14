
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .BasicModule import BasicModule
from .prediction import PredictionNet
import ipdb


class Model(BasicModule):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt

        self.model_name = self.opt.model

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

        if self.opt.ui_merge == 'cat':
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * 9
            else:
                feature_dim = self.opt.id_emb_size * 2
        else:
            if self.opt.r_id_merge == 'cat':
                feature_dim = self.opt.id_emb_size * 2
            else:
                feature_dim = self.opt.id_emb_size

        self.edge_index = torch.from_numpy(np.load(self.opt.edge_path)).cuda()
        self.opt.feature_dim = feature_dim
        self.predict_net = PredictionNet(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_docs, item_docs = datas
        user_all_feature, item_all_feature = self.user_net(user_reviews, user_docs, item_docs, self.opt.r_max_len, self.opt.u_max_r, uids, iids, user_item2id, self.opt.user_att, 'user')
        # item_all_feature = self.item_net(item_reviews, user_docs, item_docs, self.opt.r_max_len, self.opt.i_max_r, iids, uids, item_user2id, self.opt.item_att, 'item')

        # -------------- the method for merge the user feature and item feature --------------
        if self.opt.ui_merge == 'cat':
            ui_feature = torch.cat([user_all_feature, item_all_feature], 1)
        elif self.opt.ui_merge == 'add':
            ui_feature = user_all_feature + item_all_feature
        elif self.opt.ui_merge == 'dot':
            ui_feature = user_all_feature * item_all_feature

        # -------------- the method for output layer  --------------
        # ui_feature = F.relu(ui_feature)
        ui_feature = self.dropout(ui_feature)

        output = self.predict_net(ui_feature, uids, iids).squeeze(1)

        return output
