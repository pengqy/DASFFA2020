# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN
import ipdb


class NRSG(nn.Module):
    '''
    reproduce the Deepwalk
    '''
    def __init__(self, opt, uori='user'):
        super(NRSG, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        # self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32

        self.uid_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)
        self.iid_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)

        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, opt.id_emb_size)

        self.review_linear = nn.Linear(opt.r_filters_num, opt.attention_size)
        self.id_linear = nn.Linear(opt.id_emb_size, opt.attention_size, bias=False)
        self.attention_linear = nn.Linear(opt.attention_size, 1)

        self.fc_layer_u = nn.Linear(opt.r_filters_num, opt.attention_size)
        self.fc_layer_i = nn.Linear(opt.r_filters_num, opt.attention_size)
        self.fc_layer_1 = nn.Linear(opt.r_filters_num, opt.attention_size)
        self.fc_layer_2 = nn.Linear(opt.r_filters_num, opt.attention_size)

        self.user_d_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)
        self.item_d_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)

        self.doc_trans = nn.TransformerEncoderLayer(d_model=self.opt.r_filters_num, nhead=1)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_model_weight()

    def forward(self, review, user_doc, item_doc, review_len, max_num, id1, id2, ui2id, ui_att, uori):
        # --------------- word embedding ----------------------------------
        u_doc = self.word_embs(user_doc)
        i_doc = self.word_embs(item_doc)

        u_d_fea = (self.user_d_encoder(u_doc, 1, self.opt.doc_len).squeeze()).permute(2, 0, 1)
        i_d_fea = (self.item_d_encoder(i_doc, 1, self.opt.doc_len).squeeze()).permute(2, 0, 1)

        mutual_fea = torch.cat([u_d_fea, i_d_fea], 0)
        if self.opt.self_att == True:
            mutual_fea = (self.doc_trans(mutual_fea))

        u_fea, i_fea = torch.split(mutual_fea, self.opt.doc_len, 0)

        u_fea = torch.mean(u_fea, 0)
        i_fea = torch.mean(i_fea, 0)

        uid_emb = self.uid_embedding(id1)
        iid_emb = self.iid_embedding(id2)

        u_gating_feature = 1 - torch.sigmoid(uid_emb * self.fc_layer_u(u_fea))
        i_gating_feature = 1 - torch.sigmoid(iid_emb * self.fc_layer_i(i_fea))

        user_feature = u_gating_feature + uid_emb + self.fc_layer_1(u_fea)
        item_feature = i_gating_feature + iid_emb + self.fc_layer_2(i_fea)

        return user_feature, item_feature

    def init_model_weight(self):
        nn.init.xavier_normal_(self.user_d_encoder.cnn.weight)
        nn.init.constant_(self.user_d_encoder.cnn.bias, 0.1)
        nn.init.xavier_normal_(self.item_d_encoder.cnn.weight)
        nn.init.constant_(self.item_d_encoder.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer_u.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer_u.bias, 0.1)

        nn.init.uniform_(self.fc_layer_i.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer_i.bias, 0.1)

        nn.init.uniform_(self.fc_layer_1.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer_1.bias, 0.1)

        nn.init.uniform_(self.fc_layer_2.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer_2.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.uid_embedding.weight, a=-1., b=1.)
        nn.init.uniform_(self.iid_embedding.weight, a=-1., b=1.)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-1., b=1.)
