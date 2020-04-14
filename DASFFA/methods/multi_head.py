# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN
import ipdb


class Multi_Head(nn.Module):
    '''

    '''
    def __init__(self, opt, uori='user'):
        super(Multi_Head, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, opt.id_emb_size)

        self.multi_W_attention = nn.Parameter(torch.randn(self.opt.multi_size, self.opt.r_filters_num))
        self.multi_R_attention = nn.Parameter(torch.randn(self.opt.multi_size, self.opt.r_filters_num))

        self.r_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)
        self.d_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)

        self.word_layer = nn.Linear(self.opt.multi_size * self.opt.r_filters_num, self.opt.r_filters_num)
        self.review_layer1 = nn.Linear(self.opt.multi_size * self.opt.r_filters_num, self.opt.attention_size)
        self.review_layer2 = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_model_weight()

    def forward(self, review, doc, review_len, max_num, id1, id2, ui2id, ui_att, uori):
        # --------------- word embedding and CNN ---------------------------------
        # doc = self.word_embs(doc)
        # d_fea = self.d_encoder(doc, 1, self.opt.doc_len).squeeze()
        word_att = True
        review_att = True
        if ui_att != 'use':
            word_att = False
            review_att = False

        review = self.word_embs(review)  # size * 300
        if word_att == True:
            r_fea = self.r_encoder(review, max_num, review_len, "multi_att", self.multi_W_attention)
            r_fea = self.word_layer(self.dropout(r_fea))
        else:
            r_fea = self.r_encoder(review, max_num, review_len, "MAX")
        id_emb = self.id_embedding(id1)
        if review_att == True:
            att_weight = torch.matmul(r_fea, self.multi_R_attention.t())
            att_score = F.softmax(att_weight, dim=1)
            r_fea = torch.bmm(r_fea.permute(0, 2, 1), att_score)
            feature = r_fea.view(-1, self.opt.multi_size * self.opt.r_filters_num)
            feature = self.review_layer1(feature)
        else:
            r_fea = torch.mean(r_fea, 1)
            feature = self.review_layer2(r_fea)

        # ----------------revie attention ----------------------------------------
        all_feature = feature + id_emb

        return all_feature

    def init_model_weight(self):
        nn.init.xavier_normal_(self.r_encoder.cnn.weight)
        nn.init.constant_(self.r_encoder.cnn.bias, 0.1)

        nn.init.xavier_normal_(self.d_encoder.cnn.weight)
        nn.init.constant_(self.d_encoder.cnn.bias, 0.1)

        nn.init.uniform_(self.word_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.word_layer.bias, 0.1)
        nn.init.uniform_(self.review_layer1.weight, -0.1, 0.1)
        nn.init.constant_(self.review_layer1.bias, 0.1)
        nn.init.uniform_(self.review_layer2.weight, -0.1, 0.1)
        nn.init.constant_(self.review_layer2.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-1., b=1.)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-1., b=1.)
