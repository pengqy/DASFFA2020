# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN
import ipdb


class Wide_Deep(nn.Module):
    '''

    '''
    def __init__(self, opt, uori='user'):
        super(Wide_Deep, self).__init__()
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

        self.review_linear = nn.Linear(opt.r_filters_num, opt.attention_size)
        self.id_linear = nn.Linear(opt.id_emb_size, opt.attention_size, bias=False)
        self.attention_linear = nn.Linear(opt.attention_size, 1)

        self.fc_layer4 = nn.Linear(opt.r_filters_num, opt.attention_size)

        self.fc_layer2 = nn.Linear(opt.attention_size, opt.r_filters_num)

        self.wide_layer = nn.Linear(2 * opt.attention_size, opt.attention_size)
        self.deep_layer = nn.Sequential(nn.Linear(2 * opt.attention_size, opt.r_filters_num),
                                        nn.ReLU(),
                                        nn.Linear(opt.r_filters_num, opt.attention_size),
                                        nn.ReLU())

        self.r_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)
        self.d_encoder = CNN(opt.r_filters_num, opt.kernel_size, opt.word_dim)

        self.review_trans = nn.TransformerEncoderLayer(d_model=self.opt.r_filters_num, nhead=2)
        self.doc_trans = nn.TransformerEncoderLayer(d_model=self.opt.r_filters_num, nhead=2)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_model_weight()

    def forward(self, review, doc, review_len, max_num, id1, id2, ui2id, uori):
        # --------------- word embedding ----------------------------------
        doc = self.word_embs(doc)
        d_fea = self.d_encoder(doc, 1, self.opt.doc_len).squeeze() # doc feature

        if self.opt.self_att == True:
            d_fea = self.doc_trans(d_fea.unsqueeze(1)).squeeze(1) # doc self_att feature

        review = self.word_embs(review)  # size * 300

        id_emb = self.id_embedding(id1)
        u_i_id_emb = self.u_i_id_embedding(ui2id)

        # --------cnn for review--------------------
        r_fea = self.r_encoder(review, max_num, review_len)
        r_Transfea = self.review_trans(r_fea)

        # ------------------linear attention-------------------------------
        rs_mix = F.relu(self.review_linear(r_Transfea) + self.id_linear(F.relu(u_i_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_Transfea = r_Transfea * att_weight
        r_Transfea = r_Transfea.sum(1)

        # gating_feature = 1 - torch.sigmoid(id_emb * self.fc_layer4(d_fea))
        # gating_feature = id_emb * self.fc_layer4(d_fea)

        d_fea = self.fc_layer4(d_fea)
        mix_feature = torch.cat((id_emb, d_fea), 1)
        gating_feature = self.wide_layer(mix_feature) + self.deep_layer(mix_feature)

        all_feature = self.dropout(gating_feature) + id_emb + d_fea

        return all_feature

    def init_model_weight(self):
        nn.init.xavier_normal_(self.r_encoder.cnn.weight)
        nn.init.constant_(self.r_encoder.cnn.bias, 0.1)
        nn.init.xavier_normal_(self.d_encoder.cnn.weight)
        nn.init.constant_(self.d_encoder.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer2.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer2.bias, 0.1)

        nn.init.uniform_(self.fc_layer4.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer4.bias, 0.1)

        nn.init.uniform_(self.deep_layer[0].weight, -0.1, 0.1)
        nn.init.constant_(self.deep_layer[0].bias, 0.1)

        nn.init.uniform_(self.deep_layer[2].weight, -0.1, 0.1)
        nn.init.constant_(self.deep_layer[2].bias, 0.1)

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
