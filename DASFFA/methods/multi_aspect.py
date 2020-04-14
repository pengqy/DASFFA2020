# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN
import ipdb


class Multi_Aspect(nn.Module):
    '''
    Multi_Aspect recommend system
    '''
    def __init__(self, opt, uori='user'):
        super(Multi_Aspect, self).__init__()
        self.opt = opt

        user_num = self.opt.user_num
        item_num = self.opt.item_num

        id_emb_size = self.opt.id_emb_size
        att_id_emb_size = self.opt.att_id_emb_size

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)

        # -------------------attention ID embedding-------------------------
        self.user_id_emb = nn.Embedding(user_num, att_id_emb_size)  # dim is 100
        self.item_id_emb = nn.Embedding(item_num, att_id_emb_size)

        # ---------------------user/item ID embedding-----------------------
        self.uid_emb = nn.Embedding(user_num, id_emb_size)  # dim is 32
        self.iid_emb = nn.Embedding(item_num, id_emb_size)

        # ------------------word level multi aspect-------------------------
        self.w_aspectLinears = nn.ModuleList([
            nn.Linear(att_id_emb_size, att_id_emb_size) for l in range(self.opt.num_aspect)
        ])

        # --------------------review change --------------------------------
        self.review_mlp = nn.Linear(att_id_emb_size * self.opt.num_aspect, att_id_emb_size)
        self.ui_mlp = nn.Linear(att_id_emb_size, id_emb_size)

        # -----------------review level multi aspect------------------------
        self.r_aspectLinears = nn.ModuleList([
            nn.Linear(att_id_emb_size, att_id_emb_size) for l in range(self.opt.num_aspect)
        ])

        self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)

        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_word_emb()
        self.init_model_weight()

    def forward(self, review, review_len, max_num, id1, id2, ui2id, uori):
        # ------------------- word/id embedding -----------------------------
        review = self.word_embs(review)

        if self.opt.use_word_drop:
            review = self.dropout(review)

        if uori == 'user':
            attention_id_emb = self.item_id_emb(id2)
            uiIDemb = self.uid_emb(id1)
            ui2id_emb = self.item_id_emb(ui2id)
        elif uori == 'item':
            attention_id_emb = self.user_id_emb(id2)
            uiIDemb = self.iid_emb(id1)
            ui2id_emb = self.user_id_emb(ui2id)

        # ----------------user/item word/review aspect Linear---------------
        w_aspect = [l(attention_id_emb) for l in (self.w_aspectLinears)]
        word_aspect = torch.stack((w_aspect), 1)
        word_aspect = self.dropout(word_aspect)

        r_aspect = [l(attention_id_emb) for l in (self.r_aspectLinears)]
        review_aspect = torch.stack((r_aspect), 1)
        review_aspect = self.dropout(review_aspect)

        # ------------------- cnn for review -----------------------------
        r_fea = self.r_encoder(review, max_num, review_len, pooling='multi_att', qv=word_aspect)
        r_fea = self.review_mlp(r_fea) # (32, 11, 100)
        r_fea = self.dropout(r_fea) + ui2id_emb

        # --------------------review multi head attention-----------------
        att_weight = torch.bmm(r_fea, review_aspect.permute(0, 2, 1)) # 32 * 11 * 5
        att_score = F.softmax(att_weight, dim=2)
        r_fea = torch.bmm(r_fea.permute(0, 2, 1), att_score) # 32 * 100 * 5

        # ------------------user/item multi aspect attention--------------
        ui_fea = self.ui_mlp(r_fea.permute(0, 2, 1)) # 32 * 5 * 32
        ui_fea = self.dropout(ui_fea)

        att_weight = torch.bmm(ui_fea, uiIDemb.unsqueeze(2))

        att_score = F.softmax(att_weight, dim=1)
        ui_fea = torch.matmul(att_score.permute(0, 2, 1), ui_fea) # 32 * 1 * 32

        all_feature = torch.cat((ui_fea.squeeze(1), uiIDemb), 1)

        return all_feature

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        nn.init.xavier_normal_(self.user_id_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

        nn.init.xavier_normal_(self.uid_emb.weight)
        nn.init.xavier_normal_(self.iid_emb.weight)

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.r_encoder.cnn.weight)
        nn.init.uniform_(self.r_encoder.cnn.bias, a=-0.1, b=0.1)

        nn.init.xavier_uniform_(self.review_mlp.weight)
        nn.init.constant_(self.review_mlp.bias, 0.1)

        for i in self.w_aspectLinears:
            nn.init.xavier_uniform_(i.weight)
            nn.init.constant_(i.bias, 0.1)

        for j in self.r_aspectLinears:
            nn.init.xavier_uniform_(j.weight)
            nn.init.constant_(j.bias, 0.1)
