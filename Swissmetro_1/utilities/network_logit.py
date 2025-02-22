# -*- coding: UTF-8 -*-
"""
@filename:network_logit.py
@author:Chen Kunxu
@time:2023-07-18
"""

import torch.nn as nn
import torch
from utilities.TransformerLearner import Different_layer, ASU, Layer_M
import torch.nn
from utilities.coderLayer import EncoderLayer


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)


class MNL(nn.Module):
    def __init__(self, beta_num, choices_num):
        super(MNL, self).__init__()
        self.choices_num = choices_num
        self.main_model = nn.Conv2d(1, 1, kernel_size=(beta_num, 1), stride=(1, 1), bias=False)
        self.apply(_init_vit_weights)

    def forward(self, x, q):
        x = self.main_model(x)  # [choice, 1, 1]
        x = x.reshape(-1, self.choices_num)  # [choice]
        return x


class L_MNL(nn.Module):
    def __init__(self, beta_num, nExtraFeatures, choices_num, networkSize, hidden_layers=0):
        super(L_MNL, self).__init__()
        self.choices_num = choices_num
        self.main_model = nn.Conv2d(1, 1, kernel_size=(beta_num, 1), stride=(1, 1), bias=False)
        self.extra_model = learning_term(nExtraFeatures, choices_num, networkSize, hidden_layers)
        self.apply(_init_vit_weights)

    def forward(self, x, q):
        x = self.main_model(x)  # [choice, 1, 1]
        q = self.extra_model(q)  # [networkSize, 1, 1]
        x = x.reshape(-1, self.choices_num)  # [choice]
        final = torch.add(x, q)  # [choice]
        return final


class learning_term(nn.Module):
    def __init__(self, nExtraFeatures, choices_num, networkSize, hidden_layers=0):
        super(learning_term, self).__init__()
        self.networkSize = networkSize
        self.conv1 = nn.Conv2d(1, networkSize, kernel_size=(nExtraFeatures, 1))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # 非线性部分
        self.hidden_lay = hidden_layers - 1
        if self.hidden_lay:
            extra_model_full = []
            for i in range(self.hidden_lay):  # 如果隐藏层大于2
                extra_model_full += [nn.Linear(self.networkSize, self.networkSize),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3)]  # 全连接
            self.extra_model_full = nn.Sequential(*extra_model_full)
        self.Linear = nn.Linear(self.networkSize, choices_num)  # 收回 choices_num

    def forward(self, extra_input):
        extra_input = extra_input.reshape(-1, 1, extra_input.shape[1], 1)
        extra_input = self.act(self.conv1(extra_input))
        extra_input = self.dropout(extra_input)
        extra_input = extra_input.reshape(-1, self.networkSize)  # [networkSize]
        if self.hidden_lay:
            extra_input = self.extra_model_full(extra_input)
        extra_input = self.Linear(extra_input)  # [choice]
        return extra_input


class E_MNL(nn.Module):
    def __init__(self, beta_num, nExtraFeatures, choices_num, unique_cats_num):
        super(E_MNL, self).__init__()
        self.main_model = nn.Conv2d(1, 1, kernel_size=(beta_num, 1), stride=(1, 1), bias=False)
        self.extra_model = Embed_New(nExtraFeatures, choices_num, unique_cats_num)
        self.apply(_init_vit_weights)
        with torch.no_grad():
            self.extra_model.conv1.weight.data.clamp_(min=1e-9)  #

    def forward(self, x, q):
        x = self.main_model(x)
        q = self.extra_model(q)
        logits = torch.add(x, q)
        logits = logits.reshape(-1, logits.shape[3])
        return logits


class Embed_New(nn.Module):
    def __init__(self, nExtraFeatures, choices_num, unique_cats_num, dropout=0.3):
        super(Embed_New, self).__init__()
        self.Embedding = nn.Embedding(num_embeddings=unique_cats_num, embedding_dim=choices_num)
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(nExtraFeatures, 1), stride=(1, 1), bias=False)

    def forward(self, Q_train):
        Q_train = self.Embedding(Q_train)
        Q_train = self.dropout(Q_train)
        Q_train = Q_train.reshape(-1, 1, Q_train.shape[1], Q_train.shape[2])
        Q_train = self.conv1(Q_train)
        return Q_train


class T_MNL(nn.Module):
    def __init__(self, beta_num, nExtraFeatures, choices_num, unique_cats_num):
        super(T_MNL, self).__init__()
        self.choice = choices_num
        self.main_model = nn.Conv2d(1, 1, kernel_size=(beta_num, 1), stride=(1, 1), bias=False)
        self.transformer = Transformer(n_src_vocab=unique_cats_num, choice=choices_num, n_head=choices_num,
                                       nExtraFeatures=nExtraFeatures, d_model=choices_num)
        self.apply(_init_vit_weights)

    def forward(self, x, q):
        x = self.main_model(x)
        q = self.transformer(q)
        x = x.reshape(-1, x.shape[3])
        logits = torch.add(x, q)
        return logits


class Transformer(nn.Module):
    def __init__(self, n_src_vocab, choice, nExtraFeatures, d_model=3, n_layers=1, n_head=3, d_k=4, dropout=0.3):
        super(Transformer, self).__init__()
        self.src_learning_emb = nn.Embedding(n_src_vocab, d_model)
        self.drop = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(nExtraFeatures, eps=1e-6)
        self.Block = nn.ModuleList([
            EncoderLayer(nExtraFeatures, nExtraFeatures * 4, n_head, d_k, dropout=0.1)
            for _ in range(n_layers)])
        self.linear = nn.Sequential(
            nn.Linear(d_model * nExtraFeatures, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, choice),
        )

    def forward(self, x):
        # -- Forward
        x = self.src_learning_emb(x)
        x = self.drop(x)
        x = x.reshape(-1, x.shape[2], x.shape[1])
        x = self.layer_norm(x)
        for enc_layer in self.Block:
            x = enc_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


# #
class ET_MNL(nn.Module):
    def __init__(self, beta_num, nExtraFeatures, choices_num, unique_cats_num):
        super(ET_MNL, self).__init__()
        self.main_model = nn.Conv2d(1, 1, kernel_size=(beta_num, 1), stride=(1, 1), bias=False)
        self.extra_model = Embed_Transformer(n_src_vocab=unique_cats_num, choice=choices_num, n_head=choices_num,
                                             nExtraFeatures=nExtraFeatures, d_model=choices_num)
        self.apply(_init_vit_weights)
        with torch.no_grad():
            self.extra_model.conv1.weight.data.clamp_(min=1e-6)

    def forward(self, X_train, Q_train):
        out_X = self.main_model(X_train)
        out_QE, out_QT = self.extra_model(Q_train)
        logits = torch.add(out_X, out_QE)
        logits = logits.reshape(-1, logits.shape[3])
        logits = torch.add(logits, out_QT)
        return logits


class Embed_Transformer(nn.Module):
    def __init__(self, n_src_vocab, choice, nExtraFeatures, d_model=3, n_layers=1, n_head=3, d_k=4, dropout=0.3):
        super(Embed_Transformer, self).__init__()
        self.choices_num = choice

        self.src_learning_emb = nn.Embedding(n_src_vocab, choice + d_model)
        self.drop = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(nExtraFeatures, 1), bias=False)

        self.layer_norm = nn.LayerNorm(nExtraFeatures, eps=1e-6)
        self.Block = nn.ModuleList([
            EncoderLayer(nExtraFeatures, nExtraFeatures * 4, n_head, d_k, dropout=0.1)
            for _ in range(n_layers)])
        self.linear = nn.Sequential(
            nn.Linear(d_model * nExtraFeatures, 20),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(20, choice),
        )

    def forward(self, x):
        # -- Forward
        x = self.src_learning_emb(x)

        out_Q1 = x[:, :, :self.choices_num]
        out_Q2 = x[:, :, self.choices_num:]

        out_Q1 = self.drop(out_Q1)
        out_Q1 = out_Q1.reshape(-1, 1, out_Q1.shape[1], out_Q1.shape[2])
        out_Q1 = self.conv1(out_Q1)

        out_Q2 = self.drop(out_Q2)
        out_Q2 = out_Q2.reshape(-1, out_Q2.shape[2], out_Q2.shape[1])
        out_Q2 = self.layer_norm(out_Q2)
        for enc_layer in self.Block:
            out_Q2 = enc_layer(out_Q2)
        out_Q2 = out_Q2.reshape(out_Q2.shape[0], -1)
        out_Q2 = self.linear(out_Q2)

        return out_Q1, out_Q2


class ASU_DNN(nn.Module):
    def __init__(self, beta_num, nExtraFeatures, choices_num):
        super(ASU_DNN, self).__init__()
        self.cont_vars_num = beta_num  # 5
        self.emb_vars_num = nExtraFeatures  # 12
        self.choices_num = choices_num  # 3
        n_layer1 = 1
        n_layer2 = 0
        layer_nx = 60
        layer_nz = 60
        layer_n2 = 40
        dropout = 0.3

        self.x1_M = ASU(beta_num, layer_nx,
                        layer_nz, layer_n2, n_layer1, n_layer2, dropout=dropout)
        self.x2_M = ASU(beta_num, layer_nx,
                        layer_nz, layer_n2, n_layer1, n_layer2, dropout=dropout)
        self.x3_M = ASU(beta_num, layer_nx,
                        layer_nz, layer_n2, n_layer1, n_layer2, dropout=dropout)
        self.layer_z1 = Different_layer(self.emb_vars_num, layer_nz, dropout_rate=dropout)
        self.Layer_M1_z = Layer_M(layer_nz, n_layer1, dropout=dropout)
        self.apply(_init_vit_weights)

    def forward(self, x, q):
        x = x.reshape(-1, self.cont_vars_num, self.choices_num)
        q = q.reshape(-1, self.emb_vars_num)
        q = self.layer_z1(q)
        q = self.Layer_M1_z(q)
        output_1 = self.x1_M(x[:, :, 0], q)
        output_2 = self.x2_M(x[:, :, 1], q)
        output_3 = self.x3_M(x[:, :, 2], q)
        output = torch.cat([output_1, output_2, output_3], axis=1)
        return output
