# -*- coding: utf-8 -*-
"""
@filename:SubLayer.py
@author:Chen Kunxu
@Time:2023/8/5 16:38
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, dropout=0.1, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        inner_dim = n_head * d_k
        self.scale = d_k ** -0.5
        # self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=False)

        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_k)

        self.attention = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, d_model)
        self.to_out_drop = nn.Dropout(dropout)

    def forward(self, x):
        d_k, d_v, n_head = self.d_k, self.d_k, self.n_head
        q, k, v = x, x, x
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        q = self.dropout(F.softmax(q, dim=-1))
        out = torch.matmul(q, v)

        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.to_out(out)
        out = self.to_out_drop(out)

        return out


class MLP(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(MLP, self).__init__()
        self.in_l = nn.Linear(d_in, d_hid)
        self.out_l = nn.Linear(d_hid, d_in)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.in_l(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.out_l(x)
        x = self.drop(x)
        return x
