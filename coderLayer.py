# -*- coding: utf-8 -*-
"""
@filename:coderLayer.py
@author:Chen Kunxu
@Time:2023/8/5 16:35
"""
import torch
import torch.nn as nn
from utilities.SubLayer import MultiHeadAttention, MLP


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pos_ffn = MLP(d_model, d_inner, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        x = self.norm1(x+self.slf_attn(x))
        x = self.norm2(x+self.pos_ffn(x))
        return x
