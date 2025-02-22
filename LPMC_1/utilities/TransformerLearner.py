# -*- coding: utf-8 -*-
"""
@filename:TransformerLearner.py
@author:Chen Kunxu
@Time:2023/8/5 16:29
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.coderLayer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_model, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_model * 4, n_head, d_k, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_output):
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output


class Layer_M(nn.Module):
    def __init__(self, layer_n, n_layer1, dropout=0.1):
        super(Layer_M, self).__init__()
        self.Layer_M_model = nn.ModuleList(
            [
                Different_layer(layer_n, layer_n, dropout_rate=dropout)
                for _ in range(n_layer1)
            ]
        )

    def forward(self, input):
        for Layer_model in self.Layer_M_model:
            input = Layer_model(input)
        return input


class ASU(nn.Module):
    def __init__(
        self,
        cont_vars_num,
        layer_nx,
        layer_nz,
        layer_n2,
        n_layer1=0,
        n_layer2=0,
        dropout=0.1,
    ):
        super(ASU, self).__init__()
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.layer_x1 = Different_layer(cont_vars_num, layer_nx, dropout_rate=dropout)

        if self.n_layer1 > 0:
            self.Layer_M1_x = Layer_M(layer_nx, self.n_layer1, dropout=dropout)

        self.Layer_M2_begin = Different_layer(
            layer_nx + layer_nz, layer_n2, dropout_rate=dropout
        )

        if self.n_layer2 > 0:
            self.Layer_M2 = Layer_M(layer_n2, self.n_layer2, dropout=dropout)

        self.final_model = nn.Linear(layer_n2, 1)

    def forward(self, input_x, input_z):
        input_x = self.layer_x1(input_x)
        if self.n_layer1 > 0:
            input_x = self.Layer_M1_x(input_x)
        output = torch.cat((input_x, input_z), dim=1)
        output = self.Layer_M2_begin(output)
        if self.n_layer2 > 0:
            output = self.Layer_M2(output)
        output = self.final_model(output)
        return output


class Different_layer(nn.Module):
    def __init__(self, layer_n_in, layer_n_out, dropout_rate=0.2):
        super(Different_layer, self).__init__()
        self.linear = nn.Linear(in_features=layer_n_in, out_features=layer_n_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input):
        input = self.linear(input)
        input = self.act(input)
        output = self.dropout(input)
        return output
