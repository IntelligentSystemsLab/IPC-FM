# -*- coding: UTF-8 -*-
"""
@filename:client.py
@author:Chen Kunxu
@Time:2023/8/5
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional
from copy import deepcopy
from torch.utils.data import DataLoader
from utilities.data_manager import myData

class Client(nn.Module):
    """
    放置client的通用函数
    """

    def __init__(self, id_c, device, model):
        super(Client, self).__init__()
        # client 设置
        self.id_c = id_c
        self.device = device
        self.net = deepcopy(model)  # load model
        self.net = self.net.to(self.device)
        self.init_net = deepcopy(self.net)
        self.init_net = self.init_net.to(self.device)
        self.epoch = 0

    def refresh(self, model):
        # 更新全局模型
        with torch.no_grad():
            for w, w_t in zip(self.net.parameters(), model.parameters()):
                w.data.copy_(w_t.data)


class MetaClient(Client):
    def __init__(self, device, id_c, model,
                 train_epoch, test_epoch, batch_size, inner_lr, outer_lr,
                 data_main, data_extra,
                 spt_prop=0, model_name='L_MNL', mode="_fomaml"):
        super(MetaClient, self).__init__(id_c, device, model)
        self.train_mode = mode
        self.train_epoch = train_epoch
        self.test_epoch = test_epoch
        self.model_name = model_name
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.batch_size = batch_size
        if self.train_mode == "_central":
            self.inner_optim = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr)
        elif self.train_mode == "_fomaml":
            self.inner_optim = torch.optim.SGD(self.init_net.parameters(), lr=self.inner_lr)
            self.outer_optim = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr,momentum=0.9)
        else:
            self.inner_optim = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr)
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        labels = data_main[:, -1, :]
        data_main = np.delete(data_main, -1, axis=1)  # (808, 9, 3)
        data_main = np.expand_dims(data_main, axis=1)

        menu = len(data_main)
        fo_spt = int(menu * spt_prop)
        fo_qry = menu - fo_spt
        dataloader_kwargs = {
            'pin_memory': True if device.type == 'cuda' else False,
            'num_workers': 0,
            'drop_last': False,
        }
        
        if self.train_mode == "_central":
            spt_main = myData(data_main, labels)
            self.spt_main_loader = DataLoader(spt_main, batch_size=self.batch_size, **dataloader_kwargs)
            spt_extra = myData(data_extra, labels)
            self.spt_extra_loader = DataLoader(spt_extra, batch_size=self.batch_size, **dataloader_kwargs)
        else:
            spt_main = myData(data_main, labels)
            self.spt_main_loader = DataLoader(spt_main, batch_size=menu, **dataloader_kwargs)
            spt_extra = myData(data_extra, labels)
            self.spt_extra_loader = DataLoader(spt_extra, batch_size=menu, **dataloader_kwargs)

        spt_main_fo = myData(data_main[:fo_spt], labels[:fo_spt])
        self.spt_main_loader_fo = DataLoader(spt_main_fo, batch_size=fo_spt, **dataloader_kwargs)
        spt_extra_fo = myData(data_extra[:fo_spt], labels[:fo_spt])
        self.spt_extra_loader_fo = DataLoader(spt_extra_fo, batch_size=fo_spt, **dataloader_kwargs)

        qry_main_fo = myData(data_main[fo_spt:], labels[fo_spt:])
        self.qry_main_loader_fo = DataLoader(qry_main_fo, batch_size=fo_qry, **dataloader_kwargs)
        qry_extra_fo = myData(data_extra[fo_spt:], labels[fo_spt:])
        self.qry_extra_loader_fo = DataLoader(qry_extra_fo, batch_size=fo_qry, **dataloader_kwargs)

    def init_model(self):
        with torch.no_grad():
            for w, w_t in zip(self.init_net.parameters(), self.net.parameters()):
                w.data.copy_(w_t.data)

    def local_fomaml_train(self):
        self.init_model()
        self.init_net.train()
        for _ in range(self.train_epoch):
            for main_spt, extra_spt in zip(self.spt_main_loader_fo, self.spt_extra_loader_fo):
                spt_x_main = main_spt[0].float().to(self.device)
                spt_x_extra = extra_spt[0].long().to(self.device)
                spt_y = main_spt[1].to(self.device)

                self.inner_optim.zero_grad()
                output_s = self.init_net(spt_x_main, spt_x_extra)
                label_s = torch.argmax(spt_y, 1)
                loss_s = self.loss_function(output_s, label_s.long())
                loss_s.backward()
                self.inner_optim.step()

        self.inner_optim.zero_grad()
        self.init_net.eval()
        for main_qry, extra_qry in zip(self.qry_main_loader_fo, self.qry_extra_loader_fo):
            qry_x_main = main_qry[0].float().to(self.device)
            qry_x_extra = extra_qry[0].long().to(self.device)
            qry_y = extra_qry[1].float().to(self.device)

            output_q = self.init_net(qry_x_main, qry_x_extra)
            label_q = torch.argmax(qry_y, 1)
            loss_q = self.loss_function(output_q, label_q.long())
        grads = torch.autograd.grad(loss_q, self.init_net.parameters(), create_graph=False)

        self.net.train()
        self.outer_optim.zero_grad()
        for w, w_t in zip(self.net.parameters(), grads):
            if w.grad is None:
                w.grad = torch.zeros_like(w.data).to(self.device)
            w.grad.data.copy_(w_t)
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=50., norm_type=2)
        self.outer_optim.step()
        with torch.no_grad():
            self.regular_positive(self.net)
        self.outer_optim.zero_grad()

    def local_fedAvg_train(self):
        self.net.train()
        for i in range(self.train_epoch):
            for main_spt, extra_spt in zip(self.spt_main_loader, self.spt_extra_loader):
                spt_x_main = main_spt[0].float().to(self.device)
                if self.model_name == 'ASU_DNN' or self.model_name == "L_MNL":
                    spt_x_extra = extra_spt[0].float().to(self.device)
                else:
                    spt_x_extra = extra_spt[0].long().to(self.device)
                spt_y = main_spt[1].to(self.device)

                self.inner_optim.zero_grad()
                output = self.net(spt_x_main, spt_x_extra)
                label_c = torch.argmax(spt_y, 1)
                loss = self.loss_function(output, label_c.long())
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=50., norm_type=2)
                self.inner_optim.step()
                with torch.no_grad():
                    self.regular_positive(self.net)

        if self.train_mode == '_fedAvg':
            self.inner_optim.zero_grad()

    def regular_positive(self, net):
        if self.model_name == "E_MNL":
            for name, p in net.extra_model.named_parameters():
                if name == "conv1.weight":
                    p.data.clamp_(min=1e-6)
                    break

        elif self.model_name == 'T_ANN':
            for name, p in net.extra_model.named_parameters():
                if name == "conv1.weight":
                    p.data.clamp_(min=1e-6)
                    break

    def fedavg_test(self):
        support_loss = []
        self.net.eval()
        with torch.no_grad():
            for main_qry, extra_qry in zip(self.qry_main_loader_fo, self.qry_extra_loader_fo):
                qry_x_main = main_qry[0].float()
                if self.model_name == 'ASU_DNN' or self.model_name == "L_MNL":
                    qry_x_extra = extra_qry[0].float()
                else:
                    qry_x_extra = extra_qry[0].long()
                qry_y = extra_qry[1]

                if self.device.type == 'cuda':
                    qry_x_main = qry_x_main.to(self.device)
                    qry_x_extra = qry_x_extra.to(self.device)
                    qry_y = qry_y.to(self.device)

                y_hat = self.net(qry_x_main, qry_x_extra)
                label_c = torch.argmax(qry_y, 1)
                LL = self.loss_function(y_hat, label_c.long())
                support_loss.append(LL.item() * qry_x_main.size(0))
                output_c = torch.argmax(y_hat, dim=1)

        return support_loss, output_c, label_c

    def local_person(self):
        support_loss = []
        local_optim = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr)
        self.net.train()
        for main_spt, extra_spt in zip(self.spt_main_loader_fo, self.spt_extra_loader_fo):
            spt_x_main = main_spt[0].float()
            if self.model_name == 'ASU_DNN' or self.model_name == "L_MNL":
                spt_x_extra = extra_spt[0].float()
            else:
                spt_x_extra = extra_spt[0].long()
            spt_y = main_spt[1]

            if self.device.type == 'cuda':
                spt_x_main = spt_x_main.to(self.device)
                spt_x_extra = spt_x_extra.to(self.device)
                spt_y = spt_y.to(self.device)

            local_optim.zero_grad()
            output = self.net(spt_x_main, spt_x_extra)
            label = torch.argmax(spt_y, 1)
            loss = self.loss_function(output, label.long())
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=50., norm_type=2)
            local_optim.step()
            with torch.no_grad():
                self.regular_positive(self.net)

        self.net.eval()
        with torch.no_grad():
            for main_qry, extra_qry in zip(self.qry_main_loader_fo, self.qry_extra_loader_fo):
                qry_x_main = main_qry[0].float()
                if self.model_name == 'ASU_DNN' or self.model_name == "L_MNL":
                    qry_x_extra = extra_qry[0].float()
                else:
                    qry_x_extra = extra_qry[0].long()
                qry_y = extra_qry[1]

                if self.device.type == 'cuda':
                    qry_x_main = qry_x_main.to(self.device)
                    qry_x_extra = qry_x_extra.to(self.device)
                    qry_y = qry_y.to(self.device)

                y_hat = self.net(qry_x_main, qry_x_extra)
                label_c = torch.argmax(qry_y, 1)
                LL = self.loss_function(y_hat, label_c.long())
                output_c = torch.argmax(y_hat, dim=1)
                label_c = torch.argmax(qry_y, 1)
                support_loss.append(LL.item() * qry_x_main.size(0))
        return support_loss, output_c, label_c
