# -*- coding: UTF-8 -*-
"""
@Project :meta_lmnl
@IDE :PyCharm @Author : chenkunxu
@Date :2022/11/06 00:36
"""

from sklearn.metrics import f1_score, cohen_kappa_score
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from copy import deepcopy
import sys
from utilities.network_logit import MNL, E_MNL, L_MNL, ASU_DNN, T_MNL, ET_MNL
from utilities.client import MetaClient
from utilities.data_manager import load_data

sys.path.append('..')


def model_choice(model_name, beta_num, nExtraFeatures, choices_num, NUM_UNIQUE_CATS, networkSize=128,
                 hidden_layers=1):
    model = 0
    if model_name == "MNL":
        model = MNL(beta_num, choices_num)
    if model_name == "E_MNL":
        model = E_MNL(beta_num, nExtraFeatures, choices_num, NUM_UNIQUE_CATS)
    if model_name == "ASU_DNN":
        model = ASU_DNN(beta_num, nExtraFeatures, choices_num)
    if model_name == "T_MNL":
        model = T_MNL(beta_num, nExtraFeatures, choices_num, NUM_UNIQUE_CATS)
    if model_name == "ET_MNL":
        model = ET_MNL(beta_num, nExtraFeatures, choices_num, NUM_UNIQUE_CATS)
    if model_name == "L_MNL":
        model = L_MNL(beta_num, nExtraFeatures, choices_num, networkSize=networkSize,
                      hidden_layers=hidden_layers)
    return model


class Server(nn.Module):
    def __init__(self, device, choices_num, beta_num, extra_feature, networkSize, hidden_layers, model_name,
                 NUM_UNIQUE_CATS):
        super(Server, self).__init__()
        self.device = device
        # 网络结构设置
        self.choices_num = choices_num
        self.beta_num = beta_num
        self.extra_feature = extra_feature
        self.networkSize = networkSize
        self.hidden_layers = hidden_layers
        self.unique_cats_num = NUM_UNIQUE_CATS
        self.net = model_choice(model_name, beta_num, extra_feature, choices_num, NUM_UNIQUE_CATS, networkSize,
                                hidden_layers)
        print(self.net)
        self.net = self.net.to(self.device)


class MetaServer(Server):
    def __init__(self, device,  # client_num,  # 数据参数
                 spt_prop, test_client_prop, networkSize, hidden_layers,
                 beta_num, extra_feature, choices_num,
                 local_metatrain_epoch=1, local_metatest_epoch=2, batch_size=16, inner_lr=0.01, outer_lr=0.01,
                 # 训练参数
                 mode="_fomaml", centralized=0, model_name='L_MNL', NUM_UNIQUE_CATS=100):
        # fedavg fomaml reptile
        super(MetaServer, self).__init__(device, choices_num, beta_num, extra_feature, networkSize,
                                         hidden_layers, model_name, NUM_UNIQUE_CATS)
        self.local_metatrain_epoch = local_metatrain_epoch
        self.local_metatest_epoch = local_metatest_epoch
        self.spt_prop = spt_prop
        self.train_clients = []  # 存储的client
        self.test_clients = []  # 测试的client
        self.train_mode = mode
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.batch_size = batch_size
        self.model_name = model_name
        dataset_main_all, dataset_extra_all, saqure = load_data(NUM_UNIQUE_CATS, self.model_name)
        self.saqure = saqure
        num_people_LPMC = len(dataset_main_all)
        people_list = np.array(range(0, num_people_LPMC, 1))
        train_id, test_id = train_test_split(people_list, test_size=test_client_prop, random_state=3)

        dataset_main_train, dataset_main_test = [], []
        dataset_extra_train, dataset_extra_test = [], []
        for train_index in train_id:
            dataset_main_train.append(dataset_main_all[train_index])
            dataset_extra_train.append(dataset_extra_all[train_index])
        for test_index in test_id:
            dataset_main_test.append(dataset_main_all[test_index])
            dataset_extra_test.append(dataset_extra_all[test_index])

        if centralized:
            model = deepcopy(self.net)
            data_main = dataset_main_train[0]
            data_extra = dataset_extra_train[0]
            for train_data_index in range(1, len(dataset_main_train)):
                data_main = np.concatenate((data_main, dataset_main_train[train_data_index]), axis=0)
                data_extra = np.concatenate((data_extra, dataset_extra_train[train_data_index]), axis=0)
            self.train_clients.append(MetaClient(
                device=self.device, id_c=-1, mode=self.train_mode, model=model,
                train_epoch=self.local_metatrain_epoch, test_epoch=self.local_metatest_epoch,
                batch_size=self.batch_size,
                inner_lr=inner_lr, outer_lr=outer_lr, model_name=model_name,
                data_main=data_main, data_extra=data_extra,
                spt_prop=spt_prop))
        else:
            for index in range(0, len(train_id)):
                model = model_choice(model_name, beta_num, extra_feature, choices_num, NUM_UNIQUE_CATS, networkSize,
                                     hidden_layers)
                self.train_clients.append(MetaClient(
                    device=self.device, id_c=index, mode=self.train_mode, model=model,
                    train_epoch=self.local_metatrain_epoch, test_epoch=self.local_metatest_epoch,
                    batch_size=self.batch_size,
                    inner_lr=inner_lr, outer_lr=outer_lr, model_name=model_name,
                    data_main=dataset_main_train[index], data_extra=dataset_extra_train[index],
                    spt_prop=spt_prop))

        for index in range(0, len(test_id)):
            model = model_choice(model_name, beta_num, extra_feature, choices_num, NUM_UNIQUE_CATS, networkSize,hidden_layers)
            self.test_clients.append(MetaClient(
                device=self.device, id_c=index, model=model, mode=self.train_mode,
                train_epoch=self.local_metatrain_epoch, test_epoch=self.local_metatest_epoch,
                batch_size=self.batch_size,
                inner_lr=inner_lr, outer_lr=outer_lr, model_name=model_name,
                data_main=dataset_main_test[index], data_extra=dataset_extra_test[index],
                spt_prop=spt_prop))

    def centralized_training(self, round):
        Loss_all = self.train_clients[0].local_fedAvg_train()
        model_param_clients = self.train_clients[0].net.state_dict()  # 获取model的参数
        self.net.load_state_dict(model_param_clients, strict=True)
        return Loss_all

    def sync_training(self, round):
        weight = []
        id_train = list(range(len(self.train_clients)))
        for id, index in enumerate(id_train):
            self.train_clients[index].refresh(self.net)
            if self.train_mode == "_fedAvg":
                self.train_clients[index].local_fedAvg_train()  # client_cmp_time =
            else:
                self.train_clients[index].local_fomaml_train()  # client_cmp_time =
            self.train_clients[index].epoch = round
            weight.append(1)

        weight = np.array(weight)
        weight = weight / weight.sum()
        for id, index in enumerate(id_train):
            for w, w_t in zip(self.net.parameters(), self.train_clients[index].net.parameters()):
                if w is None or id == 0:
                    w.data = torch.zeros_like(w).to(self.device)
                if w_t is None:
                    w_t = torch.zeros_like(w).to(self.device)
                w.data.add_(w_t.data * weight[index])

    def fedAvg_testing(self, directory_model, acc_test):
        Flag = 0
        if (self.model_name == "E_MNL") or (self.model_name == "T_MNL") or (self.model_name == "ET_MNL"):
            state = {'model': self.net, "Label": self.saqure, "Name": self.model_name}
            torch.save(state, directory_model + '/state.pth')
            Flag = 1
        torch.save(self.net, directory_model + '/network.pth')
        id_test = list(range(len(self.test_clients)))
        init_true_total = 0
        size_total = 0

        LL_test = 0
        Predict = []
        Actual = []
        for a, id in enumerate(id_test):
            self.test_clients[id].refresh(self.net)  # 更新当前元模型
            loss_all, pre, act = self.test_clients[id].fedavg_test()
            init_true_total += (pre == act).sum()
            size_total += len(act)
            LL_test += sum(loss_all)
            for i in range(len(pre)):
                Predict.append(pre.cpu().numpy()[i])
                Actual.append(act.cpu().numpy()[i])
        acc_init = init_true_total / size_total
        f1 = f1_score(Actual, Predict, average='macro')
        kappa = cohen_kappa_score(Actual, Predict)
        acc_init = acc_init.cpu().numpy()
        if acc_init >= acc_test:
            if Flag:
                torch.save(state, directory_model + '/best_state.pth')
            else:
                torch.save(self.net, directory_model + '/best_state.pth')
        return acc_init, LL_test, f1, kappa

    def local_train(self, directory_model, max_local):
        # 本地化
        model = torch.load(directory_model + '/network.pth')
        model_param_clients = model.state_dict()
        self.net.load_state_dict(model_param_clients)
        id_test = list(range(len(self.test_clients)))
        acc_test = []
        loss_test = []
        f1_test = []
        kappa_test = []
        for _, id in enumerate(id_test):
            self.test_clients[id].refresh(self.net)  # 更新当前元模型
        for epoch in range(0, max_local):
            init_true_total = 0
            size_total = 0
            LL_test = 0
            Predict = []
            Actual = []
            for a, id in enumerate(id_test):
                loss_all, pre, act = self.test_clients[id].local_person()
                init_true_total += (pre == act).sum()
                size_total += len(act)
                LL_test += sum(loss_all)
                for i in range(len(pre)):
                    Predict.append(pre.cpu().numpy()[i])
                    Actual.append(act.cpu().numpy()[i])
            acc_init = init_true_total / size_total
            f1 = f1_score(Actual, Predict, average='macro')
            kappa = cohen_kappa_score(Actual, Predict)
            acc_init = acc_init.cpu().numpy()
            acc_test.append(acc_init)
            loss_test.append(LL_test)
            f1_test.append(f1)
            kappa_test.append(kappa)
        return acc_test, loss_test, f1_test, kappa_test
