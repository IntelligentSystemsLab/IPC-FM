# -*- coding: UTF-8 -*-
"""
@filename:Server.py
@author:Chen Kunxu
@Time:2023/8/5
"""

from sklearn.metrics import f1_score, cohen_kappa_score
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from copy import deepcopy
import sys
from utilities.network_logit import MNL, E_MNL, L_MNL, ASU_DNN, T_ANN
from utilities.client import MetaClient
from utilities.data_manager import load_data,add_positional_gaussian_noise,add_demographic_noise,add_missing_data,handle_missing_data
import concurrent.futures

sys.path.append("..")


def model_choice(
    model_name,
    beta_num,
    extra_feature,
    choices_num,
    NUM_UNIQUE_CATS,
    networkSize,
    hidden_layers,
):
    model = 0
    if model_name == "MNL":
        model = MNL(beta_num, choices_num)
    if model_name == "ASU_DNN":
        model = ASU_DNN(beta_num, extra_feature, choices_num)
    if model_name == "E_MNL":
        model = E_MNL(beta_num, extra_feature, choices_num, NUM_UNIQUE_CATS)
    if model_name == "L_MNL":
        model = L_MNL(
            beta_num,
            extra_feature,
            choices_num,
            networkSize,
            hidden_layers,
        )
    if model_name == "T_ANN":
        model = T_ANN(beta_num, extra_feature, choices_num, NUM_UNIQUE_CATS)
    return model


class Server(nn.Module):
    def __init__(
        self,
        device,
        model_name,
        networkSize,
        hidden_layers,
        choices_num,
        beta_num,
        extra_feature,
        NUM_UNIQUE_CATS,
    ):
        super(Server, self).__init__()
        self.device = device
        # 网络结构设置
        self.choices_num = choices_num
        self.beta_num = beta_num
        self.extra_feature = extra_feature
        self.networkSize = networkSize
        self.hidden_layers = hidden_layers
        self.unique_cats_num = NUM_UNIQUE_CATS
        self.net = model_choice(
            model_name,
            beta_num,
            extra_feature,
            choices_num,
            NUM_UNIQUE_CATS,
            networkSize,
            hidden_layers,
        )
        print(self.net)
        self.net = self.net.to(self.device)


class MetaServer(Server):
    def __init__(
        self,
        device,
        centralized=0,
        mode="_fomaml",
        test_client_prop=0.3,
        spt_prop=0.5,
        model_name="L_MNL",
        networkSize=128,
        hidden_layers=1,
        batch_size=16,
        beta_num=5,
        extra_feature=14,
        NUM_UNIQUE_CATS=100,
        choices_num=4,
        L1=0.0,
        L2=0.0,
        dropout=0.3,
        inner_lr=0.002,
        outer_lr=0.002,
        noise_param=0.0,
        missing_rate=0.0,
        missing_type="random",
        missing_handle="mean",
        # 性能优化参数
        max_workers=16,
        # local training
        local_metatrain_epoch=1,
        local_metatest_epoch=2,
        seed=42,
    ):
        super(MetaServer, self).__init__(
            device,
            model_name,
            networkSize,
            hidden_layers,
            choices_num,
            beta_num,
            extra_feature,
            NUM_UNIQUE_CATS,
        )
        self.local_metatrain_epoch = local_metatrain_epoch
        self.local_metatest_epoch = local_metatest_epoch
        self.spt_prop = spt_prop
        self.max_workers = max_workers  # 存储线程池大小
        self.train_clients = []  # 存储的client
        self.test_clients = []  # 测试的client
        self.train_mode = mode
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.batch_size = batch_size
        self.model_name = model_name
        dataset_main_all, dataset_extra_all, saqure = load_data(
            NUM_UNIQUE_CATS, self.model_name
        )
        self.saqure = saqure
        num_people_LPMC = len(dataset_main_all)
        people_list = np.array(range(0, num_people_LPMC, 1))
        train_id, test_id = train_test_split(
            people_list, test_size=test_client_prop, random_state=3
        )
        positions_main=[(3, 0), (3, 1), (3, 2), (4, 2), (3, 3), (4, 3)]
        dataset_main_train, dataset_main_test = [], []
        dataset_extra_train, dataset_extra_test = [], []
        for train_index in train_id:
            dataset_main_train.append(dataset_main_all[train_index])
            dataset_extra_train.append(dataset_extra_all[train_index])
        for test_index in test_id:
            # 添加缺失数据模拟
            if noise_param > 0:
                dataset_main_noise = add_positional_gaussian_noise(dataset_main_all[test_index], positions=positions_main, noise_std=noise_param, seed=seed)
                dataset_extra_noise = add_demographic_noise(dataset_extra_all[test_index], label_noise_rate=noise_param, seed=seed)
                dataset_main_test.append(dataset_main_noise)
                dataset_extra_test.append(dataset_extra_noise)
            elif missing_rate > 0:
                # 为主要特征数据添加缺失值
                dataset_main_missing = add_missing_data(dataset_main_noise, missing_rate=missing_rate, 
                                                      missing_type=missing_type, seed=seed)
                dataset_main_handled = handle_missing_data(dataset_main_missing, method=missing_handle, seed=seed)
                
                # 为额外特征数据添加缺失值
                dataset_extra_missing = add_missing_data(dataset_extra_noise, missing_rate=missing_rate, 
                                                       missing_type=missing_type, seed=seed)
                dataset_extra_handled = handle_missing_data(dataset_extra_missing, method=missing_handle, seed=seed)
                dataset_main_test.append(dataset_main_handled)
                dataset_extra_test.append(dataset_extra_handled)
            else:
                dataset_main_test.append(dataset_main_all[test_index])
                dataset_extra_test.append(dataset_extra_all[test_index])

        if centralized:
            model = deepcopy(self.net)
            data_main = dataset_main_train[0]
            data_extra = dataset_extra_train[0]
            for train_data_index in range(1, len(dataset_main_train)):
                data_main = np.concatenate(
                    (data_main, dataset_main_train[train_data_index]), axis=0
                )
                data_extra = np.concatenate(
                    (data_extra, dataset_extra_train[train_data_index]), axis=0
                )
            self.train_clients.append(
                MetaClient(
                    device=self.device,
                    id_c=-1,
                    mode=self.train_mode,
                    model=model,
                    train_epoch=self.local_metatrain_epoch,
                    test_epoch=self.local_metatest_epoch,
                    batch_size=self.batch_size,
                    inner_lr=inner_lr,
                    outer_lr=outer_lr,
                    model_name=model_name,
                    data_main=data_main,
                    data_extra=data_extra,
                    spt_prop=spt_prop,
                )
            )
        else:
            for index in range(0, len(train_id)):
                model = model_choice(
                    model_name,
                    beta_num,
                    extra_feature,
                    choices_num,
                    NUM_UNIQUE_CATS,
                    networkSize,
                    hidden_layers,
                )
                self.train_clients.append(
                    MetaClient(
                        device=self.device,
                        id_c=index,
                        mode=self.train_mode,
                        model=model,
                        train_epoch=self.local_metatrain_epoch,
                        test_epoch=self.local_metatest_epoch,
                        batch_size=self.batch_size,
                        inner_lr=inner_lr,
                        outer_lr=outer_lr,
                        model_name=model_name,
                        data_main=dataset_main_train[index],
                        data_extra=dataset_extra_train[index],
                        spt_prop=spt_prop,
                    )
                )

        for index in range(0, len(test_id)):
            model = model_choice(
                model_name,
                beta_num,
                extra_feature,
                choices_num,
                NUM_UNIQUE_CATS,
                networkSize,
                hidden_layers,
            )
            self.test_clients.append(
                MetaClient(
                    device=self.device,
                    id_c=index,
                    model=model,
                    mode=self.train_mode,
                    train_epoch=self.local_metatrain_epoch,
                    test_epoch=self.local_metatest_epoch,
                    batch_size=self.batch_size,
                    inner_lr=inner_lr,
                    outer_lr=outer_lr,
                    model_name=model_name,
                    data_main=dataset_main_test[index],
                    data_extra=dataset_extra_test[index],
                    spt_prop=spt_prop,
                )
            )

    def centralized_training(self, round):
        self.train_clients[0].local_fedAvg_train()
        model_param_clients = self.train_clients[0].net.state_dict()
        self.net.load_state_dict(model_param_clients, strict=True)


    def sync_training(self, round):
        weight = []
        id_train = list(range(len(self.train_clients)))
        def train_one_client(client, net, train_mode, round):
            client.refresh(net)
            if train_mode == "_fedAvg":
                client.local_fedAvg_train()
            else:
                client.local_fomaml_train()
            client.epoch = round
            
            # 返回数据量作为权重（更合理的聚合方式）
            return 1
        # 使用可配置的线程数
        max_workers = getattr(self, 'max_workers', 16)
        # 限制线程数不超过客户端数量
        max_workers = min(max_workers, len(self.train_clients))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for index in id_train:
                client = self.train_clients[index]
                futures.append(executor.submit(train_one_client, client, self.net, self.train_mode, round))
            weight = [future.result() for future in futures]
        # 标准化权重
        weight = np.array(weight)
        weight = weight / weight.sum()
        
        # 修复聚合逻辑：正确的联邦平均
        with torch.no_grad():
            for param_idx, server_param in enumerate(self.net.parameters()):
                # 重置服务器参数为零
                server_param.data.zero_()
                
                # 累加所有客户端的加权参数
                for client_idx, client_weight in enumerate(weight):
                    client_param = list(self.train_clients[client_idx].net.parameters())[param_idx]
                    server_param.data.add_(client_param.data * client_weight)

    def fedAvg_testing(self, directory_model, acc_test):
        if (self.model_name == "E_MNL") or (self.model_name == "T_ANN"):
            state = {"model": self.net.state_dict(), "Label": self.saqure, "Name": self.model_name}
            torch.save(state, directory_model + "/state.pth")
        torch.save(self.net.state_dict(), directory_model + "/network.pth")
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
        f1 = f1_score(Actual, Predict, average="macro")
        kappa = cohen_kappa_score(Actual, Predict)
        acc_init = acc_init.cpu().numpy()
        return acc_init, LL_test, f1, kappa

    def local_train(self, directory_model, max_local):
        # model = torch.load(directory_model + "/network.pth",weights_only=True)
        model_param_clients = torch.load(directory_model + "/network.pth",weights_only=True)
        self.net.load_state_dict(model_param_clients)
        id_test = list(range(len(self.test_clients)))
        acc_test = []
        loss_test = []
        f1_test = []
        kappa_test = []
        for _, id in enumerate(id_test):
            self.test_clients[id].refresh(self.net)
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
            f1 = f1_score(Actual, Predict, average="macro")
            kappa = cohen_kappa_score(Actual, Predict)
            acc_init = acc_init.cpu().numpy()
            acc_test.append(acc_init)
            loss_test.append(LL_test)
            f1_test.append(f1)
            kappa_test.append(kappa)
        return acc_test, loss_test, f1_test, kappa_test
