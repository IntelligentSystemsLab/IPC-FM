# -*- coding: UTF-8 -*-
"""
@Project :meta_lmnl @File :fomaml_lmnl.py
@IDE :PyCharm @Author : hejunshu
@Date :2022/9/6 19:33
"""
import numpy as np
import torch
import pandas as pd
from utilities.Server import MetaServer
import time
import os
import json
import matplotlib.pyplot as plt
from utilities.functions import fixed_initial_net
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--model_num', type=int, default=1)
parser.add_argument('--inner_lr', type=float, default=0.005)
args = parser.parse_args()


def train_test(model_name, Meta_net,inner_lr,directory,directory_model):
    epoch = args.epochs
    plt_x = np.arange(1, epoch + 1, 1)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    acc_l = []
    loss_t = []
    F1_score = []
    KAPPA = []
    acc_test = 0
    for i in range(epoch):
        Meta_net.sync_training(i)  # client, server =
        accuracy, LL_test, f1, kappa = Meta_net.fedAvg_testing(directory_model, acc_test)
        if acc_test <= accuracy:
            acc_test = accuracy
        acc_l.append(accuracy)
        loss_t.append(LL_test)
        F1_score.append(f1)
        KAPPA.append(kappa)
        print('Epoch: ' + str(i) + ' Accuracy:' + str(accuracy) + ' Loss_test:' + str(LL_test))
        print('F1-score:' + str(f1) + ' KAPPA:' + str(kappa))

    ax2.plot(plt_x, acc_l, color='orange', linestyle=':', label="accuracy_")
    ax2.legend()
    ax2.set_ylabel('accuracy value')
    ax2.set_xlabel('epoches')

    plt.title('FedAvg' + model_name)
    plt.savefig(directory + '/FedAvg_' + str(inner_lr) + model_name + '.png')

    acc_l = pd.DataFrame(acc_l, columns=['FedAvg_' + model_name])
    acc_l.to_csv(directory + '/FedAvg_acc_' + str(inner_lr) + '.csv')

    loss_t = pd.DataFrame(loss_t, columns=['FedAvg_' + model_name])
    loss_t.to_csv(directory + '/FedAvg_test_loss_' + str(inner_lr) + '.csv')

    F1_score = pd.DataFrame(F1_score, columns=['FedAvg_' + model_name])
    F1_score.to_csv(directory + '/FedAvg_F1_score_' + str(inner_lr) + '.csv')

    KAPPA = pd.DataFrame(KAPPA, columns=['FedAvg_' + model_name])
    KAPPA.to_csv(directory + '/FedAvg_KAPPA_' + str(inner_lr) + '.csv')


def local_train(model_name, Meta_net, inner_lr,directory_model, directory_local):
    local_acc, local_LL, local_f1, local_kappa = Meta_net.local_train(directory_model, 5)

    local_acc = pd.DataFrame(local_acc, columns=['FedAvg_' + model_name])
    local_acc.to_csv(directory_local + '/1_FedAvg_local_acc_' + str(inner_lr) + '.csv')

    local_LL = pd.DataFrame(local_LL, columns=['FedAvg_' + model_name])
    local_LL.to_csv(directory_local + '/1_FedAvg_local_LL_' + str(inner_lr) + '.csv')

    local_f1 = pd.DataFrame(local_f1, columns=['FedAvg_' + model_name])
    local_f1.to_csv(directory_local + '/1_FedAvg_local_f1_' + str(inner_lr) + '.csv')

    local_kappa = pd.DataFrame(local_kappa, columns=['FedAvg_' + model_name])
    local_kappa.to_csv(directory_local + '/1_FedAvg_local_kappa_' + str(inner_lr) + '.csv')


def main():
    model_name = "ET_MNL"  # "MNL" "ASU_DNN" "E_MNL" "L_MNL" "T_MNL" "ET_MNL"
    print("model_num:",args.model_num," Learning rate:",args.inner_lr)
    fixed_initial_net(args.model_num)
    print("Train model: FedAvg_" + str(model_name))
    conf_path = './conf/conf.json'
    with open(conf_path, 'r') as f:  # 文件名，读写方式
        conf = json.load(f)
    choices_num = 3
    model_num = args.model_num

    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    beta_num = 5
    extra_feature = 12
    NUM_UNIQUE_CATS = 67

    directory = str(model_num) + '/result/FedAvg_sgdmom' + model_name   # sgdmom  adam  sgd
    directory_model = str(model_num) + '/result_model/FedAvg_sgdmom' + model_name
    directory_local = str(model_num)+'/result_local_all/FedAvg_sgdmom' + model_name

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)
    if not os.path.exists(directory_local):
        os.makedirs(directory_local)

    inner_lr = args.inner_lr

    Meta_net = MetaServer(device=device, model_name=model_name, beta_num=beta_num, extra_feature=extra_feature,
                          NUM_UNIQUE_CATS=NUM_UNIQUE_CATS, choices_num=choices_num,
                          networkSize=conf['networkSize'], hidden_layers=conf['hidden_layers'],
                          batch_size=conf['batchSize'], inner_lr=inner_lr,
                          mode="_fedAvg", spt_prop=conf['spt_prop'],
                          test_client_prop=conf['test_client_prop'])

    # train_test(model_name, Meta_net, inner_lr,directory,directory_model)
    local_train(model_name, Meta_net, inner_lr,directory_model, directory_local)


if __name__ == '__main__':
    main()
