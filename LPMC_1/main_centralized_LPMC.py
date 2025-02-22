# -*- coding: UTF-8 -*-
"""
@filename:main_centralized_LPMC.py
@author:Chen Kunxu
@Time:2023/8/5 16:29
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--model", type=str, default="MNL")
parser.add_argument("--model_num", type=int, default=100)
parser.add_argument("--L1", type=float, default=0.0)
parser.add_argument("--L2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--inner_lr", type=float, default=0.002)
parser.add_argument("--save_file", type=str, default="default")
args = parser.parse_args()


def train_test(
    model_name, Meta_net, epoch, inner_lr, directory, directory_model, directory_other
):
    plt_x = np.arange(1, epoch + 1, 1)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    acc_l = []
    loss_C = []
    loss_t = []
    F1_score = []
    KAPPA = []
    acc_test = 0
    for i in range(epoch):
        Loss = Meta_net.centralized_training(i)
        accuracy, LL_test, f1, kappa = Meta_net.fedAvg_testing(
            directory_model, acc_test
        )
        if acc_test <= accuracy:
            acc_test = accuracy
        acc_l.append(accuracy)
        loss_C.append(Loss)
        loss_t.append(LL_test)
        F1_score.append(f1)
        KAPPA.append(kappa)
        print(
            "Epoch: "
            + str(i)
            + " Loss:"
            + str(Loss)
            + " Accuracy:"
            + str(accuracy)
            + " Loss_test:"
            + str(LL_test)
        )
        print("F1-score:" + str(f1) + " KAPPA:" + str(kappa))

    ax2.plot(plt_x, acc_l, color="orange", linestyle=":", label="accuracy_")
    ax2.legend()
    ax2.set_ylabel("accuracy value")
    ax2.set_xlabel("epochs")

    plt.title("Central" + model_name)
    plt.savefig(directory + "/central_" + model_name + ".png")

    acc_l = pd.DataFrame(acc_l, columns=[model_name])
    acc_l.to_csv(directory + "/central_acc_" + model_name + ".csv")

    loss_C = pd.DataFrame(loss_C, columns=[model_name])
    loss_C.to_csv(directory + "/central_train_loss_" + model_name + ".csv")

    loss_t = pd.DataFrame(loss_t, columns=[model_name])
    loss_t.to_csv(directory + "/central_test_loss_" + model_name + ".csv")

    F1_score = pd.DataFrame(F1_score, columns=[model_name])
    F1_score.to_csv(directory_other + "/central_F1_score_" + model_name + ".csv")

    KAPPA = pd.DataFrame(KAPPA)
    KAPPA.to_csv(directory_other + "/central_KAPPA_" + model_name + ".csv")


def local_train(model_name, Meta_net, inner_lr, directory_model, directory_local):
    local_acc, local_LL, local_f1, local_kappa = Meta_net.local_train(
        directory_model, 5
    )

    local_acc = pd.DataFrame(local_acc, columns=["central_" + model_name])
    local_acc.to_csv(directory_local + "/central_local_acc_" + model_name + ".csv")

    local_LL = pd.DataFrame(local_LL, columns=["central_" + model_name])
    local_LL.to_csv(directory_local + "/central_local_LL_" + model_name + ".csv")

    local_f1 = pd.DataFrame(local_f1, columns=["central_" + model_name])
    local_f1.to_csv(directory_local + "/central_local_f1_" + model_name + ".csv")

    local_kappa = pd.DataFrame(local_kappa, columns=["central_" + model_name])
    local_kappa.to_csv(directory_local + "/central_local_kappa_" + model_name + ".csv")


def main(model_name):
    model_num = args.model_num
    save_file = args.save_file

    print(model_num)
    fixed_initial_net(model_num)
    print("Train model: " + str(model_name))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = str(model_num) + "/result/sgdmom_" + model_name + "_" + save_file
    directory_model = (
        str(model_num) + "/result_model/sgdmom_" + model_name + "_" + save_file
    )
    directory_local = (
        str(model_num) + "/result_local_all/sgdmom_" + model_name + "_" + save_file
    )
    directory_other = (
        str(model_num) + "/result_other/sgdmom_" + model_name + "_" + save_file
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)
    if not os.path.exists(directory_local):
        os.makedirs(directory_local)
    if not os.path.exists(directory_other):
        os.makedirs(directory_other)

    # hyper-parameters
    conf_path = "./conf/conf.json"
    with open(conf_path, "r") as f:  # 文件名，读写方式
        conf = json.load(f)
    beta_num = 5
    extra_feature = 14
    NUM_UNIQUE_CATS = 100
    choices_num = 4
    L1 = args.L1
    L2 = args.L2
    dropout = args.dropout
    inner_lr = args.inner_lr
    epoch = args.epochs

    Meta_net = MetaServer(
        device=device,
        # mode
        centralized=1,
        mode="_central",
        # data split
        test_client_prop=conf["test_client_prop"],
        spt_prop=conf["spt_prop"],
        # model
        model_name=model_name,
        networkSize=conf["networkSize"],
        hidden_layers=conf["hidden_layers"],
        # hyper-parameters
        batch_size=conf["batchSize"],
        beta_num=beta_num,
        extra_feature=extra_feature,
        NUM_UNIQUE_CATS=NUM_UNIQUE_CATS,
        choices_num=choices_num,
        L1=L1,
        L2=L2,
        dropout=dropout,
        inner_lr=inner_lr,
    )
    train_test(
        model_name,
        Meta_net,
        epoch,
        inner_lr,
        directory,
        directory_model,
        directory_other,
    )
    local_train(model_name, Meta_net, inner_lr, directory_model, directory_local)


if __name__ == "__main__":
    model_name = args.model
    main(model_name)
