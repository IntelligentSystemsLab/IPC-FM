# -*- coding: UTF-8 -*-
"""
@Project :meta_lmnl
@author:Chen Kunxu
@Date:2023/11/06
"""
import numpy as np
import torch
import pandas as pd
from utilities.Server import MetaServer
import os
import json
import matplotlib.pyplot as plt
from utilities.functions import fixed_initial_net
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--model", type=str, default="E_MNL")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--model_num", type=int, default=19)
parser.add_argument("--inner_lr", type=float, default=0.005)
args = parser.parse_args()


def train_test(model_name, Meta_net, inner_lr, directory, directory_model):
    epoch = args.epochs
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
    plt.savefig(directory + "/central_" + str(inner_lr) + model_name + ".png")

    acc_l = pd.DataFrame(acc_l, columns=[model_name])
    acc_l.to_csv(directory + "/central_acc_" + str(inner_lr) + model_name + ".csv")

    loss_C = pd.DataFrame(loss_C, columns=[model_name])
    loss_C.to_csv(
        directory + "/central_train_loss_" + str(inner_lr) + model_name + ".csv"
    )

    loss_t = pd.DataFrame(loss_t, columns=[model_name])
    loss_t.to_csv(
        directory + "/central_test_loss_" + str(inner_lr) + model_name + ".csv"
    )

    F1_score = pd.DataFrame(F1_score, columns=[model_name])
    F1_score.to_csv(
        directory + "/central_F1_score_" + str(inner_lr) + model_name + ".csv"
    )

    KAPPA = pd.DataFrame(KAPPA)
    KAPPA.to_csv(directory + "/central_KAPPA_" + str(inner_lr) + ".csv")


def local_train(model_name, Meta_net, inner_lr, directory_model, directory_local):
    local_acc, local_LL, local_f1, local_kappa = Meta_net.local_train(
        directory_model, 5
    )

    local_acc = pd.DataFrame(local_acc, columns=["central_" + model_name])
    local_acc.to_csv(
        directory_local + "/1_central_local_acc_" + model_name + str(inner_lr) + ".csv"
    )

    local_LL = pd.DataFrame(local_LL, columns=["central_" + model_name])
    local_LL.to_csv(
        directory_local + "/1_central_local_LL_" + model_name + str(inner_lr) + ".csv"
    )

    local_f1 = pd.DataFrame(local_f1, columns=["central_" + model_name])
    local_f1.to_csv(
        directory_local + "/1_central_local_f1_" + model_name + str(inner_lr) + ".csv"
    )

    local_kappa = pd.DataFrame(local_kappa, columns=["central_" + model_name])
    local_kappa.to_csv(
        directory_local
        + "/1_central_local_kappa_"
        + model_name
        + str(inner_lr)
        + ".csv"
    )


def main(model_name):
    print("Train model: " + str(model_name))
    fixed_initial_net(args.model_num)
    print("model_num:", args.model_num, " Learning rate:", args.inner_lr)
    beta_num = 5
    extra_feature = 12
    NUM_UNIQUE_CATS = 67
    conf_path = "./conf/conf.json"
    with open(conf_path, "r") as f:  # 文件名，读写方式
        conf = json.load(f)
    choices_num = 3
    model_num = args.model_num
    device = torch.device("cpu")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    directory = str(model_num) + "/result/sgdmom_" + model_name  # adam_ sgd_ sgdmom_
    directory_model = str(model_num) + "/result_model/sgdmom_" + model_name
    directory_local = str(model_num) + "/result_local_all/sgdmom_" + model_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)
    if not os.path.exists(directory_local):
        os.makedirs(directory_local)

    inner_lr = args.inner_lr

    Meta_net = MetaServer(
        device=device,
        centralized=1,
        beta_num=beta_num,
        extra_feature=extra_feature,
        NUM_UNIQUE_CATS=NUM_UNIQUE_CATS,
        choices_num=choices_num,
        networkSize=conf["networkSize"],
        hidden_layers=conf["hidden_layers"],
        batch_size=conf["batchSize"],
        mode="_central",
        inner_lr=inner_lr,
        spt_prop=conf["spt_prop"],
        test_client_prop=conf["test_client_prop"],
        model_name=model_name,
    )

    # train_test(model_name, Meta_net, inner_lr, directory, directory_model)
    local_train(model_name, Meta_net, inner_lr, directory_model, directory_local)


if __name__ == "__main__":
    model_name = args.model
    main(model_name)
