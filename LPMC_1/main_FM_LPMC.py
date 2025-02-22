# -*- coding: UTF-8 -*-
"""
@filename:model_FM_LPMC.py
@author:Chen Kunxu
@Time:2023/8/5 16:29
"""
import argparse
import numpy as np
import torch
import pandas as pd
from utilities.Server import MetaServer
import time
import os
import json
import matplotlib.pyplot as plt
from utilities.functions import fixed_initial_net

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--model", type=str, default="MNL")
parser.add_argument("--model_num", type=int, default=100)
parser.add_argument("--L1", type=float, default=0.0)
parser.add_argument("--L2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--inner_lr", type=float, default=0.002)
parser.add_argument("--outer_lr", type=float, default=0.002)
args = parser.parse_args()


def train_test(
    model_name,
    Meta_net,
    inner_lr,
    outer_lr,
    directory,
    directory_model,
    directory_other,
):
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
        Meta_net.sync_training(i)
        accuracy, LL_test, f1, kappa = Meta_net.fedAvg_testing(
            directory_model, acc_test
        )
        if acc_test <= accuracy:
            acc_test = accuracy
        loss_t.append(LL_test)
        acc_l.append(accuracy)
        F1_score.append(f1)
        KAPPA.append(kappa)
        print(
            "Epoch: "
            + str(i)
            + " Accuracy:"
            + str(accuracy)
            + " Loss_test:"
            + str(LL_test)
        )
        print("F1-score:" + str(f1) + " KAPPA:" + str(kappa))

    ax2.plot(plt_x, acc_l, color="orange", linestyle=":", label="accuracy_")
    ax2.legend()
    ax2.set_ylabel("accuracy value")
    ax2.set_xlabel("epoches")

    plt.title("FedMeta_" + model_name)
    plt.savefig(
        directory
        + "/FedMeta_"
        + model_name
        + "_"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".png"
    )

    acc_l = pd.DataFrame(acc_l, columns=["FedMeta_" + model_name])
    acc_l.to_csv(
        directory + "/FedMeta_acc" + str(inner_lr) + "_" + str(outer_lr) + ".csv"
    )

    loss_t = pd.DataFrame(loss_t, columns=["FedMeta_" + model_name])
    loss_t.to_csv(
        directory + "/FedMeta_test_loss" + str(inner_lr) + "_" + str(outer_lr) + ".csv"
    )

    F1_score = pd.DataFrame(F1_score, columns=["FedMeta_" + model_name])
    F1_score.to_csv(
        directory_other
        + "/FedMeta_F1_score"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )

    KAPPA = pd.DataFrame(KAPPA, columns=["FedMeta_" + model_name])
    KAPPA.to_csv(
        directory_other
        + "/FedMeta_KAPPA"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )


def local_train(
    model_name, Meta_net, inner_lr, outer_lr, directory_model, directory_local
):
    local_acc, local_LL, local_f1, local_kappa = Meta_net.local_train(
        directory_model, 5
    )

    local_acc = pd.DataFrame(local_acc, columns=["FedMeta_" + model_name])
    local_acc.to_csv(
        directory_local
        + "/1_FedMeta_local_acc_"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )

    local_LL = pd.DataFrame(local_LL, columns=["FedMeta_" + model_name])
    local_LL.to_csv(
        directory_local
        + "/1_FedMeta_local_LL_"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )

    local_f1 = pd.DataFrame(local_f1, columns=["FedMeta_" + model_name])
    local_f1.to_csv(
        directory_local
        + "/1_FedMeta_local_f1_"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )

    local_kappa = pd.DataFrame(local_kappa, columns=["FedMeta_" + model_name])
    local_kappa.to_csv(
        directory_local
        + "/1_FedMeta_local_kappa_"
        + str(inner_lr)
        + "_"
        + str(outer_lr)
        + ".csv"
    )


def main():
    model_name = "ET_MNL"  # L_MNL or E_MNL or ASU_DNN or EL_MNL or T_MNL
    print(args.model_num)
    fixed_initial_net(args.model_num)
    print("Train model: FedMeta_" + str(model_name))
    conf_path = "./conf/conf.json"
    with open(conf_path, "r") as f:
        conf = json.load(f)
    choices_num = 4
    model_num = args.model_num
    epoch = args.epochs
    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    beta_num = 5
    extra_feature = 14
    NUM_UNIQUE_CATS = 100

    directory = str(model_num) + "/result/FedMeta_sgd_sgdmom" + model_name
    directory_model = str(model_num) + "/result_model/FedMeta_sgd_sgdmom" + model_name
    directory_local = (
        str(model_num) + "/result_local_all/FedMeta_sgd_sgdmom" + model_name
    )  # sgd  sgdmom   adam
    directory_other = str(model_num) + "/result_other/FedMeta_sgd_sgdmom" + model_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)
    if not os.path.exists(directory_local):
        os.makedirs(directory_local)
    if not os.path.exists(directory_other):
        os.makedirs(directory_other)

    inner_lr = 0.002
    outer_lr = 0.002

    Meta_net = MetaServer(
        device=device,
        beta_num=beta_num,
        extra_feature=extra_feature,
        NUM_UNIQUE_CATS=NUM_UNIQUE_CATS,
        choices_num=choices_num,
        networkSize=conf["networkSize"],
        hidden_layers=conf["hidden_layers"],
        batch_size=conf["batchSize"],
        mode="_fomaml",
        spt_prop=conf["spt_prop"],
        model_name=model_name,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        test_client_prop=conf["test_client_prop"],
    )  # sgd:0.1 adam:0.1
    # train_test(model_name, Meta_net, inner_lr, outer_lr, directory, directory_model, directory_other)
    local_train(
        model_name, Meta_net, inner_lr, outer_lr, directory_model, directory_local
    )


if __name__ == "__main__":
    main()
