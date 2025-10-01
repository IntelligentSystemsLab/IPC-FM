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
parser.add_argument("--model", type=str, default="T_ANN")
parser.add_argument("--model_num", type=int, default=100)
parser.add_argument("--L1", type=float, default=0.0)
parser.add_argument("--L2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--inner_lr", type=float, default=0.02)
parser.add_argument("--outer_lr", type=float, default=0.02)
parser.add_argument("--save_file", type=str, default="default")
parser.add_argument("--noise_param", type=float, default=0.0)
parser.add_argument("--missing_rate", type=float, default=0.0)
parser.add_argument("--max_workers", type=int, default=16, help="并行训练的最大线程数")
args = parser.parse_args()


def train_test(
    model_name,
    Meta_net,
    epoch,
    directory,
    directory_model,
    directory_other,
):
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

        acc_l.append(accuracy)
        loss_t.append(LL_test)
        F1_score.append(f1)
        KAPPA.append(kappa)
        
        # 优化 print：减少频率，使用 flush 确保及时输出
        if i % 10 == 0 or i == epoch - 1:  # 每10轮或最后一轮输出
            print(f"Epoch: {i:3d} | Accuracy: {accuracy:.4f}")

    plt_x = np.arange(1, epoch + 1, 1)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(plt_x, acc_l, color="orange", linestyle=":", label="accuracy_")
    ax2.legend()
    ax2.set_ylabel("accuracy value")
    ax2.set_xlabel("epoches")

    plt.title("FedMeta_" + model_name)
    plt.savefig(
        directory
        + "/FedMeta_"
        + model_name
        + ".png"
    )

    acc_l = pd.DataFrame(acc_l, columns=["FedMeta_" + model_name])
    acc_l.to_csv(
        directory + "/FedMeta_acc" + model_name + ".csv"
    )

    loss_t = pd.DataFrame(loss_t, columns=["FedMeta_" + model_name])
    loss_t.to_csv(
        directory + "/FedMeta_test_loss" + model_name + ".csv"
    )

    F1_score = pd.DataFrame(F1_score, columns=["FedMeta_" + model_name])
    F1_score.to_csv(
        directory_other
        + "/FedMeta_F1_score"
        + model_name
        + ".csv"
    )

    KAPPA = pd.DataFrame(KAPPA, columns=["FedMeta_" + model_name])
    KAPPA.to_csv(
        directory_other
        + "/FedMeta_KAPPA"
        + model_name
        + ".csv"
    )


def local_train(
    model_name, Meta_net, directory_model, directory_local
):
    local_acc, local_LL, local_f1, local_kappa = Meta_net.local_train(
        directory_model, 5
    )

    local_acc = pd.DataFrame(local_acc, columns=["FedMeta_" + model_name])
    local_acc.to_csv(
        directory_local
        + "/FedMeta_local_acc_"
        + model_name
        + ".csv"
    )

    local_LL = pd.DataFrame(local_LL, columns=["FedMeta_" + model_name])
    local_LL.to_csv(
        directory_local
        + "/FedMeta_local_LL_"
        + model_name
        + ".csv"
    )

    local_f1 = pd.DataFrame(local_f1, columns=["FedMeta_" + model_name])
    local_f1.to_csv(
        directory_local
        + "/FedMeta_local_f1_"
        + model_name
        + ".csv"
    )

    local_kappa = pd.DataFrame(local_kappa, columns=["FedMeta_" + model_name])
    local_kappa.to_csv(
        directory_local
        + "/FedMeta_local_kappa_"
        + model_name
        + ".csv"
    )


def main(model_name):
    model_num = args.model_num
    save_file = args.save_file
    noise_param = args.noise_param
    missing_rate = args.missing_rate
    max_workers = args.max_workers

    print(f"Model num: {model_num}, Noise param: {noise_param}")
    fixed_initial_net(args.model_num)
    print(f"Train model: FedMeta_{model_name}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = str(model_num) + "/result/FedMeta_sgd_sgdmom" + model_name +  "_" + save_file
    directory_model = str(model_num) + "/result_model/FedMeta_sgd_sgdmom" + model_name +  "_" + save_file
    directory_local = (
        str(model_num) + "/result_local_all/FedMeta_sgd_sgdmom" + model_name +  "_" + save_file 
    )  # sgd  sgdmom   adam
    directory_other = str(model_num) + "/result_other/FedMeta_sgd_sgdmom" + model_name +  "_" + save_file
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)
    if not os.path.exists(directory_local):
        os.makedirs(directory_local)
    if not os.path.exists(directory_other):
        os.makedirs(directory_other)

    # hyper parameters
    conf_path = "./conf/conf.json"
    with open(conf_path, "r") as f:
        conf = json.load(f)
    beta_num = 5
    extra_feature = 14
    NUM_UNIQUE_CATS = 100
    choices_num = 4
    L1 = args.L1
    L2 = args.L2
    dropout = args.dropout
    inner_lr = args.inner_lr
    outer_lr = args.outer_lr
    epoch = args.epochs

    Meta_net = MetaServer(
        device=device,
        # mode
        centralized=0,
        mode="_fomaml",
        # data split
        test_client_prop=conf["test_client_prop"],
        spt_prop=conf["spt_prop"],
        # model
        model_name=model_name,
        # hyper parameters
        beta_num=beta_num,
        extra_feature=extra_feature,
        NUM_UNIQUE_CATS=NUM_UNIQUE_CATS,
        choices_num=choices_num,
        L1=L1,
        L2=L2,
        dropout=dropout,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        noise_param=noise_param,
        # 性能优化
        max_workers=max_workers,
        # seed
        seed=model_num
    )
    train_test(model_name, Meta_net, epoch, directory, directory_model, directory_other)
    local_train(
        model_name, Meta_net, directory_model, directory_local
    )


if __name__ == "__main__":
    model_name = args.model
    main(model_name)