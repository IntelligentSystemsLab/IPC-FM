# -*- coding: utf-8 -*-
"""
@filename:model_summary.py
@author:Chen Kunxu
@Time:2024/1/20 23:15
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch.autograd import Variable


def get_stds(inv_Hess):
    if isinstance(inv_Hess, float):
        return np.nan
    else:
        stds = [inv_Hess[i][i] ** 0.5 for i in range(inv_Hess.shape[0])]
        return np.array(stds).flatten()


def data_extra_pro(dataset_extra, Label):
    x_0, y_0, z_0 = dataset_extra.shape
    data_extra_change = dataset_extra.reshape(-1, z_0)
    data_extra_change_2 = np.zeros((data_extra_change.shape[0], z_0))
    x_1, y_1 = data_extra_change.shape
    column = len(data_extra_change[1, :])
    for i in range(0, column):
        for j in range(0, x_1):
            data_extra_change_2[j, i] = Label[int(data_extra_change[j, i]), i]
    dataset_extra = data_extra_change_2.reshape(x_0, y_0, z_0)
    return dataset_extra


def model_summary(
    extra_betas_std,
    betas_extra,
    main_betas_std,
    betas_main,
    X_vars_names=None,
    Q_vars_names=None,
):
    if Q_vars_names is None:
        Q_vars_names = []
    if X_vars_names is None:
        X_vars_names = []
    if not isinstance(extra_betas_std, float) and not isinstance(main_betas_std, float):
        z_embs = betas_extra / extra_betas_std
        p_embs = (1 - norm.cdf(abs(z_embs))) * 2
        z_exog = betas_main / main_betas_std
        p_exog = (1 - norm.cdf(abs(z_exog))) * 2
        stats_main = np.array(
            list(zip(X_vars_names, betas_main, main_betas_std, z_exog, p_exog))
        )
        stats_extra = np.array(
            list(zip(Q_vars_names, betas_extra, extra_betas_std, z_embs, p_embs))
        )
        stats_all = np.vstack([stats_main, stats_extra])
        df_stats = pd.DataFrame(
            index=[i[0] for i in stats_all],
            data=np.array(
                [
                    [np.float64(i[1]) for i in stats_all],
                    [np.float64(i[2]) for i in stats_all],
                    [np.float64(i[3]) for i in stats_all],
                    [np.round(np.float64(i[4]), 4) for i in stats_all],
                ]
            ).T,
            columns=["Betas", "St errors", "t-stat", "p-value"],
        )
        return df_stats
    else:
        return np.nan


def cats2ints(Q_df_train, Label):
    cat2index = {}
    x, y = Label.shape
    for i in range(x):
        for j in range(y):
            if Q_df_train.iloc[i, j] != "nan":
                cat2index[Q_df_train.iloc[i, j]] = Label[i][j]
    return cat2index


def create_index(alfabet):  # alphabet-->number of unique categories

    index2alfa = {}
    alfa2index = {}

    for i in range(len(alfabet)):
        index2alfa[i] = alfabet[i]
        alfa2index[alfabet[i]] = i
    return index2alfa, alfa2index


def get_betas_and_embeddings(trained_model, Q_df_train):
    UNIQUE_CATS = []
    x, y = Q_df_train.shape
    for i in range(x):
        for j in range(y):
            if Q_df_train.iloc[i, j] != "nan":
                UNIQUE_CATS.append(Q_df_train.iloc[i, j])
    DICT = {}
    DICT["index2alfa_from"], DICT["alfa2index_from"] = create_index(UNIQUE_CATS)
    DICT["index2alfa_from"], DICT["alfa2index_from"] = create_index(UNIQUE_CATS)
    betas_extra = (
        trained_model.extra_model.conv1.weight.reshape(-1).cpu().detach().numpy()
    )
    betas_main = trained_model.main_model.weight.reshape(-1).cpu().detach().numpy()
    embeddings = (
        trained_model.extra_model.src_learning_emb.weight.cpu().detach().numpy()
    )

    DICT["embeddings"] = embeddings
    DICT["betas_extra"] = betas_extra
    DICT["betas_main"] = betas_main

    return DICT
